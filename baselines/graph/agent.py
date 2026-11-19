"""Run the minimal LangChain test baseline against one benchmark instance."""

import argparse
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

NETWORK_PROXY = "http://10.70.254.202:11321"


def configure_repair_git():
    git_dir = Path.home() / "xpc" / "git-2.43.0" / "bin"
    if (git_dir / "git").is_file():
        os.environ["PATH"] = os.pathsep.join((str(git_dir), os.environ.get("PATH", "")))


def configure_network_proxy():
    """Route model and GitHub requests through the configured LAN proxy."""
    values = {
        "HTTP_PROXY": NETWORK_PROXY,
        "HTTPS_PROXY": NETWORK_PROXY,
        "ALL_PROXY": NETWORK_PROXY,
        "FTP_PROXY": NETWORK_PROXY,
        "NO_PROXY": "localhost,127.0.0.1,::1",
    }
    for name, value in values.items():
        os.environ.setdefault(name, value)
        os.environ.setdefault(name.lower(), value)

def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_tools = _load_file("test_tools", Path(__file__).with_name("tools.py")).make_tools
_trace = _load_file("baseline_trace", Path(__file__).with_name("agent_trace.py"))
TraceHTTPClient, TraceRecorder, RoleTraceCallback = _trace.TraceHTTPClient, _trace.TraceRecorder, _trace.RoleTraceCallback
benchmark_toolchain = _load_file("rust_environment", Path(__file__).with_name("rust_environment.py")).benchmark_toolchain
_prompts = _load_file("agent_prompts", Path(__file__).with_name("agent_prompts.py"))
_multi_agent = _load_file("multi_agent", Path(__file__).with_name("multi_agent.py"))


def _norm_rel_path(path_val) -> str:
    if not path_val:
        return ""
    norm = str(path_val).strip().replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _compress_history_messages(messages, keep_recent_steps=6, test_cutoff_steps=2, block_size=2):
    """Compress verbose tool outputs and large edit calls from older turns to reduce prompt token bloat.

    Uses block-aligned retention to preserve LLM prompt prefix caching:
    - Recent tool turns (last keep_recent_steps) retain code reads (read_lines, search_code) in full fidelity.
    - Older reads invalidated by subsequent edit_file/write_file/revert_file/checkpoint(restore) are folded immediately.
    - Older cargo_test and cargo_check JSON payloads (> test_cutoff_steps old) are compacted to essential status/counts.
    - Older cargo_workspace and rust_symbols outputs are compacted to outline/member summaries.
    - Older non-JSON tool outputs (> keep_recent_steps old) are folded to a single summary line.
    """
    if not isinstance(messages, list) or len(messages) <= test_cutoff_steps * 2:
        return messages

    tool_indices = [
        i for i, m in enumerate(messages)
        if getattr(m, "type", None) == "tool" or (isinstance(m, dict) and m.get("role") == "tool")
    ]
    if not tool_indices:
        return messages

    total_tools = len(tool_indices)
    raw_code_cutoff = max(0, total_tools - min(keep_recent_steps, total_tools))
    raw_test_cutoff = max(0, total_tools - min(test_cutoff_steps, total_tools))

    tool_results_by_id = {}
    for idx in tool_indices:
        tm = messages[idx]
        tm_id = getattr(tm, "tool_call_id", "") or (tm.get("tool_call_id") if isinstance(tm, dict) else "")
        if tm_id:
            tm_content = getattr(tm, "content", "") if not isinstance(tm, dict) else tm.get("content", "")
            tool_results_by_id[tm_id] = tm_content if isinstance(tm_content, str) else ""

    def _mutation_succeeded(res_text: str | None) -> bool:
        if res_text is None:
            return False
        head = res_text.lstrip()
        if head.startswith(("Error", "No changes to revert")):
            return False
        if head.startswith("[earlier ") and "error omitted" in head[:80]:
            return False
        return True

    tc_map = {}
    file_modified_indices = {}
    last_restore_idx = -1
    for idx, m in enumerate(messages):
        is_ai_m = getattr(m, "type", None) == "ai" or (isinstance(m, dict) and m.get("role") == "assistant")
        if is_ai_m:
            t_calls = getattr(m, "tool_calls", None) if not isinstance(m, dict) else m.get("tool_calls")
            if isinstance(t_calls, list):
                for tc in t_calls:
                    if isinstance(tc, dict) and tc.get("id"):
                        tc_id = tc["id"]
                        tc_map[tc_id] = tc
                        t_name = tc.get("name")
                        t_args = tc.get("args") or {}
                        if isinstance(t_args, dict) and _mutation_succeeded(tool_results_by_id.get(tc_id)):
                            if t_name in ("edit_file", "write_file", "write_regression_test", "revert_file"):
                                p = _norm_rel_path(t_args.get("target_src") or t_args.get("path"))
                                if p:
                                    file_modified_indices.setdefault(p, []).append(idx)
                            elif t_name == "checkpoint" and str(t_args.get("action", "")).strip().lower() == "restore":
                                last_restore_idx = idx

    if block_size > 1:
        code_pos = (raw_code_cutoff // block_size) * block_size
        test_pos = (raw_test_cutoff // block_size) * block_size
    else:
        code_pos = raw_code_cutoff
        test_pos = raw_test_cutoff

    code_cutoff_idx = tool_indices[code_pos] if code_pos < total_tools else len(messages)
    test_cutoff_idx = tool_indices[test_pos] if test_pos < total_tools else len(messages)

    def _trim_arg_str(val):
        if not isinstance(val, str) or len(val) <= 400 or "[earlier write omitted" in val:
            return val, False
        lines = val.splitlines()
        return (
            f"[earlier write omitted: {len(lines)} lines, {len(val)} chars — "
            "do not copy this placeholder; read the file directly if needed]",
            True,
        )

    compressed = []
    for i, msg in enumerate(messages):
        is_tool = getattr(msg, "type", None) == "tool" or (isinstance(msg, dict) and msg.get("role") == "tool")
        is_ai = getattr(msg, "type", None) == "ai" or (isinstance(msg, dict) and msg.get("role") == "assistant")

        # Compact older assistant reasoning text and write/edit tool arguments
        if is_ai and i < code_cutoff_idx:
            ai_content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            content_changed = False
            if isinstance(ai_content, str) and len(ai_content) > 350 and "[earlier thought omitted" not in ai_content:
                lines = ai_content.splitlines()
                if len(lines) > 6:
                    head = "\n".join(lines[:3])
                    tail = "\n".join(lines[-2:])
                    ai_content = f"{head}\n... [earlier thought omitted ({len(lines) - 5} lines)] ...\n{tail}"
                else:
                    ai_content = ai_content[:180] + "\n... [earlier thought omitted] ...\n" + ai_content[-80:]
                content_changed = True

            tool_calls = getattr(msg, "tool_calls", None) if not isinstance(msg, dict) else msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                updated_tc = tool_calls
                if content_changed:
                    updates = {}
                    updates["content"] = ai_content
                    if hasattr(msg, "model_copy"):
                        msg = msg.model_copy(update=updates)
                    elif isinstance(msg, dict):
                        msg = dict(msg)
                        msg.update(updates)

        if not is_tool:
            compressed.append(msg)
            continue

        content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
        if not isinstance(content, str):
            compressed.append(msg)
            continue

        stripped = content.strip()
        new_content = None

        # If file was modified or checkpoint restored in a later turn, older read outputs are stale
        t_id = getattr(msg, "tool_call_id", "") or (msg.get("tool_call_id") if isinstance(msg, dict) else "")
        tc_info = tc_map.get(t_id, {})
        tool_name = tc_info.get("name") or (getattr(msg, "name", None) if not isinstance(msg, dict) else msg.get("name")) or "tool"
        if tool_name in ("read_lines", "read_file"):
            target_p = _norm_rel_path((tc_info.get("args") or {}).get("path"))
            mod_indices = file_modified_indices.get(target_p, [])
            if last_restore_idx > i or any(mod_idx > i for mod_idx in mod_indices):
                new_content = f"[earlier read of '{target_p}' omitted: file was modified in subsequent step; line numbers may have shifted]"

        if new_content is None and len(content) <= 300 and not (i < code_cutoff_idx and content.startswith("Error ") and "\n" in content):
            compressed.append(msg)
            continue

        if i < test_cutoff_idx and stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    if str(data.get("note", "")).startswith("[earlier"):
                        new_content = content
                    elif "status" in data:
                        err_codes = []
                        for d in data.get("diagnostics", []):
                            if isinstance(d, dict) and d.get("level") == "error" and d.get("code"):
                                if d["code"] not in err_codes:
                                    err_codes.append(d["code"])
                        payload = {
                            "status": data.get("status"),
                            "tests_executed": data.get("tests_executed", 0),
                            "tests_ignored": data.get("tests_ignored", 0),
                            "failed_tests": data.get("failed_tests", [])[:20],
                            "note": "[earlier test output compacted]",
                        }
                        if data.get("warning"):
                            payload["warning"] = data["warning"]
                        if err_codes:
                            payload["compiler_errors"] = err_codes[:5]
                        new_content = json.dumps(payload, ensure_ascii=False)
                    elif "exit_code" in data and "diagnostics" in data:
                        diags = data.get("diagnostics", [])
                        errs = [d for d in diags if isinstance(d, dict) and d.get("level") == "error"]
                        err_summary = []
                        for d in errs[:5]:
                            spans = d.get("spans") or []
                            first_span = spans[0] if spans else {}
                            err_summary.append({
                                "code": d.get("code"),
                                "category": d.get("category"),
                                "file": first_span.get("file_name"),
                                "line": first_span.get("line_start"),
                                "msg": (d.get("message") or "")[:120],
                            })
                        new_content = json.dumps({
                            "exit_code": data.get("exit_code"),
                            "error_count": len(errs),
                            "warning_count": len([d for d in diags if isinstance(d, dict) and d.get("level") == "warning"]),
                            "top_errors": err_summary,
                            "note": "[earlier cargo_check output compacted]",
                        }, ensure_ascii=False)
            except Exception:
                pass


        if new_content is None and i < code_cutoff_idx:
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        if str(data.get("note", "")).startswith("[earlier"):
                            new_content = content
                        elif "workspace_members" in data:
                            members = [
                                {"name": m.get("name"), "manifest_path": m.get("manifest_path")}
                                for m in data.get("workspace_members", []) if isinstance(m, dict)
                            ]
                            new_content = json.dumps({
                                "workspace_root": data.get("workspace_root"),
                                "workspace_members": members,
                                "note": "[earlier cargo_workspace targets compacted]",
                            }, ensure_ascii=False)
                        elif "results" in data or "outline" in data:
                            items = data.get("results") or data.get("outline") or []
                            slim = [
                                {"path": x.get("path"), "line": x.get("line"), "kind": x.get("kind"), "name": x.get("name")}
                                for x in items[:8] if isinstance(x, dict)
                            ]
                            new_content = json.dumps({
                                "count": len(items),
                                "top_matches": slim,
                                "truncated": data.get("truncated", False),
                                "note": "[earlier rust_symbols output compacted]",
                            }, ensure_ascii=False)
                except Exception:
                    pass

            if new_content is None:
                if content.startswith("[earlier ") or "[omitted " in content or "[earlier output truncated]" in content:
                    new_content = content
                else:
                    lines = content.splitlines()
                    first_line = next((l.strip() for l in lines if l.strip()), "")[:100]
                    if content.startswith(("Error editing ", "Error writing ", "Error reading ", "Error searching ")):
                        new_content = f"[earlier {tool_name} error omitted: {first_line}]"
                    else:
                        new_content = f"[earlier {tool_name} output omitted ({len(lines)} lines): {first_line}]"

        if new_content is not None and new_content != content:
            if hasattr(msg, "model_copy"):
                compressed.append(msg.model_copy(update={"content": new_content}))
            elif hasattr(msg, "copy"):
                compressed.append(msg.copy(update={"content": new_content}))
            elif isinstance(msg, dict):
                new_dict = dict(msg)
                new_dict["content"] = new_content
                compressed.append(new_dict)
            else:
                from langchain_core.messages import ToolMessage
                compressed.append(ToolMessage(
                    content=new_content,
                    tool_call_id=getattr(msg, "tool_call_id", ""),
                    name=getattr(msg, "name", None),
                ))
        else:
            compressed.append(msg)

    return compressed


def run_agent(repo, issue, model, api_key, base_url=None, trace=None, max_api_calls=70, rust_version=None, require_reproduction=False, instance_id=None):
    configure_repair_git()
    kwargs = {}
    if instance_id is not None:
        kwargs["instance_id"] = instance_id
    tools, state = (
        make_tools(Path(repo), rust_version, **kwargs)
        if rust_version is not None
        else make_tools(Path(repo), **kwargs)
    )
    if require_reproduction and hasattr(state, "require_reproduction"):
        state.require_reproduction = True
    try:
        if rust_version is not None:
            info = state.rust_info
            if trace:
                trace.record("rust_environment", **info)
            issue = (f"Evaluation Rust environment: {info['rustc']}; {info['cargo']} "
                     f"(source: {info['source']}).\n\n{issue}")
        client = TraceHTTPClient(trace, max_calls=max_api_calls)
        if hasattr(state, "set_budget"):
            state.set_budget(client)
        try:
            import functools
            compress_fn = functools.partial(_compress_history_messages, keep_recent_steps=6, test_cutoff_steps=2, block_size=2)
            repair_runner = _multi_agent.run_graph_agent
            patch = repair_runner(
                Path(repo),
                issue,
                model,
                api_key,
                base_url,
                tools=tools,
                state=state,
                client=client,
                trace=trace,
                max_api_calls=max_api_calls,
                compress_fn=compress_fn,
                create_agent_fn=create_agent,
                chat_model_factory=ChatOpenAI,
                prompts_mod=_prompts,
                role_callback_cls=RoleTraceCallback,
            )
            if not patch:
                patch = _multi_agent._relaxed_source_patch(state)
            if not patch:
                raise RuntimeError("Agent did not generate a non-empty patch")
            return patch
        finally:
            client.close()
    finally:
        state.close()


def _resolve_api_settings(api=None, model_override=None):
    if model_override is not None and not model_override.strip():
        return None, None, None, "--model must not be empty"
    model_val = model_override.strip() if model_override is not None else None
    if api is not None:
        prefix = api.upper()
        base_url = os.getenv(f"{prefix}_BASE_URL") or None
        api_key = os.getenv(f"{prefix}_API_KEY")
        model = model_val or os.getenv(f"{prefix}_MODEL")
        if not base_url or not api_key or not model:
            return None, None, None, f"set {prefix}_BASE_URL, {prefix}_API_KEY, and {prefix}_MODEL (or --model) in the project-root .env"
        return model, api_key, base_url, None
    api_key = os.getenv("API_KEY")
    model = model_val or os.getenv("MODEL")
    if api_key and model:
        return model, api_key, os.getenv("BASE_URL") or None, None
    base_url = os.getenv("YJ_BASE_URL") or None
    api_key = os.getenv("YJ_API_KEY")
    model = model_val or os.getenv("YJ_MODEL")
    if base_url and api_key and model:
        return model, api_key, base_url, None
    return None, None, None, "set YJ_*/DS_* (or MODEL and API_KEY) in the project-root .env"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Instance repository checked out at its base commit")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--issue-file", required=True, help="UTF-8 file containing the issue description")
    parser.add_argument("--output", required=True, help="Prediction JSONL file (appends one record)")
    parser.add_argument("--api", type=str.lower, choices=("yj", "ds"), help="API interface from .env to use (default: yj)")
    parser.add_argument("--model", help="Override the default model for the selected API interface")
    parser.add_argument("--api-limit", type=int, default=None, help="Override API_LIMIT from .env (default: 70)")
    parser.add_argument("--max-api-calls", type=int, default=None, help="Optional lower per-task limit for model HTTP calls, including retries")
    parser.add_argument("--trace-dir", help="Directory for per-run JSONL traces (default: traces beside output)")
    repro_group = parser.add_mutually_exclusive_group()
    repro_group.add_argument("--require-reproduction", dest="require_reproduction", action="store_true", default=True,
                             help="Require regression test failure before fix (paired pre-fix verification) [default: enabled]")
    repro_group.add_argument("--no-require-reproduction", dest="require_reproduction", action="store_false",
                             help="Disable pre-fix reproduction check")
    rust = parser.add_mutually_exclusive_group(required=True)
    rust.add_argument("--dataset", help="Benchmark JSONL containing the selected instance")
    rust.add_argument("--rust-toolchain", help="Explicit evaluation toolchain when dataset metadata is unavailable")
    parser.add_argument("--harness-root", default=str(Path(__file__).resolve().parents[2] / "Rust-bench"),
                        help="Rust-bench harness used for evaluation")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    configure_network_proxy()
    model, api_key, base_url, api_err = _resolve_api_settings(args.api, args.model)
    if api_err:
        parser.error(api_err)
    if args.api_limit is not None:
        if args.api_limit < 1:
            parser.error("--api-limit must be a positive integer")
        api_limit = args.api_limit
    else:
        raw_limit = os.getenv("API_LIMIT")
        try:
            api_limit = 70 if raw_limit is None else int(raw_limit)
            if api_limit < 1:
                raise ValueError
        except ValueError:
            parser.error("API_LIMIT must be a positive integer")
    if args.max_api_calls is not None and args.max_api_calls < 1:
        parser.error("--max-api-calls must be positive")
    max_api_calls = min(api_limit, args.max_api_calls) if args.max_api_calls is not None else api_limit
    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        parser.error("--repo must be an existing directory")
    issue = Path(args.issue_file).read_text(encoding="utf-8")
    output = Path(args.output)
    trace_dir = Path(args.trace_dir) if args.trace_dir else output.parent / "traces"
    trace = TraceRecorder(trace_dir / f"{uuid.uuid4().hex}.jsonl", secrets=[api_key])
    trace.record("run_start", instance_id=args.instance_id, model=model, repo=str(repo), issue=issue)
    print(f"Trace: {trace.path}", flush=True)
    started = time.perf_counter()
    status = "failed"
    stage = "agent"
    try:
        rust_version = (benchmark_toolchain(args.dataset, args.instance_id, args.harness_root)
                        if args.dataset else args.rust_toolchain)
        run_kwargs = {
            "trace": trace,
            "max_api_calls": max_api_calls,
            "rust_version": rust_version,
            "require_reproduction": args.require_reproduction,
            "instance_id": args.instance_id,
        }
        patch = run_agent(repo, issue, model, api_key, base_url, **run_kwargs)
        trace.record("patch", patch=patch)
        stage = "prediction"
        prediction = {"instance_id": args.instance_id, "model_name_or_path": f"test__{model}",
                      "model_patch": patch.replace("\r\n", "\n")}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        print(f"Wrote prediction for {args.instance_id} to {output}")
        status = "completed"
    except BaseException as exc:
        trace.record("run_error", stage=stage, error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        trace.record("agent_metrics", status=status, elapsed_seconds=time.perf_counter() - started,
                     **trace.metrics())
    trace.record("run_end", status="completed")


if __name__ == "__main__":
    main()
