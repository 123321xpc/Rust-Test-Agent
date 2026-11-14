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
TraceHTTPClient, TraceRecorder = _trace.TraceHTTPClient, _trace.TraceRecorder
benchmark_toolchain = _load_file("rust_environment", Path(__file__).with_name("rust_environment.py")).benchmark_toolchain


SYSTEM_PROMPT = """You are a software engineer resolving an issue in a Rust repository.
Follow this order:
1. Analyze the issue and relevant code. Identify the cause, expected behavior,
   and callers/trait constraints using search_code (with context_lines when helpful,
   or summarize=True for a file overview, or empty query with file_glob to locate files),
   rust_symbols (including callers=True / references=True, inspecting vis and enclosing trait/impl),
   read_lines, and cargo_workspace. Keep initial exploration focused (typically within 10-15
   read/search calls) before writing the regression test.
2. Use write_regression_test to create a new temporary integration test in the
   relevant package's tests/test_regression_<name>.rs. You may set run_now=True to
   immediately execute cargo_test on the written test in one step. Reproduce the issue via
   the public API (or for binary-only CLI crates, via `env!("CARGO_BIN_EXE_<bin>")`)
   and cover boundary cases described in the issue. Run cargo_test with test_target
   set to the regression test before fixing to confirm reproduction (Red phase:
   runtime failure on buggy code, or expected API-missing compile failure for new
   APIs). Do not add or modify inline tests or other test files; never modify files in tests/ or benches/ (modifications to existing test files are blocked to prevent benchmark conflicts).
3. Implement a targeted fix using edit_file (single replacement or edits=[...]
   for multiple hunks), preserving unrelated code. Before changing any function
   signature or trait bound, check callers with rust_symbols(callers=True). Use
   checkpoint(action="save"/"restore") before risky refactors and revert_file to
   cleanly discard unwanted changes. If a strategy_warning appears, step back and
   change approach instead of repeating similar local tweaks.
4. Use layered verification after edits: first run cargo_check (or cargo_test
   with no_run=True to check test target compilation), then run cargo_test with
   test_target set to each regression test (no filter). Then run targeted existing
   tests (lib=True with test_filter or relevant test_target); run the full package
   suite (tests=True) only when fast enough, avoiding repeated timeouts on huge
   crates. Each temporary regression test must pass with at least one test executed
   and 0 ignored. Existing tests may fail; report their failures, but do not
   modify or remove them to satisfy the gate. If the issue states an existing test
   asserts outdated behavior contrary to the issue, do not weaken your fix to pass it.
   When modifying public APIs, types, or doc comments, you may also run cargo_test with doc=True to verify documentation examples.
5. After verification passes, use generate_patch to preview changes, or call
   generate_patch(submit=True) to automatically submit all verified source changes in one call.
   Alternatively, submit explicitly with paths=[...]. Temporary
   regression tests are excluded automatically, and a post-strip verification
   checks affected crates and downstream workspace dependents under both cargo check and cargo check --tests.
   Any further write invalidates regression verification: rerun the required tests. Watch for budget
   notices and submit before API calls run out.
Briefly describe the analysis, tests, and change.
Do not run git or external shell commands to inspect history or host files.
Focus strictly on reproducing and fixing the issue using the crate's Rust API
(or `CARGO_BIN_EXE_<bin>` for binary crates). The Rust toolchain is configured by
the runner; do not change repository toolchain files merely to configure the environment.
"""



def _compress_history_messages(messages, keep_recent_steps=6, test_cutoff_steps=2, block_size=2):
    """Compress verbose tool outputs and large edit calls from older turns to reduce prompt token bloat.

    Uses block-aligned retention to preserve LLM prompt prefix caching:
    - Recent tool turns (last keep_recent_steps) retain code reads (read_lines, search_code) in full fidelity.
    - Older cargo_test and cargo_check JSON payloads (> test_cutoff_steps old) are compacted to essential status/counts.
    - Older cargo_workspace and rust_symbols outputs are compacted to outline/member summaries.
    - Large write_file/edit_file tool arguments (> 800 chars) in older assistant messages are truncated.
    - Code reads older than keep_recent_steps have excessive middle lines compacted.
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

    tc_map = {}
    file_modified_indices = {}
    for idx, m in enumerate(messages):
        is_ai_m = getattr(m, "type", None) == "ai" or (isinstance(m, dict) and m.get("role") == "assistant")
        if is_ai_m:
            t_calls = getattr(m, "tool_calls", None) if not isinstance(m, dict) else m.get("tool_calls")
            if isinstance(t_calls, list):
                for tc in t_calls:
                    if isinstance(tc, dict) and tc.get("id"):
                        tc_map[tc["id"]] = tc
                        t_name = tc.get("name")
                        t_args = tc.get("args") or {}
                        if t_name in ("edit_file", "write_file", "write_regression_test") and isinstance(t_args, dict):
                            p = t_args.get("path")
                            if p:
                                file_modified_indices.setdefault(str(p), []).append(idx)

    if block_size > 1:
        code_pos = (raw_code_cutoff // block_size) * block_size
        test_pos = (raw_test_cutoff // block_size) * block_size
    else:
        code_pos = raw_code_cutoff
        test_pos = raw_test_cutoff

    code_cutoff_idx = tool_indices[code_pos] if code_pos < total_tools else len(messages)
    test_cutoff_idx = tool_indices[test_pos] if test_pos < total_tools else len(messages)

    def _trim_arg_str(val):
        if not isinstance(val, str) or len(val) <= 400 or "[earlier write omitted]" in val:
            return val, False
        lines = val.splitlines()
        if len(lines) > 8:
            head = "\n".join(lines[:4])
            tail = "\n".join(lines[-3:])
            return f"{head}\n... [earlier write omitted ({len(lines) - 7} lines)] ...\n{tail}", True
        return val[:250] + "\n... [earlier write omitted] ...", True

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
                updated_tc = []
                tc_changed = False
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        updated_tc.append(tc)
                        continue
                    name = tc.get("name")
                    args = tc.get("args")
                    if name in ("write_file", "write_regression_test", "edit_file") and isinstance(args, dict):
                        args_copy = dict(args)
                        for k in ("content", "old_text", "new_text"):
                            new_val, changed = _trim_arg_str(args_copy.get(k))
                            if changed:
                                args_copy[k] = new_val
                                tc_changed = True
                        if isinstance(args_copy.get("edits"), list):
                            new_edits = []
                            edits_changed = False
                            for item in args_copy["edits"]:
                                if isinstance(item, dict):
                                    item_copy = dict(item)
                                    for ek in ("old_text", "new_text"):
                                        nv, ch = _trim_arg_str(item_copy.get(ek))
                                        if ch:
                                            item_copy[ek] = nv
                                            edits_changed = True
                                    new_edits.append(item_copy)
                                else:
                                    new_edits.append(item)
                            if edits_changed:
                                args_copy["edits"] = new_edits
                                tc_changed = True
                        updated_tc.append({**tc, "args": args_copy})
                    else:
                        updated_tc.append(tc)
                if content_changed or tc_changed:
                    updates = {}
                    if content_changed:
                        updates["content"] = ai_content
                    if tc_changed:
                        updates["tool_calls"] = updated_tc
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

        # If file was modified in a later turn, older read outputs are stale
        t_id = getattr(msg, "tool_call_id", "") or (msg.get("tool_call_id") if isinstance(msg, dict) else "")
        tc_info = tc_map.get(t_id, {})
        if tc_info.get("name") in ("read_lines", "read_file"):
            target_p = str((tc_info.get("args") or {}).get("path") or "")
            mod_indices = file_modified_indices.get(target_p, [])
            if any(mod_idx > i for mod_idx in mod_indices):
                new_content = f"[earlier read of '{target_p}' omitted: file was modified in subsequent step; line numbers may have shifted]"

        if new_content is None and len(content) <= 300:
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
                if "[omitted " in content or "[earlier output truncated]" in content:
                    new_content = content
                else:
                    lines = content.splitlines()
                    if len(lines) > 10:
                        head = "\n".join(lines[:5])
                        tail = "\n".join(lines[-3:])
                        omitted = len(lines) - 8
                        new_content = f"{head}\n... [omitted {omitted} intermediate lines of earlier reading] ...\n{tail}"
                    elif len(content) > 800:
                        new_content = content[:500] + "\n... [earlier output truncated] ..."

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


def _continuation_prompt(state, client):
    prompt = (
        "The repair is not submitted. Continue the required workflow: "
        "write a regression test, fix the issue, and pass each temporary regression "
        "test before submitting the patch. Report existing test failures; they do "
        "not block submission. Investigate if regression tests cannot run."
    )
    if getattr(state, "_last_submit_error", ""):
        prompt += f" Previous submission attempt failed verification: {state._last_submit_error}."
    if hasattr(state, "verified") and hasattr(state, "changed_paths"):
        regression_tests = getattr(state, "regression_tests", set())
        source_changes = [p.as_posix() for p in state.changed_paths() if p not in regression_tests]
        if state.verified() and source_changes:
            prompt += f" Verification is already complete—call generate_patch with paths={source_changes} now."
        elif hasattr(state, "missing_verification_steps"):
            missing = state.missing_verification_steps()
            if missing:
                prompt += f" Pending verification: {missing}."
    remaining = getattr(client, "remaining_calls", None)
    if isinstance(remaining, int):
        prompt += f" Remaining API calls: {remaining}."
    return prompt


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
            llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url,
                             temperature=0, timeout=180, max_retries=3, http_client=client)
            import functools
            compress_fn = functools.partial(_compress_history_messages, keep_recent_steps=6, test_cutoff_steps=2, block_size=2)
            if hasattr(llm, "bind_tools") and callable(getattr(llm, "bind_tools")):
                from langchain_core.runnables import RunnableLambda
                orig_bind_tools = llm.bind_tools
                def _bind_tools_with_compression(*args, **kwargs):
                    bound = orig_bind_tools(*args, **kwargs)
                    return RunnableLambda(compress_fn) | bound
                object.__setattr__(llm, "bind_tools", _bind_tools_with_compression)
            agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
            config = {"recursion_limit": sys.maxsize}
            if trace:
                config["callbacks"] = [trace]
            messages = [{"role": "user", "content": issue}]
            consecutive_no_tools = 0
            while True:
                try:
                    result = agent.invoke({"messages": messages}, config=config)
                except RuntimeError as exc:
                    if "Model API call limit reached" in str(exc):
                        fallback = getattr(state, "patch", "") or (
                            state.auto_submit_verified_patch() if hasattr(state, "auto_submit_verified_patch") else ""
                        )
                        if fallback:
                            if trace:
                                trace.record("auto_submit_on_limit", patch=fallback)
                            return fallback
                    raise
                if state.patch:
                    break
                if not isinstance(result, dict) or not isinstance(result.get("messages"), list):
                    raise RuntimeError("Agent did not generate a non-empty patch")
                # Auto-submit if current revision is verified and model stopped without explicit generate_patch
                if hasattr(state, "auto_submit_verified_patch"):
                    try:
                        fallback = state.auto_submit_verified_patch(allow_snapshot_fallback=False)
                    except TypeError:
                        fallback = state.auto_submit_verified_patch()
                    if fallback:
                        if trace:
                            trace.record("auto_submit_verified", patch=fallback)
                        return fallback

                # Detect if model is returning plain text repeatedly without calling tools
                last_msg = result["messages"][-1]
                has_tool_call = bool(getattr(last_msg, "tool_calls", None) or (isinstance(last_msg, dict) and last_msg.get("tool_calls")))
                if not has_tool_call:
                    consecutive_no_tools += 1
                    if consecutive_no_tools >= 4:
                        if hasattr(state, "auto_submit_verified_patch"):
                            fallback = state.auto_submit_verified_patch()
                            if fallback:
                                if trace:
                                    trace.record("auto_submit_on_idle", patch=fallback)
                                return fallback
                        raise RuntimeError("Agent stopped calling tools without submitting a verified patch")
                else:
                    consecutive_no_tools = 0

                messages = compress_fn(result["messages"]) + [{"role": "user", "content": _continuation_prompt(state, client)}]
        finally:
            client.close()
        if not state.patch:
            raise RuntimeError("Agent did not generate a non-empty patch")
        return state.patch
    finally:
        state.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Instance repository checked out at its base commit")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--issue-file", required=True, help="UTF-8 file containing the issue description")
    parser.add_argument("--output", required=True, help="Prediction JSONL file (appends one record)")
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
    model = os.getenv("MODEL")
    api_key = os.getenv("API_KEY")
    if not model or not api_key:
        parser.error("set MODEL and API_KEY in the project-root .env")
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
        patch = run_agent(repo, issue, model, api_key, os.getenv("BASE_URL") or None,
                          trace=trace, max_api_calls=max_api_calls, rust_version=rust_version,
                          require_reproduction=args.require_reproduction, instance_id=args.instance_id)
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
