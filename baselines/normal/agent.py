"""Run the generic LangChain ReAct normal baseline against one benchmark instance."""

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


make_tools = _load_file("normal_tools", Path(__file__).with_name("tools.py")).make_tools
_trace = _load_file("baseline_trace", Path(__file__).with_name("agent_trace.py"))
TraceHTTPClient, TraceRecorder = _trace.TraceHTTPClient, _trace.TraceRecorder
benchmark_toolchain = _load_file("rust_environment", Path(__file__).with_name("rust_environment.py")).benchmark_toolchain


SYSTEM_PROMPT = """You are a software engineer resolving an issue in a repository.
Use the available tools to search and read files, edit code to fix the issue, run commands or tests via run_command to verify your changes, and call generate_patch to submit your final patch.
Do not include temporary test files or unrelated changes in the final submitted patch."""


def _compress_history_messages(messages, keep_recent_steps=6, block_size=2):
    """Compress verbose tool outputs and large edit calls from older turns to reduce prompt token bloat."""
    if not isinstance(messages, list) or len(messages) <= keep_recent_steps * 2:
        return messages

    tool_indices = [
        i for i, m in enumerate(messages)
        if getattr(m, "type", None) == "tool" or (isinstance(m, dict) and m.get("role") == "tool")
    ]
    if not tool_indices:
        return messages

    total_tools = len(tool_indices)
    raw_code_cutoff = max(0, total_tools - min(keep_recent_steps, total_tools))
    code_pos = (raw_code_cutoff // block_size) * block_size if block_size > 1 else raw_code_cutoff
    code_cutoff_idx = tool_indices[code_pos] if code_pos < total_tools else len(messages)

    def _trim_arg_str(val):
        if not isinstance(val, str) or len(val) <= 400 or "[earlier write omitted" in val:
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
                    if name in ("write_file", "edit_file") and isinstance(args, dict):
                        args_copy = dict(args)
                        for k in ("content", "old_text", "new_text"):
                            new_val, changed = _trim_arg_str(args_copy.get(k))
                            if changed:
                                args_copy[k] = new_val
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
            elif content_changed:
                if hasattr(msg, "model_copy"):
                    msg = msg.model_copy(update={"content": ai_content})
                elif isinstance(msg, dict):
                    msg = dict(msg)
                    msg["content"] = ai_content

        if not is_tool:
            compressed.append(msg)
            continue

        content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
        if not isinstance(content, str) or len(content) <= 300:
            compressed.append(msg)
            continue

        new_content = None
        if i < code_cutoff_idx and "[omitted " not in content and "[earlier output truncated]" not in content:
            lines = content.splitlines()
            if len(lines) > 10:
                head = "\n".join(lines[:5])
                tail = "\n".join(lines[-3:])
                omitted = len(lines) - 8
                new_content = f"{head}\n... [omitted {omitted} intermediate lines of earlier output] ...\n{tail}"
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
        "The repair patch is not submitted yet. Continue resolving the issue "
        "and call generate_patch(paths=[...]) with the files to include in the final patch."
    )
    if hasattr(state, "changed_paths"):
        changed = [p.as_posix() for p in state.changed_paths()]
        if changed:
            prompt += f" Currently modified files: {changed}."
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
            compress_fn = functools.partial(_compress_history_messages, keep_recent_steps=6, block_size=2)
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
                            state.auto_submit_patch() if hasattr(state, "auto_submit_patch") else ""
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

                last_msg = result["messages"][-1]
                has_tool_call = bool(getattr(last_msg, "tool_calls", None) or (isinstance(last_msg, dict) and last_msg.get("tool_calls")))
                if not has_tool_call:
                    consecutive_no_tools += 1
                    if consecutive_no_tools >= 4:
                        if hasattr(state, "auto_submit_patch"):
                            fallback = state.auto_submit_patch()
                            if fallback:
                                if trace:
                                    trace.record("auto_submit_on_idle", patch=fallback)
                                return fallback
                        raise RuntimeError("Agent stopped calling tools without submitting a patch")
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
    repro_group.add_argument("--require-reproduction", dest="require_reproduction", action="store_true", default=False,
                             help="Compatibility flag (no-op in generic normal baseline) [default: disabled]")
    repro_group.add_argument("--no-require-reproduction", dest="require_reproduction", action="store_false",
                             help="Compatibility flag (no-op in generic normal baseline)")
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
        patch = run_agent(repo, issue, model, api_key, base_url,
                          trace=trace, max_api_calls=max_api_calls, rust_version=rust_version,
                          require_reproduction=args.require_reproduction, instance_id=args.instance_id)
        trace.record("patch", patch=patch)
        stage = "prediction"
        prediction = {"instance_id": args.instance_id, "model_name_or_path": f"normal__{model}",
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
