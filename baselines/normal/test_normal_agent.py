"""Offline contract tests for the generic LangChain normal baseline runner."""
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def agent_module(monkeypatch):
    baseline_dir = Path(__file__).resolve().parent
    monkeypatch.syspath_prepend(str(baseline_dir))
    spec = importlib.util.spec_from_file_location("normal_baseline_agent", baseline_dir / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_prefers_xpc_git(agent_module, monkeypatch, tmp_path):
    git_dir = tmp_path / "xpc" / "git-2.43.0" / "bin"
    git_dir.mkdir(parents=True)
    (git_dir / "git").touch()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PATH", "existing-path")

    agent_module.configure_repair_git()

    assert os.environ["PATH"] == os.pathsep.join((str(git_dir), "existing-path"))


def test_repair_keeps_path_when_xpc_git_is_missing(agent_module, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PATH", "existing-path")

    agent_module.configure_repair_git()

    assert os.environ["PATH"] == "existing-path"


def test_run_agent_returns_tool_submission(agent_module, monkeypatch, tmp_path):
    state = SimpleNamespace(patch="", close=Mock())
    tools = [object()]
    monkeypatch.setattr(agent_module, "make_tools", lambda repo: (tools, state))
    llm = object()
    llm_factory = Mock(return_value=llm)
    monkeypatch.setattr(agent_module, "ChatOpenAI", llm_factory)
    submitted_patch = "diff --git a/a.rs b/a.rs\n"

    def invoke(inputs, config):
        assert inputs == {"messages": [{"role": "user", "content": "fix issue"}]}
        assert config == {"recursion_limit": sys.maxsize}
        state.patch = submitted_patch

    create = Mock(return_value=SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(agent_module, "create_agent", create)
    assert agent_module.run_agent(tmp_path, "fix issue", "model", "key", "https://example.com", max_api_calls=3) == submitted_patch
    state.close.assert_called_once()
    create.assert_called_once_with(model=llm, tools=tools, system_prompt=agent_module.SYSTEM_PROMPT)
    assert llm_factory.call_args.kwargs["base_url"] == "https://example.com"


def test_run_agent_reports_verified_toolchain(agent_module, monkeypatch, tmp_path):
    state = SimpleNamespace(patch="", close=Mock(), rust_info={"source": "benchmark", "rustc": "rustc 1.72.0", "cargo": "cargo 1.72.0"})
    make_tools = Mock(return_value=([], state))
    monkeypatch.setattr(agent_module, "make_tools", make_tools)
    monkeypatch.setattr(agent_module, "ChatOpenAI", Mock())
    trace = Mock()

    def invoke(inputs, config):
        assert "rustc 1.72.0" in inputs["messages"][0]["content"]
        state.patch = "patch"

    monkeypatch.setattr(agent_module, "create_agent", Mock(return_value=SimpleNamespace(invoke=invoke)))
    assert agent_module.run_agent(tmp_path, "issue", "model", "key", trace=trace, rust_version="1.72.0") == "patch"
    make_tools.assert_called_once_with(tmp_path, "1.72.0")
    trace.record.assert_called_once_with("rust_environment", **state.rust_info)


def test_run_agent_rejects_missing_submission(agent_module, monkeypatch, tmp_path):
    state = SimpleNamespace(patch="", close=Mock())
    monkeypatch.setattr(agent_module, "make_tools", lambda repo: ([], state))
    monkeypatch.setattr(agent_module, "ChatOpenAI", Mock())
    monkeypatch.setattr(agent_module, "create_agent", Mock(return_value=SimpleNamespace(invoke=Mock())))
    with pytest.raises(RuntimeError, match="non-empty patch"):
        agent_module.run_agent(tmp_path, "issue", "model", "key")
    state.close.assert_called_once()


def test_run_agent_uses_selected_patch_and_requires_resubmission(agent_module, monkeypatch, tmp_path):
    (tmp_path / "lib.rs").write_text("before\n", encoding="utf-8")
    monkeypatch.setattr(agent_module, "ChatOpenAI", Mock())
    make_tools = agent_module.make_tools
    states = []

    def capture_state(repo):
        tools, state = make_tools(repo)
        states.append(state)
        return tools, state

    monkeypatch.setattr(agent_module, "make_tools", capture_state)

    def invoke(inputs, config):
        tools = {tool.name: tool for tool in create.call_args.kwargs["tools"]}
        tools["write_file"].invoke({"path": "lib.rs", "content": "after\n"})
        preview = tools["generate_patch"].invoke({})
        assert "lib.rs" in preview
        tools["generate_patch"].invoke({"paths": ["lib.rs"]})

    create = Mock(return_value=SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(agent_module, "create_agent", create)
    patch = agent_module.run_agent(tmp_path, "issue", "model", "key")
    assert "after" in patch

    def invoke_without_resubmission(inputs, config):
        tools = {tool.name: tool for tool in create.call_args.kwargs["tools"]}
        tools["write_file"].invoke({"path": "lib.rs", "content": "after\n"})
        tools["generate_patch"].invoke({"paths": ["lib.rs"]})
        tools["write_file"].invoke({"path": "scratch.rs", "content": "temporary\n"})

    create.return_value = SimpleNamespace(invoke=invoke_without_resubmission)
    with pytest.raises(RuntimeError, match="non-empty patch"):
        agent_module.run_agent(tmp_path, "issue", "model", "key")


def test_agent_continues_when_model_stops_without_submission(agent_module, monkeypatch, tmp_path):
    state = SimpleNamespace(patch="", close=Mock())
    monkeypatch.setattr(agent_module, "make_tools", lambda repo: ([], state))
    monkeypatch.setattr(agent_module, "ChatOpenAI", Mock())
    calls = []

    def invoke(inputs, config):
        calls.append(inputs["messages"])
        if len(calls) == 1:
            return {"messages": inputs["messages"] + [{"role": "assistant", "content": "Tests failed."}]}
        state.patch = "diff --git a/a b/a\n"
        return {"messages": inputs["messages"]}

    monkeypatch.setattr(agent_module, "create_agent", Mock(return_value=SimpleNamespace(invoke=invoke)))
    assert agent_module.run_agent(tmp_path, "issue", "model", "key") == state.patch
    assert len(calls) == 2 and "continue" in calls[1][-1]["content"].lower()


def test_cli_appends_prediction(agent_module, monkeypatch, tmp_path):
    issue = tmp_path / "issue.txt"
    issue.write_text("issue text", encoding="utf-8")
    output = tmp_path / "results" / "predictions.jsonl"
    monkeypatch.delenv("API_LIMIT", raising=False)
    monkeypatch.setenv("MODEL", "example-model")
    monkeypatch.setenv("API_KEY", "example-key")
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(agent_module, "configure_network_proxy", lambda: None)
    run = Mock(return_value="diff --git a/a.rs b/a.rs\n")
    monkeypatch.setattr(agent_module, "run_agent", run)
    monkeypatch.setattr(sys, "argv", ["agent.py", "--repo", str(tmp_path), "--instance-id", "owner__repo-1", "--issue-file", str(issue), "--output", str(output), "--rust-toolchain", "1.72.0"])
    agent_module.main()
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "instance_id": "owner__repo-1",
        "model_name_or_path": "normal__example-model",
        "model_patch": "diff --git a/a.rs b/a.rs\n",
    }
    assert run.call_args.args == (tmp_path.resolve(), "issue text", "example-model", "example-key", None)
    assert isinstance(run.call_args.kwargs["trace"], agent_module.TraceRecorder)
    assert run.call_args.kwargs["max_api_calls"] == 70
    assert run.call_args.kwargs["rust_version"] == "1.72.0"


def test_cli_does_not_write_on_agent_failure(agent_module, monkeypatch, tmp_path):
    issue = tmp_path / "issue.txt"
    issue.write_text("issue", encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    monkeypatch.delenv("API_LIMIT", raising=False)
    monkeypatch.setenv("MODEL", "model")
    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(agent_module, "configure_network_proxy", lambda: None)
    monkeypatch.setattr(agent_module, "run_agent", Mock(side_effect=RuntimeError("failed")))
    monkeypatch.setattr(sys, "argv", ["agent.py", "--repo", str(tmp_path), "--instance-id", "id", "--issue-file", str(issue), "--output", str(output), "--rust-toolchain", "1.72.0"])
    with pytest.raises(RuntimeError, match="failed"):
        agent_module.main()
    assert not output.exists()
    trace = next((tmp_path / "traces").glob("*.jsonl"))
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["run_start", "run_error", "agent_metrics"]
    assert events[-2]["stage"] == "agent"
    assert events[-1]["status"] == "failed"
    assert events[-1]["elapsed_seconds"] >= 0
    assert events[-1]["api_calls"] == 0


@pytest.mark.parametrize(("configured", "cli", "expected"), [("5", None, 5), ("5", "9", 5), ("5", "2", 2)])
def test_api_limit_for_one_repair(agent_module, monkeypatch, tmp_path, configured, cli, expected):
    issue = tmp_path / "issue.txt"
    issue.write_text("issue", encoding="utf-8")
    monkeypatch.setenv("API_LIMIT", configured)
    monkeypatch.setenv("MODEL", "model")
    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(agent_module, "configure_network_proxy", lambda: None)
    run = Mock(return_value="diff --git a/a b/a\n")
    monkeypatch.setattr(agent_module, "run_agent", run)
    argv = ["agent.py", "--repo", str(tmp_path), "--instance-id", "id", "--issue-file", str(issue), "--output", str(tmp_path / "out.jsonl"), "--rust-toolchain", "1.72.0"]
    if cli is not None:
        argv.extend(["--max-api-calls", cli])
    monkeypatch.setattr(sys, "argv", argv)
    agent_module.main()
    assert run.call_args.kwargs["max_api_calls"] == expected


@pytest.mark.parametrize("value", ["", "0", "-1", "abc", "1.5"])
def test_invalid_api_limit(agent_module, monkeypatch, tmp_path, value):
    monkeypatch.setenv("API_LIMIT", value)
    monkeypatch.setenv("MODEL", "model")
    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(agent_module, "configure_network_proxy", lambda: None)
    monkeypatch.setattr(sys, "argv", ["agent.py", "--repo", str(tmp_path), "--instance-id", "id", "--issue-file", "issue", "--output", "out", "--rust-toolchain", "1.72.0"])
    with pytest.raises(SystemExit):
        agent_module.main()


def test_cli_api_limit_and_default_yj_interface(agent_module, monkeypatch, tmp_path):
    issue = tmp_path / "issue.txt"
    issue.write_text("issue", encoding="utf-8")
    for k in ("API_KEY", "MODEL", "BASE_URL", "API_LIMIT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("API_LIMIT", "10")
    monkeypatch.setenv("YJ_BASE_URL", "https://yj.example/v1")
    monkeypatch.setenv("YJ_API_KEY", "yj-key")
    monkeypatch.setenv("YJ_MODEL", "yj-model")
    monkeypatch.setenv("DS_BASE_URL", "https://ds.example/v1")
    monkeypatch.setenv("DS_API_KEY", "ds-key")
    monkeypatch.setenv("DS_MODEL", "ds-model")
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(agent_module, "configure_network_proxy", lambda: None)
    run = Mock(return_value="diff --git a/a b/a\n")
    monkeypatch.setattr(agent_module, "run_agent", run)

    argv = ["agent.py", "--repo", str(tmp_path), "--instance-id", "id", "--issue-file", str(issue),
            "--output", str(tmp_path / "out.jsonl"), "--rust-toolchain", "1.72.0", "--api-limit", "25"]
    monkeypatch.setattr(sys, "argv", argv)
    agent_module.main()
    assert run.call_args.args == (tmp_path.resolve(), "issue", "yj-model", "yj-key", "https://yj.example/v1")
    assert run.call_args.kwargs["max_api_calls"] == 25

    monkeypatch.setattr(sys, "argv", argv[:-2] + ["--api", "ds"])
    agent_module.main()
    assert run.call_args.args == (tmp_path.resolve(), "issue", "ds-model", "ds-key", "https://ds.example/v1")

    monkeypatch.setattr(sys, "argv", argv[:-2] + ["--api", "ds", "--api-limit", "0"])
    with pytest.raises(SystemExit):
        agent_module.main()


def test_run_agent_auto_submits_changes_when_api_limit_reached(agent_module, monkeypatch, tmp_path):
    (tmp_path / "lib.rs").write_text("before\n", encoding="utf-8")
    monkeypatch.setattr(agent_module, "ChatOpenAI", Mock())
    make_tools = agent_module.make_tools

    monkeypatch.setattr(agent_module, "make_tools", lambda repo: make_tools(repo))

    def invoke_edited_then_limit(inputs, config):
        tools = {tool.name: tool for tool in create.call_args.kwargs["tools"]}
        tools["write_file"].invoke({"path": "lib.rs", "content": "after\n"})
        raise RuntimeError("Model API call limit reached (5)")

    create = Mock(return_value=SimpleNamespace(invoke=invoke_edited_then_limit))
    monkeypatch.setattr(agent_module, "create_agent", create)
    trace = Mock()
    patch = agent_module.run_agent(tmp_path, "issue", "model", "key", trace=trace, max_api_calls=5)
    assert "+after" in patch
    trace.record.assert_called_once_with("auto_submit_on_limit", patch=patch)

    def invoke_no_changes_then_limit(inputs, config):
        raise RuntimeError("Model API call limit reached (5)")

    create.return_value = SimpleNamespace(invoke=invoke_no_changes_then_limit)
    with pytest.raises(RuntimeError, match="Model API call limit reached"):
        agent_module.run_agent(tmp_path, "issue", "model", "key", max_api_calls=5)


def test_compress_history_messages_trims_old_tool_outputs_and_args(agent_module):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    big_code = "fn big() {\n" + "".join(f"    let v_{i} = {i};\n" for i in range(100)) + "}\n"
    long_thought = "Let me think.\n" + "\n".join(f"Step {i}: checking..." for i in range(1, 25)) + "\nDone."
    msgs = [
        HumanMessage(content="issue text"),
        AIMessage(content=long_thought, tool_calls=[{"id": "call_0", "name": "write_file", "args": {"path": "src/lib.rs", "content": big_code}}]),
        ToolMessage(content="\n".join(f"{j}: code line" for j in range(1, 40)), tool_call_id="call_0"),
    ]
    for i in range(1, 10):
        msgs.append(AIMessage(content="", tool_calls=[{"id": f"call_{i}", "name": "read_file", "args": {}}]))
        msgs.append(ToolMessage(content="\n".join(f"{j}: code line" for j in range(1, 40)), tool_call_id=f"call_{i}"))

    compressed = agent_module._compress_history_messages(msgs, keep_recent_steps=3, block_size=2)
    assert len(compressed) == len(msgs)
    assert "earlier thought omitted" in compressed[1].content
    assert "earlier write omitted" in compressed[1].tool_calls[0]["args"]["content"]
    assert "omitted" in compressed[2].content
    assert len(compressed[2].content) < len(msgs[2].content)
    assert compressed[-1].content == msgs[-1].content
    assert compressed[-3].content == msgs[-3].content
    assert compressed[-5].content == msgs[-5].content


def test_compress_history_messages_block_size_stability_and_idempotency(agent_module):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    def make_msgs(n):
        m = [HumanMessage(content="issue")]
        for i in range(n):
            m.append(AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "read_file", "args": {}}]))
            m.append(ToolMessage(content="\n".join(f"{j}: code line" for j in range(1, 30)), tool_call_id=f"c{i}"))
        return m

    c18 = agent_module._compress_history_messages(make_msgs(18), keep_recent_steps=12, block_size=5)
    c19 = agent_module._compress_history_messages(make_msgs(19), keep_recent_steps=12, block_size=5)
    c21 = agent_module._compress_history_messages(make_msgs(21), keep_recent_steps=12, block_size=5)
    for idx in range(11):
        assert c18[idx].content == c19[idx].content == c21[idx].content

    again = agent_module._compress_history_messages(c18, keep_recent_steps=12, block_size=5)
    for m1, m2 in zip(c18, again):
        assert m1.content == m2.content


def test_cli_defaults_require_reproduction_false_and_accepts_flags(agent_module, monkeypatch, tmp_path):
    issue = tmp_path / "issue.txt"
    issue.write_text("issue text", encoding="utf-8")
    monkeypatch.setenv("API_LIMIT", "70")
    monkeypatch.setenv("MODEL", "model")
    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(agent_module, "configure_network_proxy", lambda: None)
    run = Mock(return_value="diff --git patch\n")
    monkeypatch.setattr(agent_module, "run_agent", run)

    argv_default = ["agent.py", "--repo", str(tmp_path), "--instance-id", "id", "--issue-file", str(issue), "--output", str(tmp_path / "out.jsonl"), "--rust-toolchain", "1.72.0"]
    monkeypatch.setattr(sys, "argv", argv_default)
    agent_module.main()
    assert run.call_args.kwargs["require_reproduction"] is False

    argv_yes = [*argv_default, "--require-reproduction"]
    monkeypatch.setattr(sys, "argv", argv_yes)
    agent_module.main()
    assert run.call_args.kwargs["require_reproduction"] is True

    argv_no = [*argv_default, "--no-require-reproduction"]
    monkeypatch.setattr(sys, "argv", argv_no)
    agent_module.main()
    assert run.call_args.kwargs["require_reproduction"] is False


def test_run_agent_stops_after_repeated_idle_responses(agent_module, monkeypatch, tmp_path):
    state = SimpleNamespace(patch="", close=Mock())
    monkeypatch.setattr(agent_module, "make_tools", lambda repo: ([], state))
    monkeypatch.setattr(agent_module, "ChatOpenAI", Mock())
    calls = []

    def invoke(inputs, config):
        calls.append(inputs["messages"])
        return {"messages": inputs["messages"] + [{"role": "assistant", "content": "I am done."}]}

    monkeypatch.setattr(agent_module, "create_agent", Mock(return_value=SimpleNamespace(invoke=invoke)))
    with pytest.raises(RuntimeError, match="stopped calling tools"):
        agent_module.run_agent(tmp_path, "issue", "model", "key")
    assert len(calls) == 4
