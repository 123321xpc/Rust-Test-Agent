"""Offline contract tests for the LangChain baseline runner."""
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
    spec = importlib.util.spec_from_file_location("test_baseline_agent", baseline_dir / "agent.py")
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


def test_run_agent_defaults_to_staged_multi_agent_and_reports_toolchain(agent_module, monkeypatch, tmp_path):
    state = SimpleNamespace(patch="", close=Mock(), rust_info={"source": "benchmark", "rustc": "rustc 1.72.0", "cargo": "cargo 1.72.0"})
    make_tools = Mock(return_value=([], state))
    monkeypatch.setattr(agent_module, "make_tools", make_tools)
    monkeypatch.delenv("RUSTBENCH_REPAIR_MODE", raising=False)
    trace = Mock()
    staged_mock = Mock(return_value="diff --git a/lib.rs b/lib.rs\n")
    continuous_mock = Mock(return_value="continuous-patch")
    monkeypatch.setattr(agent_module._multi_agent, "run_multi_agent", staged_mock)
    monkeypatch.setattr(agent_module._multi_agent, "run_continuous_repair", continuous_mock)

    patch = agent_module.run_agent(tmp_path, "issue", "model", "key", "https://example.com", trace=trace, max_api_calls=12, rust_version="1.72.0")
    assert patch == "diff --git a/lib.rs b/lib.rs\n"
    make_tools.assert_called_once_with(tmp_path, "1.72.0")
    trace.record.assert_called_once_with("rust_environment", **state.rust_info)
    staged_mock.assert_called_once()
    continuous_mock.assert_not_called()
    assert "rustc 1.72.0" in staged_mock.call_args.args[1]
    assert staged_mock.call_args.args[4] == "https://example.com"
    state.close.assert_called_once()

    monkeypatch.setenv("RUSTBENCH_REPAIR_MODE", "continuous")
    assert agent_module.run_agent(tmp_path, "issue", "model", "key") == "continuous-patch"
    continuous_mock.assert_called_once()


def test_run_agent_rejects_missing_patch(agent_module, monkeypatch, tmp_path):
    state = SimpleNamespace(patch="", close=Mock(), changed_paths=Mock(return_value=[]))
    monkeypatch.setattr(agent_module, "make_tools", lambda repo: ([], state))
    monkeypatch.setattr(agent_module._multi_agent, "run_multi_agent", Mock(return_value=""))
    with pytest.raises(RuntimeError, match="non-empty patch"):
        agent_module.run_agent(tmp_path, "issue", "model", "key")
    state.close.assert_called_once()


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
        "model_name_or_path": "test__example-model",
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


def test_compress_history_messages_trims_old_tool_outputs(agent_module):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    msgs = [HumanMessage(content="issue text")]
    for i in range(10):
        msgs.append(AIMessage(content="", tool_calls=[{"id": f"call_{i}", "name": "read_lines", "args": {}}]))
        if i == 0:
            content = json.dumps({
                "status": "passed",
                "tests_executed": 42,
                "failed_tests": [],
                "test_output": "test " * 300,
                "cargo_output": "Compiling " * 100,
            })
        else:
            content = "\n".join(f"{j}: code line" for j in range(1, 40))
        msgs.append(ToolMessage(content=content, tool_call_id=f"call_{i}"))

    compressed = agent_module._compress_history_messages(msgs, keep_recent_steps=3)
    assert len(compressed) == len(msgs)
    # First tool message (index 2) was cargo_test JSON -> compressed JSON
    old_json = json.loads(compressed[2].content)
    assert old_json["status"] == "passed"
    assert old_json["tests_executed"] == 42
    assert "note" in old_json
    assert compressed[2].tool_call_id == "call_0"
    # Second tool message (index 4) was multiline text -> compressed preview
    assert "omitted" in compressed[4].content
    assert len(compressed[4].content) < len(msgs[4].content)
    # Recent tool messages (last 3: indices -5, -3, -1) kept intact
    assert compressed[-1].content == msgs[-1].content
    assert compressed[-3].content == msgs[-3].content
    assert compressed[-5].content == msgs[-5].content



def test_compress_history_messages_handles_cargo_check_and_assistant_tool_args(agent_module):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    msgs = [HumanMessage(content="start")]
    # Turn 0: AI with huge write_file args
    big_code = "fn big() {\n" + "".join(f"    let v_{i} = {i};\n" for i in range(100)) + "}\n"
    msgs.append(AIMessage(content="", tool_calls=[{"id": "c0", "name": "write_file", "args": {"path": "src/lib.rs", "content": big_code}}]))
    msgs.append(ToolMessage(content="Updated src/lib.rs", tool_call_id="c0"))

    # Turn 1: cargo_check JSON tool output
    check_json = json.dumps({
        "exit_code": 101,
        "diagnostics": [
            {"level": "error", "code": "E0308", "category": "type_mismatch", "message": "mismatched types", "spans": [{"file_name": "src/lib.rs", "line_start": 5}]}
        ] * 10,
        "cargo_output": "error: could not compile\n" * 50,
    })
    msgs.append(AIMessage(content="", tool_calls=[{"id": "c1", "name": "cargo_check", "args": {}}]))
    msgs.append(ToolMessage(content=check_json, tool_call_id="c1"))

    # Turn 2: cargo_workspace JSON
    ws_json = json.dumps({
        "workspace_root": "/workspace",
        "workspace_members": [{"name": "pkg1", "manifest_path": "/workspace/Cargo.toml", "targets": [{"name": f"target_{i}", "kind": ["lib", "test"], "src_path": f"src/target_{i}.rs"} for i in range(10)]}],
    }, indent=2)
    msgs.append(AIMessage(content="", tool_calls=[{"id": "c2", "name": "cargo_workspace", "args": {}}]))
    msgs.append(ToolMessage(content=ws_json, tool_call_id="c2"))

    # Add 4 dummy tool turns to push turns 0-2 past cutoff
    for k in range(3, 7):
        msgs.append(AIMessage(content="", tool_calls=[{"id": f"c{k}", "name": "read_lines", "args": {}}]))
        msgs.append(ToolMessage(content=f"{k}: code line", tool_call_id=f"c{k}"))

    compressed = agent_module._compress_history_messages(msgs, keep_recent_steps=3, test_cutoff_steps=2)
    assert len(compressed) == len(msgs)

    # Mutation payloads remain intact so a later model turn cannot copy a history placeholder into a file.
    ai_args = compressed[1].tool_calls[0]["args"]
    assert ai_args["content"] == big_code

    # Check cargo_check tool message (index 4) is compacted
    check_obj = json.loads(compressed[4].content)
    assert check_obj["exit_code"] == 101
    assert check_obj["error_count"] == 10
    assert "earlier cargo_check" in check_obj["note"]

    # Check cargo_workspace tool message (index 6) is compacted
    ws_obj = json.loads(compressed[6].content)
    assert ws_obj["workspace_members"][0]["name"] == "pkg1"
    assert "targets" not in ws_obj["workspace_members"][0]


def test_cli_defaults_require_reproduction_and_accepts_no_require_reproduction(agent_module, monkeypatch, tmp_path):
    issue = tmp_path / "issue.txt"
    issue.write_text("issue text", encoding="utf-8")
    monkeypatch.setenv("API_LIMIT", "70")
    monkeypatch.setenv("MODEL", "model")
    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(agent_module, "configure_network_proxy", lambda: None)
    run = Mock(return_value="diff --git patch\n")
    monkeypatch.setattr(agent_module, "run_agent", run)

    # Test default: require_reproduction is True
    argv_default = ["agent.py", "--repo", str(tmp_path), "--instance-id", "id", "--issue-file", str(issue), "--output", str(tmp_path / "out.jsonl"), "--rust-toolchain", "1.72.0"]
    monkeypatch.setattr(sys, "argv", argv_default)
    agent_module.main()
    assert run.call_args.kwargs["require_reproduction"] is True

    # Test explicit flag: --no-require-reproduction sets False
    argv_no = ["agent.py", "--repo", str(tmp_path), "--instance-id", "id", "--issue-file", str(issue), "--output", str(tmp_path / "out.jsonl"), "--rust-toolchain", "1.72.0", "--no-require-reproduction"]
    monkeypatch.setattr(sys, "argv", argv_no)
    agent_module.main()
    assert run.call_args.kwargs["require_reproduction"] is False



def test_compress_history_messages_block_size_stability_and_idempotency(agent_module):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    def make_msgs(n):
        m = [HumanMessage(content="issue")]
        for i in range(n):
            m.append(AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "read_lines", "args": {}}]))
            m.append(ToolMessage(content="\n".join(f"{j}: code line" for j in range(1, 30)), tool_call_id=f"c{i}"))
        return m

    # With block_size=5, keep_recent_steps=12, at n=18 vs n=19 vs n=20:
    # raw_code_cutoff: 18-12=6 (pos=5), 19-12=7 (pos=5), 20-12=8 (pos=5), 21-12=9 (pos=5)
    # Prefix up to tool 17 must be byte-for-byte identical across turns 18..21 (preserving prompt cache)!
    c18 = agent_module._compress_history_messages(make_msgs(18), keep_recent_steps=12, test_cutoff_steps=3, block_size=5)
    c19 = agent_module._compress_history_messages(make_msgs(19), keep_recent_steps=12, test_cutoff_steps=3, block_size=5)
    c21 = agent_module._compress_history_messages(make_msgs(21), keep_recent_steps=12, test_cutoff_steps=3, block_size=5)
    # First 5 tools (indices 1..10) were compacted at n=18 and stay identical at n=19 and n=21!
    for idx in range(11):
        assert c18[idx].content == c19[idx].content == c21[idx].content

    # Idempotency check
    again = agent_module._compress_history_messages(c18, keep_recent_steps=12, test_cutoff_steps=3, block_size=5)
    for m1, m2 in zip(c18, again):
        assert m1.content == m2.content


def test_compress_history_messages_compacts_older_assistant_thoughts(agent_module):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    long_thought = "Let me think about this bug.\n" + "\n".join(f"Step {i}: inspecting invariants..." for i in range(1, 25)) + "\nReady to edit."
    msgs = [
        HumanMessage(content="issue text"),
        AIMessage(content=long_thought, tool_calls=[{"id": "c0", "name": "read_lines", "args": {}}]),
        ToolMessage(content="1: code", tool_call_id="c0"),
    ]
    # Add dummy tool turns to push turn 0 past cutoff
    for k in range(1, 8):
        msgs.append(AIMessage(content="ok", tool_calls=[{"id": f"c{k}", "name": "read_lines", "args": {}}]))
        msgs.append(ToolMessage(content=f"{k}: code line", tool_call_id=f"c{k}"))

    compressed = agent_module._compress_history_messages(msgs, keep_recent_steps=3, test_cutoff_steps=2, block_size=2)
    old_ai = compressed[1]
    assert "earlier thought omitted" in old_ai.content
    assert len(old_ai.content) < len(long_thought)


def test_compress_history_messages_compacts_stale_file_reads(agent_module):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    msgs = [
        HumanMessage(content="start"),
        # Turn 0: read_lines src/lib.rs
        AIMessage(content="", tool_calls=[{"id": "c0", "name": "read_lines", "args": {"path": "src/lib.rs", "start_line": 1, "end_line": 20}}]),
        ToolMessage(content="1: fn foo() {}\n" * 10, tool_call_id="c0"),
        # Turn 1: edit_file src/lib.rs
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "edit_file", "args": {"path": "src/lib.rs", "old_text": "foo", "new_text": "bar"}}]),
        ToolMessage(content="Updated src/lib.rs", tool_call_id="c1"),
        # Turn 2: read_lines src/lib.rs (fresh read after edit)
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "read_lines", "args": {"path": "src/lib.rs", "start_line": 1, "end_line": 20}}]),
        ToolMessage(content="1: fn bar() {}\n" * 10, tool_call_id="c2"),
    ]
    # keep_recent_steps=6 keeps all 3 turns in window, but turn 0 read should be compacted as stale!
    compressed = agent_module._compress_history_messages(msgs, keep_recent_steps=6, test_cutoff_steps=1)
    assert "subsequent step" in compressed[2].content
    assert "1: fn bar()" in compressed[6].content


def test_cli_rejects_removed_single_agent_flag(agent_module, monkeypatch, tmp_path):
    issue = tmp_path / "issue.txt"
    issue.write_text("issue text", encoding="utf-8")
    output = tmp_path / "out.jsonl"
    monkeypatch.setenv("API_LIMIT", "70")
    monkeypatch.setenv("MODEL", "model")
    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(agent_module, "configure_network_proxy", lambda: None)
    monkeypatch.setattr(sys, "argv", [
        "agent.py", "--repo", str(tmp_path), "--instance-id", "id",
        "--issue-file", str(issue), "--output", str(output),
        "--rust-toolchain", "1.72.0", "--single-agent",
    ])
    with pytest.raises(SystemExit):
        agent_module.main()


def test_compress_history_invalidates_on_revert_and_folds_old_errors_to_one_line(agent_module):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    msgs = [
        HumanMessage(content="start"),
        AIMessage(content="", tool_calls=[{"id": "c0", "name": "read_lines", "args": {"path": "./src/lib.rs"}}]),
        ToolMessage(content="1: fn a() {}\n" * 30, tool_call_id="c0"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "edit_file", "args": {"path": "src/lib.rs"}}]),
        ToolMessage(content="Error editing src/lib.rs: old_text was not found.\n[Region hint: lines 1-5:\n1: a\n2: b\n3: c\n4: d\n5: e\n6: f\n7: g]", tool_call_id="c1"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "revert_file", "args": {"path": "src/lib.rs"}}]),
        ToolMessage(content="Reverted src/lib.rs to its original state.", tool_call_id="c2"),
        AIMessage(content="", tool_calls=[{"id": "c3", "name": "read_lines", "args": {"path": "src/other.rs"}}]),
        ToolMessage(content="1: ok\n", tool_call_id="c3"),
    ]
    compressed = agent_module._compress_history_messages(msgs, keep_recent_steps=2, test_cutoff_steps=1)
    assert "subsequent step" in compressed[2].content
    assert "\n" not in compressed[4].content
    assert "earlier edit_file error omitted" in compressed[4].content
