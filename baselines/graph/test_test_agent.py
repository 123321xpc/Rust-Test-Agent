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


def test_run_agent_defaults_to_graph_mode_and_reports_toolchain(agent_module, monkeypatch, tmp_path):
    state = SimpleNamespace(patch="", close=Mock(), rust_info={"source": "benchmark", "rustc": "rustc 1.72.0", "cargo": "cargo 1.72.0"})
    make_tools = Mock(return_value=([], state))
    monkeypatch.setattr(agent_module, "make_tools", make_tools)
    trace = Mock()
    graph_mock = Mock(return_value="diff --git a/lib.rs b/lib.rs\n")
    monkeypatch.setattr(agent_module._multi_agent, "run_graph_agent", graph_mock)

    patch = agent_module.run_agent(tmp_path, "issue", "model", "key", "https://example.com", trace=trace, max_api_calls=12, rust_version="1.72.0")
    assert patch == "diff --git a/lib.rs b/lib.rs\n"
    make_tools.assert_called_once_with(tmp_path, "1.72.0")
    trace.record.assert_called_once_with("rust_environment", **state.rust_info)
    graph_mock.assert_called_once()
    assert "rustc 1.72.0" in graph_mock.call_args.args[1]
    assert graph_mock.call_args.args[4] == "https://example.com"
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
