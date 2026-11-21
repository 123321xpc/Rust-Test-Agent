"""Tests for the default continuous repair path and existing-test smoke checks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_continuous_test", Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path):
    repo = tmp_path / "continuous_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text('[package]\nname="continuous_demo"\nversion="0.1.0"\nedition="2021"\n', encoding="utf-8")
    (repo / "src/lib.rs").write_text("pub fn value() -> i32 { 0 }\n", encoding="utf-8")
    return repo


@pytest.mark.parametrize("consult", [False, True])
def test_default_continuous_repair_preserves_context_and_changes(tmp_path, monkeypatch, consult):
    agent = load("agent")
    repo = make_repo(tmp_path)
    monkeypatch.delenv("RUSTBENCH_REPAIR_MODE", raising=False)
    monkeypatch.setattr(agent, "ChatOpenAI", Mock())
    make_tools = agent.make_tools
    states = []
    calls = []

    def capture(*args, **kwargs):
        tools, state = make_tools(*args, **kwargs)
        state.post_strip_check = False
        state.require_reproduction = False
        state.run_existing_test_smoke = lambda _selected: []
        states.append(state)
        return tools, state

    monkeypatch.setattr(agent, "make_tools", capture)

    def create(model, tools, system_prompt):
        tool_map = {item.name: item for item in tools}
        if "one continuous repair conversation" in system_prompt:
            role = "repair"
        elif "Rust Architect / Suggester" in system_prompt:
            role = "advisor"
        else:
            role = "verifier"
        calls.append(role)

        def invoke(inputs, config):
            messages = inputs["messages"]
            if role == "repair" and calls.count("repair") == 1:
                tool_map["write_regression_test"].invoke({
                    "path": "tests/test_regression_value.rs", "content": "#[test] fn regression() {}",
                })
                tool_map["edit_file"].invoke({
                    "path": "src/lib.rs", "old_text": "{ 0 }", "new_text": "{ 2 }" if consult else "{ 1 }",
                })
                if consult:
                    tool_map["escalate_for_replan"].invoke({"reason": "repeated mismatch in the value contract"})
                else:
                    states[-1].pass_test(states[-1].start_test(), test_target="test_regression_value")
                    tool_map["submit_for_qa"].invoke({"summary": "fixed"})
            elif role == "advisor":
                assert "edit_file" not in tool_map and "write_file" not in tool_map
                assert "+pub fn value() -> i32 { 2 }" in messages[-1]["content"]
                tool_map["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1", "arbitration_rationale": "return the required value",
                    "ordered_edits": [{"file": "src/lib.rs", "description": "replace 2 with 1"}],
                })
            elif role == "repair":
                assert any("prior repair reasoning" in str(message) for message in messages)
                assert "{ 2 }" in tool_map["read_file"].invoke({"path": "src/lib.rs"})
                tool_map["edit_file"].invoke({"path": "src/lib.rs", "old_text": "{ 2 }", "new_text": "{ 1 }"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_value")
                tool_map["submit_for_qa"].invoke({"summary": "fixed after advice"})
            else:
                assert "Candidate patch" in messages[-1]["content"]
                tool_map["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "matches issue"})
            return {"messages": messages + [{"role": "assistant", "content": "prior repair reasoning"}]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent, "create_agent", create)
    patch = agent.run_agent(repo, "Make value return 1", "unused", "unused", max_api_calls=70)
    assert "+pub fn value() -> i32 { 1 }" in patch
    assert calls == (["repair", "advisor", "repair", "verifier"] if consult else ["repair", "verifier"])


def test_existing_smoke_executes_tests_and_restores_regression(tmp_path):
    tools_mod = load("tools")
    repo = make_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests/contract.rs").write_text(
        "#[test] fn contract_fails() { assert_eq!(continuous_demo::value(), 9); }\n", encoding="utf-8"
    )
    tool_list, state = tools_mod.make_tools(repo, post_strip_check=False)
    tool_map = {item.name: item for item in tool_list}
    try:
        tool_map["edit_file"].invoke({"path": "src/lib.rs", "old_text": "{ 0 }", "new_text": "{ 1 }"})
        tool_map["write_regression_test"].invoke({
            "path": "tests/test_regression_temp.rs", "content": 'compile_error!("temporary test must be stripped");',
        })
        results = state.run_existing_test_smoke([Path("src/lib.rs")], timeout=90)
        failure = next(item for item in results if item["target"] == "test:contract")
        assert failure["status"] == "failed" and failure["tests_executed"] == 1
        assert "contract_fails" in failure["details"]
        assert (state.root / "tests/test_regression_temp.rs").is_file()
        assert state.run_existing_test_smoke([Path("src/lib.rs")], timeout=90) == results
    finally:
        state.close()
