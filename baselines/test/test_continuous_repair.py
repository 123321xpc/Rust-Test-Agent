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
    monkeypatch.setenv("RUSTBENCH_REPAIR_MODE", "continuous")
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


@pytest.mark.parametrize("baseline_broken", [False, True])
def test_workspace_all_features_distinguishes_new_and_baseline_failures(tmp_path, baseline_broken):
    tools_mod = load("tools")
    repo = make_repo(tmp_path)
    (repo / "Cargo.toml").write_text(
        '[package]\nname="continuous_demo"\nversion="0.1.0"\nedition="2021"\n'
        '[features]\nextra=[]\n', encoding="utf-8"
    )
    (repo / "src/lib.rs").write_text(
        'pub fn value() -> i32 { 0 }\n'
        '#[cfg(feature = "extra")]\n'
        + ('compile_error!("baseline feature failure");\n' if baseline_broken else
           'pub fn extra() -> i32 { 1 }\n'), encoding="utf-8"
    )
    tool_list, state = tools_mod.make_tools(repo, post_strip_check=True)
    tool_map = {item.name: item for item in tool_list}
    try:
        if baseline_broken:
            tool_map["edit_file"].invoke({
                "path": "src/lib.rs", "old_text": "pub fn value() -> i32 { 0 }",
                "new_text": "pub fn value() -> i32 { 1 }",
            })
            report = state.run_post_strip_check([Path("src/lib.rs")])
            assert report["status"] == "inconclusive"
            assert "baseline also failed" in " ".join(report["warnings"])
        else:
            tool_map["edit_file"].invoke({
                "path": "src/lib.rs", "old_text": "pub fn extra() -> i32 { 1 }",
                "new_text": 'pub fn extra() -> i32 { "broken" }',
            })
            with pytest.raises(ValueError, match="Post-strip verification failed|Workspace compilation regressed"):
                state.run_post_strip_check([Path("src/lib.rs")])
        assert (state.root / "src/lib.rs").read_text(encoding="utf-8") != (repo / "src/lib.rs").read_text(encoding="utf-8")
    finally:
        state.close()


def test_continuous_repair_uses_remaining_budget_for_multiple_qa_rejections(tmp_path, monkeypatch):
    agent = load("agent")
    repo = make_repo(tmp_path)
    monkeypatch.setenv("RUSTBENCH_REPAIR_MODE", "continuous")
    monkeypatch.setattr(agent, "ChatOpenAI", Mock())
    original_make_tools = agent.make_tools
    states = []
    calls = {"fixer": 0, "verifier": 0}

    def capture(*args, **kwargs):
        tools, state = original_make_tools(*args, **kwargs)
        state.post_strip_check = False
        state.require_reproduction = False
        state.run_existing_test_smoke = lambda selected: []
        states.append(state)
        return tools, state

    def create(model, tools, system_prompt):
        tool_map = {item.name: item for item in tools}
        role = "fixer" if "one continuous repair conversation" in system_prompt else "verifier"

        def invoke(inputs, config):
            calls[role] += 1
            if role == "fixer":
                if calls[role] == 1:
                    tool_map["write_regression_test"].invoke({
                        "path": "tests/test_regression_value.rs", "content": "#[test] fn regression() {}",
                    })
                old = calls[role] - 1
                tool_map["edit_file"].invoke({
                    "path": "src/lib.rs", "old_text": f"{{ {old} }}", "new_text": f"{{ {old + 1} }}",
                })
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_value")
                tool_map["submit_for_qa"].invoke({"summary": "revised"})
            else:
                decision = "APPROVE" if calls[role] == 4 else "REJECT"
                tool_map["submit_qa_decision"].invoke({
                    "decision": decision, "summary": "reviewed", "objections": [f"objection {calls[role]}"] if decision == "REJECT" else [],
                })
            return {"messages": inputs["messages"] + [{"role": "assistant", "content": "continued"}]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent, "make_tools", capture)
    monkeypatch.setattr(agent, "create_agent", create)
    patch = agent.run_agent(repo, "Fix value", "unused", "unused", max_api_calls=100)
    assert "+pub fn value() -> i32 { 4 }" in patch
    assert calls == {"fixer": 4, "verifier": 4}


def test_trace_recorder_counts_responses_api_usage(tmp_path):
    import httpx
    trace_mod = load("agent_trace")
    recorder = trace_mod.TraceRecorder(tmp_path / "resp_metrics.jsonl")
    client = trace_mod.TraceHTTPClient(recorder)
    responses = iter([
        httpx.Response(200, json={"usage": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}}),
        httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
    ])
    client._transport = httpx.MockTransport(lambda request: next(responses))
    client._mounts = {}
    try:
        client.post("https://example.test/v1/responses", json={})
        client.post("https://example.test/v1/chat/completions", json={})
    finally:
        client.close()
    metrics = recorder.metrics()
    assert metrics["input_tokens"] == 130
    assert metrics["output_tokens"] == 35
    assert metrics["total_tokens"] == 165
    assert metrics["responses_without_usage"] == 0


def test_continuous_repair_verifier_gets_full_signals_tools_and_escalates_repeated_qa_rejects(tmp_path, monkeypatch):
    agent = load("agent")
    repo = make_repo(tmp_path)
    (repo / "src/lexer.rs").write_text("pub fn lex() -> i32 { 0 }\n", encoding="utf-8")
    (repo / "src/lib.rs").write_text("pub mod lexer;\npub fn value() -> i32 { lexer::lex() }\n", encoding="utf-8")
    monkeypatch.setenv("RUSTBENCH_REPAIR_MODE", "continuous")
    monkeypatch.setattr(agent, "ChatOpenAI", Mock())
    make_tools = agent.make_tools
    states = []
    seen = []
    verifier_prompts = []
    advisor_prompts = []

    def capture(*args, **kwargs):
        tools, state = make_tools(*args, **kwargs)
        state.post_strip_check = False
        state.require_reproduction = False
        state.run_existing_test_smoke = lambda _selected: []
        states.append(state)
        return tools, state

    monkeypatch.setattr(agent, "make_tools", capture)
    ver_calls = {"n": 0}
    fix_calls = {"n": 0}

    def create(model, tools, system_prompt):
        tool_map = {item.name: item for item in tools}
        if "one continuous repair conversation" in system_prompt:
            role = "fixer"
        elif "Rust Architect / Suggester" in system_prompt:
            role = "advisor"
        else:
            role = "verifier"
        seen.append(role)

        def invoke(inputs, config):
            msg = inputs["messages"][-1]["content"]
            if role == "fixer":
                fix_calls["n"] += 1
                if fix_calls["n"] == 1:
                    tool_map["read_lines"].invoke({"path": "src/lexer.rs", "start_line": 1, "end_line": 1})
                    tool_map["write_regression_test"].invoke({
                        "path": "tests/test_regression_lex.rs", "content": "#[test] fn t() {}",
                    })
                    tool_map["edit_file"].invoke({
                        "path": "src/lib.rs", "old_text": "lexer::lex()", "new_text": "lexer::lex() + 1",
                    })
                elif fix_calls["n"] == 2:
                    tool_map["edit_file"].invoke({
                        "path": "src/lib.rs", "old_text": "lexer::lex() + 1", "new_text": "lexer::lex() + 2",
                    })
                else:
                    tool_map["edit_file"].invoke({
                        "path": "src/lexer.rs", "old_text": "{ 0 }", "new_text": "{ 1 }",
                    })
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_lex")
                tool_map["submit_for_qa"].invoke({"summary": "updated"})
            elif role == "verifier":
                ver_calls["n"] += 1
                verifier_prompts.append(msg)
                assert "search_code" in tool_map and "read_file" in tool_map
                if ver_calls["n"] <= 2:
                    assert "Issue-Referenced Source Files Untouched by Diff: ['src/lexer.rs']" in msg or "src/lexer.rs" in msg
                    tool_map["submit_qa_decision"].invoke({
                        "decision": "REJECT",
                        "summary": "Core lexer.rs file untouched",
                        "missing_requirements": ["Update src/lexer.rs directly"],
                        "objections": ["Downstream shim in lib.rs only"],
                    })
                else:
                    tool_map["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "Fixed root cause"})
            else:
                advisor_prompts.append(msg)
                assert "Issue Contract:" in msg
                assert "Recent Verifier QA Rejections:" in msg
                assert "Pre-loaded Source Excerpts" in msg
                tool_map["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1",
                    "arbitration_rationale": "Fix root cause in src/lexer.rs",
                    "ordered_edits": [{"file": "src/lexer.rs", "description": "return 1"}],
                })
            return {"messages": inputs["messages"] + [{"role": "assistant", "content": "done"}]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent, "create_agent", create)
    patch = agent.run_agent(repo, "Bug in `src/lexer.rs`: `lex` returns wrong value", "unused", "unused", max_api_calls=100)
    assert "+pub fn lex() -> i32 { 1 }" in patch
    assert seen == ["fixer", "verifier", "fixer", "verifier", "advisor", "fixer", "verifier"]
    assert len(advisor_prompts) == 1
    assert "Untouched" in verifier_prompts[0]


def test_continuous_repair_bounces_passed_on_original_regression_test(tmp_path, monkeypatch):
    agent = load("agent")
    repo = make_repo(tmp_path)
    monkeypatch.setenv("RUSTBENCH_REPAIR_MODE", "continuous")
    monkeypatch.setattr(agent, "ChatOpenAI", Mock())
    make_tools = agent.make_tools
    states = []
    seen = []

    def capture(*args, **kwargs):
        tools, state = make_tools(*args, **kwargs)
        state.post_strip_check = False
        state.require_reproduction = True
        state.run_existing_test_smoke = lambda _selected: []
        # Simulate pre-fix reproduction check reporting the regression test passed on unpatched original code
        def fake_verify(targets=None):
            for k in (targets or state.get_regression_targets().keys()):
                state._reproduced_regressions.pop(k, None)
                state._reproduction_notes[k] = "passed on unpatched original code"
            raise ValueError("Paired reproduction check failed: passed on original")
        state.verify_pre_fix_reproduction = fake_verify
        states.append(state)
        return tools, state

    monkeypatch.setattr(agent, "make_tools", capture)
    fix_calls = {"n": 0}

    def create(model, tools, system_prompt):
        tool_map = {item.name: item for item in tools}
        role = "fixer" if "one continuous repair conversation" in system_prompt else "verifier"
        seen.append(role)

        def invoke(inputs, config):
            if role == "fixer":
                fix_calls["n"] += 1
                if fix_calls["n"] == 1:
                    tool_map["write_regression_test"].invoke({
                        "path": "tests/test_regression_value.rs", "content": "#[test] fn t() { assert!(true); }",
                    })
                    tool_map["edit_file"].invoke({"path": "src/lib.rs", "old_text": "{ 0 }", "new_text": "{ 1 }"})
                    states[-1].pass_test(states[-1].start_test(), test_target="test_regression_value")
                    tool_map["submit_for_qa"].invoke({"summary": "first try"})
                else:
                    assert "PASSED on the unpatched original checkout" in inputs["messages"][-1]["content"]
                    # Now simulate genuine Red-Green reproduction
                    key = next(iter(states[-1].get_regression_targets().keys()))
                    states[-1].verify_pre_fix_reproduction = lambda _t=None: None
                    states[-1]._reproduction_notes.pop(key, None)
                    states[-1]._reproduced_regressions[key] = {
                        "clean_red": True, "runtime_failure": True, "compile_reproduction": False,
                        "test_sha256": states[-1]._target_sha256(key),
                    }
                    states[-1].pass_test(states[-1].start_test(), test_target="test_regression_value")
                    tool_map["submit_for_qa"].invoke({"summary": "reproduced and fixed"})
            else:
                tool_map["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "ok"})
            return {"messages": inputs["messages"] + [{"role": "assistant", "content": "ok"}]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent, "create_agent", create)
    patch = agent.run_agent(repo, "Make value return 1", "unused", "unused", max_api_calls=100)
    assert "+pub fn value() -> i32 { 1 }" in patch
    assert seen == ["fixer", "fixer", "verifier"]
