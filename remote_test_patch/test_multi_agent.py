"""Offline contract tests for the adaptive multi-agent architecture in baselines/test."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def staged_repair_mode(monkeypatch):
    monkeypatch.setenv("RUSTBENCH_REPAIR_MODE", "staged")


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_ma_test", Path(__file__).with_name(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_issue_contract_and_adaptive_routing(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")

    issue = "Add \x60timestamp_micros\x60 method to \x60Utc\x60 implementing \x60TimeZone\x60 trait with lifetime 'a."
    contract = ma.extract_issue_contract(issue)
    assert "timestamp_micros" in contract["requested_symbols"]
    assert "timestamp_micros" in contract["new_api_candidates"]
    assert contract["signals"]["mentions_trait_or_lifetime"] is True
    assert contract["signals"]["mentions_new_api"] is True

    # Simple single-file localized bug -> Fast Path
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("pub fn f() {}\n", encoding="utf-8")
    _, state = tools_mod.make_tools(tmp_path)
    try:
        bb_simple = {
            "issue_contract": ma.extract_issue_contract("Fix off-by-one in \x60parse_count\x60"),
            "localization": {
                "candidate_chains": [{"chain_id": "C1", "nodes": [{"path": "src/lib.rs", "symbol": "parse_count"}]}],
                "reproduction_status": "runtime_failure",
                "needs_architecture_planning": False,
            },
        }
        route_fast, _ = ma.route_after_navigation(bb_simple, state)
        assert route_fast == "fast"

        # Multi-file / multi-chain -> Planned Path
        bb_complex = {
            "issue_contract": contract,
            "localization": {
                "candidate_chains": [
                    {"chain_id": "C1", "nodes": [{"path": "src/a.rs", "symbol": "f1"}]},
                    {"chain_id": "C2", "nodes": [{"path": "src/b.rs", "symbol": "f2"}]},
                ],
                "reproduction_status": "runtime_failure",
                "needs_architecture_planning": False,
            },
        }
        route_planned, reasons = ma.route_after_navigation(bb_complex, state)
        assert route_planned == "planned"
        assert any("multiple candidate" in r for r in reasons)
    finally:
        state.close()


def test_allocate_stage_budgets_scales_cleanly():
    ma = _load("multi_agent")
    b150 = ma.allocate_stage_budgets(150)
    assert b150["navigator_cap"] == 32
    assert b150["suggester_cap"] == 8
    assert b150["fixer_attempt_cap"] == 57
    assert b150["verifier_reserve"] == 5

    b70 = ma.allocate_stage_budgets(70)
    assert b70["navigator_cap"] == 15
    assert b70["suggester_cap"] == 5
    assert b70["verifier_reserve"] == 3


def test_verifier_conflict_requires_evidence_and_supports_no_plan():
    ma = _load("multi_agent")
    tools, holder = ma._make_handoff_tools("verifier", None, {}, {})
    submit = tools[0]
    response = submit.invoke({"decision": "APPROVE", "summary": "ok", "plan_consistency": "conflict"})
    assert "provide" in response.lower() or "describe" in response.lower()
    assert not holder["done"]
    response = submit.invoke({"decision": "APPROVE", "summary": "fast path", "plan_consistency": "not_applicable"})
    assert "recorded" in response.lower()
    assert holder["payload"]["plan_consistency"] == "not_applicable"
    assert holder["payload"]["plan_conflicts"] == []
    tools, holder = ma._make_handoff_tools("verifier", None, {"repair_plan": {"action": "plan"}}, {})
    response = tools[0].invoke({"decision": "APPROVE", "summary": "ok", "plan_consistency": "not_applicable"})
    assert "plan exists" in response.lower()
    assert not holder["done"]


def test_qa_repair_feedback_is_actionable_and_blocks_unverified_fallback():
    ma = _load("multi_agent")
    payload = {
        "decision": "REJECT",
        "objections": ["diff shifts doctest line 108 to 121"],
        "missing_requirements": ["document grow behavior"],
        "plan_conflicts": ["ordered_edits[2] is absent from the diff"],
    }
    feedback = ma._format_qa_repair_feedback(payload, round_number=2)
    assert "Do not resubmit the same diff" in feedback
    assert "diff shifts doctest line 108 to 121" in feedback
    assert "document grow behavior" in feedback
    assert "ordered_edits[2] is absent from the diff" in feedback

    blackboard = {}
    ma._record_consistency_blocked(None, blackboard, "diff", payload,
                                   reason="budget exhausted", attempt=2)
    assert blackboard["consistency_blocked"] is True
    assert blackboard["consistency_block_reason"] == "budget exhausted"


def _setup_fake_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn bug() -> i32 { 0 }\n", encoding="utf-8")
    return repo


def test_multi_agent_fast_path_skips_suggester_and_submits(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)

    captured_roles = []
    states = []
    orig_make_tools = agent_mod.make_tools

    def cap_make_tools(r, *a, **kw):
        t, s = orig_make_tools(r, *a, **kw)
        s.post_strip_check = False
        s.require_reproduction = False
        states.append(s)
        return t, s

    monkeypatch.setattr(agent_mod, "make_tools", cap_make_tools)
    monkeypatch.setattr(agent_mod, "ChatOpenAI", Mock())

    def fake_create_agent(model, tools, system_prompt):
        tmap = {t.name: t for t in tools}
        if "Navigator & Reproducer" in system_prompt:
            role = "navigator"
        elif "Rust Architect / Suggester" in system_prompt:
            role = "suggester"
        elif "Rust Fixer" in system_prompt:
            role = "fixer"
        else:
            role = "verifier"
        captured_roles.append(role)

        def invoke(inputs, config):
            if role == "navigator":
                assert "edit_file" not in tmap
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_bug.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "bug in src/lib.rs",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "wrong return", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer":
                assert "generate_patch" not in tmap
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "fixed return value"})
            elif role == "verifier":
                tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "looks good"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    trace_path = tmp_path / "trace.jsonl"
    trace = agent_mod.TraceRecorder(trace_path)
    patch = agent_mod.run_agent(repo, "Fix bug() returning 0 instead of 1.", "m", "k", trace=trace, max_api_calls=70)
    assert "+pub fn bug() -> i32 { 1 }" in patch
    assert captured_roles == ["navigator", "fixer", "verifier"]
    events = [json.loads(line)["event"] for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert "multi_agent_start" in events
    assert "adaptive_route" in events
    assert "qa_gate" in events


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_visible_panic_mismatch_reaches_fixer_and_verifier(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "visible.rs").write_text(
        '#[test]\n#[should_panic(expected = "visible contract")]\n'
        'fn message_contract() { panic!("actual failure"); }\n', encoding="utf-8"
    )
    states = []
    original = agent_mod.make_tools

    def capture_tools(*args, **kwargs):
        tools, state = original(*args, **kwargs)
        state.post_strip_check = False
        state.require_reproduction = False
        states.append(state)
        return tools, state

    monkeypatch.setattr(agent_mod, "make_tools", capture_tools)
    monkeypatch.setattr(agent_mod, "ChatOpenAI", Mock())
    seen = []

    def create(model, tools, system_prompt):
        tmap = {t.name: t for t in tools}
        if "Navigator & Reproducer" in system_prompt:
            role = "navigator"
        elif "Rust Fixer" in system_prompt:
            role = "fixer"
        else:
            role = "verifier"
        seen.append(role)

        def invoke(inputs, config):
            prompt = inputs["messages"][-1]["content"]
            if role == "navigator":
                result = json.loads(tmap["cargo_test"].invoke({"test_target": "visible"}))
                assert result["failure_details"][0]["kind"] == "panic_message_mismatch"
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_bug.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "bug", "candidate_chains": [{"chain_id": "C1", "hypothesis": "bad result", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure", "needs_architecture_planning": False,
                })
            elif role == "fixer":
                assert "visible contract" in prompt and "actual failure" in prompt
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
                result = json.loads(tmap["cargo_test"].invoke({"test_target": "visible"}))
                assert result["failure_details"][0]["kind"] == "panic_message_mismatch"
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "fixed"})
            else:
                assert "visible contract" in prompt and "actual failure" in prompt
                assert "panic_message_mismatch" in prompt
                tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "ok"})
            return {"messages": inputs["messages"]}
        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", create)
    patch = agent_mod.run_agent(repo, "Fix bug()", "m", "k", max_api_calls=70)
    assert "+pub fn bug() -> i32 { 1 }" in patch
    assert seen == ["navigator", "fixer", "verifier"]


def test_multi_agent_planned_path_escalation_and_qa_rejection(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)

    captured_roles = []
    states = []
    orig_make_tools = agent_mod.make_tools

    def cap_make_tools(r, *a, **kw):
        t, s = orig_make_tools(r, *a, **kw)
        s.post_strip_check = False
        s.require_reproduction = False
        states.append(s)
        return t, s

    monkeypatch.setattr(agent_mod, "make_tools", cap_make_tools)
    monkeypatch.setattr(agent_mod, "ChatOpenAI", Mock())

    fixer_runs = {"n": 0}
    verifier_runs = {"n": 0}

    def fake_create_agent(model, tools, system_prompt):
        tmap = {t.name: t for t in tools}
        if "Navigator & Reproducer" in system_prompt:
            role = "navigator"
        elif "Rust Architect / Suggester" in system_prompt:
            role = "suggester"
        elif "Rust Fixer" in system_prompt:
            role = "fixer"
        else:
            role = "verifier"
        captured_roles.append(role)

        def invoke(inputs, config):
            if role == "navigator":
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_api.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "need new method",
                    "candidate_chains": [
                        {"chain_id": "C1", "hypothesis": "h1", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]},
                            {"chain_id": "C2", "hypothesis": "h2", "nodes": [{"path": "src/other.rs", "symbol": "bug"}]},
                    ],
                    "reproduction_status": "compile_reproduction",
                    "needs_architecture_planning": True,
                })
            elif role == "suggester":
                tmap["submit_repair_plan"].invoke({
                    "selected_chain_id": "C2" if fixer_runs["n"] > 0 else "C1",
                    "arbitration_rationale": "pick root and add the exact method",
                    "ordered_edits": [{"file": "src/lib.rs", "symbol": "bug", "description": "add timestamp_micros"}],
                    "forbidden_edits": ["do not rename bug"],
                })
            elif role == "fixer":
                fixer_runs["n"] += 1
                if fixer_runs["n"] == 1:
                    # Attempt 1 makes a bad edit then escalates
                    tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 999 }"})
                    tmap["escalate_for_replan"].invoke({"reason": "stuck on trait bound"})
                elif fixer_runs["n"] == 2:
                    # Attempt 2 forgot timestamp_micros -> QA should reject!
                    assert "pub fn bug() -> i32 { 999 }" in tmap["read_file"].invoke({"path": "src/lib.rs"})  # current edits are preserved across replanning
                    tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 999 }", "new_text": "pub fn timestamp_micros_opt() -> i32 { 1 }"})
                    states[-1].pass_test(states[-1].start_test(), test_target="test_regression_api")
                    tmap["submit_for_qa"].invoke({"summary": "added opt variant"})
                else:
                    # Attempt 3 fixes QA objection
                    assert "plan_conflicts" in inputs["messages"][-1]["content"]
                    assert "forbidden_edits" in inputs["messages"][-1]["content"]
                    tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn timestamp_micros_opt() -> i32 { 1 }", "new_text": "pub fn timestamp_micros() -> i32 { 1 }"})
                    states[-1].pass_test(states[-1].start_test(), test_target="test_regression_api")
                    tmap["submit_for_qa"].invoke({"summary": "added exact requested method"})
            elif role == "verifier":
                verifier_runs["n"] += 1
                if verifier_runs["n"] == 1:
                    # Confirm deterministic contract warning was injected into prompt!
                    assert "timestamp_micros" in inputs["messages"][-1]["content"]
                    assert "pick root and add the exact method" in inputs["messages"][-1]["content"]
                    assert "do not rename bug" in inputs["messages"][-1]["content"]
                    assert "+pub fn timestamp_micros_opt()" in inputs["messages"][-1]["content"]
                    tmap["submit_qa_decision"].invoke({
                        "decision": "REJECT",
                        "summary": "Missing exact public method",
                        "missing_requirements": ["timestamp_micros"],
                        "objections": ["Issue asked for timestamp_micros, only timestamp_micros_opt was added."],
                        "plan_consistency": "conflict",
                        "plan_conflicts": ["forbidden_edits says do not rename bug; diff replaces bug in src/lib.rs"],
                    })
                else:
                    tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "All requirements satisfied.",
                        "plan_consistency": "conflict",
                        "plan_conflicts": ["forbidden_edits says do not rename bug; diff replaces bug in src/lib.rs"]})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    issue = "Add \x60timestamp_micros\x60 method to \x60Utc\x60."
    trace_path = tmp_path / "plan_audit.jsonl"
    patch = agent_mod.run_agent(repo, issue, "m", "k", max_api_calls=70,
                                trace=agent_mod.TraceRecorder(trace_path))
    assert "pub fn timestamp_micros() -> i32 { 1 }" in patch
    assert captured_roles == ["navigator", "suggester", "fixer", "suggester", "fixer", "verifier", "fixer", "verifier"]
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    audits = [event for event in events if event["event"] == "plan_consistency_audit"]
    assert len(audits) == 2 and all(event["status"] == "conflict" for event in audits)
    assert "forbidden_edits" in audits[0]["conflicts"][0]
    assert audits[0]["plan"]["present"] and audits[0]["patch"]["present"]
    handoffs = [event for event in events if event["event"] == "handoff_feature_snapshot" and event.get("stage") == "verifier_handoff"]
    assert handoffs[0]["features"]["plan_conflicts"]["items"] == audits[0]["conflicts"]


def test_deterministic_qa_rejects_dbg_macro_leftover(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        tools["write_regression_test"].invoke({"path": "tests/test_regression_x.rs", "content": "#[test] fn t() {}"})
        tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "pub fn bug() -> i32 { 0 }",
            "new_text": "pub fn bug() -> i32 { dbg!(1) }",
        })
        state.pass_test(state.start_test(), test_target="test_regression_x")
        res = ma.run_deterministic_qa_checks(state, ma.extract_issue_contract("fix bug"))
        assert res["ok"] is False
        assert "dbg!" in res["reason"]
    finally:
        state.close()


def test_multi_agent_auto_escalates_on_repeated_compile_errors_and_recovers(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    captured_roles = []
    states = []
    orig_make_tools = agent_mod.make_tools

    def cap_make_tools(r, *a, **kw):
        t, s = orig_make_tools(r, *a, **kw)
        s.post_strip_check = False
        s.require_reproduction = False
        states.append(s)
        return t, s

    monkeypatch.setattr(agent_mod, "make_tools", cap_make_tools)
    monkeypatch.setattr(agent_mod, "ChatOpenAI", Mock())

    # Mock cargo_check subprocess to fail with identical E0308 for first 3 edits
    diag_mod_spec = importlib.util.spec_from_file_location("diag_stub", Path(__file__).with_name("rust_diagnostics.py"))
    import subprocess
    fake_err = json.dumps({
        "reason": "compiler-message",
        "message": {
            "level": "error",
            "code": {"code": "E0308"},
            "message": "mismatched types: expected i32, found ()",
            "spans": [{"file_name": "src/lib.rs", "line_start": 1, "is_primary": True}],
            "children": [],
        },
    })

    fixer_runs = {"n": 0}

    def fake_create_agent(model, tools, system_prompt):
        tmap = {t.name: t for t in tools}
        if "Navigator & Reproducer" in system_prompt:
            role = "navigator"
        elif "Rust Architect / Suggester" in system_prompt:
            role = "suggester"
        elif "Rust Fixer" in system_prompt:
            role = "fixer"
        else:
            role = "verifier"
        captured_roles.append(role)

        def invoke(inputs, config):
            if role == "navigator":
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_bug.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "single file",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "h1", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "suggester":
                # Triggered after Fast Path Fixer escalates!
                assert "E0308" in inputs["messages"][-1]["content"]
                tmap["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1",
                    "arbitration_rationale": "return i32 instead of unit",
                    "ordered_edits": [{"file": "src/lib.rs", "symbol": "bug", "description": "return 1"}],
                })
            elif role == "fixer":
                fixer_runs["n"] += 1
                if fixer_runs["n"] == 1:
                    real_run = subprocess.run
                    failures_left = [3]
                    def cargo_fail_run(cmd, *a, **kw):
                        if cmd and str(cmd[0]) == "cargo" and failures_left[0]:
                            failures_left[0] -= 1
                            return SimpleNamespace(stdout=fake_err + "\n", stderr="error", returncode=101)
                        return real_run(cmd, *a, **kw)
                    monkeypatch.setattr(subprocess, "run", cargo_fail_run)
                    for v in ("()", "();", "() "):
                        tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": tmap["read_lines"].invoke({"path": "src/lib.rs"}).split(": ", 1)[1], "new_text": f"pub fn bug() -> i32 {{ {v} }}"})
                        tmap["cargo_check"].invoke({})
                else:
                    assert states[-1].stuck_status()["compile_repeats"] <= 1
                    tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { ()  }", "new_text": "pub fn bug() -> i32 { 1 }"})
                    states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                    tmap["submit_for_qa"].invoke({"summary": "fixed"})
            elif role == "verifier":
                tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "ok"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    patch = agent_mod.run_agent(repo, "Fix bug() returning wrong value", "m", "k", max_api_calls=70)
    assert "+pub fn bug() -> i32 { 1 }" in patch
    # Started on Fast Path (navigator -> fixer), auto-escalated to suggester -> fixer -> verifier!
    assert captured_roles == ["navigator", "fixer", "suggester", "fixer", "verifier"]


def test_multi_agent_auto_submits_verified_snapshot_on_api_limit(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    states = []
    orig_make_tools = agent_mod.make_tools

    def cap_make_tools(r, *a, **kw):
        t, s = orig_make_tools(r, *a, **kw)
        s.post_strip_check = False
        s.require_reproduction = False
        states.append(s)
        return t, s

    monkeypatch.setattr(agent_mod, "make_tools", cap_make_tools)
    monkeypatch.setattr(agent_mod, "ChatOpenAI", Mock())

    def fake_create_agent(model, tools, system_prompt):
        tmap = {t.name: t for t in tools}
        if "Navigator & Reproducer" in system_prompt:
            role = "navigator"
        elif "Rust Fixer" in system_prompt:
            role = "fixer"
        else:
            role = "verifier"

        def invoke(inputs, config):
            if role == "navigator":
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_bug.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "single file",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "h", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer":
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "fixed"})
            elif role == "verifier":
                raise RuntimeError("Model API call limit reached (10)")
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    trace = Mock()
    patch = agent_mod.run_agent(repo, "Fix bug()", "m", "k", trace=trace, max_api_calls=10)
    assert "+pub fn bug() -> i32 { 1 }" in patch
    trace.record.assert_any_call("auto_submit_on_limit", patch=patch)


def test_role_trace_callback_and_stage_budget_guard(tmp_path):
    agent_mod = _load("agent")
    ma = _load("multi_agent")
    trace_mod = _load("agent_trace")
    from langchain_core.runnables import RunnableLambda

    trace_path = tmp_path / "role_trace.jsonl"
    rec = trace_mod.TraceRecorder(trace_path)
    cb = trace_mod.RoleTraceCallback(rec, role="suggester", attempt=2)
    cb.on_tool_start({"name": "inspect_rust_neighborhood"}, "Worker", run_id="00000000-0000-0000-0000-000000000001")
    events = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["agent_role"] == "suggester"
    assert events[0]["attempt"] == 2

    # Verify _guarded_bind raises _StageBudgetReached cleanly when stage cap is hit
    class FakeLLM:
        def bind_tools(self, tools, **kw):
            return RunnableLambda(lambda msgs: "ok")

    fake_client = SimpleNamespace(calls_made=10, remaining_calls=50)
    prompts_mod = _load("agent_prompts")
    holder = {"done": False, "payload": None}

    def fake_create_agent(model, tools, system_prompt):
        bound = model.bind_tools(tools)
        def invoke(inputs, config):
            # Simulate 5 calls used inside this stage (cap=5 -> should trigger _StageBudgetReached!)
            fake_client.calls_made = 15
            bound.invoke(inputs["messages"])
            return inputs
        return SimpleNamespace(invoke=invoke)

    payload, outcome = ma._run_role_agent(
        "navigator", prompts_mod.NAVIGATOR_PROMPT, "hi", [], holder,
        model="m", api_key="k", base_url=None, client=fake_client, trace=rec,
        compress_fn=lambda m: m, create_agent_fn=fake_create_agent,
        chat_model_factory=lambda **kw: FakeLLM(), role_callback_cls=trace_mod.RoleTraceCallback,
        stage_cap=5, reserve_floor=4,
    )
    assert payload is None
    assert outcome == "stage_cap_reached"


def test_issue_contract_avoids_false_trait_flag_and_captures_expected_result(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)

    issue = (
        "advance_mut implementation of BufMut for Vec<u8> is inconsistent.\n\n"
        "### Expected result\n"
        "Unterminated string literal should be `JsonBogusValue`."
    )
    contract = ma.extract_issue_contract(issue)
    assert contract["signals"]["mentions_trait_or_lifetime"] is False
    assert "JsonBogusValue" in contract["expected_result_excerpt"]

    _, state = tools_mod.make_tools(repo)
    state._find_downstream_workspace_packages = lambda ms: ["cli_pkg"]
    try:
        bb = {
            "issue_contract": contract,
            "localization": {
                "candidate_chains": [{"chain_id": "C1", "nodes": [{"path": "src/lib.rs", "symbol": "advance_mut"}]}],
                "reproduction_status": "runtime_failure",
                "needs_architecture_planning": False,
            },
        }
        route, _ = ma.route_after_navigation(bb, state)
        assert route == "fast"
    finally:
        state.close()


def test_forced_handoff_wrapup_captures_repair_plan_on_stage_cap(tmp_path):
    ma = _load("multi_agent")
    trace_mod = _load("agent_trace")
    prompts_mod = _load("agent_prompts")
    from langchain_core.runnables import RunnableLambda

    rec = trace_mod.TraceRecorder(tmp_path / "wrap_trace.jsonl")
    fake_client = SimpleNamespace(calls_made=0, remaining_calls=40)

    class FakeLLM:
        def bind_tools(self, tools, **kw):
            return RunnableLambda(lambda msgs: {"tools": [t.name for t in tools]})

    blackboard = {"issue_contract": {"requested_symbols": []}, "localization": {}}
    sug_activity = {"files_read": [], "symbols_queried": []}
    sug_handoff, sug_holder = ma._make_handoff_tools("suggester", SimpleNamespace(), blackboard, sug_activity)
    seen_wrap_tools = []

    def fake_create_agent(model, tools, system_prompt):
        bound = model.bind_tools(tools)
        tmap = {t.name: t for t in tools}

        def invoke(inputs, config):
            if "submit_repair_plan" in tmap and len(tmap) == 2:
                # Forced wrap-up invocation: ONLY handoff tools are provided!
                seen_wrap_tools.append(sorted(tmap.keys()))
                fake_client.calls_made += 1
                tmap["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1",
                    "arbitration_rationale": "root cause in lib.rs",
                    "ordered_edits": [{"file": "src/lib.rs", "symbol": "bug", "description": "fix"}],
                })
                return inputs
            # Main exploration loop hits explore_cap (stage_cap - 1 = 3)
            fake_client.calls_made += 3
            bound.invoke(inputs["messages"])
            return inputs

        return SimpleNamespace(invoke=invoke)

    payload, outcome = ma._run_role_agent(
        "suggester", prompts_mod.SUGGESTER_PROMPT, "plan", [], sug_holder,
        model="m", api_key="k", base_url=None, client=fake_client, trace=rec,
        compress_fn=lambda m: m, create_agent_fn=fake_create_agent,
        chat_model_factory=lambda **kw: FakeLLM(), role_callback_cls=trace_mod.RoleTraceCallback,
        stage_cap=4, reserve_floor=3, handoff_only_tools=sug_handoff,
        wrapup_prompt="Call submit_repair_plan now.",
    )
    assert outcome == "completed"
    assert payload is not None
    assert payload["ordered_edits"][0]["file"] == "src/lib.rs"
    assert seen_wrap_tools == [["request_relocalization", "submit_repair_plan"]]


def test_prune_unreproduced_targets_and_qa_surfaces_existing_failures_and_inline_tests(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=True)
    tools = {t.name: t for t in tools_list}
    try:
        # Navigator creates one dead orphan test and one valid Red test
        tools["write_regression_test"].invoke({"path": "tests/test_regression_orphan.rs", "content": "#[test] fn x() {}"})
        tools["write_regression_test"].invoke({"path": "tests/test_regression_valid.rs", "content": "#[test] fn y() {}"})
        state.fail_test(state.start_test(), test_target="test_regression_valid", executed=1, failures=["y"], returncode=101)

        pruned = ma._prune_unreproduced_regression_targets(state)
        assert pruned == ["tests/test_regression_orphan.rs"]
        assert not (state.root / "tests" / "test_regression_orphan.rs").exists()
        assert list(state.get_regression_targets().keys()) == [(Path("tests/../Cargo.toml".replace("tests/../", "")), "test_regression_valid")]

        # Fixer edits src/lib.rs adding inline #[cfg(test)] -> must be deterministically rejected
        tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "pub fn bug() -> i32 { 0 }",
            "new_text": "pub fn bug() -> i32 { 1 }\n#[cfg(test)]\nmod extra { #[test] fn t() {} }",
        })
        state.pass_test(state.start_test(), test_target="test_regression_valid")
        det_qa_inline = ma.run_deterministic_qa_checks(state, ma.extract_issue_contract("Fix bug()"))
        assert det_qa_inline["ok"] is False
        assert "Inline test(s)" in det_qa_inline["reason"]

        # Clean edit without inline tests passes det_qa while still surfacing existing_test_failures
        tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "pub fn bug() -> i32 { 1 }\n#[cfg(test)]\nmod extra { #[test] fn t() {} }",
            "new_text": "pub fn bug() -> i32 { 1 }",
        })
        state.pass_test(state.start_test(), test_target="test_regression_valid")
        state.fail_test(state.start_test(), test_target="test_existing", executed=2, failures=["test_contract_panic"], returncode=101)

        det_qa = ma.run_deterministic_qa_checks(state, ma.extract_issue_contract("Fix bug()"))
        assert det_qa["ok"] is True
        assert det_qa["existing_test_failures"]
        assert det_qa["existing_test_failures"][0]["failures"] == ["test_contract_panic"]
    finally:
        state.close()


def test_reserve_reached_runs_single_rescue_without_zero_call_spin(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    states = []
    orig_make_tools = agent_mod.make_tools

    def cap_make_tools(r, *a, **kw):
        t, s = orig_make_tools(r, *a, **kw)
        s.post_strip_check = False
        s.require_reproduction = False
        states.append(s)
        return t, s

    monkeypatch.setattr(agent_mod, "make_tools", cap_make_tools)
    monkeypatch.setattr(agent_mod, "ChatOpenAI", Mock())

    # Simulate client hitting reserve=3 on first Fixer pass while edit already exists, then finishing on 1 rescue call!
    fake_client = SimpleNamespace(calls_made=0, remaining_calls=30, max_calls=30, close=lambda: None)
    ma_mod = agent_mod._multi_agent

    fixer_calls = []

    def fake_create_agent(model, tools, system_prompt):
        tmap = {t.name: t for t in tools}
        if "Navigator & Reproducer" in system_prompt:
            role = "navigator"
        elif "Rust Fixer" in system_prompt:
            role = "fixer"
        else:
            role = "verifier"

        def invoke(inputs, config):
            if role == "navigator":
                fake_client.calls_made += 5
                fake_client.remaining_calls = 25
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_bug.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "single file",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "h", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer":
                fixer_calls.append(fake_client.remaining_calls)
                if len(fixer_calls) == 1:
                    # Edit source, consume calls down to reserve_floor=3, and trigger _StageReserveReached
                    tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
                    fake_client.calls_made = 27
                    fake_client.remaining_calls = 3
                    raise ma_mod._StageReserveReached("fixer", 3, 3)
                else:
                    # Rescue pass must NOT have escalate_for_replan and has remaining=3
                    assert "escalate_for_replan" not in tmap
                    assert "LOW BUDGET RESCUE" in inputs["messages"][-1]["content"]
                    fake_client.calls_made = 29
                    fake_client.remaining_calls = 1
                    states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                    tmap["submit_for_qa"].invoke({"summary": "rescue passed"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    monkeypatch.setattr(agent_mod, "TraceHTTPClient", lambda trace=None, max_calls=30: fake_client)
    patch = agent_mod.run_agent(repo, "Fix bug()", "m", "k", max_api_calls=30)
    assert "+pub fn bug() -> i32 { 1 }" in patch
    assert fixer_calls == [25, 3]



def test_single_quoted_prose_not_treated_as_lifetime_and_fast_path_ignores_test_and_intermediate_nodes(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")

    issue = "Installer prints 'command not found' when 'cache' is missing; works on 'a' too."
    contract = ma.extract_issue_contract(issue)
    assert contract["signals"]["mentions_trait_or_lifetime"] is False

    # Workspace with a downstream crate depending on core_pkg
    repo = tmp_path / "ws"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[workspace]\nmembers = ["core_pkg", "cli_pkg"]\n', encoding="utf-8")
    (repo / "core_pkg" / "src").mkdir(parents=True)
    (repo / "core_pkg" / "Cargo.toml").write_text('[package]\nname = "core_pkg"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "core_pkg" / "src" / "lib.rs").write_text("pub fn core() -> i32 { 1 }\n", encoding="utf-8")
    (repo / "cli_pkg" / "src").mkdir(parents=True)
    (repo / "cli_pkg" / "Cargo.toml").write_text(
        '[package]\nname = "cli_pkg"\nversion = "0.1.0"\n[dependencies]\ncore_pkg = { path = "../core_pkg" }\n',
        encoding="utf-8",
    )
    (repo / "cli_pkg" / "src" / "main.rs").write_text("fn main() { let _ = core_pkg::core(); }\n", encoding="utf-8")

    _, state = tools_mod.make_tools(repo)
    try:
        bb = {
            "issue_contract": contract,
            "localization": {
                "candidate_chains": [{
                    "chain_id": "C1",
                    "nodes": [
                        {"path": "core_pkg/src/lib.rs", "role": "root_candidate"},
                        {"path": "cli_pkg/src/main.rs", "role": "entry_or_symptom"},
                        {"path": "tests/test_regression_x.rs", "role": "candidate"},
                    ],
                }],
                "reproduction_status": "not_reproduced",
                "needs_architecture_planning": False,
            },
        }
        route, reasons = ma.route_after_navigation(bb, state)
        assert route == "fast", reasons
        assert "cli_pkg" in bb.get("downstream_dependents", [])
    finally:
        state.close()


def test_mixed_code_doctest_shift_is_nonblocking_and_secondary_prune_works(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = tmp_path / "doc_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "doc_demo"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text(
        "/// Doc note line 1\n"
        "/// Doc note line 2\n"
        "pub fn bug() -> i32 { 0 }\n\n"
        "/// ```\n"
        "/// assert_eq!(doc_demo::other(), 2);\n"
        "/// ```\n"
        "pub fn other() -> i32 { 2 }\n",
        encoding="utf-8",
    )
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        tools["write_regression_test"].invoke({"path": "tests/test_regression_main.rs", "content": "#[test] fn a() {}"})
        tools["write_regression_test"].invoke({"path": "tests/test_regression_extra_tmp.rs", "content": "#[test] fn b() {}"})
        # A mixed code/doc hunk is reported as a warning but must remain submittable.
        tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "edits": [
                {"old_text": "/// Doc note line 1\n/// Doc note line 2", "new_text": "/// Doc note line 1\n/// Doc note line 2\n/// Extra wrapped line"},
                {"old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"},
            ],
        })
        state.pass_test(state.start_test(), test_target="test_regression_main")

        # Secondary unverified test should be pruned cleanly once a primary test passed
        pruned = ma._prune_secondary_unverified_regression_targets(state)
        assert pruned == ["tests/test_regression_extra_tmp.rs"]
        assert not (state.root / "tests" / "test_regression_extra_tmp.rs").exists()

        det_mixed = ma.run_deterministic_qa_checks(state, ma.extract_issue_contract("Fix bug()"), run_doctest_on_shift=False)
        assert det_mixed["ok"] is True

        # Revert doc comment to line-neutral (delta=0) -> passes deterministic QA!
        tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "/// Doc note line 1\n/// Doc note line 2\n/// Extra wrapped line",
            "new_text": "/// Doc note updated 1\n/// Doc note updated 2",
        })
        state.pass_test(state.start_test(), test_target="test_regression_main")
        det_ok = ma.run_deterministic_qa_checks(state, ma.extract_issue_contract("Fix bug()"), run_doctest_on_shift=False)
        assert det_ok["ok"] is True

        # Verify degraded fallback returns patch when at least 1 regression test passed
        patch = ma._try_degraded_patch_fallback(state)
        assert "+pub fn bug() -> i32 { 1 }" in patch
    finally:
        state.close()
def test_wrapup_retry_recovers_after_malformed_handoff_and_synthesizes_edits(tmp_path):
    ma = _load("multi_agent")
    trace_mod = _load("agent_trace")
    prompts_mod = _load("agent_prompts")
    from langchain_core.runnables import RunnableLambda

    rec = trace_mod.TraceRecorder(tmp_path / "wrap_retry.jsonl")
    fake_client = SimpleNamespace(calls_made=0, remaining_calls=40)

    class FakeLLM:
        def bind_tools(self, tools, **kw):
            return RunnableLambda(lambda msgs: {"tools": [t.name for t in tools]})

    blackboard = {"issue_contract": {"requested_symbols": []}, "localization": {}}
    sug_handoff, sug_holder = ma._make_handoff_tools("suggester", SimpleNamespace(), blackboard, {})
    wrap_attempts = {"n": 0}

    def fake_create_agent(model, tools, system_prompt):
        bound = model.bind_tools(tools)
        tmap = {t.name: t for t in tools}

        def invoke(inputs, config):
            if "submit_repair_plan" in tmap and len(tmap) == 2:
                wrap_attempts["n"] += 1
                fake_client.calls_made += 1
                if wrap_attempts["n"] == 1:
                    # Simulate malformed tool call JSON (holder['done'] stays False)
                    return inputs
                assert "failed JSON parsing" in inputs["messages"][-1]["content"]
                tmap["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1",
                    "arbitration_rationale": "concise rationale",
                    "ordered_edits": [{"file": "crates/config/src/combined.rs", "symbol": "scan", "description": "fix suppression"}],
                })
                return inputs
            fake_client.calls_made += 3
            bound.invoke(inputs["messages"])
            return inputs

        return SimpleNamespace(invoke=invoke)

    payload, outcome = ma._run_role_agent(
        "suggester", prompts_mod.SUGGESTER_PROMPT, "plan", [], sug_holder,
        model="m", api_key="k", base_url=None, client=fake_client, trace=rec,
        compress_fn=lambda m: m, create_agent_fn=fake_create_agent,
        chat_model_factory=lambda **kw: FakeLLM(), role_callback_cls=trace_mod.RoleTraceCallback,
        stage_cap=4, reserve_floor=3, handoff_only_tools=sug_handoff,
    )
    assert outcome == "completed"
    assert wrap_attempts["n"] == 2
    assert payload["ordered_edits"][0]["file"] == "crates/config/src/combined.rs"

    synth = ma._synthesize_ordered_edits_from_chains([
        {
            "chain_id": "C1",
            "hypothesis": "bug here",
            "nodes": [
                {"path": "tests/test_a.rs", "symbol": "t"},
                {"path": "crates/cli/src/scan.rs", "symbol": "run_scan", "line_start": 10, "line_end": 20},
            ],
        }
    ])
    assert len(synth) == 1
    assert synth[0]["file"] == "crates/cli/src/scan.rs"


def test_deterministic_qa_rejects_modified_inline_tests_for_additive_cli_and_cleans_scratch_fallback(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")

    repo = tmp_path / "cli_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "cli_demo"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text(
        "pub struct Dep { pub branch: Option<String> }\n"
        "impl Dep {\n"
        "    pub fn set_git(mut self, branch: Option<String>) -> Self { self.branch = branch; self }\n"
        "}\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    #[test]\n"
        "    fn test_set_git() {\n"
        "        let _ = super::Dep { branch: None }.set_git(None);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "templates").mkdir()
    (repo / "templates" / "install.sh.j2").write_text("echo ok\n", encoding="utf-8")

    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        tools["write_regression_test"].invoke({"path": "tests/test_regression_rev.rs", "content": "#[test] fn t() {}"})
        tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "edits": [
                {
                    "old_text": "    pub fn set_git(mut self, branch: Option<String>) -> Self { self.branch = branch; self }",
                    "new_text": "    pub fn set_git(mut self, branch: Option<String>, _rev: Option<String>) -> Self { self.branch = branch; self }",
                },
                {
                    "old_text": "        let _ = super::Dep { branch: None }.set_git(None);",
                    "new_text": "        let _ = super::Dep { branch: None }.set_git(None, None);",
                },
            ],
        })
        state.pass_test(state.start_test(), test_target="test_regression_rev")
        contract = ma.extract_issue_contract("Add another argument like `--branch` to specify `--rev`")
        assert contract["signals"]["mentions_cli_or_additive_flag"] is True
        det_bad = ma.run_deterministic_qa_checks(state, contract, run_doctest_on_shift=False)
        assert det_bad["ok"] is False
        assert "Modified existing inline test(s)" in det_bad["reason"]

        # Fallback cleanup strips scratch helper script and preserves valid template edit
        tools["revert_file"].invoke({"path": "src/lib.rs"})
        tools["edit_file"].invoke({"path": "templates/install.sh.j2", "old_text": "echo ok", "new_text": "echo fixed"})
        tools["write_file"].invoke({"path": "_splice_snaps.py", "content": "print(1)\n"})
        state.pass_test(state.start_test(), test_target="test_regression_rev")
        raw_fb = state.auto_submit_verified_patch(allow_snapshot_fallback=True)
        assert "_splice_snaps.py" in raw_fb
        cleaned = ma._clean_and_audit_fallback_patch(state, ma.extract_issue_contract("Fix template"), raw_fb)
        assert "_splice_snaps.py" not in cleaned
        assert "templates/install.sh.j2" in cleaned
        assert not (state.root / "_splice_snaps.py").exists()
    finally:
        state.close()


def test_prune_invalid_regression_targets_and_env_blocker_auto_escalation(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")

    repo = _setup_fake_repo(tmp_path)
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        # Simulate a zombie target marked with "no test target named" in _reproduction_notes
        tools["write_regression_test"].invoke({"path": "tests/test_regression_valid.rs", "content": "#[test] fn a() {}"})
        tools["write_regression_test"].invoke({"path": "tests/test_regression_bad.rs", "content": "#[test] fn b() {}"})
        bad_key = (Path("Cargo.toml"), "test_regression_bad")
        state._reproduction_notes[bad_key] = "no test target named `test_regression_bad`"
        pruned = ma._prune_invalid_regression_targets(state)
        assert pruned == ["tests/test_regression_bad.rs"]
        assert not (state.root / "tests" / "test_regression_bad.rs").exists()

        # Verify repeated env dependency blocker in _wrap_tools_for_role triggers _AutoEscalate
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def fake_cargo_test() -> str:
            """Fake cargo_test."""
            return 'error: failed to select a version for the requirement `git-config = "~0.16.3"`'

        blackboard = {"environment_blocker": False}
        activity = {}
        wrapped = ma._wrap_tools_for_role(
            "fixer",
            {"cargo_test": fake_cargo_test},
            ["cargo_test"],
            [],
            {},
            state,
            SimpleNamespace(calls_made=0),
            0,
            20,
            activity,
            auto_escalate_on_stuck=True,
            blackboard=blackboard,
        )[0]
        r1 = wrapped.invoke({})
        assert "Environment Blocker Hint" not in r1
        assert blackboard["environment_blocker"] is True
        r2 = wrapped.invoke({})
        assert "Environment Blocker Hint" in r2
        with pytest.raises(ma._AutoEscalate):
            wrapped.invoke({})
    finally:
        state.close()



def test_degraded_compiling_snapshot_fallback_and_issue_audit_signals(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")

    issue = (
        "Bug in https://github.com/org/repo/blob/abc1234/crates/lexer/src/mod.rs#L10-L20: "
        "unterminated string emits wrong token."
    )
    contract = ma.extract_issue_contract(issue)
    assert "crates/lexer/src/mod.rs" in contract["referenced_files"]

    repo = _setup_fake_repo(tmp_path)
    (repo / "src" / "extra.rs").write_text("pub fn extra() -> i32 { 1 }\n", encoding="utf-8")
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        # Secondary in-module target pruning strips the dead block while keeping primary verified target
        tools["write_regression_test"].invoke({"path": "tests/test_regression_main.rs", "content": "#[test] fn a() {}"})
        tools["write_regression_test"].invoke({
            "path": "src/extra.rs",
            "target_src": "src/extra.rs",
            "content": "#[test] fn dead_in_mod() {}",
        })
        tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
        state.pass_test(state.start_test(), test_target="test_regression_main")
        pruned = ma._prune_secondary_unverified_regression_targets(state)
        assert pruned == ["src/extra.rs::__test_regression_extra"]
        assert "__TEST_REGRESSION_START" not in (state.root / "src" / "extra.rs").read_text(encoding="utf-8")

        # Simulate clean compiling snapshot followed by accidental restore('pre_fix') and recovery via allow_unverified_compiling
        state.save_checkpoint("pre_fix")
        state.record_check_result(0, [])
        assert state._last_compiling_prod_snapshot is not None
        state.restore_checkpoint("pre_fix")
        state.revert_file_unused = None
        # Restore original src/lib.rs manually to simulate pre_fix state
        (state.root / "src" / "lib.rs").write_text("pub fn bug() -> i32 { 0 }\n", encoding="utf-8")
        state._passed_regressions.clear()
        state.post_strip_check = True
        state.run_post_strip_check = lambda selected: None
        recovered = ma._try_degraded_patch_fallback(state, allow_unverified_compiling=True)
        assert "+pub fn bug() -> i32 { 1 }" in recovered
    finally:
        state.close()

def test_p0_handoff_feature_snapshots_recorded_on_fast_and_planned_paths(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    states = []
    orig_make_tools = agent_mod.make_tools

    def cap_make_tools(r, *a, **kw):
        t, s = orig_make_tools(r, *a, **kw)
        s.post_strip_check = False
        s.require_reproduction = False
        states.append(s)
        return t, s

    monkeypatch.setattr(agent_mod, "make_tools", cap_make_tools)
    monkeypatch.setattr(agent_mod, "ChatOpenAI", Mock())

    fixer_runs = {"n": 0}
    verifier_runs = {"n": 0}

    def fake_create_agent(model, tools, system_prompt):
        tmap = {t.name: t for t in tools}
        if "Navigator & Reproducer" in system_prompt:
            role = "navigator"
        elif "Rust Architect / Suggester" in system_prompt:
            role = "suggester"
        elif "Rust Fixer" in system_prompt:
            role = "fixer"
        else:
            role = "verifier"

        def invoke(inputs, config):
            if role == "navigator":
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_api.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "need new method",
                    "candidate_chains": [
                        {"chain_id": "C1", "hypothesis": "h1", "nodes": [{"path": "src/lib.rs", "symbol": "bug", "role": "root_candidate"}]},
                            {"chain_id": "C2", "hypothesis": "h2", "nodes": [{"path": "src/other.rs", "symbol": "bug"}]},
                    ],
                    "reproduction_status": "compile_reproduction",
                    "needs_architecture_planning": True,
                })
            elif role == "suggester":
                tmap["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1",
                    "arbitration_rationale": "add missing public api",
                    "ordered_edits": [{"file": "src/lib.rs", "symbol": "bug", "description": "add timestamp_micros"}],
                    "rust_invariants": ["preserve doctest line numbers"],
                    "api_checklist": ["timestamp_micros"],
                    "forbidden_edits": ["tests/"],
                })
            elif role == "fixer":
                fixer_runs["n"] += 1
                if fixer_runs["n"] == 1:
                    tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 2 }"})
                    tmap["escalate_for_replan"].invoke({"reason": "need exact api name"})
                else:
                    tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 2 }", "new_text": "pub fn timestamp_micros() -> i32 { 1 }"})
                    states[-1].pass_test(states[-1].start_test(), test_target="test_regression_api")
                    tmap["submit_for_qa"].invoke({"summary": "added timestamp_micros"})
            elif role == "verifier":
                verifier_runs["n"] += 1
                if verifier_runs["n"] == 1:
                    tmap["submit_qa_decision"].invoke({
                        "decision": "REJECT",
                        "summary": "Check edge case",
                        "missing_requirements": ["edge_case"],
                        "objections": ["Verify edge case"],
                    })
                else:
                    tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "verified"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    trace_path = tmp_path / "p0_trace.jsonl"
    trace = agent_mod.TraceRecorder(trace_path)
    issue = "Add `timestamp_micros` method to `Utc`."
    patch = agent_mod.run_agent(repo, issue, "m", "k", trace=trace, max_api_calls=70)
    assert "pub fn timestamp_micros() -> i32 { 1 }" in patch

    snapshots = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("event") == "handoff_feature_snapshot"
    ]
    stages = [s["stage"] for s in snapshots]
    assert "issue_contract" in stages
    assert "navigator_handoff" in stages
    assert "suggester_handoff" in stages
    assert "fixer_handoff" in stages
    assert "verifier_handoff" in stages

    issue_snap = next(s for s in snapshots if s["stage"] == "issue_contract")
    assert issue_snap["features"]["new_api_candidates"]["present"] is True
    assert "timestamp_micros" in issue_snap["features"]["new_api_candidates"]["items"]
    assert len(issue_snap["features"]["issue_text"]["sha256"]) == 16

    nav_snap = next(s for s in snapshots if s["stage"] == "navigator_handoff")
    assert nav_snap["route"] == "planned"
    assert nav_snap["outcome"] == "completed"
    assert nav_snap["features"]["candidate_chains"]["chain_count"] == 2
    assert nav_snap["features"]["candidate_chains"]["root_candidates"][0]["path"] == "src/lib.rs"

    sug_snap = next(s for s in snapshots if s["stage"] == "suggester_handoff")
    assert sug_snap["action"] == "plan"
    assert sug_snap["sug_prompt_breakdown"]["total_chars"] > 0
    assert sug_snap["features"]["api_checklist"]["items"] == ["timestamp_micros"]
    assert sug_snap["features"]["forbidden_edits"]["count"] == 1

    fix_snaps = [s for s in snapshots if s["stage"] == "fixer_handoff"]
    assert any(s["action"] == "escalate" and s["to_role"] == "suggester" for s in fix_snaps)
    qa_fix_snap = next(s for s in fix_snaps if s["to_role"] == "qa")
    assert qa_fix_snap["fix_prompt_breakdown"]["total_chars"] > 0
    assert qa_fix_snap["features"]["deterministic_qa"]["ok"] is True
    assert qa_fix_snap["features"]["changed_files"]["paths"] == ["src/lib.rs"]

    ver_snaps = [s for s in snapshots if s["stage"] == "verifier_handoff"]
    assert [s["decision"] for s in ver_snaps] == ["REJECT", "APPROVE"]
    assert ver_snaps[0]["features"]["objections"]["items"] == ["Verify edge case"]
    assert ver_snaps[1]["ver_prompt_chars"] > 0



