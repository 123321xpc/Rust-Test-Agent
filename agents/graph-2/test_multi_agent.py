"""Offline contract tests for the adaptive multi-agent architecture in baselines/test."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


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

    blackboard = {
        "repair_plan": {"ordered_edits": [{"file": "src/lib.rs", "symbol": "timestamp_micros"}]},
        "issue_contract": ma.extract_issue_contract("Add `timestamp_micros` to satisfy the requested behavior."),
    }
    tools, holder = ma._make_handoff_tools("verifier", None, blackboard, {})
    submit = tools[0]
    conflict = {
        "decision": "APPROVE",
        "summary": "The plan is stale but implementation is correct.",
        "plan_consistency": "conflict",
        "plan_conflicts": ["The plan did not include the final API edit."],
    }
    response = submit.invoke(conflict)
    assert "explain how the conflict relates" in response.lower()
    assert not holder["done"]
    conflict["explanation"] = (
        "The issue contract requests timestamp_micros, and the regression test verifies the final API behavior."
    )
    response = submit.invoke(conflict)
    assert "recorded" in response.lower()
    assert holder["payload"]["explanation"] == conflict["explanation"]


def test_verifier_consistent_is_normalized_when_no_architect_plan_exists():
    """Without a plan, 'consistent' is meaningless; it must not burn a verifier round."""
    ma = _load("multi_agent")
    tools, holder = ma._make_handoff_tools("verifier", None, {}, {})
    submit = tools[0]
    response = submit.invoke({
        "decision": "APPROVE", "summary": "fast path looks right",
        "plan_consistency": "consistent",
    })
    assert "recorded" in response.lower()
    assert holder["done"]
    assert holder["payload"]["plan_consistency"] == "not_applicable"
    assert holder["payload"]["plan_conflicts"] == []


def test_suggester_can_release_a_locked_regression_once():
    ma = _load("multi_agent")
    released = {}

    def unlock(name, reason, approved_by=""):
        if released:
            return {"ok": False, "error": "already unlocked"}
        released["name"] = name
        released["reason"] = reason
        released["approved_by"] = approved_by
        return {"ok": True, "unlocked": name, "path": "tests/test_regression_x.rs", "reason": reason}

    state = SimpleNamespace(unlock_regression=unlock)
    tools, holder = ma._make_handoff_tools("suggester", state, {}, {})
    release = next(t for t in tools if t.name == "release_locked_regression")
    response = release.invoke({
        "name": "test_regression_x",
        "reason": "The locked test asserted the wrong crate path for the public API.",
    })
    assert "Released" in response
    assert holder["done"]
    assert holder["payload"]["action"] == "unlock"
    assert holder["payload"]["unlocked"] == "test_regression_x"
    assert released["approved_by"] == "architect"


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


def test_submission_gates_reject_docs_and_existing_tests_but_allow_production_edits():
    ma = _load("multi_agent")
    contract = ma.extract_issue_contract("Fix `bug` behavior")
    docs = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n"
    assert ma._detect_non_substantive_patch(docs, issue_contract=contract)

    comments = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-// old\n+// new\n"
    assert ma._detect_non_substantive_patch(comments, issue_contract=contract)

    string_literal = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-pub fn bug() { let _ = 1; }\n+pub fn bug() { let _ = \"//\"; }\n"
    assert ma._detect_non_substantive_patch(string_literal, issue_contract=contract) == []

    template = "--- a/templates/install.sh.j2\n+++ b/templates/install.sh.j2\n@@ -1 +1 @@\n-echo old\n+echo new\n"
    assert ma._detect_non_substantive_patch(template, issue_contract=contract) == []

    state = SimpleNamespace(regression_tests=set(), in_module_regressions={}, originals={})
    existing_test = "--- a/src/tests.rs\n+++ b/src/tests.rs\n@@ -1 +1 @@\n-#[test]\n+#[test]\n"
    assert ma._detect_benchmark_test_conflicts(existing_test, state, contract)
    test_contract = ma.extract_issue_contract("Update the existing snapshot test expected output")
    assert ma._detect_benchmark_test_conflicts(existing_test, state, test_contract) == []


def test_submission_gate_blocks_current_code_smoke_but_warns_baseline_and_environment():
    ma = _load("multi_agent")
    state = SimpleNamespace(_existing_test_failures=[], _revision=4)
    patch = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-pub fn bug() { 0 }\n+pub fn bug() { 1 }\n"
    selected = [Path("src/lib.rs")]
    code = {"target": "test:unit", "status": "failed", "classification": "code", "baseline_status": "passed", "observed_revision": 4, "details": "error[E0061]: this function takes 1 argument"}
    gate = ma._submission_gate_report(state, ma.extract_issue_contract("Fix bug"), patch, selected, [code])
    assert gate["blocking"] and "test:unit" in gate["blocking"][0]
    runtime = {"target": "test:snapshot", "status": "failed", "classification": "code", "baseline_status": "passed", "observed_revision": 4, "details": "test snapshot_powershell ... FAILED"}
    gate_rt = ma._submission_gate_report(state, ma.extract_issue_contract("Fix bug"), patch, selected, [runtime])
    assert gate_rt["blocking"] and "New existing-test regression" in gate_rt["blocking"][0]
    new_fail = {"target": "test:snap", "status": "failed", "classification": "code", "baseline_status": "failed", "observed_revision": 4, "new_failed_tests": ["snap_b"], "details": "FAILED"}
    assert ma._submission_gate_report(state, ma.extract_issue_contract("Fix bug"), patch, selected, [new_fail])["blocking"]
    baseline = dict(code, baseline_status="failed")
    env = {"target": "test:network", "status": "environment_error", "classification": "environment", "observed_revision": 4, "details": "download failed"}
    gate = ma._submission_gate_report(state, ma.extract_issue_contract("Fix bug"), patch, selected, [baseline, env])
    assert gate["blocking"] == []
    assert len(gate["warnings"]) == 2


def test_inline_test_overlap_ignores_context_only_hunks():
    """Production code added next to a `#[cfg(test)]` block is not an inline-test edit."""
    ma = _load("multi_agent")
    original = (
        "pub fn compute() -> i32 {\n"
        "    1\n"
        "}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    #[test]\n"
        "    fn keeps_working() {}\n"
        "}\n"
    )
    state = SimpleNamespace(
        regression_tests=set(), in_module_regressions={},
        originals={Path("src/lib.rs"): (original.encode("utf-8"), 0, 0)},
    )
    # A hunk whose context reaches into the test module but only edits production code.
    patch = (
        "--- a/src/lib.rs\n"
        "+++ b/src/lib.rs\n"
        "@@ -1,6 +1,8 @@\n"
        " pub fn compute() -> i32 {\n"
        "-    1\n"
        "+    2\n"
        "+    let extra = 3;\n"
        " }\n"
        " \n"
        " #[cfg(test)]\n"
    )
    assert ma._detect_modified_inline_tests(patch, state=state) == []


def test_inline_test_overlap_still_flags_real_edits_inside_the_test_module():
    ma = _load("multi_agent")
    original = (
        "pub fn compute() -> i32 {\n"
        "    1\n"
        "}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    #[test]\n"
        "    fn keeps_working() {}\n"
        "}\n"
    )
    state = SimpleNamespace(
        regression_tests=set(), in_module_regressions={},
        originals={Path("src/lib.rs"): (original.encode("utf-8"), 0, 0)},
    )
    patch = (
        "--- a/src/lib.rs\n"
        "+++ b/src/lib.rs\n"
        "@@ -6,5 +6,5 @@\n"
        " mod tests {\n"
        "     #[test]\n"
        "-    fn keeps_working() {}\n"
        "+    fn keeps_working() { assert!(true); }\n"
        " }\n"
    )
    findings = ma._detect_modified_inline_tests(patch, state=state)
    assert findings, "an edit inside `mod tests` must still be reported"


def test_touched_line_hits_inline_tests_uses_gap_and_line_semantics():
    ma = _load("multi_agent")
    ranges = [(5, 9)]
    # Removing a line inside the range is a hit.
    assert ma._touched_line_hits_inline_tests((("line", 7),), ranges)
    # Inserting after a line inside the range is a hit.
    assert ma._touched_line_hits_inline_tests((("gap", 5),), ranges)
    # Inserting after the last line of the range is a hit.
    assert ma._touched_line_hits_inline_tests((("gap", 8),), ranges)
    # Inserting after the line just past the range is not a hit.
    assert not ma._touched_line_hits_inline_tests((("gap", 9),), ranges)
    # A pure context line is not a hit.
    assert not ma._touched_line_hits_inline_tests((("line", 3),), ranges)


def test_deterministic_qa_and_verified_fallback_cannot_bypass_smoke_gate(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {item.name: item for item in tools_list}
    try:
        tools["write_regression_test"].invoke({
            "path": "tests/test_regression_gate.rs",
            "content": "#[test] fn regression_gate() { assert_eq!(demo::bug(), 1); }",
        })
        tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
        state.pass_test(state.start_test(), test_target="test_regression_gate")
        state.run_existing_test_smoke = lambda selected: [{
            "target": "test:unit", "status": "failed", "classification": "code",
            "baseline_status": "passed", "details": "error[E0061]: wrong argument count",
        }]

        qa = ma.run_deterministic_qa_checks(state, ma.extract_issue_contract("Fix bug()"), run_doctest_on_shift=False)
        assert qa["ok"] is False
        assert "Submission gate failed" in qa["reason"]

        raw = state.auto_submit_verified_patch(allow_snapshot_fallback=True)
        assert raw
        assert ma._clean_and_audit_fallback_patch(state, ma.extract_issue_contract("Fix bug()"), raw) == ""
    finally:
        state.close()


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
                        "explanation": "The issue contract requires timestamp_micros, and verification shows the diff replaces bug contrary to the plan's forbidden edit.",
                    })
                else:
                    tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "All requirements satisfied.",
                        "plan_consistency": "conflict",
                        "plan_conflicts": ["forbidden_edits says do not rename bug; diff replaces bug in src/lib.rs"],
                        "explanation": "The issue contract is satisfied by the final API, while verification confirms the stale plan conflict does not affect the requested behavior."})
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
                        tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": tmap["read_lines"].invoke({"path": "src/lib.rs"}).split("\n1: ", 1)[1], "new_text": f"pub fn bug() -> i32 {{ {v} }}"})
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
    sug_handoff_names = {t.name for t in sug_handoff}
    seen_wrap_tools = []

    def fake_create_agent(model, tools, system_prompt):
        bound = model.bind_tools(tools)
        tmap = {t.name: t for t in tools}

        def invoke(inputs, config):
            if "submit_repair_plan" in tmap and set(tmap) == set(sug_handoff_names):
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
    assert seen_wrap_tools == [sorted(sug_handoff_names)]


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
    sug_handoff_names = {t.name for t in sug_handoff}
    wrap_attempts = {"n": 0}

    def fake_create_agent(model, tools, system_prompt):
        bound = model.bind_tools(tools)
        tmap = {t.name: t for t in tools}

        def invoke(inputs, config):
            if "submit_repair_plan" in tmap and set(tmap) == sug_handoff_names:
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


def test_fixer_read_only_streak_warns_and_auto_escalates(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        # auto_escalate_on_stuck=False emits pacing warning at 10+ reads and resets on edit_file
        activity_warn = {"files_read": [], "symbols_queried": []}
        wrapped_warn = {
            t.name: t
            for t in ma._wrap_tools_for_role(
                "fixer", tools, ["read_lines", "edit_file"], [], {}, state,
                SimpleNamespace(calls_made=0), 0, 100, activity_warn,
                auto_escalate_on_stuck=False, blackboard={},
            )
        }
        for _ in range(9):
            out = wrapped_warn["read_lines"].invoke({"path": "src/lib.rs", "start_line": 1, "end_line": 1})
            assert "Pacing warning" not in out
        out10 = wrapped_warn["read_lines"].invoke({"path": "src/lib.rs", "start_line": 1, "end_line": 1})
        assert "Pacing warning" in out10
        wrapped_warn["edit_file"].invoke({"path": "src/lib.rs", "old_text": "{ 0 }", "new_text": "{ 1 }"})
        assert activity_warn["read_only_streak"] == 0

        # auto_escalate_on_stuck=True triggers _AutoEscalate at 16 consecutive read-only calls
        activity_esc = {"files_read": [], "symbols_queried": []}
        wrapped_esc = ma._wrap_tools_for_role(
            "fixer", tools, ["read_lines"], [], {}, state,
            SimpleNamespace(calls_made=0), 0, 100, activity_esc,
            auto_escalate_on_stuck=True, blackboard={},
        )[0]
        for _ in range(15):
            wrapped_esc.invoke({"path": "src/lib.rs", "start_line": 1, "end_line": 1})
        with pytest.raises(ma._AutoEscalate, match="read-only exploration stalled"):
            wrapped_esc.invoke({"path": "src/lib.rs", "start_line": 1, "end_line": 1})
    finally:
        state.close()


def test_unreproduced_reason_requires_evidence_and_degraded_guards_hygiene(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    trace_mod = _load("agent_trace")
    repo = _setup_fake_repo(tmp_path)
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        contract = ma.extract_issue_contract("Fix `bug` returning 0 instead of 1")
        bb = {"environment_blocker": False, "qa_history": [{"decision": "REJECT", "objections": ["wrong return"]}]}
        # Fabricated excuses without corroborating evidence must be rejected
        assert not ma._is_valid_external_unreproduced_reason(
            "Cannot run tests because the environment is offline and network is blocked.",
            contract, state=state, blackboard=bb, issue_text="Fix bug returning 0 instead of 1",
        )
        assert not ma._is_valid_external_unreproduced_reason(
            "This bug only happens on windows terminal gui.",
            contract, state=state, blackboard=bb, issue_text="Fix bug returning 0 instead of 1",
        )
        # Real environment_blocker on blackboard makes offline claim valid
        bb["environment_blocker"] = True
        assert ma._is_valid_external_unreproduced_reason(
            "Cannot run tests because the offline Cargo registry lacks dependency.",
            contract, state=state, blackboard=bb, issue_text="Fix bug returning 0 instead of 1",
        )

        # Degraded fallback rejects dbg! leftovers, then accepts clean edit with QA metadata
        tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { dbg!(1) }"})
        assert ma._try_degraded_patch_fallback(state, issue_contract=contract, blackboard=bb) == ""

        tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { dbg!(1) }", "new_text": "pub fn bug() -> i32 { 1 }"})
        rec_path = tmp_path / "degraded.jsonl"
        rec = trace_mod.TraceRecorder(rec_path)
        patch = ma._try_degraded_patch_fallback(state, trace=rec, issue_contract=contract, blackboard=bb)
        assert "+pub fn bug() -> i32 { 1 }" in patch
        events = [json.loads(ln) for ln in rec_path.read_text(encoding="utf-8").splitlines()]
        ev = next(e for e in events if e["event"] == "auto_submit_degraded")
        assert ev["verification"] == "unverified_after_qa_reject"
        assert ev["last_qa_decision"] == "REJECT"
    finally:
        state.close()


def test_handoff_card_unit_scoped_and_shared_memory_prioritizes_root_files():
    ma = _load("multi_agent")
    contract = ma.extract_issue_contract("Update `src/core.rs` to add `api_a` and `api_b`.")
    bb = {
        "issue_contract": contract,
        "localization": {
            "summary": "root in core.rs",
            "candidate_chains": [{"chain_id": "C1", "nodes": [{"path": "src/core.rs", "role": "root_candidate"}]}],
        },
        "repair_plan": {
            "selected_chain_id": "C1",
            "ordered_edits": [{"file": f"src/f{i}.rs", "symbol": f"s{i}", "action": "modify", "description": f"step {i}"} for i in range(7)],
            "api_checklist": ["api_a", "api_b", "api_c"],
            "rust_invariants": [f"inv_{i}" for i in range(5)],
            "forbidden_edits": [f"forb_{i}" for i in range(5)],
        },
        "repair_units": {"units": [{"unit_id": "u0", "status": "verified"}]},
    }
    active_unit = {
        "unit_id": "u1",
        "expected_outcome": "api_b compiles cleanly",
        "work": [{"file": f"src/f{i}.rs", "symbol": f"s{i}", "action": "modify", "description": f"step {i}"} for i in range(6)],
        "paths": ["src/f4.rs", "src/f5.rs"],
    }
    card = ma.build_handoff_card(contract, bb, active_unit=active_unit)
    assert "api_c" in card
    assert "step 5" in card
    assert "verified_units=['u0']" in card
    assert "inv_4" in card and "forb_4" in card

    # Shared memory eviction keeps root_candidate/referenced_files over older non-priority reads
    bb["shared_read_excerpts"] = {"src/core.rs:1-20": "core code"}
    activity = {"read_excerpts": {f"src/other_{i}.rs:1-10": f"other {i}" for i in range(15)}}
    ma._merge_activity_into_blackboard(bb, activity)
    assert "src/core.rs:1-20" in bb["shared_read_excerpts"]
    assert len(bb["shared_read_excerpts"]) == 14


def test_graph_agent_direct_fixer_to_navigator_skips_suggester(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    (repo / "src" / "other.rs").write_text("pub fn real_bug() -> i32 { 0 }\n", encoding="utf-8")

    roles = []
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
        roles.append(role)
        nav_count = sum(1 for r in roles if r == "navigator")
        fix_count = sum(1 for r in roles if r == "fixer")

        def invoke(inputs, config):
            if role == "navigator" and nav_count == 1:
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_bug.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "initial",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "in lib", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer" and fix_count == 1:
                tmap["escalate_for_replan"].invoke({"reason": "wrong file localization; real_bug is elsewhere"})
            elif role == "navigator" and nav_count == 2:
                tmap["submit_localization_bundle"].invoke({
                    "summary": "relocalized to other.rs",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "in other", "nodes": [{"path": "src/other.rs", "symbol": "real_bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer" and fix_count == 2:
                tmap["edit_file"].invoke({"path": "src/other.rs", "old_text": "pub fn real_bug() -> i32 { 0 }", "new_text": "pub fn real_bug() -> i32 { 1 }"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "fixed real_bug"})
            elif role == "verifier":
                tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "ok"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    trace_path = tmp_path / "graph_trace.jsonl"
    trace = agent_mod.TraceRecorder(trace_path)
    patch = agent_mod.run_agent(repo, "Fix real_bug()", "m", "k", trace=trace, max_api_calls=70)
    assert "+pub fn real_bug() -> i32 { 1 }" in patch
    assert roles == ["navigator", "fixer", "navigator", "fixer", "verifier"]
    events = [json.loads(ln) for ln in trace_path.read_text(encoding="utf-8").splitlines()]
    transitions = [(e["from_role"], e["to_role"], e["relation"]) for e in events if e["event"] == "graph_transition"]
    assert ("fixer", "navigator", "relocalize_direct") in transitions
    assert ("navigator", "fixer", "relocalized_fast") in transitions


def test_graph_agent_unlocks_by_test_fn_name_and_preserves_plan(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)

    roles = []
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
        roles.append(role)
        fix_count = sum(1 for r in roles if r == "fixer")

        def invoke(inputs, config):
            prompt = inputs["messages"][-1]["content"]
            if role == "navigator":
                tmap["write_regression_test"].invoke({
                    "path": "tests/test_regression_vec_advance_mut.rs",
                    "content": "#[test]\nfn vec_advance_mut_wrong_oracle() { assert_eq!(1, 2); }\n",
                })
                states[-1].fail_test(
                    states[-1].start_test(),
                    test_target="test_regression_vec_advance_mut",
                    executed=1,
                    failures=["vec_advance_mut_wrong_oracle"],
                    returncode=101,
                )
                states[-1].lock_regressions()
                tmap["submit_localization_bundle"].invoke({
                    "summary": "bug in src/lib.rs",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "return 1", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer" and fix_count == 1:
                tmap["escalate_for_replan"].invoke({"reason": "Locked regression oracle vec_advance_mut_wrong_oracle is wrong"})
            elif role == "suggester":
                assert "Locked Red-phase Regression Test(s)" in prompt
                # Unlock using the Rust #[test] fn name AND submit a repair plan in the same turn
                res = tmap["release_locked_regression"].invoke({
                    "name": "vec_advance_mut_wrong_oracle",
                    "reason": "The locked test asserts 1 == 2 which is factually impossible under the API contract.",
                })
                assert "Released the Red-phase lock" in res
                tmap["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1",
                    "arbitration_rationale": "Update oracle and fix bug() to return 1.",
                    "ordered_edits": [{"file": "src/lib.rs", "symbol": "bug", "action": "modify", "description": "Return 1"}],
                })
            elif role == "fixer" and fix_count == 2:
                assert "RED-PHASE LOCK RELEASED BY THE ARCHITECT" in prompt
                tmap["write_regression_test"].invoke({
                    "path": "tests/test_regression_vec_advance_mut.rs",
                    "content": "#[test]\nfn vec_advance_mut_fixed_oracle() { assert_eq!(1, 1); }\n",
                })
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_vec_advance_mut")
                tmap["submit_for_qa"].invoke({"summary": "fixed bug with updated oracle"})
            elif role == "verifier":
                tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "ok", "plan_consistency": "consistent"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    patch = agent_mod.run_agent(repo, "Fix bug()", "m", "k", max_api_calls=70)
    assert "+pub fn bug() -> i32 { 1 }" in patch
    assert roles == ["navigator", "fixer", "suggester", "fixer", "verifier"]


def test_graph_agent_verifier_escalates_directly_to_navigator(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    (repo / "src" / "other.rs").write_text("pub fn target() -> i32 { 0 }\n", encoding="utf-8")

    roles = []
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
        roles.append(role)
        nav_count = sum(1 for r in roles if r == "navigator")
        fix_count = sum(1 for r in roles if r == "fixer")
        ver_count = sum(1 for r in roles if r == "verifier")

        def invoke(inputs, config):
            if role == "navigator" and nav_count == 1:
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_bug.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "wrong file",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "lib", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer" and fix_count == 1:
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "edited lib"})
            elif role == "verifier" and ver_count in (1, 2):
                tmap["submit_qa_decision"].invoke({
                    "decision": "REJECT",
                    "summary": "wrong module",
                    "missing_requirements": ["fix target in other.rs (mislocalized wrong file)"],
                    "objections": ["wrong file src/lib.rs edited; mislocalized"],
                    "plan_consistency": "not_applicable",
                })
            elif role == "fixer" and fix_count == 2:
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "resubmit"})
            elif role == "navigator" and nav_count == 2:
                tmap["submit_localization_bundle"].invoke({
                    "summary": "found target in other.rs",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "other", "nodes": [{"path": "src/other.rs", "symbol": "target"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer" and fix_count == 3:
                tmap["edit_file"].invoke({"path": "src/other.rs", "old_text": "pub fn target() -> i32 { 0 }", "new_text": "pub fn target() -> i32 { 1 }"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "fixed other.rs"})
            elif role == "verifier" and ver_count == 3:
                tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "ok", "plan_consistency": "not_applicable"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    trace_path = tmp_path / "ver_nav.jsonl"
    trace = agent_mod.TraceRecorder(trace_path)
    patch = agent_mod.run_agent(repo, "Fix `src/other.rs` target()", "m", "k", trace=trace, max_api_calls=70)
    assert "+pub fn target() -> i32 { 1 }" in patch
    assert roles == ["navigator", "fixer", "verifier", "fixer", "verifier", "navigator", "fixer", "verifier"]
    events = [json.loads(ln) for ln in trace_path.read_text(encoding="utf-8").splitlines()]
    transitions = [(e["from_role"], e["to_role"], e["relation"]) for e in events if e["event"] == "graph_transition"]
    assert ("verifier", "navigator", "qa_escalate_navigator") in transitions


def test_route_after_navigation_respects_signals_planning_and_ckg_consumers(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    _, state = tools_mod.make_tools(repo)
    try:
        base_loc = {
            "candidate_chains": [{"chain_id": "C1", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
            "reproduction_status": "runtime_failure",
            "needs_architecture_planning": False,
        }
        # needs_architecture_planning=True forces planned
        route, reasons = ma.route_after_navigation({
            "issue_contract": ma.extract_issue_contract("Fix off-by-one in `bug`"),
            "localization": {**base_loc, "needs_architecture_planning": True},
        }, state)
        assert route == "planned" and any("architecture planning" in r for r in reasons)

        # Single-file CLI additive option stays on Fast Path when 1 file / 1 chain / no new API
        route_cli_single, _ = ma.route_after_navigation({
            "issue_contract": ma.extract_issue_contract("Support `--allow-dirty` CLI option in `run`"),
            "localization": base_loc,
        }, state)
        assert route_cli_single == "fast"

        # Single-file mentions_new_api stays on Fast Path; multi-file mentions_new_api routes to Planned Path
        route_api_single, _ = ma.route_after_navigation({
            "issue_contract": {"signals": {"mentions_new_api": True}},
            "localization": base_loc,
        }, state)
        assert route_api_single == "fast"

        multi_loc = {
            "candidate_chains": [{
                "chain_id": "C1",
                "nodes": [
                    {"path": "src/lib.rs", "symbol": "bug", "role": "root_candidate"},
                    {"path": "src/cli.rs", "symbol": "run", "role": "root_candidate"},
                ],
            }],
            "reproduction_status": "runtime_failure",
            "needs_architecture_planning": False,
        }
        route_api, api_reasons = ma.route_after_navigation({
            "issue_contract": {"signals": {"mentions_new_api": True}},
            "localization": multi_loc,
        }, state)
        assert route_api == "planned" and any("new public API" in r for r in api_reasons)

        # Multi-file CKG impact surface forces planned, while 1-file CKG consumer fan-out stays fast
        orig_builder = ma._rust_impact.build_impact_surface
        try:
            state.ckg = SimpleNamespace()
            ma._rust_impact.build_impact_surface = lambda _ckg, _paths: {
                "consumer_paths": ["src/other1.rs", "src/other2.rs"],
                "trait_impls": [],
            }
            route_ckg_single, _ = ma.route_after_navigation({
                "issue_contract": ma.extract_issue_contract("Fix off-by-one in `bug`"),
                "localization": base_loc,
            }, state)
            assert route_ckg_single == "fast"
            route_ckg_multi, ckg_reasons = ma.route_after_navigation({
                "issue_contract": ma.extract_issue_contract("Fix off-by-one in `bug`"),
                "localization": multi_loc,
            }, state)
            assert route_ckg_multi == "planned" and any("CKG impact" in r for r in ckg_reasons)
        finally:
            ma._rust_impact.build_impact_surface = orig_builder
    finally:
        state.close()


def test_scheduled_fixer_omits_duplicate_user_prompt_on_subsequent_slices(tmp_path, monkeypatch):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    (repo / "src" / "b.rs").write_text("pub fn b() -> i32 { 0 }\n", encoding="utf-8")
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        board = {"localization": {"candidate_chains": []}, "repair_units": {}}
        coord = ma._repair_units.RepairUnitCoordinator(state, tools, board)
        coord.configure({
            "ordered_edits": [
                {"file": "src/lib.rs", "description": "step 1"},
                {"file": "src/b.rs", "description": "step 2"},
            ],
            "repair_units": [
                {"unit_id": "u1", "edit_indices": [0]},
                {"unit_id": "u2", "edit_indices": [1], "depends_on": ["u1"]},
            ],
        }, [])
        prompts_seen = []

        def fake_role(_role, _sys, user_prompt, _tools, holder, **kw):
            prompts_seen.append(user_prompt)
            if len(prompts_seen) == 1:
                tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "{ 0 }", "new_text": "{ 1 }"})
                holder["history"] = [{"role": "user", "content": user_prompt}]
                return {"action": "unit_complete", "summary": "u1"}, "completed"
            return {"action": "qa", "summary": "done"}, "completed"

        monkeypatch.setattr(ma, "_run_role_agent", fake_role)
        monkeypatch.setattr(coord, "verify", lambda complete=False, summary="": (True, "ok"))
        payload, _ = ma._run_scheduled_fixer(
            "sys", "UNIQUE_INITIAL_USER_PROMPT", [], {},
            coordinator=coord, allow_escalate=True, client=SimpleNamespace(calls_made=0), stage_cap=20,
        )
        assert payload["action"] == "qa"
        assert "UNIQUE_INITIAL_USER_PROMPT" in prompts_seen[0]
        assert "UNIQUE_INITIAL_USER_PROMPT" not in prompts_seen[1]
        assert "Continue" in prompts_seen[1]
    finally:
        state.close()


def test_graph_agent_red_lock_routes_unlock_followup_and_stall_breaker(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)

    roles = []
    states = []
    fixer_prompts = []
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
        roles.append(role)
        sug_count = sum(1 for r in roles if r == "suggester")
        fix_count = sum(1 for r in roles if r == "fixer")
        ver_count = sum(1 for r in roles if r == "verifier")

        def invoke(inputs, config):
            prompt = inputs["messages"][-1]["content"]
            if role == "navigator":
                tmap["write_regression_test"].invoke({
                    "path": "tests/test_regression_x.rs",
                    "content": "#[test]\nfn oracle_x() { assert_eq!(1, 2); }\n",
                })
                states[-1].fail_test(
                    states[-1].start_test(), test_target="test_regression_x",
                    executed=1, failures=["oracle_x"], returncode=101,
                )
                states[-1].lock_regressions()
                tmap["submit_localization_bundle"].invoke({
                    "summary": "bug in lib.rs",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "in lib", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer" and fix_count == 1:
                fixer_prompts.append(prompt)
                # Fixer makes a partial edit, hits Red-phase lock violation, and submits unverified
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { -1 }"})
                tmap["write_regression_test"].invoke({
                    "path": "tests/test_regression_x.rs",
                    "content": "#[test]\nfn oracle_x() { assert_eq!(1, 1); }\n",
                })
                tmap["submit_for_qa"].invoke({"summary": "partial edit blocked by locked test"})
            elif role == "suggester" and sug_count == 1:
                # First call unlocks ONLY (terminates turn with action='unlock')
                tmap["release_locked_regression"].invoke({
                    "name": "oracle_x",
                    "reason": "The locked test asserted 1 == 2 which violates the documented API.",
                })
            elif role == "suggester" and sug_count == 2:
                # Automatic follow-up turn submits the RepairPlan
                assert "Red-phase regression lock has been released" in prompt
                tmap["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1",
                    "arbitration_rationale": "Update test and fix bug()",
                    "ordered_edits": [{"file": "src/lib.rs", "symbol": "bug", "action": "modify", "description": "Return 1"}],
                })
            elif role == "fixer" and fix_count == 2:
                fixer_prompts.append(prompt)
                tmap["write_regression_test"].invoke({
                    "path": "tests/test_regression_x.rs",
                    "content": "#[test]\nfn oracle_x() { assert_eq!(1, 1); }\n",
                })
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { -1 }", "new_text": "pub fn bug() -> i32 { 1 }"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_x")
                tmap["submit_for_qa"].invoke({"summary": "fixed"})
            elif role == "verifier" and ver_count >= 1:
                # Rejections with no progress: #1 -> qa_reject_fix, #2 -> qa_escalate_architect, #3 -> qa_stall_stop
                tmap["submit_qa_decision"].invoke({
                    "decision": "REJECT",
                    "summary": "missing edge check",
                    "missing_requirements": ["check overflow"],
                    "objections": ["edge case"],
                    "plan_consistency": "consistent",
                })
            elif role == "suggester" and sug_count == 3:
                tmap["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1",
                    "arbitration_rationale": "Verify edge case without reverting working edit",
                    "ordered_edits": [{"file": "src/lib.rs", "symbol": "bug", "action": "modify", "description": "Keep 1"}],
                })
            elif role == "fixer" and fix_count >= 3:
                fixer_prompts.append(prompt)
                # Resubmit same state without edits -> same _repair_progress_signature
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_x")
                tmap["submit_for_qa"].invoke({"summary": "resubmit unchanged"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    trace_path = tmp_path / "opt_trace.jsonl"
    trace = agent_mod.TraceRecorder(trace_path)
    patch = agent_mod.run_agent(repo, "Fix `bug`", "m", "k", trace=trace, max_api_calls=100)
    assert "+pub fn bug() -> i32 { 1 }" in patch
    assert "Current Candidate Unified Diff" in fixer_prompts[-1]
    events = [json.loads(ln) for ln in trace_path.read_text(encoding="utf-8").splitlines()]
    transitions = [(e["from_role"], e["to_role"], e["relation"]) for e in events if e["event"] == "graph_transition"]
    assert ("fixer", "suggester", "replan_red_lock") in transitions
    assert ("verifier", "__end__", "qa_stall_stop") in transitions


def test_graph_agent_prunes_zero_progress_replan_and_emits_coordination_metrics(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)

    roles = []
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
        roles.append(role)
        sug_count = sum(1 for r in roles if r == "suggester")
        fix_count = sum(1 for r in roles if r == "fixer")

        def invoke(inputs, config):
            if role == "navigator":
                tmap["read_lines"].invoke({"path": "src/lib.rs", "start_line": 1, "end_line": 5})
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_bug.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "bug in lib.rs",
                    "candidate_chains": [
                        {"chain_id": "C1", "hypothesis": "h1", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]},
                        {"chain_id": "C2", "hypothesis": "h2", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]},
                    ],
                    "needs_architecture_planning": True,
                    "reproduction_status": "runtime_failure",
                })
            elif role == "suggester":
                tmap["submit_repair_plan"].invoke({
                    "selected_chain_id": "C1",
                    "arbitration_rationale": f"Plan attempt {sug_count}",
                    "ordered_edits": [{"file": "src/lib.rs", "symbol": "bug", "action": "modify", "description": "Return 1"}],
                })
            elif role == "fixer" and fix_count == 1:
                tmap["read_lines"].invoke({"path": "src/lib.rs", "start_line": 1, "end_line": 5})
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
                tmap["escalate_for_replan"].invoke({"reason": "trait bound mismatch on attempt 1"})
            elif role == "fixer" and fix_count == 2:
                # Zero milestone progress since replan -> second escalation to suggester must be pruned
                tmap["escalate_for_replan"].invoke({"reason": "trait bound still stuck on attempt 2"})
            elif role == "fixer" and fix_count >= 3:
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "fixed"})
            elif role == "verifier":
                tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "ok", "plan_consistency": "consistent"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    trace_path = tmp_path / "prune_trace.jsonl"
    trace = agent_mod.TraceRecorder(trace_path)
    patch = agent_mod.run_agent(repo, "Fix `bug`", "m", "k", trace=trace, max_api_calls=120)
    assert "+pub fn bug() -> i32 { 1 }" in patch
    assert roles.count("suggester") == 2  # initial plan + 1 replan; 2nd replan pruned
    events = [json.loads(ln) for ln in trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(e["event"] == "graph_edge_pruned" and e["to_role"] == "suggester" for e in events)
    coord_ev = next(e for e in events if e["event"] == "coordination_metrics")
    assert coord_ev["pruned_replans"] >= 1
    assert coord_ev["duplicate_reads"] >= 1
    assert "overall_kpi" in coord_ev


def test_graph_agent_relocalize_fast_keeps_history_clean_and_injects_sibling_contracts(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    (repo / "src" / "other.rs").write_text(
        'pub struct Buf;\nimpl Buf {\n    pub fn advance_mut(&mut self, cnt: usize) {\n        if cnt > 0 { panic!("cannot advance past capacity"); }\n    }\n}\n',
        encoding="utf-8",
    )
    (repo / "src" / "lib.rs").write_text(
        "pub mod other;\npub fn bug() -> i32 { 0 }\npub fn advance_mut(_cnt: usize) {}\n",
        encoding="utf-8",
    )

    roles = []
    fixer_histories_seen = []
    fixer_prompts_seen = []
    verifier_prompts_seen = []
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
        roles.append(role)
        nav_count = sum(1 for r in roles if r == "navigator")
        fix_count = sum(1 for r in roles if r == "fixer")
        ver_count = sum(1 for r in roles if r == "verifier")

        def invoke(inputs, config):
            msgs = inputs["messages"]
            prompt = msgs[-1]["content"]
            if role == "navigator" and nav_count == 1:
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_bug.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "wrong file",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "h", "nodes": [{"path": "src/lib.rs", "symbol": "advance_mut"}]}],
                    "reproduction_status": "runtime_failure",
                })
                return {"messages": msgs + [{"role": "assistant", "content": "NAV_MARKER_INITIAL"}]}
            if role == "fixer" and fix_count == 1:
                tmap["escalate_for_replan"].invoke({"reason": "wrong file localization"})
                return {"messages": msgs}
            if role == "navigator" and nav_count == 2:
                tmap["submit_localization_bundle"].invoke({
                    "summary": "relocalized",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "h2", "nodes": [{"path": "src/other.rs", "symbol": "advance_mut"}]}],
                    "reproduction_status": "runtime_failure",
                })
                return {"messages": msgs + [{"role": "assistant", "content": "NAV_RELOC_SECRET_MARKER"}]}
            if role == "fixer" and fix_count == 2:
                fixer_histories_seen.append(json.dumps(msgs))
                fixer_prompts_seen.append(prompt)
                tmap["edit_file"].invoke({"path": "src/other.rs", "old_text": "if cnt > 0", "new_text": "if cnt > 1"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "updated"})
                return {"messages": msgs + [{"role": "assistant", "content": "FIXER_ROUND1_MEMORY"}]}
            if role == "verifier" and ver_count == 1:
                verifier_prompts_seen.append(prompt)
                tmap["submit_qa_decision"].invoke({
                    "decision": "REJECT",
                    "summary": "check edge",
                    "missing_requirements": ["verify boundary cnt == 1"],
                    "objections": ["boundary off-by-one"],
                    "plan_consistency": "not_applicable",
                })
                return {"messages": msgs}
            if role == "fixer" and fix_count == 3:
                fixer_histories_seen.append(json.dumps(msgs))
                fixer_prompts_seen.append(prompt)
                tmap["edit_file"].invoke({"path": "src/other.rs", "old_text": "if cnt > 1", "new_text": "if cnt > 2"})
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_bug")
                tmap["submit_for_qa"].invoke({"summary": "fixed boundary"})
                return {"messages": msgs}
            if role == "verifier" and ver_count >= 2:
                tmap["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "ok", "plan_consistency": "not_applicable"})
            return {"messages": msgs}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    patch = agent_mod.run_agent(repo, "Fix `advance_mut`", "m", "k", max_api_calls=100)
    assert "if cnt > 2" in patch
    # Fixer after relocalized_fast must NOT inherit Navigator's relocalize conversation
    assert "NAV_RELOC_SECRET_MARKER" not in fixer_histories_seen[0]
    # Fixer after qa_reject_fix MUST retain compressed previous Fixer conversation
    assert "FIXER_ROUND1_MEMORY" in fixer_histories_seen[1]
    # Sibling contract panic message surfaced to Fixer & Verifier
    assert "cannot advance past capacity" in fixer_prompts_seen[0]
    assert "cannot advance past capacity" in verifier_prompts_seen[0]


def test_scheduled_fixer_omits_unchanged_file_excerpts_across_slices(tmp_path, monkeypatch):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        board = {"localization": {"candidate_chains": []}, "repair_units": {}}
        coord = ma._repair_units.RepairUnitCoordinator(state, tools, board)
        coord.configure({
            "ordered_edits": [{"file": "src/lib.rs", "description": "step 1"}],
            "repair_units": [{"unit_id": "u1", "edit_indices": [0]}],
        }, [])
        prompts_seen = []

        def fake_role(_role, _sys, user_prompt, _tools, holder, **kw):
            prompts_seen.append(user_prompt)
            if len(prompts_seen) == 1:
                tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "{ 0 }", "new_text": "{ 1 }"})
                holder["history"] = [{"role": "user", "content": user_prompt}]
                return None, "unit_boundary"
            if len(prompts_seen) == 2:
                holder["history"] = [{"role": "user", "content": user_prompt}]
                return None, "unit_boundary"
            return {"action": "qa", "summary": "done"}, "completed"

        monkeypatch.setattr(ma, "_run_role_agent", fake_role)
        monkeypatch.setattr(coord, "verify", lambda complete=False, summary="": (False, "check failed"))
        ma._run_scheduled_fixer(
            "sys", "INIT_PROMPT", [], {},
            coordinator=coord, allow_escalate=True, client=SimpleNamespace(calls_made=0), stage_cap=20,
        )
        assert "src/lib.rs:" in prompts_seen[1]
        assert "unchanged since previous slice" in prompts_seen[2]
        assert "src/lib.rs:" not in prompts_seen[2]
    finally:
        state.close()


def test_extract_issue_contract_adds_is_empty_companion_hint_and_verifier_conflict_basis():
    ma = _load("multi_agent")
    contract = ma.extract_issue_contract("Add `len` method to `broadcast::Sender`")
    assert any("is_empty" in h for h in contract["companion_api_hints"])
    card = ma.build_handoff_card(contract, {"localization": {}, "repair_plan": {}})
    assert "is_empty" in card

    tools, holder = ma._make_handoff_tools("verifier", None, {
        "repair_plan": {"ordered_edits": [{"file": "src/lib.rs"}]},
        "issue_contract": ma.extract_issue_contract("Support `--only` flag"),
    }, {})
    submit = tools[0]
    err = submit.invoke({
        "decision": "APPROVE",
        "summary": "ok",
        "plan_consistency": "conflict",
        "plan_conflicts": ["diff edited src/ui.rs directly"],
    })
    assert "explanation" in err.lower()
    assert not holder["done"]

    ok = submit.invoke({
        "decision": "APPROVE",
        "summary": "ok",
        "plan_consistency": "conflict",
        "plan_conflicts": ["diff edited src/ui.rs directly"],
        "explanation": "The diff satisfies the requested `--only` flag and cargo check passed.",
    })
    assert "recorded" in ok.lower()
    assert holder["done"]


def test_extract_issue_contract_ignores_inline_shell_command_flags():
    ma = _load("multi_agent")
    issue = (
        "Installer.sh sourcing ~/.cargo/bin in ~/.profile is not always sufficient.\n"
        "One-liner: `curl --proto '=https' --tlsv1.2 -LsSf https://example.com/install.sh?a=--ignored | sh`\n"
        "```bash\n"
        "cargo test --all-features -- --nocapture\n"
        "```\n"
        "- Running `curl --proto '=https' --tlsv1.2` should work in non-interactive shells.\n"
    )
    contract = ma.extract_issue_contract(issue)
    assert contract["signals"]["mentions_cli_or_additive_flag"] is False
    assert "requirement_checklist" not in contract

    real_cli = ma.extract_issue_contract("Support `--config <path>` and `--only` options.")
    assert real_cli["signals"]["mentions_cli_or_additive_flag"] is True


def test_localization_plan_and_fixer_zero_edit_circuit_breakers(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    tools_list, state = tools_mod.make_tools(repo)
    tools_by_name = {t.name: t for t in tools_list}
    try:
        # 1. Navigator cannot silently hand off a temporary regression test that passed on original code.
        tools_by_name["write_regression_test"].invoke({"path": "tests/test_regression_fake.rs", "content": "#[test] fn fake() {}"})
        state.pass_test(state.start_test(), test_target="test_regression_fake")
        nav_tools, nav_holder = ma._make_handoff_tools(
            "navigator",
            state,
            {"issue_contract": {"requested_symbols": []}},
            {"files_read": [], "symbols_queried": [], "calls_remaining_in_stage": 5},
        )
        nav_err = nav_tools[0].invoke({
            "summary": "s",
            "candidate_chains": [{"chain_id": "C1", "hypothesis": "h", "nodes": [{"path": "src/lib.rs", "role": "root_candidate"}]}],
        })
        assert "already PASSED" in nav_err and not nav_holder["done"]

        # 2. Suggester cannot submit empty or preserve-only plans.
        sug_tools, sug_holder = ma._make_handoff_tools(
            "suggester",
            state,
            {"issue_contract": {"requested_symbols": []}, "localization": {"candidate_chains": []}},
            {"files_read": [], "symbols_queried": []},
        )
        submit_plan = next(t for t in sug_tools if t.name == "submit_repair_plan")
        plan_err = submit_plan.invoke({
            "selected_chain_id": "C1",
            "arbitration_rationale": "none",
            "ordered_edits": [{"file": "src/lib.rs", "action": "preserve", "description": "keep"}],
        })
        assert "concrete 'modify' or 'add'" in plan_err and not sug_holder["done"]

        # 3. Fixer triggers _AutoEscalate on repeated submit_for_qa with 0 source edits.
        fix_act = {"files_read": [], "symbols_queried": [], "unit_coordinator": SimpleNamespace(before_tool=lambda *a: None, observe=lambda *a: None)}
        fix_tools, _ = ma._make_handoff_tools("fixer", state, {"issue_contract": {}, "localization": {}}, fix_act, allow_escalate=True)
        submit_qa = next(t for t in fix_tools if t.name == "submit_for_qa")
        assert "no source code changes" in submit_qa.invoke({"summary": "noop"})
        try:
            submit_qa.invoke({"summary": "noop2"})
            assert False, "expected _AutoEscalate on repeated zero-edit submit_for_qa"
        except ma._AutoEscalate as exc:
            assert exc.detail["no_edit_qa_attempts"] == 2

        # 4. Fixer triggers _AutoEscalate on >=3 write_regression_test calls without source edits even when coordinated.
        dummy_write_reg = ma._clone_tool_with_func(
            tools_by_name["write_regression_test"],
            lambda **kw: "ok",
        )
        wrapped = ma._wrap_tools_for_role(
            "fixer",
            {"write_regression_test": dummy_write_reg},
            ["write_regression_test"],
            [],
            {},
            state,
            SimpleNamespace(calls_made=0, remaining_calls=20),
            0,
            10,
            fix_act,
            auto_escalate_on_stuck=True,
        )
        wtool = wrapped[0]
        wtool.invoke({"path": "tests/test_regression_fake.rs", "content": "#[test] fn a() {}"})
        wtool.invoke({"path": "tests/test_regression_fake.rs", "content": "#[test] fn b() {}"})
        try:
            wtool.invoke({"path": "tests/test_regression_fake.rs", "content": "#[test] fn c() {}"})
            assert False, "expected _AutoEscalate on zero-source-edit reg_write_streak"
        except ma._AutoEscalate as exc:
            assert exc.detail["reg_write_streak"] == 3 and not exc.detail["has_source_edits"]
    finally:
        state.close()


def test_deterministic_qa_blocks_dead_enum_variant_and_warns_bare_assert(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = tmp_path / "enum_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text('[package]\nname = "enum_demo"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src" / "lib.rs").write_text(
        "pub enum Kind {\n    Num,\n}\n"
        "pub trait BufLike {\n    fn adv(&mut self, n: usize);\n}\n"
        "impl BufLike for Vec<u8> {\n"
        "    fn adv(&mut self, n: usize) { assert!(n <= 4, \"cannot advance\"); }\n"
        "}\n"
        "pub fn handle(k: Kind) -> i32 {\n"
        "    match k {\n        Kind::Num => 1,\n        _ => 0,\n    }\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "src" / "other.rs").write_text(
        "pub fn adv_other(n: usize) {\n    if n > 4 {}\n}\n",
        encoding="utf-8",
    )
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        tools["write_regression_test"].invoke({"path": "tests/test_regression_e.rs", "content": "#[test] fn t() {}"})
        tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "pub enum Kind {\n    Num,\n}",
            "new_text": "pub enum Kind {\n    Num,\n    Bogus,\n}",
        })
        state.pass_test(state.start_test(), test_target="test_regression_e")
        res = ma.run_deterministic_qa_checks(state, ma.extract_issue_contract("Fix parser"), run_doctest_on_shift=False)
        assert res["ok"] is False
        assert "One-sided enum edit" in res["reason"] and "Bogus" in res["reason"]

        tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "pub enum Kind {\n    Num,\n    Bogus,\n}",
            "new_text": "pub enum Kind {\n    Num,\n}",
        })
        tools["edit_file"].invoke({
            "path": "src/other.rs",
            "old_text": "pub fn adv_other(n: usize) {\n    if n > 4 {}\n}",
            "new_text": "pub fn adv_other(n: usize) {\n    assert!(n <= 4);\n}",
        })
        # Check sibling bare assert warning directly
        bare_warn = ma._detect_bare_assert_sibling_mismatch(
            state,
            "+++ b/src/other.rs\n@@ -1,2 +1,2 @@ pub fn adv(n: usize) {\n+    assert!(n <= 4);\n",
        )
        assert "Bare assert!(...)" in bare_warn and "cannot advance" in bare_warn
    finally:
        state.close()


def test_conditional_dynamic_trace_triggers_only_on_real_logic_failure(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = tmp_path / "trace_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text('[package]\nname = "tr"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src" / "lib.rs").write_text("pub fn bug(x: i32) -> i32 { x }\n", encoding="utf-8")
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=True)
    try:
        (state.root / "target").mkdir()
        tools = {t.name: t for t in tools_list}
        tools["write_regression_test"].invoke({"path": "tests/test_regression_tr.rs", "content": "#[test] fn t() {}"})
        state.fail_test(state.start_test(), test_target="test_regression_tr", executed=1, failures=["t"], returncode=101)
        called = []
        fake_trace = SimpleNamespace(name="rust_trace", func=lambda **kw: (called.append(kw), "[Rust Dynamic Trace Report]\nCall Hierarchy:\nbug (src/lib.rs:1)")[1])
        ctx = SimpleNamespace(
            tools_by_name={"rust_trace": fake_trace},
            state=state,
            trace=None,
            blackboard={"localization": {"candidate_chains": [{"chain_id": "C1", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}]}},
        )
        rep = ma._maybe_run_conditional_trace(ctx, on_escalation=False)
        assert "Rust Dynamic Trace Report" in rep
        assert called and called[0]["instrument"] == ["src/lib.rs:bug"]
        assert ctx.blackboard["dynamic_trace_report"] == rep
    finally:
        state.close()


def test_conditional_dynamic_trace_uses_cargo_target_dir_and_active_features(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = tmp_path / "trace_env_repo"
    ext_target = tmp_path / "shared_cargo_target"
    ext_target.mkdir()
    (repo / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text('[package]\nname = "tr_env"\nversion = "0.1.0"\n[features]\nserde = []\n', encoding="utf-8")
    (repo / "src" / "lib.rs").write_text("pub fn bug(x: i32) -> i32 { x }\n", encoding="utf-8")
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=True)
    try:
        assert not (state.root / "target").exists()
        state.rust_env = {"CARGO_TARGET_DIR": str(ext_target)}
        state._active_cfg_features[(state.root / "Cargo.toml").resolve()] = ["serde"]
        tools = {t.name: t for t in tools_list}
        tools["write_regression_test"].invoke({"path": "tests/test_regression_env.rs", "content": "#[test] fn t() {}"})
        state.fail_test(state.start_test(), test_target="test_regression_env", executed=1, failures=["t"], returncode=101)
        called = []
        fake_trace = SimpleNamespace(name="rust_trace", func=lambda **kw: (called.append(kw), "[Rust Dynamic Trace Report]\nok")[1])
        ctx = SimpleNamespace(
            tools_by_name={"rust_trace": fake_trace},
            state=state,
            trace=None,
            blackboard={"localization": {"candidate_chains": [{"chain_id": "C1", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}]}},
        )
        rep = ma._maybe_run_conditional_trace(ctx, on_escalation=False)
        assert "Rust Dynamic Trace Report" in rep
        assert called and called[0].get("features") == ["serde"]
    finally:
        state.close()


def test_should_prune_suggester_replan_keeps_locked_oracle_escalation_and_tracks_qa_streak(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = tmp_path / "prune_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text('[package]\nname = "pr"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src" / "lib.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=True)
    try:
        tools = {t.name: t for t in tools_list}
        tools["write_regression_test"].invoke({"path": "tests/test_regression_bad.rs", "content": "#[test] fn t() {}"})
        state.fail_test(state.start_test(), test_target="test_regression_bad", executed=1, failures=["t"], returncode=101)
        ctx = SimpleNamespace(
            state=state,
            candidate_policy=None,
            blackboard={
                "replan_count": 1,
                "last_replan_score": (1, 1, 1, 1),
                "escalations": [{"reason": "Locked regression test oracle asserts wrong expected value"}],
            },
        )
        prune, _ = ma._should_prune_suggester_replan(ctx)
        assert prune is False

        ctx.blackboard["escalations"] = [{"reason": "Compile failed unrelated"}]
        prune2, _ = ma._should_prune_suggester_replan(ctx)
        assert prune2 is True

        assert ma._record_det_qa_failure_streak(ctx, "Rust changes contain only comments/blank lines: src/lib.rs") == 1
        assert ma._record_det_qa_failure_streak(ctx, "Rust changes contain only comments/blank lines: src/other.rs") == 2
        assert ma._record_det_qa_failure_streak(ctx, "Different failure reason: x") == 1
    finally:
        state.close()


def test_smoke_gate_clusters_failed_tests_and_allows_legacy_audit_after_first_round():
    ma = _load("multi_agent")
    state = SimpleNamespace(_existing_test_failures=[], _revision=2)
    patch = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n"
    selected = [Path("src/lib.rs")]
    many_failed = [f"err::string_{i}" for i in range(12)] + ["err::object_comma_json", "err::object_null_json"]
    smoke = [{
        "target": "test:spec_tests",
        "status": "failed",
        "classification": "code",
        "baseline_status": "passed",
        "observed_revision": 2,
        "failed_tests": many_failed,
        "details": "snapshot mismatch",
    }]
    first = ma._submission_gate_report(state, ma.extract_issue_contract("Fix parser"), patch, selected, smoke)
    assert first["blocking"] and "New existing-test regression" in first["blocking"][0]
    assert "err::object_comma_json" in first["blocking"][0]

    retry = ma._submission_gate_report(
        state, ma.extract_issue_contract("Fix parser"), patch, selected, smoke,
        allow_legacy_regression_audit=True,
    )
    assert retry["blocking"] == []
    assert any("Legacy/existing-test failure for Verifier audit" in w for w in retry["warnings"])

    compile_smoke = [{
        "target": "lib",
        "status": "failed",
        "classification": "code",
        "baseline_status": "passed",
        "observed_revision": 2,
        "details": "error[E0308]: mismatched types",
    }]
    still_blocked = ma._submission_gate_report(
        state, ma.extract_issue_contract("Fix parser"), patch, selected, compile_smoke,
        allow_legacy_regression_audit=True,
    )
    assert still_blocked["blocking"] and "New compilation failure" in still_blocked["blocking"][0]


def test_graph_agent_enforces_integration_gate_once_on_template_edit(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    (repo / "templates").mkdir()
    (repo / "templates" / "install.sh.j2").write_text("echo old\n", encoding="utf-8")
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "integration.rs").write_text("#[test] fn snap() {}\n", encoding="utf-8")

    roles = []
    orig_make_tools = agent_mod.make_tools

    def cap_make_tools(r, *a, **kw):
        t, s = orig_make_tools(r, *a, **kw)
        s.post_strip_check = False
        s.require_reproduction = False
        s.run_existing_test_smoke = lambda _sel: [{"target": "test:integration", "status": "timeout", "classification": "unknown"}]
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
        roles.append(role)
        fix_count = sum(1 for r in roles if r == "fixer")

        def invoke(inputs, config):
            if role == "navigator":
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_tpl.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "template",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "template", "nodes": [{"path": "templates/install.sh.j2", "symbol": ""}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer":
                if fix_count == 1:
                    tmap["write_file"].invoke({"path": "templates/install.sh.j2", "content": "echo mid\n"})
                else:
                    tmap["write_file"].invoke({"path": "templates/install.sh.j2", "content": "echo final\n"})
                tmap["submit_for_qa"].invoke({"summary": "updated template", "unreproduced_reason": "template"})
            elif role == "verifier":
                tmap["submit_qa_decision"].invoke({
                    "decision": "APPROVE",
                    "summary": "ok",
                    "plan_consistency": "not_applicable",
                })
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    patch = agent_mod.run_agent(repo, "Update install.sh.j2 template", "m", "k", max_api_calls=30)
    assert "+echo final" in patch
    assert roles.count("verifier") == 2
    assert roles.count("fixer") == 2


def test_graph_agent_allows_paired_repro_legacy_smoke_audit_and_checks_untouched_ref_files(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    (repo / "src" / "lexer").mkdir(parents=True)
    (repo / "src" / "lexer" / "mod.rs").write_text("pub fn lex() {}\n", encoding="utf-8")

    roles = []
    orig_make_tools = agent_mod.make_tools

    def cap_make_tools(r, *a, **kw):
        t, s = orig_make_tools(r, *a, **kw)
        s.post_strip_check = False
        s.require_reproduction = False
        s.reproduced_cleanly = lambda: True
        s.verified = lambda: True
        s.run_existing_test_smoke = lambda _sel: [{
            "target": "test:legacy",
            "status": "failed",
            "classification": "code",
            "baseline_status": "passed",
            "observed_revision": s._revision,
            "failed_tests": ["legacy::old_behavior"],
            "details": "assertion failed",
        }]
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
        roles.append(role)
        ver_count = sum(1 for r in roles if r == "verifier")

        def invoke(inputs, config):
            if role == "navigator":
                tmap["write_regression_test"].invoke({"path": "tests/test_regression_ref.rs", "content": "#[test] fn t() {}"})
                tmap["submit_localization_bundle"].invoke({
                    "summary": "in lib",
                    "candidate_chains": [{"chain_id": "C1", "hypothesis": "lib", "nodes": [{"path": "src/lib.rs", "symbol": "bug"}]}],
                    "reproduction_status": "runtime_failure",
                })
            elif role == "fixer":
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
                tmap["submit_for_qa"].invoke({"summary": "updated lib", "unreproduced_reason": ""})
            elif role == "verifier":
                if ver_count == 1:
                    tmap["submit_qa_decision"].invoke({
                        "decision": "APPROVE",
                        "summary": "downstream change looks good",
                        "plan_consistency": "not_applicable",
                    })
                else:
                    tmap["submit_qa_decision"].invoke({
                        "decision": "APPROVE",
                        "summary": "verified lexer/mod.rs stays unchanged intentionally",
                        "plan_consistency": "not_applicable",
                    })
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    patch = agent_mod.run_agent(repo, "Bug in src/lexer/mod.rs causes wrong output", "m", "k", max_api_calls=30)
    assert "pub fn bug() -> i32 { 1 }" in patch
    assert roles.count("verifier") == 2
    assert roles.count("fixer") == 2


def test_verifier_blocks_false_approve(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    _, state = tools_mod.make_tools(repo)
    try:
        assert ma._is_env_dependency_blocker("error: failed to parse lock file at: Cargo.lock\nlock file version 4 requires `-Znext-lockfile-bump`")

        ver_act = {"probe_results": [{"probe_type": "assertion", "status": "failed", "tests_failed": 1, "failed_tests": ["contract"]}]}
        ver_tools, ver_holder = ma._make_handoff_tools("verifier", state, {}, ver_act)
        blocked = ver_tools[0].invoke({"decision": "APPROVE", "summary": "looks good", "plan_consistency": "not_applicable"})
        assert "Error: cannot APPROVE" in blocked and not ver_holder["done"]

        ver_act["probe_results"].append({"probe_type": "assertion", "status": "passed", "tests_failed": 0, "failed_tests": []})
        ok = ver_tools[0].invoke({"decision": "APPROVE", "summary": "verified", "plan_consistency": "not_applicable"})
        assert "QA decision recorded: APPROVE" in ok and ver_holder["done"]

        state._locked_regressions[("Cargo.toml", "test_regression_x")] = {"name": "test_regression_x"}
        state.regression_tests.add(Path("tests/test_regression_x.rs"))
        ver_tools2, ver_holder2 = ma._make_handoff_tools("verifier", state, {"_last_det_qa": {"unreproduced_reason": ""}}, {})
        blocked_lock = ver_tools2[0].invoke({"decision": "APPROVE", "summary": "ok", "plan_consistency": "not_applicable"})
        assert "locked Red-phase regression test" in blocked_lock and not ver_holder2["done"]
    finally:
        state.close()


def test_graph_budget_200_and_fallback_strips_inline_tests_and_uses_compiling_snapshot(tmp_path, monkeypatch):
    ma = _load("multi_agent")
    tools_mod = _load("tools")

    b200 = ma.allocate_stage_budgets(200)
    assert b200["navigator_cap"] == 44
    assert b200["suggester_cap"] == 12
    assert b200["verifier_cap"] == 12

    repo = tmp_path / "graph_inline_repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "gi"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (repo / "src" / "lib.rs").write_text(
        "pub fn val() -> i32 { 1 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() { assert_eq!(super::val(), 1); }\n}\n",
        encoding="utf-8",
    )
    tool_list, state = tools_mod.make_tools(repo, post_strip_check=False)
    tmap = {t.name: t for t in tool_list}
    try:
        tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn val() -> i32 { 1 }", "new_text": "pub fn val() -> i32 { 2 }"})
        tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "assert_eq!(super::val(), 1);", "new_text": "assert_eq!(super::val(), 2);"})
        state._last_compiling_prod_snapshot = state._capture_snapshot()
        recovered = ma._try_degraded_patch_fallback(
            state, allow_unverified_compiling=True, issue_contract={"signals": {}}, blackboard={},
        )
        assert "+pub fn val() -> i32 { 2 }" in recovered
        assert "assert_eq!(super::val(), 2);" not in recovered

        state.post_strip_check = True
        monkeypatch.setattr(ma.shutil, "which", lambda cmd: "/usr/bin/cargo" if cmd == "cargo" else None)
        state._last_compiling_prod_snapshot = state._capture_snapshot()
        tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn val() -> i32 { 2 }", "new_text": "pub fn val() -> i32 { broken }"})

        def fake_post_strip(sel):
            if "broken" in (state.root / "src" / "lib.rs").read_text(encoding="utf-8"):
                raise ValueError("compile failed")
            return {"status": "passed", "warnings": []}

        monkeypatch.setattr(state, "run_post_strip_check", fake_post_strip)
        snap_patch = ma._try_degraded_patch_fallback(
            state, allow_unverified_compiling=True, issue_contract={"signals": {}}, blackboard={},
        )
        assert "+pub fn val() -> i32 { 2 }" in snap_patch
        assert "broken" not in snap_patch
    finally:
        state.close()


def test_test_qa_cap_tool_allowlist_session_dedup_and_doctest_trigger(tmp_path, monkeypatch):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    trace_mod = _load("agent_trace")
    repo = _setup_fake_repo(tmp_path)
    (repo / "src" / "lib.rs").write_text(
        "/// Example:\n/// ```\n/// assert_eq!(demo::val(), 2);\n/// ```\npub fn val() -> i32 { 1 }\n",
        encoding="utf-8",
    )
    tools_list, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    tools = {t.name: t for t in tools_list}
    try:
        for removed in ("checkpoint", "rust_error_explain", "rust_trace"):
            assert removed not in ma._NAV_ALLOWED and removed not in ma._FIX_ALLOWED

        ctx = ma._init_orchestration_ctx(
            repo, "Fix val", "m", "k", None,
            tools=tools_list, state=state, client=SimpleNamespace(calls_made=0),
            trace=None, max_api_calls=70, compress_fn=lambda m: m,
            create_agent_fn=Mock(), chat_model_factory=Mock(),
            prompts_mod=SimpleNamespace(), role_callback_cls=None,
        )
        assert ctx.max_qa_rounds == 2

        activity = {"files_read": [], "symbols_queried": []}
        wrapped = {
            t.name: t
            for t in ma._wrap_tools_for_role(
                "fixer", tools, ["read_lines", "edit_file"], [], {}, state,
                SimpleNamespace(calls_made=0), 0, 50, activity,
                auto_escalate_on_stuck=False, blackboard={},
            )
        }
        first = wrapped["read_lines"].invoke({"path": "src/lib.rs", "start_line": 1, "end_line": 5})
        second = wrapped["read_lines"].invoke({"path": "src/lib.rs", "start_line": 1, "end_line": 5})
        assert "pub fn val" in first
        assert "Duplicate call skipped" in second

        tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn val() -> i32 { 1 }", "new_text": "pub fn val() -> i32 { 2 }"})
        after_edit = wrapped["read_lines"].invoke({"path": "src/lib.rs", "start_line": 1, "end_line": 5})
        assert "pub fn val() -> i32 { 2 }" in after_edit

        doc_cmds = []
        monkeypatch.setattr(ma.shutil, "which", lambda cmd: "/usr/bin/cargo" if cmd == "cargo" else None)
        import subprocess as sp_mod
        orig_sp_run = sp_mod.run
        monkeypatch.setattr(
            sp_mod, "run",
            lambda cmd, **kw: (
                (doc_cmds.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
                if cmd and cmd[0] == "cargo"
                else orig_sp_run(cmd, **kw)
            ),
        )
        state.run_existing_test_smoke = lambda sel: []
        qa = ma.run_deterministic_qa_checks(state, ma.extract_issue_contract("Fix val"), run_doctest_on_shift=True)
        assert qa["ok"] is True and qa["doctest_status"] == "passed"
        assert doc_cmds and "--doc" in doc_cmds[0]

        trace = trace_mod.TraceRecorder(tmp_path / "bypass.jsonl")
        ctx_bypass = SimpleNamespace(
            state=state,
            blackboard={"repair_plan": {"selected_chain_id": "C1"}, "_qa_repair_pending": False},
            trace=trace,
            qa_rounds=0,
        )
        ma._handle_low_budget_det_qa_bypass(ctx_bypass, {"selected": [Path("src/lib.rs")], "hygiene": {"patch": "+x"}, "missing_contract_apis": []})
        assert not ctx_bypass.blackboard.get("consistency_blocked")
    finally:
        state.close()


def test_graph_agent_reraises_api_server_error_without_degraded_fallback(monkeypatch, tmp_path):
    agent_mod = _load("agent")
    repo = _setup_fake_repo(tmp_path)
    orig_make_tools = agent_mod.make_tools

    def cap_make_tools(r, *a, **kw):
        t, s = orig_make_tools(r, *a, **kw)
        s.post_strip_check = False
        s.require_reproduction = False
        return t, s

    monkeypatch.setattr(agent_mod, "make_tools", cap_make_tools)
    monkeypatch.setattr(agent_mod, "ChatOpenAI", Mock())

    def fake_create_agent(model, tools, system_prompt):
        tmap = {t.name: t for t in tools}

        def invoke(inputs, config):
            if "edit_file" in tmap:
                tmap["edit_file"].invoke({"path": "src/lib.rs", "old_text": "pub fn bug() -> i32 { 0 }", "new_text": "pub fn bug() -> i32 { 1 }"})
            raise agent_mod._trace.APIServerError(503, 3)

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
    with pytest.raises(agent_mod._trace.APIServerError):
        agent_mod.run_agent(repo, "Fix `bug`", "m", "k", max_api_calls=70)






