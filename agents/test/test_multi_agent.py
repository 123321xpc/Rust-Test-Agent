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


def test_write_regression_test_pacing_nudges_earlier_in_navigator_and_fixer(tmp_path):
    ma = _load("multi_agent")
    tools_mod = _load("tools")
    repo = _setup_fake_repo(tmp_path)
    tlist, state = tools_mod.make_tools(repo, post_strip_check=False, require_reproduction=False)
    try:
        by_name = {t.name: t for t in tlist}
        nav_activity = {"files_read": [], "symbols_queried": [], "calls_remaining_in_stage": 20}
        nav_wrapped = ma._wrap_tools_for_role(
            "navigator", by_name, {"write_regression_test"}, [], {}, state, None, 0, 20, nav_activity,
        )
        nav_wrt = next(t for t in nav_wrapped if t.name == "write_regression_test")
        nav_wrt.invoke({"path": "tests/test_regression_a.rs", "content": "#[test] fn a() { assert_eq!(1, 1); }\n"})
        nav_out2 = nav_wrt.invoke({"path": "tests/test_regression_a.rs", "content": "#[test] fn a() { assert_eq!(1, 2); }\n"})
        assert "2+ write_regression_test attempts without clean reproduction" in nav_out2

        fix_activity = {"files_read": [], "symbols_queried": [], "calls_remaining_in_stage": 20}
        fix_wrapped = ma._wrap_tools_for_role(
            "fixer", by_name, {"write_regression_test"}, [], {}, state, None, 0, 20, fix_activity,
        )
        fix_wrt = next(t for t in fix_wrapped if t.name == "write_regression_test")
        fix_wrt.invoke({"path": "tests/test_regression_b.rs", "content": "#[test] fn b() { assert_eq!(1, 1); }\n"})
        fix_out2 = fix_wrt.invoke({"path": "tests/test_regression_b.rs", "content": "#[test] fn b() { assert_eq!(1, 2); }\n"})
        assert "stop rewriting temporary tests" in fix_out2
    finally:
        state.close()
