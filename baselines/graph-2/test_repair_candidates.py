"""Candidate gating/selection and orchestration tests; no model API calls."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_candidate_test", Path(__file__).with_name(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def board():
    return {"issue_contract": {}, "localization": {"candidate_chains": [
        {"chain_id": "C1", "hypothesis": "Wrong registry state", "nodes": [{"path": "src/a.rs"}]},
        {"chain_id": "C2", "hypothesis": "Context lost in propagation", "nodes": [{"path": "src/b.rs"}]}]}}


def cards(**changes):
    result = [{"candidate_id": c, "chain_id": chain, "mechanism": mechanism,
               "contract": "satisfies", "contract_reason": "Implements the requested state/propagation contract",
               "execution": "untested", "dependencies": "compatible", "dependency_reason": "Uses the current trait and locked features",
               "affected_paths": paths, "verification_cost": "targeted_tests"}
              for c, chain, mechanism, paths in [("A", "C1", "Repair registry state at its owner", ["src/a.rs", "src/shared.rs"]),
                                                 ("B", "C2", "Propagate context through the caller", ["src/b.rs"])]]
    for key, value in changes.items():
        idx, field = key.split("_", 1)
        result[0 if idx == "a" else 1][field] = value
    return result


@pytest.fixture
def policy():
    mod = load("repair_candidates")
    state = SimpleNamespace(_revision=1, stuck_status=lambda: {"compile_repeats": 0, "test_repeats": 0})
    return mod.CandidatePolicy(board(), state), state


def ambiguous(policy):
    assert policy.ambiguity(["C1", "C2"], "Existing evidence cannot distinguish state owner from propagation") is None


def cargo(policy, state, *, code=101, runtime=True, args=None):
    return policy.observe("cargo_test" if runtime else "cargo_check", args or {"manifest_path": "Cargo.toml"},
                          json.dumps({"exit_code": code, "status": "failed" if code else "passed",
                                      "tests_executed": 1 if runtime else 0, "failures": ["contract_assertion"] if code else []}))


def test_default_one_plan_even_with_multiple_chains(policy):
    p, _ = policy
    payload = {"ordered_edits": [{"file": "src/a.rs", "description": "Fix state"}], "selected_chain_id": "C1"}
    assert p.plan_gate(payload, "") is None and payload["candidate_id"] == "A"
    assert not p.data["comparisons"] and not p.data["pending"]
    with pytest.raises(ValueError, match="disabled"):
        p.compare(cards())
    assert "requires a triggered" in p.plan_gate(payload, "B")


def test_ambiguity_needs_distinct_known_chains_and_reason(policy):
    p, _ = policy
    assert p.ambiguity(["C1", "missing"], "unclear")
    assert p.ambiguity(["C1", "C2"], "")
    ambiguous(p)
    assert p.data["pending"]["kind"] == "ambiguous_chains"


@pytest.mark.parametrize("criterion", ["contract", "execution", "dependencies"])
def test_primary_evidence_beats_smaller_cheaper_patch(policy, criterion):
    p, state = policy
    ambiguous(p)
    c = cards(b_affected_paths=["src/small.rs"], a_verification_cost="workspace_tests", b_verification_cost="targeted_check")
    if criterion == "contract":
        c[1]["contract"] = "conflict"
    elif criterion == "dependencies":
        c[1]["dependencies"] = "incompatible"
    else:
        eid = cargo(p, state)
        c[1].update(execution="refuted", evidence_ids=[eid], execution_reason="Observed failure contradicts the caller-only hypothesis")
    result = p.compare(c)
    assert result["selected_candidate"] == "A"
    assert "take priority" in result["reason"]


def test_equal_primary_evidence_uses_scope_then_verification_cost(policy):
    p, _ = policy
    ambiguous(p)
    assert p.compare(cards())["selected_candidate"] == "B"
    ambiguous(p)
    result = p.compare(cards(a_affected_paths=["src/a.rs"], a_verification_cost="targeted_check", b_verification_cost="workspace_tests"))
    assert result["selected_candidate"] == "A" and "equivalent primary" in result["reason"]


def test_unknown_alternative_does_not_trigger_speculative_implementation(policy):
    p, _ = policy
    ambiguous(p)
    result = p.compare(cards(a_contract="uncertain", b_contract="uncertain"), next_check="Check where Context changes on the failing path")
    assert result["selected_candidate"] == "A"
    assert "lacks decisive" in result["reason"] and result["next_check"]


def test_both_inadmissible_requires_relocalization_and_new_episode(policy):
    p, _ = policy
    ambiguous(p)
    assert p.compare(cards(a_contract="conflict", b_dependencies="incompatible"))["selected_candidate"] is None
    payload = {"ordered_edits": [{"file": "src/c.rs"}], "selected_chain_id": "C3"}
    assert "Neither mechanism" in p.plan_gate(payload, "A")
    p.relocalized()
    assert p.plan_gate(payload, "") is None
    assert len(p.data["comparisons"]) == 1  # Historical comparison remains auditable.


def test_counterevidence_is_observed_current_and_not_superseded(policy):
    p, state = policy
    assert p.counterevidence("contradiction", ["invented"])
    failure = cargo(p, state)
    state._revision += 1
    assert p.counterevidence("old result might no longer apply", [failure])
    failure2 = cargo(p, state)
    assert p.counterevidence("Current assertion contradicts registry-only fix", [failure2]) is None
    state._revision += 1
    cargo(p, state, code=0)
    assert p.data["pending"] is None
    assert p.counterevidence("already resolved", [failure2])


def test_latest_result_at_same_revision_and_new_episode_get_fresh_evidence(policy):
    p, state = policy
    first = cargo(p, state)
    passed = cargo(p, state, code=0)
    failed_again = cargo(p, state)
    assert len({first, passed, failed_again}) == 3
    assert p.data["evidence"][0]["superseded_by"] == passed
    assert p.counterevidence("The latest rerun contradicts the prediction", [failed_again]) is None
    assert cargo(p, state) == failed_again  # Cache reuse does not multiply evidence.
    p.data.update(retain_current_patch=True, switch_paths=["src/a.rs"])
    p.relocalized()
    assert not p.data["retain_current_patch"] and not p.data["switch_paths"]
    assert cargo(p, state) != failed_again


def test_empty_test_success_does_not_supersede_behavioral_failure(policy):
    p, state = policy
    failed = cargo(p, state)
    p.counterevidence("Assertion contradicts the current prediction", [failed])
    p.observe("cargo_test", {"manifest_path": "Cargo.toml"}, '{"exit_code":0,"tests_executed":0}')
    assert p.data["evidence"][0]["superseded_by"] is None
    assert p.data["pending"]["evidence_ids"] == [failed]
    cargo(p, state, code=0)
    assert p.data["pending"] is None


def test_fake_and_zero_test_evidence_cannot_rank_mechanisms(policy):
    p, state = policy
    ambiguous(p)
    with pytest.raises(ValueError, match="existing"):
        p.compare(cards(b_execution="supported", b_evidence_ids=["invented"], b_execution_reason="pretend pass"))
    eid = p.observe("cargo_test", {"test_filter": "absent"}, '{"exit_code":0,"tests_executed":0}')
    with pytest.raises(ValueError, match="zero-test"):
        p.compare(cards(b_execution="supported", b_evidence_ids=[eid], b_execution_reason="No assertion actually executed"))


def test_regression_auto_run_is_recorded_without_repeating_test_source(policy):
    p, _ = policy
    output = 'Updated temporary test.\n\n[Auto-run result for red]:\n{"exit_code":101,"tests_executed":1,"failures":["contract"]}'
    eid = p.observe("write_regression_test", {"run_now": True, "path": "tests/test_regression_a.rs", "content": "large test body"}, output)
    assert eid == "E1" and p.data["evidence"][0]["runtime"]
    assert "content" not in p.data["evidence"][0]["arguments"]
    assert p.data["pending"] is None  # A failed test is not automatically a refuted hypothesis.


def test_repeated_behavior_failure_opens_gate_but_compile_noise_does_not(policy):
    p, state = policy
    state.stuck_status = lambda: {"compile_repeats": 3, "test_repeats": 0}
    cargo(p, state, runtime=False)
    state._revision += 1
    cargo(p, state, runtime=False)
    p.repeated_test_failure()
    assert p.data["pending"] is None
    state.stuck_status = lambda: {"compile_repeats": 0, "test_repeats": 2}
    cargo(p, state)
    state._revision += 1
    cargo(p, state)
    p.repeated_test_failure()
    assert p.data["pending"]["kind"] == "verification_counterevidence"


def test_scope_growth_requires_two_expansions_not_just_a_large_plan(policy):
    p, _ = policy
    p.observe_scope({f"src/{i}.rs" for i in range(10)})
    assert p.data["pending"] is None
    p.observe_scope(set(p.data["scope"]) | {"src/extra.rs"})
    assert p.data["pending"] is None
    p.observe_scope(set(p.data["scope"]) | {"src/more.rs"})
    assert p.data["pending"]["kind"] == "scope_growth"


@pytest.fixture
def scheduled(tmp_path):
    ma, fixture = load("multi_agent"), load("test_repair_units")
    tools, state = load("tools").make_tools(fixture.make_repo(tmp_path), post_strip_check=False)
    tools = {t.name: t for t in tools}
    tools["cargo_check"] = SimpleNamespace(invoke=lambda _: '{"exit_code":0,"diagnostics":[]}')
    blackboard = board()
    p = ma._repair_candidates.CandidatePolicy(blackboard, state)
    coord = ma._repair_units.RepairUnitCoordinator(state, tools, blackboard, candidate_policy=p)
    try:
        yield ma, fixture, state, tools, p, coord
    finally:
        state.close()


@pytest.mark.parametrize("handoff", [True, False])
def test_retaining_a_preserves_fast_path_edits_and_their_rollback_baseline(scheduled, handoff):
    _, fixture, state, tools, p, coord = scheduled
    coord.configure(None, ["src/a.rs"])
    coord.select()
    fixture.edit(tools, "a")
    ambiguous(p)
    assert p.compare(cards(b_contract="conflict"))["selected_candidate"] == "A"
    formal = {"selected_chain_id": "C1", "ordered_edits": [{"file": "src/a.rs", "description": "Keep repaired state"}],
              "repair_units": [{"unit_id": "state", "description": "State owner", "edit_indices": [0]}]}
    if handoff:
        assert p.plan_gate(formal, "A") is None
    else:
        assert p.fallback_plan(None) is None  # Same retention rule at planner cap.
    coord.configure(formal, [])
    assert "{ 1 }" in (state.root / "src/a.rs").read_text()
    assert coord.select()["unit_id"] == "state"
    assert coord.verify(complete=True)[0]
    coord.rollback({"state"})
    assert "{ 1 }" in (state.root / "src/a.rs").read_text()


def test_b_switch_invalidates_affected_units_even_with_identical_plan_specs(scheduled):
    _, fixture, state, tools, p, coord = scheduled
    plan = dict(fixture.plan(), selected_chain_id="C1")
    assert p.plan_gate(plan, "A") is None
    coord.configure(plan, [])
    for name in "abcd":
        fixture.close_unit(tools, coord, name)
    ambiguous(p)
    assert p.compare(cards(b_affected_paths=["src/b.rs"]))["selected_candidate"] == "B"
    plan["selected_chain_id"] = "C2"
    assert p.plan_gate(plan, "B") is None
    coord.configure(plan, [])
    assert [u["unit_id"] for u in coord.units if u["status"] == "verified"] == ["a", "c"]
    for name in "ac":
        assert "{ 1 }" in (state.root / f"src/{name}.rs").read_text()
    for name in "bd":
        assert "{ 1 }" in (state.root / f"src/{name}.rs").read_text()


def test_repeated_scope_requests_extend_current_unit(scheduled):
    ma, fixture, state, _, p, coord = scheduled
    coord.configure(fixture.plan(), [])
    coord.select()
    assert coord.before_tool("edit_file", {"path": "src/extra.rs"}) is None
    assert coord.before_tool("edit_file", {"path": "src/more.rs"}) is None
    assert {"src/extra.rs", "src/more.rs"} <= set(coord._unit(coord.active)["paths"])
    assert not (state.root / "src/extra.rs").exists() and not (state.root / "src/more.rs").exists()


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo unavailable")
def test_orchestrator_implements_only_selected_b_after_brief_comparison(tmp_path, monkeypatch):
    agent = load("agent")
    fixture = load("test_repair_units")
    repo = fixture.make_repo(tmp_path)
    states, roles = [], []
    original = agent.make_tools

    def make(*args, **kwargs):
        tools, state = original(*args, **kwargs)
        state.post_strip_check = False
        state.require_reproduction = False
        states.append(state)
        return tools, state

    monkeypatch.setattr(agent, "make_tools", make)
    monkeypatch.setattr(agent, "ChatOpenAI", Mock())

    def create(model, tools, system_prompt):
        t = {x.name: x for x in tools}
        role = "navigator" if "Navigator & Reproducer" in system_prompt else "suggester" if "Rust Architect / Suggester" in system_prompt else "fixer" if "Rust Fixer" in system_prompt else "verifier"
        roles.append(role)

        def invoke(inputs, config):
            if role == "navigator":
                t["write_regression_test"].invoke({"path": "tests/test_regression_choice.rs", "content": "#[test] fn result() {}"})
                t["submit_localization_bundle"].invoke({"summary": "ambiguous", "candidate_chains": board()["localization"]["candidate_chains"],
                    "unresolved_chain_ids": ["C1", "C2"], "ambiguity_reason": "Either owner or propagation might explain the issue"})
            elif role == "suggester":
                assert '"kind": "ambiguous_chains"' in inputs["messages"][-1]["content"]
                result = json.loads(t["compare_repair_candidates"].invoke({"candidates_brief": cards(a_contract="conflict", b_affected_paths=["src/b.rs", "src/c.rs"])}))
                assert result["selected_candidate"] == "B"
                t["submit_repair_plan"].invoke({"selected_chain_id": "C2", "candidate_id": "B", "arbitration_rationale": "B meets requested contract",
                    "ordered_edits": [{"file": "src/b.rs", "description": "Fix value"}, {"file": "src/c.rs", "description": "Propagate value"}]})
            elif role == "fixer":
                fixture.edit(t, "b")
                fixture.edit(t, "c")
                states[-1].pass_test(states[-1].start_test(), test_target="test_regression_choice")
                t["submit_for_qa"].invoke({"summary": "selected B implemented"})
            else:
                t["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "meets contract"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent, "create_agent", create)
    trace_path = tmp_path / "candidate_trace.jsonl"
    patch = agent.run_agent(repo, "Fix state propagation", "unused", "unused", max_api_calls=150, trace=agent.TraceRecorder(trace_path))
    assert roles == ["navigator", "suggester", "fixer", "verifier"]
    assert "src/a.rs" not in patch and "src/b.rs" in patch and "src/c.rs" in patch
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert sum(e["event"] == "candidate_comparison" for e in rows) == 1


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo unavailable")
def test_orchestrator_counterevidence_retains_a_without_reimplementing_it(tmp_path, monkeypatch):
    agent, fixture = load("agent"), load("test_repair_units")
    repo = fixture.make_repo(tmp_path)
    states, roles, observations = [], [], []
    original = agent.make_tools

    def make(*args, **kwargs):
        tools, state = original(*args, **kwargs)
        state.post_strip_check = False
        state.require_reproduction = False
        states.append(state)
        return tools, state

    monkeypatch.setattr(agent, "make_tools", make)
    monkeypatch.setattr(agent, "ChatOpenAI", Mock())

    def create(model, tools, system_prompt):
        t = {x.name: x for x in tools}
        role = "navigator" if "Navigator & Reproducer" in system_prompt else "suggester" if "Rust Architect / Suggester" in system_prompt else "fixer" if "Rust Fixer" in system_prompt else "verifier"
        roles.append(role)

        def invoke(inputs, config):
            if role == "navigator":
                t["write_regression_test"].invoke({"path": "tests/test_regression_choice.rs",
                    "content": "#[test] fn result() { assert_eq!(unit_fixture::a::value(), 2); }"})
                t["submit_localization_bundle"].invoke({"summary": "localized state", "candidate_chains": [board()["localization"]["candidate_chains"][0]]})
            elif role == "suggester":
                assert "verification_counterevidence" in inputs["messages"][-1]["content"]
                result = json.loads(t["compare_repair_candidates"].invoke({"candidates_brief": cards(
                    a_affected_paths=["src/a.rs"], b_contract="conflict",
                    a_execution="supported", a_evidence_ids=observations,
                    a_execution_reason="Failure shows value=1 as required; generated oracle wrongly expects 2"),
                    "next_check": "Correct generated oracle to the issue contract, then run it"}))
                assert result["selected_candidate"] == "A"
                t["submit_repair_plan"].invoke({"selected_chain_id": "C1", "candidate_id": "A",
                    "arbitration_rationale": "State owner fix is correct; test oracle was wrong",
                    "ordered_edits": [{"file": "src/a.rs", "description": "Keep value=1"}],
                    "repair_units": [{"unit_id": "owner", "description": "State owner", "edit_indices": [0]}]})
            elif role == "fixer" and not observations:
                fixture.edit(t, "a")
                result = json.loads(t["cargo_test"].invoke({"test_target": "test_regression_choice"}))
                assert result["exit_code"] != 0 and result["tests_executed"] == 1
                observations.append(result["candidate_evidence_id"])
                t["request_candidate_comparison"].invoke({"trigger": "verification_counterevidence",
                    "reason": "Actual value is 1 but oracle expects 2; distinguish wrong root cause from wrong oracle",
                    "evidence_ids": observations})
            elif role == "fixer":
                assert "{ 1 }" in (states[-1].root / "src/a.rs").read_text()
                out = t["write_regression_test"].invoke({"path": "tests/test_regression_choice.rs", "run_now": True,
                    "content": "#[test] fn result() { assert_eq!(unit_fixture::a::value(), 1); }"})
                assert "Coordinator execution evidence" in out and states[-1].verified()
                t["submit_for_qa"].invoke({"summary": "A retained; generated regression corrected and passed"})
            else:
                t["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "matches issue contract"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent, "create_agent", create)
    patch = agent.run_agent(repo, "Fix the state owner's value to return 1", "unused", "unused", max_api_calls=150)
    assert roles == ["navigator", "fixer", "suggester", "fixer", "verifier"]
    assert "src/a.rs" in patch and "src/b.rs" not in patch and "src/shared.rs" not in patch
