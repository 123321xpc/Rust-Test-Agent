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
