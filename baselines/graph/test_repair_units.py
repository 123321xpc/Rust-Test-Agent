"""Offline scheduling contracts plus real Cargo fixtures; no model API calls."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_units_test", Path(__file__).with_name(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text('[package]\nname="unit_fixture"\nversion="0.1.0"\nedition="2021"\n')
    (repo / "src/lib.rs").write_text("pub mod a; pub mod b; pub mod c; pub mod d;\n")
    for name in "abcd":
        (repo / f"src/{name}.rs").write_text("pub fn value() -> i32 { 0 }\n")
    return repo


def plan(*, b_description="Fix b"):
    return {"ordered_edits": [{"file": f"src/{n}.rs", "description": f"Fix {n}" if n != "b" else b_description}
                              for n in "abcd"],
            "repair_units": [{"unit_id": n, "description": f"Unit {n}", "edit_indices": [i],
                              "depends_on": (["a"] if n == "b" else ["b"] if n == "d" else [])}
                             for i, n in enumerate("abcd")]}


@pytest.fixture
def setup(tmp_path):
    ma = load("multi_agent")
    tool_list, state = load("tools").make_tools(make_repo(tmp_path), post_strip_check=False)
    tools = {t.name: t for t in tool_list}
    checks = []

    def check(args):
        checks.append(dict(args))
        return json.dumps({"exit_code": 0, "diagnostics": []})

    tools["cargo_check"] = SimpleNamespace(invoke=check)
    board = {}
    coord = ma._repair_units.RepairUnitCoordinator(state, tools, board)
    coord.configure(plan(), [])
    try:
        yield ma, state, tools, coord, checks, board
    finally:
        state.close()


def edit(tools, name, value=1):
    return tools["edit_file"].invoke({"path": f"src/{name}.rs", "old_text": "{ 0 }", "new_text": f"{{ {value} }}"})


def close_unit(tools, coord, name):
    assert coord.select()["unit_id"] == name
    edit(tools, name)
    assert coord.verify(complete=True, summary=f"Fixed {name}")[0]


def test_dependency_order_and_partial_state_ledger(setup):
    _, state, tools, coord, checks, board = setup
    close_unit(tools, coord, "a")
    assert coord.select()["unit_id"] == "b"
    assert not coord.can_submit()
    assert checks == [{"manifest_path": "Cargo.toml"}]
    ledger = board["repair_units"]
    assert ledger["current_unit"] == "b"
    assert ledger["units"][0]["completed_changes"] == ["src/a.rs"]
    assert ledger["units"][0]["evidence"][0]["kind"] == "compile"
    assert ledger["units"][1]["depends_on"] == ["a"]
    assert not state.verified()  # A type check is not final behavioral evidence.


def test_rollback_preserves_independent_completed_work(setup):
    _, state, tools, coord, _, _ = setup
    for name in "abcd":
        close_unit(tools, coord, name)
    assert coord.all_done()
    assert coord.rollback({"b"}, "bad registry design") == ["b", "d"]
    for name in "ac":
        assert "{ 1 }" in (state.root / f"src/{name}.rs").read_text()
        assert coord._unit(name)["status"] == "verified" and coord.evidence_valid(coord._unit(name))
    for name in "bd":
        assert "{ 0 }" in (state.root / f"src/{name}.rs").read_text()
        assert coord._unit(name)["evidence"] == []
    assert not state.verified()
    assert coord.select()["unit_id"] == "b"


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo unavailable")
def test_full_orchestrator_closes_units_and_preserves_prerequisite_on_escalation(tmp_path, monkeypatch):
    agent = load("agent")
    repo = make_repo(tmp_path)
    states, events, fixer_units = [], [], []
    original_make = agent.make_tools

    def make(*args, **kwargs):
        tools, state = original_make(*args, **kwargs)
        state.post_strip_check = False
        state.require_reproduction = False
        states.append(state)
        return tools, state

    monkeypatch.setattr(agent, "make_tools", make)
    monkeypatch.setattr(agent, "ChatOpenAI", Mock())
    failed_b = False

    def create(model, tools, system_prompt):
        nonlocal failed_b
        t = {x.name: x for x in tools}
        role = "navigator" if "Navigator & Reproducer" in system_prompt else "suggester" if "Rust Architect / Suggester" in system_prompt else "fixer" if "Rust Fixer" in system_prompt else "verifier"
        events.append(role)

        def invoke(inputs, config):
            nonlocal failed_b
            if role == "navigator":
                t["write_regression_test"].invoke({"path": "tests/test_regression_unit.rs", "content": "#[test] fn result() {}"})
                t["submit_localization_bundle"].invoke({"summary": "cross-module repair", "needs_architecture_planning": True,
                        "candidate_chains": [
                            {"chain_id": "C1", "hypothesis": "fix dependent modules", "nodes": [
                                {"path": "src/a.rs", "symbol": "value"}, {"path": "src/b.rs", "symbol": "value"}]},
                            {"chain_id": "C2", "hypothesis": "fix dependent callers", "nodes": [
                                {"path": "src/c.rs", "symbol": "value"}]},
                        ]})
            elif role == "suggester":
                p = plan()
                t["submit_repair_plan"].invoke(dict(p, selected_chain_id="C1", arbitration_rationale="a precedes b, b precedes d"))
                if failed_b:
                    assert "repair_units" in inputs["messages"][0]["content"]
                    assert "{ 1 }" in (states[-1].root / "src/a.rs").read_text()
            elif role == "fixer":
                prompt = inputs["messages"][-1]["content"]
                ledger = prompt.split("Coordinator Repair Units (only edit the current unit; do not restore pre_fix):\n", 1)[1].split("\n", 1)[0]
                unit = json.loads(ledger)["current_unit"]
                fixer_units.append(unit)
                if unit == "b" and not failed_b:
                    edit(t, "b", 2)
                    failed_b = True
                    t["escalate_for_replan"].invoke({"reason": "b contract wrong; keep verified a"})
                elif unit:
                    if unit == "b":
                        assert "{ 2 }" in (states[-1].root / "src/b.rs").read_text()
                        assert "{ 1 }" in (states[-1].root / "src/a.rs").read_text()
                        t["edit_file"].invoke({"path": "src/b.rs", "old_text": "{ 2 }", "new_text": "{ 1 }"})
                    else:
                        edit(t, unit)
                    t["complete_repair_unit"].invoke({"summary": f"fixed {unit}"})
                else:
                    states[-1].pass_test(states[-1].start_test(), test_target="test_regression_unit")
                    t["submit_for_qa"].invoke({"summary": "all units complete"})
            else:
                t["submit_qa_decision"].invoke({"decision": "APPROVE", "summary": "complete", "plan_consistency": "consistent"})
            return {"messages": inputs["messages"]}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent, "create_agent", create)
    trace_path = tmp_path / "unit_trace.jsonl"
    trace = agent.TraceRecorder(trace_path)
    patch = agent.run_agent(repo, "Fix cross-module value implementations", "unused", "unused", max_api_calls=150, trace=trace)
    assert fixer_units == ["a", "b", "b", "c", "d", None]
    assert events.count("suggester") == 2 and events[-1] == "verifier"
    assert all(f"src/{name}.rs" in patch for name in "abcd")
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert not any(e for e in records if e["event"] == "repair_unit_rollback" and e["affected_units"])
    assert any(e["event"] == "repair_unit_verification" and e["complete"] for e in records)
