"""Offline scheduling contracts plus real Cargo fixtures; no model API calls."""

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


def test_cycles_and_shared_files_form_joint_units(setup):
    ma, _, _, _, _, _ = setup
    p = plan()
    p["repair_units"][0]["depends_on"] = ["b"]
    units = ma._repair_units.normalize_units(p, [])
    assert {u["unit_id"] for u in units} == {"a+b", "c", "d"}
    assert next(u for u in units if u["unit_id"] == "d")["depends_on"] == ["a+b"]
    p["ordered_edits"][2]["file"] = "src/a.rs"
    units = ma._repair_units.normalize_units(p, [])
    assert {u["unit_id"] for u in units} == {"a+b+c", "d"}


@pytest.mark.parametrize("change", ["unknown_dependency", "uncovered_edit", "bad_index", "duplicate_id"])
def test_invalid_plans_rejected_before_handoff(setup, change):
    ma, state, _, _, _, _ = setup
    p = plan()
    if change == "unknown_dependency":
        p["repair_units"][0]["depends_on"] = ["missing"]
    elif change == "uncovered_edit":
        p["repair_units"].pop()
    elif change == "bad_index":
        p["repair_units"][0]["edit_indices"] = [100]
    else:
        p["repair_units"][1]["unit_id"] = "a"
    handoffs, holder = ma._make_handoff_tools("suggester", state, {}, {})
    result = handoffs[0].invoke(dict(p, selected_chain_id="C1", arbitration_rationale="root cause"))
    assert "Error in repair units" in result
    assert not holder["done"]


def test_legacy_plan_is_one_coherent_unit(setup):
    ma, _, _, _, _, _ = setup
    units = ma._repair_units.normalize_units({"ordered_edits": plan()["ordered_edits"]}, [])
    assert len(units) == 1 and not units[0]["strict_scope"]
    assert len(units[0]["paths"]) == 4


def test_failed_check_blocks_dependents_and_records_diagnostics(setup):
    _, _, tools, coord, _, board = setup
    coord.select()
    edit(tools, "a")
    tools["cargo_check"] = SimpleNamespace(invoke=lambda _: '{"exit_code":101,"diagnostics":[{"code":"E0308"}]}')
    assert not coord.verify(complete=True)[0]
    assert coord.active == "a" and coord.select()["unit_id"] == "a"
    u = board["repair_units"]["units"][0]
    assert u["status"] == "active" and u["blocker"]
    assert "E0308" in u["evidence"][0]["output"]


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


def test_replan_retains_matching_evidence_and_preserves_current_work(setup):
    _, state, tools, coord, _, _ = setup
    for name in "abcd":
        close_unit(tools, coord, name)
    coord.configure(plan(b_description="Different registry contract"), [])
    assert [u["unit_id"] for u in coord.units if u["status"] == "verified"] == ["a", "c"]
    assert "{ 1 }" in (state.root / "src/c.rs").read_text()
    assert "{ 1 }" in (state.root / "src/d.rs").read_text()


def test_evidence_reused_only_at_current_revision_and_exact_config(setup):
    _, state, tools, coord, checks, _ = setup
    coord.select()
    edit(tools, "a")
    result = '{"exit_code":0,"diagnostics":[]}'
    coord.observe("cargo_check", {"manifest_path": "Cargo.toml"}, result, state._revision)
    assert coord.verify()[0] and checks == []
    tools["edit_file"].invoke({"path": "src/a.rs", "old_text": "{ 1 }", "new_text": "{ 2 }"})
    assert coord.verify()[0] and len(checks) == 1
    coord._unit("a")["verification"] = [{"tool": "cargo_check", "arguments": {"features": ["extra"]}}]
    assert coord.verify()[0] and len(checks) == 2


def test_zero_test_execution_never_completes_behavior_target(setup):
    _, _, tools, coord, _, _ = setup
    coord.select()
    edit(tools, "a")
    coord._unit("a")["verification"] = [{"tool": "cargo_test", "arguments": {"test_filter": "not_found"}}]
    tools["cargo_test"] = SimpleNamespace(invoke=lambda _: '{"exit_code":0,"status":"passed","tests_executed":0}')
    assert not coord.verify(complete=True)[0]
    assert coord.active == "a"


def test_scope_global_restore_and_handoff_gates(setup):
    ma, state, _, coord, _, board = setup
    coord.select()
    assert coord.before_tool("edit_file", {"path": "src/b.rs"}) is None
    assert coord.before_tool("checkpoint", {"action": "restore", "name": "pre_fix"}) is None
    assert coord.before_tool("write_regression_test", {"path": "tests/test_regression_a.rs"}) is None
    assert coord.before_tool("write_regression_test", {"path": "src/b.rs"}) is None
    tools, holder = ma._make_handoff_tools("fixer", state, board, {"unit_coordinator": coord})
    handoffs = {t.name: t for t in tools}
    assert "pending" in handoffs["submit_for_qa"].invoke({"summary": "not done"})
    assert not holder["done"]
    handoffs["complete_repair_unit"].invoke({"summary": "unit ready"})
    assert holder["payload"]["action"] == "unit_complete"
    assert coord.active == "a"  # Claiming done is not verification.


def test_browsing_does_not_consume_repair_budget(setup):
    ma, state, tools, coord, _, board = setup
    coord.select()
    activity = {"files_read": [], "symbols_queried": [], "unit_coordinator": coord}
    wrapped = ma._wrap_tools_for_role("fixer", tools, ["read_lines"], [], {}, state,
                                      SimpleNamespace(calls_made=0), 0, 100, activity, blackboard=board)[0]
    for _ in range(100):
        wrapped.invoke({"path": "src/a.rs"})
    assert coord.steps == 0


def test_fixer_preserves_attempt_budget_without_forcing_read_slices(setup, monkeypatch):
    ma, _, _, coord, _, _ = setup
    client = SimpleNamespace(calls_made=0, remaining_calls=100)
    caps = []

    def role(*args, **kwargs):
        caps.append(kwargs["stage_cap"])
        client.calls_made += kwargs["stage_cap"]
        return None, "stage_cap_reached"

    monkeypatch.setattr(ma, "_run_role_agent", role)
    holder = {}
    result, outcome = ma._run_scheduled_fixer("system", "issue", [], holder, coordinator=coord,
                                             allow_escalate=True, client=client, stage_cap=40)
    assert caps == [40] and client.calls_made == 40
    assert outcome == "stage_cap_reached" and result is None


def test_repeated_coordinator_checks_also_trigger_replanning(setup, monkeypatch):
    ma, state, tools, coord, _, _ = setup

    def role(*args, **kwargs):
        edit(tools, "a")
        return {"action": "unit_complete", "summary": "candidate"}, "completed"

    tools["cargo_check"] = SimpleNamespace(invoke=lambda _: '{"exit_code":101,"diagnostics":[{"code":"E0308"}]}')
    monkeypatch.setattr(ma, "_run_role_agent", role)
    monkeypatch.setattr(state, "stuck_status", lambda: {"compile_repeats": 3, "test_repeats": 0})
    result, outcome = ma._run_scheduled_fixer("system", "issue", [], {}, coordinator=coord,
                                             allow_escalate=True, client=SimpleNamespace(calls_made=0), stage_cap=40)
    assert result["action"] == "escalate" and "E0308" in result["reason"]
    assert outcome == "unit_blocked"


def test_qa_edit_invalidates_unit_and_transitive_dependents(setup):
    _, _, tools, coord, _, _ = setup
    for name in "abcd":
        close_unit(tools, coord, name)
    assert coord.before_tool("edit_file", {"path": "src/a.rs"}) is None
    assert coord.active == "a"
    assert coord._unit("b")["status"] == "pending" and coord._unit("d")["status"] == "pending"
    assert coord._unit("c")["status"] == "verified"


def test_final_verification_can_remove_navigator_temporary_test(setup):
    _, _, tools, coord, _, _ = setup
    tools["write_regression_test"].invoke({"path": "tests/test_regression_navigator.rs", "content": "#[test] fn red() {}"})
    for name in "abcd":
        close_unit(tools, coord, name)
    assert coord.before_tool("revert_file", {"path": "tests/test_regression_navigator.rs"}) is None
    assert coord.active == "d"  # Only final verification unit is reopened.
    assert coord._unit("a")["status"] == "verified"


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo unavailable")
def test_real_cargo_failure_then_dependency_closed_success(setup):
    _, state, tools, coord, _, _ = setup
    diag = load("rust_diagnostics")
    tools["cargo_check"] = diag.make_cargo_check_tool(state.root, lambda p: state.root / p)
    coord.select()
    tools["edit_file"].invoke({"path": "src/a.rs", "old_text": "{ 0 }", "new_text": "{ missing_value() }"})
    assert not coord.verify(complete=True)[0]
    assert "E0425" in coord._unit("a")["evidence"][0]["output"]
    tools["edit_file"].invoke({"path": "src/a.rs", "old_text": "{ missing_value() }", "new_text": "{ 1 }"})
    assert coord.verify(complete=True)[0]
    assert coord.select()["unit_id"] == "b"


def test_unit_rollback_restores_test_oracle_and_clears_global_green(setup):
    _, state, tools, coord, _, _ = setup
    tools["write_regression_test"].invoke({"path": "tests/test_regression_a.rs", "content": "#[test] fn before() {}"})
    state.pass_test(state.start_test(), test_target="test_regression_a")
    assert state.verified()
    coord.select()
    edit(tools, "a")
    tools["write_regression_test"].invoke({"path": "tests/test_regression_a.rs", "content": "#[test] fn after() {}"})
    coord.observe("write_regression_test", {"path": "tests/test_regression_a.rs"}, "ok", state._revision - 1)
    tools["write_regression_test"].invoke({"path": "tests/test_regression_new.rs", "content": "#[test] fn new() {}"})
    coord.observe("write_regression_test", {"path": "tests/test_regression_new.rs"}, "ok", state._revision - 1)
    coord.rollback()
    assert "fn before" in (state.root / "tests/test_regression_a.rs").read_text()
    assert not (state.root / "tests/test_regression_new.rs").exists()
    assert Path("tests/test_regression_new.rs") not in state.regression_tests
    assert not state.verified() and not state._executed_test_targets


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


def test_unit_refutation_signals_block_completion_and_trigger_escalation(setup, monkeypatch):
    ma, state, tools, _, _, board = setup
    p = {
        "ordered_edits": [
            {"file": "src/a.rs", "description": "Fix trait impl in a"},
            {"file": "src/b.rs", "description": "Use trait in b"},
        ],
        "repair_units": [
            {
                "unit_id": "a",
                "description": "Unit a",
                "edit_indices": [0],
                "depends_on": [],
                "expected_outcome": "src/a.rs compiles cleanly",
                "refutation_signals": ["E0277"],
            },
            {"unit_id": "b", "description": "Unit b", "edit_indices": [1], "depends_on": ["a"]},
        ],
    }
    policy = ma._repair_candidates.CandidatePolicy(board, state)
    coord = ma._repair_units.RepairUnitCoordinator(state, tools, board, candidate_policy=policy)
    coord.configure(p, [])
    assert coord.select()["expected_outcome"] == "src/a.rs compiles cleanly"
    tools["cargo_check"] = SimpleNamespace(invoke=lambda _: '{"exit_code":101,"diagnostics":[{"code":"E0277","message":"the trait bound is not satisfied"}]}')

    def role(*args, **kwargs):
        edit(tools, "a")
        return {"action": "unit_complete", "summary": "done"}, "completed"

    monkeypatch.setattr(ma, "_run_role_agent", role)
    result, outcome = ma._run_scheduled_fixer(
        "system", "issue", [], {}, coordinator=coord,
        allow_escalate=True, client=SimpleNamespace(calls_made=0), stage_cap=20,
    )
    assert outcome == "unit_blocked"
    assert result["action"] == "escalate" and "E0277" in result["reason"]
    assert coord.active == "a"
    assert coord._unit("a")["status"] == "active"
    assert coord._unit("b")["status"] == "pending"
    assert policy.data["pending"]["kind"] == "verification_counterevidence"


def test_evomac_freezes_verified_units_preserves_across_empty_and_wording_replans_and_appends(setup):
    _, state, tools, coord, _, board = setup
    close_unit(tools, coord, "a")
    assert board["repair_units"]["units"][0]["frozen"] is True
    assert coord.select()["unit_id"] == "b"

    # Freeze guard blocks reverting verified unit files while another unit is active.
    blocked = coord.before_tool("revert_file", {"path": "src/a.rs"})
    assert blocked and "verified repair unit 'a'" in blocked

    # Empty replan must not wipe out an established unit DAG.
    coord.configure({"ordered_edits": [], "repair_units": []}, [])
    assert [u["unit_id"] for u in coord.units] == ["a", "b", "c", "d"]
    assert coord._unit("a")["status"] == "verified"

    # Wording tweaks on unit description/expected_outcome keep verified unit "a".
    tweaked = plan()
    tweaked["repair_units"][0]["description"] = "Reworded unit a"
    tweaked["repair_units"][0]["expected_outcome"] = "a compiles"
    coord.configure(tweaked, [])
    assert coord._unit("a")["status"] == "verified"

    # Incremental replan omitting verified unit "a" carries "a" forward cleanly.
    inc = {
        "ordered_edits": [{"file": "src/b.rs", "description": "Revised b"}],
        "repair_units": [{"unit_id": "b", "description": "Unit b", "edit_indices": [0], "depends_on": []}],
    }
    coord.configure(inc, [])
    assert [u["unit_id"] for u in coord.units] == ["a", "b"]
    assert coord._unit("a")["status"] == "verified"
