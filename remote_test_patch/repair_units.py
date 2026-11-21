"""Dependency-closed repair scheduling; no extra model role or shell execution.

Units own disjoint files (overlaps and dependency cycles are joined). Evidence is
local to a unit and its declared prerequisites, never a replacement for final QA.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class VerificationTarget(BaseModel):
    tool: Literal["cargo_check", "cargo_test"] = "cargo_check"
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments of the named Cargo tool, not a shell command")


class RepairUnitSpec(BaseModel):
    unit_id: str = Field(description="Stable short identifier, retained across replans for unchanged work")
    description: str
    edit_indices: list[int] = Field(description="Zero-based indices into ordered_edits; group mutually dependent hunks")
    depends_on: list[str] = Field(default_factory=list)
    verification: list[VerificationTarget] = Field(default_factory=list, description="Targeted Cargo checks after this unit closes; defaults to affected manifests")


class UnitBoundary(Exception):
    """Yield the long-running tool loop to the deterministic Coordinator."""


def normalize_units(plan: dict | None, fallback_paths: list[str]) -> list[dict]:
    plan = plan or {}
    edits = plan.get("ordered_edits") or []
    raw = plan.get("repair_units") or []
    explicit = bool(raw)
    if not raw:
        raw = [{"unit_id": "repair", "description": "Coherent direct/legacy repair", "edit_indices": list(range(len(edits)))}]
    units = []
    ids = [str(x.get("unit_id", "")).strip() for x in raw]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("repair_units must have unique non-empty unit_id values")
    covered = set()
    for item in raw:
        indices = item.get("edit_indices") or []
        if any(type(i) is not int or not 0 <= i < len(edits) for i in indices):
            raise ValueError("edit_indices must refer to zero-based ordered_edits indices")
        covered.update(indices)
        deps = set(item.get("depends_on") or [])
        if deps - set(ids):
            raise ValueError(f"Unknown repair unit prerequisites: {sorted(deps - set(ids))}")
        work = [e for i, e in enumerate(edits) if i in indices and e.get("action") != "preserve"]
        paths = {str(e["file"]).replace("\\", "/") for e in work if e.get("file")}
        if not explicit and not paths:
            paths = set(fallback_paths)
        for path in paths:
            if Path(path).is_absolute() or ".." in Path(path).parts or ":" in path:
                raise ValueError(f"Unit path must be repository-relative: {path}")
        checks = item.get("verification") or []
        for check in checks:
            VerificationTarget.model_validate(check)
        units.append({"members": {item["unit_id"]}, "description": item.get("description", ""),
                      "paths": paths, "work": work, "deps": deps, "verification": checks,
                      "strict_scope": explicit})
    required = {i for i, e in enumerate(edits) if e.get("action") != "preserve"}
    if explicit and required - covered:
        raise ValueError("Every non-preserve ordered_edits step must belong to a repair unit")

    # Join file overlaps and SCCs. Recompute reachability after each join because
    # an overlap itself can create a new dependency cycle.
    while True:
        owner = {member: i for i, u in enumerate(units) for member in u["members"]}
        edges = [{owner[d] for d in u["deps"]} - {i} for i, u in enumerate(units)]
        reach = [set(e) for e in edges]
        for _ in units:
            reach = [r | set().union(*(reach[j] for j in r)) for r in reach]
        pair = next(((i, j) for i in range(len(units)) for j in range(i + 1, len(units))
                     if units[i]["paths"] & units[j]["paths"] or (j in reach[i] and i in reach[j])), None)
        if pair is None:
            break
        i, j = pair
        a, b = units[i], units.pop(j)
        for key in ("members", "paths", "deps"):
            a[key] |= b[key]
        a["description"] += " / " + b["description"]
        a["work"] += b["work"]
        a["verification"] += b["verification"]
    owner = {m: "+".join(sorted(u["members"])) for u in units for m in u["members"]}
    for u in units:
        u["unit_id"] = "+".join(sorted(u.pop("members")))
        u["depends_on"] = sorted({owner[d] for d in u.pop("deps")} - {u["unit_id"]})
        u["paths"] = sorted(u["paths"])
    return units


class RepairUnitCoordinator:
    UNCHECKED_EDITS = 6
    READ_ONLY_TOOLS = frozenset({
        "read_file", "read_lines", "search_code", "cargo_workspace", "rust_symbols",
        "inspect_rust_neighborhood", "rust_error_explain", "rust_trace",
    })

    def __init__(self, state, tools_by_name, blackboard, trace=None, candidate_policy=None):
        self.state, self.tools, self.blackboard, self.trace = state, tools_by_name, blackboard, trace
        self.candidate_policy = candidate_policy
        self.units: list[dict] = []
        self.active: str | None = None
        self.plan_key = None
        self.steps = self.unchecked_edits = 0
        self.check_cache: dict = {}

    def _unit(self, uid):
        return next(u for u in self.units if u["unit_id"] == uid)

    def status(self):
        return {"current_unit": self.active, "units": [
            {k: u.get(k) for k in ("unit_id", "description", "paths", "depends_on", "status",
                                   "completed_changes", "summary", "evidence", "blocker")}
            for u in self.units]}

    def publish(self, event, **extra):
        self.blackboard["repair_units"] = self.status()
        if self.trace:
            self.trace.record(event, **self.status(), **extra)

    def _images(self, paths):
        images = {}
        for path in paths:
            p = self.state.root / path
            images[str(path)] = p.read_bytes() if p.is_file() else None
        return images

    def _closure(self, uid):
        result = {uid}
        while True:
            more = set().union(*(set(self._unit(i)["depends_on"]) for i in result)) - result
            if not more:
                return result
            result |= more

    def _dependents(self, ids):
        result = set(ids)
        while True:
            more = {u["unit_id"] for u in self.units if set(u["depends_on"]) & result} - result
            if not more:
                return result
            result |= more

    def configure(self, plan, fallback_paths):
        specs = normalize_units(plan, fallback_paths)
        self.planned_scope = {p for u in specs for p in u["paths"]}
        if self.candidate_policy and not self.candidate_policy.data["scope_initialized"]:
            self.candidate_policy.observe_scope({p for u in specs for p in u["paths"]})
        key = json.dumps(specs, sort_keys=True)
        policy = self.candidate_policy
        switch_paths = set(policy.data.pop("switch_paths", []) if policy else [])
        if policy:
            policy.data.pop("retain_current_patch", None)
        if key == self.plan_key and not switch_paths:
            return
        old = {u["unit_id"]: u for u in self.units}
        replan_baseline = self.state._capture_snapshot() if self.units else None
        keep = {s["unit_id"] for s in specs if s["unit_id"] in old
                and old[s["unit_id"]]["spec"] == s
                and old[s["unit_id"]]["status"] == "verified"
                and self.evidence_valid(old[s["unit_id"]])
                and not set(s["paths"]) & switch_paths}
        # A retained unit must also retain all of its verified prerequisites.
        while True:
            reduced = {uid for uid in keep if set(old[uid]["depends_on"]) <= keep}
            if reduced == keep:
                break
            keep = reduced
        self.units = [old[s["unit_id"]] if s["unit_id"] in keep else
                      dict(s, spec=s, status="pending", completed_changes=[], evidence=[], blocker="", baseline=replan_baseline, touched=set())
                      for s in specs]
        partial_paths = [p.as_posix() for p in self.state.changed_paths()]
        self.check_cache.clear()
        self.active = None
        self.plan_key = key
        self.publish("repair_unit_plan", retained_units=sorted(keep), preserved_partial_paths=partial_paths)

    def evidence_valid(self, unit):
        return unit.get("inputs") == self._images((unit.get("inputs") or {}).keys())

    def select(self):
        if self.active:
            return self._unit(self.active)
        stale = {u["unit_id"] for u in self.units if u["status"] == "verified" and not self.evidence_valid(u)}
        if stale:
            self.invalidate(stale, "Unit or prerequisite inputs changed")
        done = {u["unit_id"] for u in self.units if u["status"] == "verified"}
        ready = next((u for u in self.units if u["status"] != "verified" and set(u["depends_on"]) <= done), None)
        if ready:
            self.active = ready["unit_id"]
            ready["status"] = "active"
            if ready.get("baseline") is None:
                ready["baseline"] = self.state._capture_snapshot()
            self.steps = self.unchecked_edits = 0
            self.publish("repair_unit_start")
        return ready

    def invalidate(self, ids, reason):
        for uid in self._dependents(ids):
            u = self._unit(uid)
            u.update(status="pending", evidence=[], blocker=reason)
        self.active = None
        self.publish("repair_unit_invalidated")

    def all_done(self):
        return bool(self.units) and all(u["status"] == "verified" and self.evidence_valid(u) for u in self.units)

    def can_submit(self):
        if self.all_done():
            return True
        return bool(self.active) and all(u["status"] == "verified" or u["unit_id"] == self.active for u in self.units)

    def before_tool(self, name, kwargs):
        if name not in ("edit_file", "write_file", "revert_file", "write_regression_test"):
            return None
        path = str(kwargs.get("target_src") or kwargs.get("path") or "").replace("\\", "/")
        relative = Path(path)
        source_test = bool(kwargs.get("target_src")) or ("src" in relative.parts and relative.suffix == ".rs")
        temporary = ((name == "write_regression_test" and not source_test)
                     or (relative in self.state.regression_tests and relative not in self.state.in_module_regressions))
        if not self.active:
            owners = {u["unit_id"] for u in self.units if path in u["paths"] or path in u["completed_changes"]}
            if not owners and (temporary or not self.units[-1]["strict_scope"]):
                owners = {self.units[-1]["unit_id"]}
            if owners:
                self.invalidate(owners, "QA/refinement changed unit inputs; recheck dependent units")
                self.select()
        unit = self._unit(self.active) if self.active else None
        if unit and path and not temporary and path not in unit["paths"]:
            unit["paths"] = sorted(set(unit["paths"]) | {path})
            self.planned_scope.add(path)
            stale = {other["unit_id"] for other in self.units
                     if other is not unit and path in other["paths"]}
            for uid in self._dependents(stale):
                other = self._unit(uid)
                if other is not unit:
                    other.update(status="pending", evidence=[], blocker="Overlapping file changed")
        return None

    def observe(self, name, arguments, out, revision_before):
        evidence_id = None
        if self.candidate_policy:
            evidence_id = self.candidate_policy.observe(name, arguments, out)
        if name not in self.READ_ONLY_TOOLS:
            self.steps += 1
        if self.state._revision != revision_before:
            if name == "checkpoint" and str(arguments.get("action", "save")).lower() == "restore":
                self.invalidate({u["unit_id"] for u in self.units}, "Checkpoint restored; recheck units")
                self.check_cache.clear()
                return evidence_id
            self.unchecked_edits += 1
            if self.active:
                path = arguments.get("target_src") or arguments.get("path")
                if path:
                    self._unit(self.active)["touched"].add(str(path).replace("\\", "/"))
                self._unit(self.active)["completed_changes"] = [p.as_posix() for p in self.delta_paths(self._unit(self.active))]
                self.publish("repair_unit_edit")
            if self.candidate_policy:
                self.candidate_policy.observe_scope(self.planned_scope | {
                    p.as_posix() for p in self.state.changed_paths()
                    if p not in self.state.regression_tests or p in self.state.in_module_regressions})
        if name in ("cargo_check", "cargo_test"):
            self.check_cache[(name, json.dumps(arguments, sort_keys=True))] = (self.state._revision, out)
            self.steps = self.unchecked_edits = 0
        return evidence_id

    def delta_paths(self, unit):
        snap = unit.get("baseline")
        if not snap:
            return []
        result = []
        for path, original in self.state.originals.items():
            if path.as_posix() not in set(unit["paths"]) | unit.get("touched", set()):
                continue
            before = snap["files"].get(path, original)
            now = self.state.root / path
            data = now.read_bytes() if now.is_file() else None
            if data != (before[0] if before else None):
                result.append(path)
        return result

    def targets(self, unit):
        if unit["verification"]:
            return unit["verification"]
        manifests = set()
        for path in unit["paths"] or ["src/lib.rs"]:
            parent = (self.state.root / path).parent
            while parent.is_relative_to(self.state.root):
                if (parent / "Cargo.toml").is_file():
                    manifests.add((parent / "Cargo.toml").relative_to(self.state.root).as_posix())
                    break
                parent = parent.parent
        return [{"tool": "cargo_check", "arguments": {"manifest_path": m}} for m in sorted(manifests or {"Cargo.toml"})]

    def verify(self, *, complete=False, summary=""):
        if not self.active:
            return self.all_done(), "No active unit; perform final regression and independent QA."
        unit = self._unit(self.active)
        evidence = []
        for target in self.targets(unit):
            evidence_id = None
            name, arguments = target["tool"], target.get("arguments") or {}
            cache_key = (name, json.dumps(arguments, sort_keys=True))
            cached = self.check_cache.get(cache_key)
            try:
                if cached and cached[0] == self.state._revision:
                    out = cached[1]
                else:
                    out = self.tools[name].invoke(arguments)
                    self.check_cache[cache_key] = (self.state._revision, out)
                if self.candidate_policy:
                    evidence_id = self.candidate_policy.observe(name, arguments, out)
                result = json.loads(out)
                ok = result.get("exit_code") == 0 and "error" not in result
                behavior = name == "cargo_test" and not arguments.get("no_run", False)
                if behavior:
                    ok = ok and result.get("tests_executed", 0) > 0 and result.get("status") == "passed"
            except (KeyError, TypeError, ValueError) as exc:
                out, ok, behavior = str(exc), False, False
            evidence.append({"tool": name, "arguments": arguments, "revision": self.state._revision,
                             "candidate_evidence_id": evidence_id,
                             "ok": bool(ok), "kind": "behavior" if behavior else "compile",
                             "output": str(out)[-1800:]})
        ok = bool(evidence) and all(e["ok"] for e in evidence)
        unit["evidence"] = evidence
        unit["completed_changes"] = [p.as_posix() for p in self.delta_paths(unit)]
        unit["blocker"] = "" if ok else "Targeted unit verification failed"
        self.steps = self.unchecked_edits = 0
        if ok and complete:
            paths = {p for uid in self._closure(self.active) for p in self._unit(uid)["paths"]}
            paths.update(unit["completed_changes"])
            # Manifest/lockfile changes also invalidate earlier compilation evidence.
            for path in list(paths):
                for parent in (self.state.root / path).parents:
                    if not parent.is_relative_to(self.state.root):
                        break
                    for file in ("Cargo.toml", "Cargo.lock"):
                        if (parent / file).is_file():
                            paths.add((parent / file).relative_to(self.state.root).as_posix())
            unit.update(status="verified", inputs=self._images(paths), summary=summary)
            self.active = None
        self.publish("repair_unit_verification", complete=bool(ok and complete))
        return ok, json.dumps({"unit_id": unit["unit_id"], "checks": evidence, "complete": bool(ok and complete)}, ensure_ascii=False)

    def rollback(self, ids=None, reason="Unit failed", *, expand=True):
        if ids is None:
            if self.active:
                ids = {self.active}
            else:
                diagnostics = json.dumps(getattr(self.state, "_last_check_diagnostics", [])) + reason
                ids = {u["unit_id"] for u in self.units if any(p in diagnostics for p in u["paths"])}
                if not ids and self.units:
                    ids = {self.units[-1]["unit_id"]}
        affected = self._dependents(ids) if expand else set(ids)
        # Units have disjoint production files. Reverse order also restores shared
        # temporary tests to the earliest affected baseline.
        for unit in reversed(self.units):
            if unit["unit_id"] in affected:
                paths = self.delta_paths(unit)
                if paths:
                    self.state.restore_partial_snapshot(unit["baseline"], paths)
                    cache = self.blackboard.get("shared_read_excerpts", {})
                    for path in paths:
                        prefix = path.as_posix()
                        for key in list(cache):
                            if key == prefix or key.startswith(prefix + ":"):
                                cache.pop(key, None)
                unit.update(status="pending", baseline=None, completed_changes=[], evidence=[], blocker=reason, touched=set())
        self.active = None
        self.check_cache.clear()
        self.publish("repair_unit_rollback", affected_units=sorted(affected))
        return sorted(affected)

    def prompt(self):
        active = self._unit(self.active) if self.active else None
        ledger = self.status()
        for unit in ledger["units"]:
            # Keep full evidence in traces, but do not resend successful Cargo
            # boilerplate in every clean-context slice.
            unit["evidence"] = [{k: v for k, v in evidence.items() if k != "output" or not evidence["ok"]}
                                for evidence in unit["evidence"]]
        return ("\n\nCoordinator Repair Units (only edit the current unit; do not restore pre_fix):\n"
                + json.dumps(ledger, ensure_ascii=False)
                + ("\nCurrent unit work:\n" + json.dumps(active["work"], ensure_ascii=False) if active else "")
                + "\nFinish a dependency-closed unit with complete_repair_unit(summary). Coordinator runs targeted Cargo checks. "
                  "Edit related files as needed. Checkpoints may be restored explicitly. "
                  "After all units close, run final regression/existing tests and submit_for_qa.")
