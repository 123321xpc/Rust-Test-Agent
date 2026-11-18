"""Dependency-closed repair scheduling; no extra model role or shell execution.

Units own disjoint files (overlaps and dependency cycles are joined). Evidence is
local to a unit and its declared prerequisites, never a replacement for final QA.
"""

import json
import re
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
    expected_outcome: str = Field(default="", description="Expected compiler or test outcome when this unit closes")
    refutation_signals: list[str] = Field(default_factory=list, description="Error codes or failure substrings that refute this unit's causal hypothesis")


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
        ref_signals = [str(s).strip() for s in (item.get("refutation_signals") or []) if str(s).strip()]
        units.append({"members": {item["unit_id"]}, "description": item.get("description", ""),
                      "paths": paths, "work": work, "deps": deps, "verification": checks,
                      "expected_outcome": str(item.get("expected_outcome") or "").strip(),
                      "refutation_signals": ref_signals,
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
        a["expected_outcome"] = " / ".join(x for x in (a["expected_outcome"], b["expected_outcome"]) if x)
        a["refutation_signals"] = list(dict.fromkeys(a["refutation_signals"] + b["refutation_signals"]))
    owner = {m: "+".join(sorted(u["members"])) for u in units for m in u["members"]}
    for u in units:
        u["unit_id"] = "+".join(sorted(u.pop("members")))
        u["depends_on"] = sorted({owner[d] for d in u.pop("deps")} - {u["unit_id"]})
        u["paths"] = sorted(u["paths"])
    return units


def _spec_core(spec: dict[str, Any]) -> tuple:
    """Return execution-relevant fields of a normalized unit spec for retention across wording edits."""
    return (
        tuple(spec.get("paths") or []),
        spec.get("work") or [],
        tuple(sorted(spec.get("depends_on") or spec.get("deps") or [])),
        spec.get("verification") or [],
        bool(spec.get("strict_scope", False)),
    )


class RepairUnitCoordinator:
    UNCHECKED_EDITS = 6
    READ_ONLY_TOOLS = frozenset({
        "read_file", "read_lines", "search_code", "cargo_workspace", "rust_symbols",
        "inspect_rust_neighborhood", "rust_error_explain",
    })

    def __init__(self, state, tools_by_name, blackboard, trace=None, candidate_policy=None):
        self.state, self.tools, self.blackboard, self.trace = state, tools_by_name, blackboard, trace
        self.candidate_policy = candidate_policy
        self.units: list[dict] = []
        self.planned_scope: set[str] = set()
        self.active: str | None = None
        self.plan_key = None
        self.steps = self.unchecked_edits = 0
        self.check_cache: dict = {}

    def _unit(self, uid):
        return next(u for u in self.units if u["unit_id"] == uid)

    def status(self):
        return {"current_unit": self.active, "units": [
            {**{k: u.get(k) for k in ("unit_id", "description", "paths", "depends_on", "status",
                                      "expected_outcome", "refutation_signals", "refuted_by",
                                      "completed_changes", "summary", "evidence", "blocker")},
             "frozen": u.get("status") == "verified"}
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
        raw_plan = plan or {}
        policy = self.candidate_policy
        switch_paths = set(policy.data.pop("switch_paths", []) if policy else [])
        if policy:
            policy.data.pop("retain_current_patch", None)
        if self.units and not (raw_plan.get("ordered_edits") or []) and not (raw_plan.get("repair_units") or []):
            # Guard against empty replans wiping out an established unit DAG.
            if switch_paths:
                stale = {u["unit_id"] for u in self.units if set(u["paths"]) & switch_paths}
                if stale:
                    self.invalidate(stale, "Candidate hypothesis switched")
            return
        specs = normalize_units(plan, fallback_paths)
        if policy and not policy.data["scope_initialized"]:
            policy.observe_scope({p for u in specs for p in u["paths"]})
        key = json.dumps(specs, sort_keys=True)
        if key == self.plan_key and not switch_paths:
            return
        old = {u["unit_id"]: u for u in self.units}
        replan_baseline = self.state._capture_snapshot() if self.units else None
        keep = {s["unit_id"] for s in specs if s["unit_id"] in old
                and _spec_core(old[s["unit_id"]]["spec"]) == _spec_core(s)
                and old[s["unit_id"]]["status"] in {"verified", "completed_unverified"}
                and self.evidence_valid(old[s["unit_id"]])
                and not set(s["paths"]) & switch_paths}
        new_ids = {s["unit_id"] for s in specs}
        new_paths = {p for s in specs for p in s["paths"]}
        # Carry over verified units omitted from an incremental replan when paths do not conflict.
        carried_ids = {
            uid for uid, prev in old.items()
            if uid not in new_ids
            and prev.get("status") == "verified"
            and self.evidence_valid(prev)
            and not (set(prev.get("paths") or []) & (switch_paths | new_paths))
        }
        retained = keep | carried_ids
        # A retained unit must also retain all of its verified/retained prerequisites.
        while True:
            reduced = {uid for uid in retained if set(old[uid]["depends_on"]) <= retained}
            if reduced == retained:
                break
            retained = reduced
        keep = retained & new_ids
        carried = [old[uid] for uid in old if uid in (retained & carried_ids)]
        self.units = carried + [
            old[s["unit_id"]] if s["unit_id"] in keep else
            dict(s, spec=s, status="pending", completed_changes=[], evidence=[], blocker="", baseline=replan_baseline, touched=set())
            for s in specs
        ]
        self.planned_scope = {p for u in self.units for p in u["paths"]}
        partial_paths = [p.as_posix() for p in self.state.changed_paths()]
        self.check_cache.clear()
        self.active = None
        self.plan_key = key
        self.publish("repair_unit_plan", retained_units=sorted(retained), preserved_partial_paths=partial_paths)

    def evidence_valid(self, unit):
        return unit.get("inputs") == self._images((unit.get("inputs") or {}).keys())

    def select(self):
        if self.active:
            return self._unit(self.active)
        stale = {u["unit_id"] for u in self.units if u["status"] in {"verified", "completed_unverified"} and not self.evidence_valid(u)}
        if stale:
            self.invalidate(stale, "Unit or prerequisite inputs changed")
        done = {u["unit_id"] for u in self.units if u["status"] in {"verified", "completed_unverified"}}
        ready = next((u for u in self.units if u["status"] not in {"verified", "completed_unverified"} and set(u["depends_on"]) <= done), None)
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
        return bool(self.units) and all(u["status"] in {"verified", "completed_unverified"} and self.evidence_valid(u) for u in self.units)

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
        if name == "revert_file" and self.active and path and not temporary:
            frozen = next(
                (other for other in self.units
                 if other["unit_id"] != self.active
                 and other.get("status") == "verified"
                 and path in (set(other.get("paths") or []) | set(other.get("completed_changes") or []))),
                None,
            )
            if frozen is not None:
                return (
                    f"Error: Cannot revert '{path}' because it belongs to verified repair unit "
                    f"'{frozen['unit_id']}'. Keep verified units intact; edit only active unit '{self.active}' "
                    "or call escalate_for_replan."
                )
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

    def detect_refutation(self, unit, evidence):
        """Return the matching refutation signal when failed check output refutes the unit hypothesis."""
        signals = unit.get("refutation_signals") or []
        if not signals:
            return ""
        failed_text = "\n".join(str(e.get("output") or "") for e in evidence if not e.get("ok")).lower()
        if not failed_text:
            return ""
        for sig in signals:
            if sig.lower() in failed_text:
                return sig
        return ""

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
        refuted_by = "" if ok else self.detect_refutation(unit, evidence)
        unit["evidence"] = evidence
        unit["completed_changes"] = [p.as_posix() for p in self.delta_paths(unit)]
        unit["refuted_by"] = refuted_by
        unit["blocker"] = (
            f"Unit hypothesis refuted by observed output ({refuted_by})"
            if refuted_by else ("" if ok else "Targeted unit verification failed")
        )
        self.steps = self.unchecked_edits = 0
        if refuted_by:
            if self.candidate_policy:
                ev_ids = [e["candidate_evidence_id"] for e in evidence if not e["ok"] and e.get("candidate_evidence_id")]
                if ev_ids:
                    self.candidate_policy.counterevidence(
                        f"Unit '{unit['unit_id']}' matched refutation signal '{refuted_by}'.", ev_ids
                    )
            self.publish("repair_unit_verification", complete=False, verified=False, refuted_by=refuted_by)
            return False, json.dumps({"unit_id": unit["unit_id"], "checks": evidence, "complete": False,
                                      "verified": False, "refuted_by": refuted_by}, ensure_ascii=False)
        if complete:
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
            unit.update(status="verified" if ok else "completed_unverified", inputs=self._images(paths), summary=summary)
            self.active = None
        self.publish("repair_unit_verification", complete=complete, verified=bool(ok))
        return ok, json.dumps({"unit_id": unit["unit_id"], "checks": evidence, "complete": complete, "verified": bool(ok)}, ensure_ascii=False)

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

    def append_unit(
        self,
        unit_id: str,
        description: str,
        paths: list[str],
        *,
        symbol: str = "",
        depends_on: list[str] | None = None,
        expected_outcome: str = "",
    ) -> dict[str, Any]:
        """Dynamically append a missing subtask unit (or merge into an overlapping owner) without wiping verified units."""
        norm_paths: list[str] = []
        for raw_p in paths:
            p = str(raw_p or "").strip().replace("\\", "/")
            if not p:
                continue
            rel = Path(p)
            if rel.is_absolute() or ".." in rel.parts or ":" in p:
                raise ValueError(f"Unit path must be repository-relative: {p}")
            if p not in norm_paths:
                norm_paths.append(p)
        if not norm_paths:
            raise ValueError("append_unit requires at least one repository-relative file path")
        uid = str(unit_id or "").strip()
        if not uid:
            raise ValueError("append_unit requires a non-empty unit_id")
        work_items = [
            {"file": p, "symbol": symbol, "action": "modify", "description": description}
            for p in norm_paths
        ]
        overlapping = [u for u in self.units if set(u["paths"]) & set(norm_paths)]
        if overlapping:
            target = next((u for u in overlapping if u["unit_id"] == self.active), overlapping[0])
            old_status = target.get("status")
            for w in work_items:
                if w not in target["work"]:
                    target["work"].append(w)
            target["paths"] = sorted(set(target["paths"]) | set(norm_paths))
            if expected_outcome:
                target["expected_outcome"] = " / ".join(
                    x for x in (target.get("expected_outcome", ""), expected_outcome) if x
                )
            self.planned_scope |= set(norm_paths)
            if target["unit_id"] != self.active and old_status in {"verified", "completed_unverified"}:
                self.invalidate({target["unit_id"]}, f"Appended follow-up work: {description[:120]}")
                if not self.active:
                    self.select()
            self.plan_key = None
            self.publish("repair_unit_appended", action="merged", target_unit=target["unit_id"], paths=norm_paths)
            return {"action": "merged", "unit_id": target["unit_id"], "paths": norm_paths, "reopened": old_status == "verified"}

        existing_ids = {u["unit_id"] for u in self.units}
        base_uid = uid
        suffix = 2
        while uid in existing_ids:
            uid = f"{base_uid}_{suffix}"
            suffix += 1
        if depends_on is None:
            deps = sorted(u["unit_id"] for u in self.units if u.get("status") == "verified")
        else:
            deps = sorted({str(d).strip() for d in depends_on if str(d).strip()} & existing_ids)
        new_spec = {
            "unit_id": uid,
            "description": description,
            "paths": sorted(norm_paths),
            "work": work_items,
            "depends_on": deps,
            "verification": [],
            "expected_outcome": expected_outcome.strip(),
            "refutation_signals": [],
            "strict_scope": bool(self.units and self.units[0].get("strict_scope")),
        }
        new_unit = dict(
            new_spec,
            spec=new_spec,
            status="pending",
            completed_changes=[],
            evidence=[],
            blocker="",
            baseline=self.state._capture_snapshot() if hasattr(self.state, "_capture_snapshot") else None,
            touched=set(),
        )
        self.units.append(new_unit)
        self.planned_scope |= set(norm_paths)
        self.plan_key = None
        if self.candidate_policy:
            self.candidate_policy.observe_scope(self.planned_scope)
        if not self.active:
            self.select()
        self.publish("repair_unit_appended", action="appended", target_unit=uid, paths=norm_paths, depends_on=deps)
        return {"action": "appended", "unit_id": uid, "paths": norm_paths, "depends_on": deps}

    def textual_gradient_summary(
        self,
        *,
        last_error: str = "",
        missing_requirements: list[str] | None = None,
        candidate_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute a deterministic EvoMAC-style 3-part attribution delta (frozen, faulty, missing)."""
        frozen = [
            {
                "unit_id": u["unit_id"],
                "paths": list(u.get("completed_changes") or u.get("paths") or []),
                "summary": str(u.get("summary") or "")[:160],
            }
            for u in self.units
            if u.get("status") == "verified"
        ]
        faulty = [
            {
                "unit_id": u["unit_id"],
                "status": u.get("status"),
                "paths": list(u.get("paths") or []),
                "refuted_by": u.get("refuted_by") or "",
                "blocker": str(u.get("blocker") or "")[:200],
            }
            for u in self.units
            if u.get("status") in {"active", "completed_unverified"} or u.get("blocker") or u.get("refuted_by")
        ]
        planned = set(self.planned_scope) | {p for u in self.units for p in u.get("paths") or []}
        missing_paths: list[str] = []

        def _consider_path(raw: str) -> None:
            p = str(raw or "").strip().replace("\\", "/").lstrip("./")
            if not p or not p.endswith(".rs") or p in planned or p in missing_paths:
                return
            parts = Path(p).parts
            if any(seg in parts for seg in ("tests", "benches", "target", ".cargo")):
                return
            if (self.state.root / p).is_file():
                missing_paths.append(p)

        for diag in getattr(self.state, "_last_check_diagnostics", []) or []:
            if isinstance(diag, dict):
                if diag.get("file"):
                    _consider_path(diag["file"])
                for sp in diag.get("spans") or []:
                    if isinstance(sp, dict) and sp.get("file_name"):
                        _consider_path(sp["file_name"])
        for m in re.finditer(r"([A-Za-z0-9_./-]+\.rs):\d+", last_error or ""):
            _consider_path(m.group(1))
        for cp in candidate_paths or []:
            _consider_path(cp)

        return {
            "frozen_verified_units": frozen,
            "faulty_units": faulty[:4],
            "missing_subtask_candidates": {
                "unplanned_files": missing_paths[:6],
                "unmet_requirements": [str(r)[:180] for r in (missing_requirements or []) if str(r).strip()][:5],
            },
        }

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
                  "If a missing companion/consumer file must be updated separately, use append_repair_unit(...) rather than reverting verified units. "
                  "After all units close, run final regression/existing tests and submit_for_qa.")
