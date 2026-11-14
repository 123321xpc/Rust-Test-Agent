"""Demand-driven comparison of two short mechanisms, within existing role budgets."""

import json
from typing import Literal

from pydantic import BaseModel, Field


class RepairCandidate(BaseModel):
    candidate_id: Literal["A", "B"] = Field(description="A is the initial/default mechanism; B is the alternative. See active_candidate for the current choice")
    chain_id: str = ""
    mechanism: str = Field(min_length=1, max_length=400, description="Brief causal mechanism, not a second full edit plan")
    contract: Literal["satisfies", "uncertain", "conflict"]
    contract_reason: str = Field(min_length=1, max_length=300, description="Map to explicit issue requirements; no scope/cost arguments")
    execution: Literal["supported", "untested", "refuted"] = "untested"
    evidence_ids: list[str] = Field(default_factory=list, description="Observed Cargo evidence IDs; required for supported/refuted")
    execution_reason: str = Field(default="", max_length=300, description="Explain how observed results distinguish this mechanism; this is not proof that unimplemented B passes")
    dependencies: Literal["compatible", "uncertain", "incompatible"] = "uncertain"
    dependency_reason: str = Field(min_length=1, max_length=300, description="Concrete Rust interface/feature/lockfile compatibility evidence")
    affected_paths: list[str] = Field(min_length=1, max_length=24)
    verification_cost: Literal["targeted_check", "targeted_tests", "workspace_tests"] = "targeted_tests"


class CandidatePolicy:
    """Track why another mechanism is warranted; never run another model/patch itself."""

    def __init__(self, blackboard, state, trace=None):
        self.board, self.state, self.trace = blackboard, state, trace
        saved = blackboard.get("candidate_policy") or {}
        self.data = {"active_candidate": "A", "selected_candidate": "A", "pending": None,
                     "evidence": [], "comparisons": [], "scope": [], "scope_expansions": 0,
                     "scope_initialized": False, "scope_attempts": [], "episode": 0, "comparison_offset": 0, **saved}
        blackboard["candidate_policy"] = self.data

    def emit(self, event, **extra):
        if self.trace:
            self.trace.record(event, **extra)

    def current_comparison(self):
        return self.data["comparisons"][-1] if len(self.data["comparisons"]) > self.data["comparison_offset"] else None

    def relocalized(self):
        self.data.update(active_candidate="A", selected_candidate="A", pending=None, scope=[],
                         scope_expansions=0, scope_initialized=False, scope_attempts=[],
                         retain_current_patch=False, switch_paths=[],
                         episode=self.data["episode"] + 1, comparison_offset=len(self.data["comparisons"]))
        self.emit("candidate_relocalized", episode=self.data["episode"])

    def trigger(self, kind, reason, evidence_ids=None):
        request = {"kind": kind, "reason": reason[:800], "evidence_ids": evidence_ids or []}
        if request != self.data["pending"]:
            self.data["pending"] = request
            self.emit("candidate_gate", **request)

    def ambiguity(self, chain_ids, reason):
        chains = self.board.get("localization", {}).get("candidate_chains") or []
        selected = [c for c in chains if c.get("chain_id") in set(chain_ids or [])]
        if len({c.get("chain_id") for c in selected}) < 2 or len({c.get("hypothesis", "").strip() for c in selected}) < 2 or not reason.strip():
            return "Ambiguity requires two distinct known root-cause hypotheses and a concrete unresolved reason."
        self.trigger("ambiguous_chains", reason)
        return None

    def observe_scope(self, paths):
        paths = set(paths)
        if not self.data["scope_initialized"]:
            self.data.update(scope=sorted(paths), scope_initialized=True)
            return
        previous = set(self.data["scope"])
        if paths > previous:
            self.data["scope_expansions"] += 1
            self.data["scope"] = sorted(paths)
            if self.data["scope_expansions"] >= 2:
                self.trigger("scope_growth", f"Production scope grew repeatedly to {len(paths)} files: {sorted(paths)}")
        elif paths != previous:
            self.data.update(scope=sorted(paths), scope_expansions=0)

    def scope_attempt(self, path):
        if path not in self.data["scope_attempts"] and path not in self.data["scope"]:
            self.data["scope_attempts"].append(path)
        if len(self.data["scope_attempts"]) >= 2:
            self.trigger("scope_growth", "Repeated requests to extend unit scope: " + ", ".join(self.data["scope_attempts"]))

    def observe(self, name, arguments, output):
        """Also handles Cargo results embedded in write_regression_test(run_now=True)."""
        if name == "write_regression_test":
            if not arguments.get("run_now") or "[Auto-run result" not in str(output):
                return None
            raw = str(output).split("[Auto-run result", 1)[1]
            raw = raw[raw.find("{"):]
            name = "regression_auto_run"
            arguments = {k: v for k, v in arguments.items() if k != "content"}
        elif name in ("cargo_check", "cargo_test"):
            raw = str(output)
        else:
            return None
        try:
            result, _ = json.JSONDecoder().raw_decode(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(result, dict) or result.get("exit_code") is None:
            return None
        # A cache reuse of the same revision/config is one observation.
        runtime = (name != "cargo_check" and not arguments.get("no_run") and result.get("tests_executed", 0) > 0)
        signature = (name, arguments, self.state._revision)
        existing = next((e for e in reversed(self.data["evidence"])
                         if (e["tool"], e["arguments"], e["revision"]) == signature
                         and e["episode"] == self.data["episode"]
                         and e["candidate_id"] == self.data["active_candidate"]), None)
        if existing and existing["exit_code"] == result["exit_code"] and existing["runtime"] == runtime:
            return existing["evidence_id"]
        entry = {"evidence_id": f"E{len(self.data['evidence']) + 1}", "tool": name,
                 "arguments": dict(arguments), "revision": self.state._revision,
                 "episode": self.data["episode"],
                 "candidate_id": self.data["active_candidate"], "exit_code": result["exit_code"],
                 "runtime": runtime, "output": raw[-1800:], "superseded_by": None}
        self.data["evidence"].append(entry)
        # A later successful rerun resolves earlier failures of that exact check;
        # historical failures are no longer grounds to reject the current mechanism.
        meaningful_success = runtime or name == "cargo_check" or arguments.get("no_run")
        if result["exit_code"] == 0 and meaningful_success:
            for old in self.data["evidence"][:-1]:
                if (old["tool"] == name and old["arguments"] == arguments and old["exit_code"] != 0
                        and old["episode"] == self.data["episode"]
                        and old["candidate_id"] == self.data["active_candidate"]):
                    old["superseded_by"] = entry["evidence_id"]
            pending = self.data["pending"]
            if pending and pending["kind"] == "verification_counterevidence":
                active = {e["evidence_id"] for e in self.data["evidence"] if not e["superseded_by"]}
                if not active.intersection(pending["evidence_ids"]):
                    self.data["pending"] = None
        self.emit("candidate_evidence", **entry)
        return entry["evidence_id"]

    def counterevidence(self, reason, evidence_ids):
        known = {e["evidence_id"]: e for e in self.data["evidence"]}
        if not evidence_ids or any(i not in known for i in evidence_ids):
            return "Counterevidence must cite actual Coordinator Cargo evidence IDs."
        if not any(known[i]["exit_code"] != 0 and not known[i]["superseded_by"]
                   and known[i]["candidate_id"] == self.data["active_candidate"]
                   and known[i]["episode"] == self.data["episode"]
                   and known[i]["revision"] == self.state._revision for i in evidence_ids):
            return "No unresolved failing execution evidence in these IDs; ordinary replanning remains available."
        if not reason.strip():
            return "Explain which prediction of the current mechanism the observed result contradicts."
        self.trigger("verification_counterevidence", reason, evidence_ids)
        return None

    def repeated_test_failure(self):
        failure = self.state.stuck_status()
        observations = [e for e in self.data["evidence"] if e["runtime"] and e["exit_code"] != 0
                        and not e["superseded_by"] and e["candidate_id"] == self.data["active_candidate"]
                        and e["episode"] == self.data["episode"]]
        if (failure["test_repeats"] >= 2 and len({e["revision"] for e in observations}) >= 2
                and observations[-1]["revision"] == self.state._revision):
            self.trigger("verification_counterevidence", "Behavioral failure persisted across edited revisions; compare whether it refutes the current mechanism.",
                         [e["evidence_id"] for e in observations[-2:]])

    def compare(self, candidates, *, next_check=""):
        if not self.data["pending"]:
            raise ValueError("Second mechanism is disabled: use one plan unless ambiguity, counterevidence, or sustained scope growth is observed")
        cards = [RepairCandidate.model_validate(c).model_dump() for c in candidates]
        if len(cards) != 2 or {c["candidate_id"] for c in cards} != {"A", "B"}:
            raise ValueError("Compare exactly two brief mechanisms: current/default A and alternative B")
        cards.sort(key=lambda c: c["candidate_id"])
        if cards[0]["mechanism"].strip().casefold() == cards[1]["mechanism"].strip().casefold():
            raise ValueError("B must describe a different causal mechanism, not a renamed copy of A")
        evidence = {e["evidence_id"]: e for e in self.data["evidence"]}
        for c in cards:
            if any(i not in evidence or evidence[i]["superseded_by"] or evidence[i]["episode"] != self.data["episode"] for i in c["evidence_ids"]):
                raise ValueError("Execution claims must cite existing, unresolved Cargo evidence IDs")
            if c["execution"] != "untested" and (not c["evidence_ids"] or not c["execution_reason"].strip()):
                raise ValueError("supported/refuted requires actual execution evidence and a causal explanation")
            if c["execution"] != "untested" and not any(
                evidence[i]["tool"] == "cargo_check" or evidence[i]["arguments"].get("no_run")
                or evidence[i]["runtime"] or evidence[i]["exit_code"] != 0 for i in c["evidence_ids"]
            ):
                raise ValueError("A zero-test run is not distinguishing execution evidence")
            if c["execution"] == "refuted" and not any(evidence[i]["exit_code"] != 0 for i in c["evidence_ids"]):
                raise ValueError("A successful check alone cannot be reported as a failed prediction")
        eligible = [c for c in cards if c["contract"] != "conflict" and c["dependencies"] != "incompatible" and c["execution"] != "refuted"]

        def primary(c):
            return (int(c["contract"] == "satisfies"), int(c["execution"] == "supported"), int(c["dependencies"] == "compatible"))

        cost = {"targeted_check": 0, "targeted_tests": 1, "workspace_tests": 2}
        if eligible:
            current = self.data["active_candidate"]
            chosen = max(eligible, key=lambda c: (*primary(c), -len(set(c["affected_paths"])), -cost[c["verification_cost"]], c["candidate_id"] == current))
            other = next(c for c in cards if c is not chosen)
            tie = len(eligible) == 2 and primary(chosen) == primary(other)
            reason = "equivalent primary evidence; compare scope then verification cost" if tie else "contract, execution evidence, and dependency compatibility take priority"
            selected = chosen["candidate_id"]
            # Do not spend another implementation on an entirely uncertain rival.
            if selected != current and chosen["contract"] == "uncertain" and chosen["execution"] == "untested" and other in eligible:
                selected, reason = current, "Alternative lacks decisive contract/execution evidence; keep current mechanism and obtain a distinguishing check"
        else:
            selected, reason = None, "Both mechanisms conflict with requirements/evidence/compatibility; relocalize"
        result = {"trigger": self.data["pending"], "candidates": cards, "selected_candidate": selected,
                  "reason": reason, "next_check": next_check[:600], "implementation": "Brief comparison executes neither mechanism; only the selected mechanism may receive a detailed plan"}
        self.data["comparisons"].append(result)
        self.data["selected_candidate"] = selected
        self.data["pending"] = None
        self.data.update(scope_expansions=0, scope_attempts=[])
        self.emit("candidate_comparison", **result)
        return result

    def plan_gate(self, payload, candidate_id):
        paths = {e["file"] for e in payload["ordered_edits"] if e.get("file") and e.get("action") != "preserve"}
        self.observe_scope(paths)
        if self.data["pending"]:
            return "A candidate comparison is required before another detailed plan. Call compare_repair_candidates with two brief mechanisms."
        if self.data["selected_candidate"] is None:
            return "Neither mechanism is admissible. Request relocalization before proposing another plan."
        selected = self.data["selected_candidate"]
        comparison = self.current_comparison()
        if comparison:
            if candidate_id != selected:
                return f"Only selected candidate {selected} may be implemented; pass candidate_id='{selected}'."
            card = next(c for c in comparison["candidates"] if c["candidate_id"] == selected)
            if card["chain_id"] and payload["selected_chain_id"] != card["chain_id"]:
                return "Detailed plan must use the selected mechanism's chain_id."
        elif candidate_id not in ("", "A"):
            return "Default is one mechanism A; B requires a triggered comparison first."
        payload["candidate_id"] = selected
        active = self.data["active_candidate"]
        self.data["retain_current_patch"] = bool(comparison and selected == active)
        self.data["switch_paths"] = card["affected_paths"] if comparison and selected != active else []
        self.data["active_candidate"] = selected
        self.emit("candidate_plan", candidate_id=selected, selected_chain_id=payload["selected_chain_id"], affected_paths=sorted(paths))
        return None

    def fallback_plan(self, current_plan):
        """Planner cap cannot silently authorize an unplanned B or a refuted A."""
        active = self.data["active_candidate"]
        comparison = self.current_comparison()
        if comparison:
            card = next(c for c in comparison["candidates"] if c["candidate_id"] == active)
            if card["contract"] == "conflict" or card["execution"] == "refuted" or card["dependencies"] == "incompatible":
                raise ValueError("Planner ended without an admissible selected plan; current mechanism is refuted/incompatible")
        self.data["selected_candidate"] = active
        self.data["pending"] = None
        self.data.update(retain_current_patch=True, switch_paths=[])
        self.emit("candidate_comparison_deferred", reason="Architect budget/no handoff; keep current mechanism, do not implement another candidate", candidate_id=active)
        return current_plan

    def prompt(self):
        pending = self.data["pending"]
        if not pending and not self.current_comparison() and not self.data["evidence"]:
            return "\nCandidate policy: one mechanism by default. Multiple chains alone do not require a second candidate."
        observations = [{k: v for k, v in e.items() if k != "output" or e["exit_code"] != 0}
                        for e in self.data["evidence"][-6:]]
        return ("\nCandidate comparison gate:\n" + json.dumps({"active_candidate": self.data["active_candidate"],
                "pending": pending, "last_comparison": self.current_comparison(),
                "current_revision": self.state._revision, "execution_observations": observations}, ensure_ascii=False)
                + "\nWhen pending, compare exactly two brief mechanisms before submitting a detailed plan. "
                  "Use contract, actual execution evidence, and dependency compatibility first; only equal primary evidence permits scope/cost tiebreaks. "
                  "Historical observations concern the recorded revision/current patch, not an unimplemented B. "
                  "Keep the current mechanism when evidence is insufficient and name a cheap distinguishing check.")
