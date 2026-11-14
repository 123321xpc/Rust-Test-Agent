#!/usr/bin/env python3
"""Correlate multi-agent handoff features in traces with Rust-bench evaluation outcomes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _resolve_report_path(
    instance_entry: dict[str, Any],
    run_dir: Path,
    agent: str,
    harness_logs: Path | None,
) -> Path | None:
    raw = instance_entry.get("report")
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
        # Remap remote paths (e.g. /home/xupeicai/xpc/Rust-bench/logs/run_evaluation/<eval_id>/<agent>/<iid>/report.json)
        parts = p.parts
        if "run_evaluation" in parts:
            idx = parts.index("run_evaluation")
            rel = Path(*parts[idx + 1 :])
            if harness_logs is not None and (harness_logs / rel).is_file():
                return harness_logs / rel
    if harness_logs is not None:
        eval_id = f"{run_dir.name}_{agent.replace('-', '_')}"
        cand = harness_logs / eval_id / agent / instance_entry["instance_id"] / "report.json"
        if cand.is_file():
            return cand
    return None


def _parse_report(report_path: Path | None, instance_id: str) -> dict[str, Any]:
    if report_path is None:
        return {}
    data = _safe_load_json(report_path)
    if not isinstance(data, dict):
        return {}
    row = data.get(instance_id)
    if not isinstance(row, dict):
        return {}
    tests = row.get("tests_status") or {}
    f2p = tests.get("FAIL_TO_PASS") or {}
    p2p = tests.get("PASS_TO_PASS") or {}
    f2p_succ = f2p.get("success") or []
    f2p_fail = f2p.get("failure") or []
    p2p_succ = p2p.get("success") or []
    p2p_fail = p2p.get("failure") or []
    return {
        "patch_successfully_applied": row.get("patch_successfully_applied"),
        "detailed_status": row.get("detailed_status"),
        "fail_to_pass_passed": len(f2p_succ),
        "fail_to_pass_total": len(f2p_succ) + len(f2p_fail),
        "pass_to_pass_passed": len(p2p_succ),
        "pass_to_pass_total": len(p2p_succ) + len(p2p_fail),
        "pass_to_pass_failed_items": [str(x) for x in p2p_fail[:10]],
    }


def _find_trace_file(task_dir: Path) -> Path | None:
    traces_dir = task_dir / "traces"
    if not traces_dir.is_dir():
        return None
    files = sorted(traces_dir.glob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
    return files[0] if files else None


def parse_trace_file(trace_path: Path | None) -> dict[str, Any]:
    """Stream one JSONL trace and extract P0 handoff features (with legacy fallback)."""
    info: dict[str, Any] = {
        "trace_file": str(trace_path) if trace_path else None,
        "has_handoff_snapshots": False,
        "snapshots": [],
        "stages": [],
        "route": None,
        "route_reasons": [],
        "escalations": [],
        "qa_gates": [],
        "run_error": None,
        "metrics": {},
        "issue_contract": {},
        "legacy_localization": None,
        "legacy_repair_plan": None,
        "legacy_env_blocker": False,
        "legacy_api_limit": False,
        "tool_counts": Counter(),
    }
    if trace_path is None or not trace_path.is_file():
        return info

    pending_tools: dict[str, tuple[str, Any]] = {}
    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            etype = ev.get("event")
            if etype == "multi_agent_start":
                info["issue_contract"] = ev.get("issue_contract") or {}
            elif etype == "handoff_feature_snapshot":
                info["has_handoff_snapshots"] = True
                info["snapshots"].append(ev)
            elif etype == "adaptive_route":
                info["route"] = ev.get("route")
                info["route_reasons"] = ev.get("reasons") or []
            elif etype == "stage_end":
                info["stages"].append({
                    "role": ev.get("agent_role"),
                    "attempt": ev.get("attempt"),
                    "outcome": ev.get("outcome"),
                    "calls_used": ev.get("calls_used"),
                    "elapsed_seconds": ev.get("elapsed_seconds"),
                })
            elif etype == "escalation":
                info["escalations"].append({
                    "from_role": ev.get("from_role"),
                    "to_role": ev.get("to_role"),
                    "reason": ev.get("reason"),
                    "attempt": ev.get("attempt"),
                })
            elif etype == "qa_gate":
                info["qa_gates"].append({
                    "deterministic_ok": ev.get("deterministic_ok"),
                    "llm_decision": ev.get("llm_decision"),
                    "reason": ev.get("reason"),
                    "qa_round": ev.get("qa_round"),
                    "missing_apis": ev.get("missing_apis"),
                })
            elif etype == "run_error":
                info["run_error"] = {
                    "stage": ev.get("stage"),
                    "error_type": ev.get("error_type"),
                    "error": ev.get("error"),
                }
            elif etype == "agent_metrics":
                info["metrics"] = {
                    "status": ev.get("status"),
                    "elapsed_seconds": ev.get("elapsed_seconds"),
                    "api_calls": ev.get("api_calls"),
                    "input_tokens": ev.get("input_tokens"),
                    "output_tokens": ev.get("output_tokens"),
                    "total_tokens": ev.get("total_tokens"),
                }
            elif etype == "api_limit":
                info["legacy_api_limit"] = True
            elif etype == "tool_start":
                tname = str(ev.get("tool") or "")
                if tname:
                    info["tool_counts"][tname] += 1
                    rid = str(ev.get("run_id") or "")
                    if rid:
                        pending_tools[rid] = (tname, ev.get("input"))
            elif etype == "tool_end":
                rid = str(ev.get("run_id") or "")
                out = ev.get("output")
                tname, tinput = pending_tools.pop(rid, ("", None))
                if isinstance(out, str) and (
                    "modifying cargo.lock is prohibited" in out.lower()
                    or "failed to select a version" in out.lower()
                ):
                    info["legacy_env_blocker"] = True
                if tname == "submit_localization_bundle" and isinstance(tinput, dict):
                    info["legacy_localization"] = tinput
                elif tname == "submit_repair_plan" and isinstance(tinput, dict):
                    info["legacy_repair_plan"] = tinput

    info["tool_counts"] = dict(info["tool_counts"])
    return info


def _latest_snapshot(snapshots: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    matched = [s for s in snapshots if s.get("stage") == stage]
    return matched[-1] if matched else None


def _extract_feature_record(trace_info: dict[str, Any]) -> dict[str, Any]:
    """Normalize P0 features from handoff_feature_snapshot or legacy trace reconstruction."""
    snaps = trace_info["snapshots"]
    issue_contract = trace_info.get("issue_contract") or {}
    signals = issue_contract.get("signals") or {}

    issue_snap = _latest_snapshot(snaps, "issue_contract")
    nav_snap = _latest_snapshot(snaps, "navigator_handoff")
    sug_snap = _latest_snapshot(snaps, "suggester_handoff")
    fix_snap = _latest_snapshot(snaps, "fixer_handoff")
    ver_snap = _latest_snapshot(snaps, "verifier_handoff")

    if issue_snap:
        if_feats = issue_snap.get("features") or {}
        has_new_apis = bool((if_feats.get("new_api_candidates") or {}).get("present"))
        has_expected_result = bool((if_feats.get("expected_result_excerpt") or {}).get("present"))
        has_ref_files = bool((if_feats.get("referenced_files") or {}).get("present"))
        req_sym_count = int((if_feats.get("requested_symbols") or {}).get("count", 0))
    else:
        has_new_apis = bool(issue_contract.get("new_api_candidates"))
        has_expected_result = bool((issue_contract.get("expected_result_excerpt") or "").strip())
        has_ref_files = bool(issue_contract.get("referenced_files"))
        req_sym_count = len(issue_contract.get("requested_symbols") or [])

    if nav_snap:
        nf = nav_snap.get("features") or {}
        repro_meta = nf.get("reproduction_status") or {}
        repro_status = repro_meta.get("status") or "not_reproduced"
        clean_red_verified = int(repro_meta.get("clean_red_count", 0)) > 0
        chains_meta = nf.get("candidate_chains") or {}
        candidate_chains_count = int(chains_meta.get("chain_count", 0))
        root_candidates = [x.get("path", "") for x in (chains_meta.get("root_candidates") or []) if x.get("path")]
        repro_test_chars = int((nf.get("repro_test_contents") or {}).get("char_len", 0))
        shared_reads_chars = int((nf.get("shared_read_excerpts") or {}).get("char_len", 0))
        shared_ckg_chars = int((nf.get("shared_ckg_notes") or {}).get("char_len", 0))
        nav_outcome = nav_snap.get("outcome")
        nav_auto_synthesized = bool(nav_snap.get("auto_synthesized"))
        route = nav_snap.get("route") or trace_info.get("route")
    else:
        loc = trace_info.get("legacy_localization") or {}
        repro_status = loc.get("reproduction_status") or "not_reproduced"
        clean_red_verified = repro_status in ("runtime_failure", "compile_reproduction")
        chains = loc.get("candidate_chains") or []
        candidate_chains_count = len(chains)
        root_candidates = [
            node.get("path", "")
            for ch in chains
            for node in (ch.get("nodes") or [])
            if str(node.get("role") or "").lower() in ("root_candidate", "modify") and node.get("path")
        ]
        repro_test_chars = 0
        shared_reads_chars = 0
        shared_ckg_chars = 0
        nav_stage = next((s for s in trace_info["stages"] if s.get("role") == "navigator"), {})
        nav_outcome = nav_stage.get("outcome")
        nav_auto_synthesized = False
        route = trace_info.get("route")

    if sug_snap:
        sf = sug_snap.get("features") or {}
        repair_plan_present = sug_snap.get("action") in ("plan", "synthesized_fallback")
        repair_plan_synthesized = bool(sug_snap.get("synthesized"))
        ordered_edits_count = int((sf.get("ordered_edits") or {}).get("count", 0))
        api_checklist_count = int((sf.get("api_checklist") or {}).get("count", 0))
        rust_invariants_count = int((sf.get("rust_invariants") or {}).get("count", 0))
        forbidden_edits_count = int((sf.get("forbidden_edits") or {}).get("count", 0))
        sug_prompt_breakdown = sug_snap.get("sug_prompt_breakdown") or {}
    else:
        rp = trace_info.get("legacy_repair_plan")
        repair_plan_present = isinstance(rp, dict)
        repair_plan_synthesized = False
        ordered_edits_count = len((rp or {}).get("ordered_edits") or [])
        api_checklist_count = len((rp or {}).get("api_checklist") or [])
        rust_invariants_count = len((rp or {}).get("rust_invariants") or [])
        forbidden_edits_count = len((rp or {}).get("forbidden_edits") or [])
        sug_prompt_breakdown = {}

    if fix_snap:
        ff = fix_snap.get("features") or {}
        det = ff.get("deterministic_qa") or {}
        stuck = ff.get("stuck_status") or {}
        reg_ver = ff.get("regression_verification") or {}
        changed_files = list((ff.get("changed_files") or {}).get("paths") or [])
        fixer_verified = bool(reg_ver.get("verified"))
        compile_repeats = int(stuck.get("compile_repeats", 0))
        test_repeats = int(stuck.get("test_repeats", 0))
        env_blocker = bool((ff.get("env_blocker") or {}).get("value"))
        det_qa_ok = det.get("ok") if det.get("present") else None
        det_qa_reason = det.get("reason") or ""
        missing_contract_apis = list(det.get("missing_contract_apis") or [])
        doctest_shifts = list(det.get("doctest_shifts") or [])
        added_inline_tests_count = int(det.get("added_inline_tests_count", 0) or 0)
        modified_inline_tests_count = int(det.get("modified_inline_tests_count", 0) or 0)
        fix_prompt_breakdown = fix_snap.get("fix_prompt_breakdown") or {}
    else:
        changed_files = []
        fixer_verified = any(g.get("deterministic_ok") for g in trace_info["qa_gates"])
        compile_repeats = 0
        test_repeats = 0
        env_blocker = bool(trace_info["legacy_env_blocker"]) or any(
            "environment blocker" in str(e.get("reason") or "").lower()
            for e in trace_info["escalations"]
        )
        det_qa_ok = any(g.get("deterministic_ok") for g in trace_info["qa_gates"]) if trace_info["qa_gates"] else None
        det_qa_reason = next((str(g.get("reason") or "") for g in reversed(trace_info["qa_gates"]) if g.get("reason")), "")
        missing_contract_apis = []
        for g in trace_info["qa_gates"]:
            if g.get("missing_apis"):
                missing_contract_apis = list(g["missing_apis"])
        doctest_shifts = []
        added_inline_tests_count = 0
        modified_inline_tests_count = 0
        fix_prompt_breakdown = {}

    if ver_snap:
        vf = ver_snap.get("features") or {}
        final_qa_decision = ver_snap.get("decision")
        qa_missing_reqs = list((vf.get("missing_requirements") or {}).get("items") or [])
        qa_objections = list((vf.get("objections") or {}).get("items") or [])
        ver_prompt_chars = int(ver_snap.get("ver_prompt_chars", 0) or 0)
    else:
        final_qa_decision = trace_info["qa_gates"][-1].get("llm_decision") if trace_info["qa_gates"] else None
        qa_missing_reqs = []
        qa_objections = []
        ver_prompt_chars = 0

    qa_rejected_any = any(g.get("llm_decision") == "REJECT" for g in trace_info["qa_gates"]) or any(
        s.get("decision") == "REJECT" for s in snaps if s.get("stage") == "verifier_handoff"
    )

    return {
        "route": route or "unknown",
        "has_new_api_candidates": has_new_apis,
        "has_expected_result": has_expected_result,
        "has_referenced_files": has_ref_files,
        "requested_symbols_count": req_sym_count,
        "mentions_trait_or_lifetime": bool(signals.get("mentions_trait_or_lifetime")),
        "mentions_macro": bool(signals.get("mentions_macro")),
        "mentions_new_api": bool(signals.get("mentions_new_api")),
        "nav_outcome": nav_outcome,
        "nav_auto_synthesized": nav_auto_synthesized,
        "candidate_chains_count": candidate_chains_count,
        "root_candidate_files": root_candidates,
        "reproduction_status": repro_status,
        "clean_red_verified": clean_red_verified,
        "repro_test_chars": repro_test_chars,
        "shared_read_excerpts_chars": shared_reads_chars,
        "shared_ckg_notes_chars": shared_ckg_chars,
        "repair_plan_present": repair_plan_present,
        "repair_plan_synthesized": repair_plan_synthesized,
        "ordered_edits_count": ordered_edits_count,
        "api_checklist_count": api_checklist_count,
        "rust_invariants_count": rust_invariants_count,
        "forbidden_edits_count": forbidden_edits_count,
        "escalations_count": len(trace_info["escalations"]),
        "escalated": len(trace_info["escalations"]) > 0,
        "changed_files": changed_files,
        "fixer_verified": fixer_verified,
        "compile_repeats": compile_repeats,
        "test_repeats": test_repeats,
        "env_blocker": env_blocker,
        "det_qa_ok": det_qa_ok,
        "det_qa_reason": det_qa_reason,
        "missing_contract_apis": missing_contract_apis,
        "missing_contract_apis_detected": len(missing_contract_apis) > 0,
        "doctest_shifts": doctest_shifts,
        "doctest_shifts_detected": len(doctest_shifts) > 0,
        "added_inline_tests_count": added_inline_tests_count,
        "modified_inline_tests_count": modified_inline_tests_count,
        "final_qa_decision": final_qa_decision,
        "qa_rejected_any": qa_rejected_any,
        "qa_missing_requirements": qa_missing_reqs,
        "qa_objections": qa_objections,
        "sug_prompt_breakdown": sug_prompt_breakdown,
        "fix_prompt_breakdown": fix_prompt_breakdown,
        "ver_prompt_chars": ver_prompt_chars,
    }


def _classify_failure_attribution(
    verdict: str,
    features: dict[str, Any],
    trace_info: dict[str, Any],
    report_info: dict[str, Any],
) -> str:
    if verdict == "resolved":
        return "resolved"
    if verdict in ("evaluation_error", "error"):
        return verdict

    run_err = str((trace_info.get("run_error") or {}).get("error") or "").lower()
    if verdict in ("agent_error", "no_patch"):
        if features.get("env_blocker"):
            return "env_blocker"
        if "no production source changes" in run_err:
            return "no_source_edits"
        if "no temporary regression test" in run_err or features.get("reproduction_status") == "not_reproduced":
            return "no_reproduction_test"
        if trace_info.get("legacy_api_limit") or "api call limit" in run_err:
            return "api_limit_exhausted"
        if "deterministic qa" in run_err or features.get("det_qa_ok") is False:
            return "deterministic_qa_failed"
        return "other_agent_error"

    # verdict == "unresolved"
    if report_info.get("patch_successfully_applied") is False:
        return "patch_apply_failed"
    if features.get("missing_contract_apis_detected"):
        return "missing_contract_apis"
    p2p_failed = report_info.get("pass_to_pass_failed_items") or []
    if p2p_failed:
        if features.get("doctest_shifts_detected") or any("(line " in item for item in p2p_failed):
            return "doctest_or_pass_to_pass_regression"
        return "pass_to_pass_regression"
    return "fail_to_pass_unfixed"


def analyze_runs(
    run_dirs: list[Path],
    *,
    agent: str = "test",
    harness_logs: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        summary = _safe_load_json(summary_path)
        if not isinstance(summary, dict) or agent not in summary:
            raise RuntimeError(f"Missing agent '{agent}' in {summary_path}")
        agent_summary = summary[agent]
        inferred_logs = harness_logs
        if inferred_logs is None:
            cand = run_dir.parents[1] / "Rust-bench" / "logs" / "run_evaluation"
            if cand.is_dir():
                inferred_logs = cand

        for inst in agent_summary.get("instances") or []:
            iid = inst["instance_id"]
            task_dir = run_dir / agent / iid
            trace_path = _find_trace_file(task_dir)
            trace_info = parse_trace_file(trace_path)
            report_path = _resolve_report_path(inst, run_dir, agent, inferred_logs)
            report_info = _parse_report(report_path, iid)
            features = _extract_feature_record(trace_info)
            verdict = inst.get("verdict") or "unknown"
            attribution = _classify_failure_attribution(verdict, features, trace_info, report_info)
            metrics = trace_info.get("metrics") or {}
            rows.append({
                "run_id": run_dir.name,
                "instance_id": iid,
                "verdict": verdict,
                "failure_attribution": attribution,
                "seconds": inst.get("seconds"),
                "patch_bytes": inst.get("patch_bytes", 0),
                "api_calls": metrics.get("api_calls"),
                "input_tokens": metrics.get("input_tokens"),
                "output_tokens": metrics.get("output_tokens"),
                "total_tokens": metrics.get("total_tokens"),
                "has_handoff_snapshots": trace_info["has_handoff_snapshots"],
                "stages": trace_info["stages"],
                "run_error": trace_info["run_error"],
                "report": report_info,
                "features": features,
            })

    verdict_counts = Counter(r["verdict"] for r in rows)
    attr_counts = Counter(r["failure_attribution"] for r in rows)
    snapshot_coverage = sum(1 for r in rows if r["has_handoff_snapshots"])

    correlation_keys = [
        "route",
        "has_new_api_candidates",
        "has_expected_result",
        "has_referenced_files",
        "mentions_trait_or_lifetime",
        "reproduction_status",
        "clean_red_verified",
        "repair_plan_present",
        "repair_plan_synthesized",
        "escalated",
        "env_blocker",
        "missing_contract_apis_detected",
        "doctest_shifts_detected",
        "qa_rejected_any",
    ]
    correlations: dict[str, dict[str, Any]] = {}
    for key in correlation_keys:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            val = r["features"].get(key)
            grouped[str(val)].append(r)
        bucket_stats: dict[str, Any] = {}
        for bval, items in sorted(grouped.items()):
            n = len(items)
            resolved = sum(1 for x in items if x["verdict"] == "resolved")
            unresolved = sum(1 for x in items if x["verdict"] == "unresolved")
            err_or_empty = n - resolved - unresolved
            calls = [x["api_calls"] for x in items if isinstance(x.get("api_calls"), (int, float))]
            toks = [x["total_tokens"] for x in items if isinstance(x.get("total_tokens"), (int, float))]
            bucket_stats[bval] = {
                "total": n,
                "resolved": resolved,
                "unresolved": unresolved,
                "other": err_or_empty,
                "resolve_rate": round(resolved / n, 4) if n else 0.0,
                "avg_api_calls": round(statistics.mean(calls), 2) if calls else None,
                "avg_total_tokens": round(statistics.mean(toks), 1) if toks else None,
            }
        correlations[key] = bucket_stats

    prompt_stats: dict[str, dict[str, float]] = {}
    for pkey in ("sug_prompt_breakdown", "fix_prompt_breakdown"):
        acc: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            bd = r["features"].get(pkey) or {}
            if isinstance(bd, dict):
                for sk, sv in bd.items():
                    if isinstance(sv, (int, float)):
                        acc[sk].append(int(sv))
        if acc:
            prompt_stats[pkey] = {sk: round(statistics.mean(vals), 1) for sk, vals in sorted(acc.items())}

    total = len(rows)
    resolved_n = verdict_counts.get("resolved", 0)
    return {
        "agent": agent,
        "run_dirs": [str(p) for p in run_dirs],
        "total_instances": total,
        "snapshot_coverage": snapshot_coverage,
        "verdict_counts": dict(verdict_counts),
        "resolve_rate": round(resolved_n / total, 4) if total else 0.0,
        "failure_attribution_counts": dict(attr_counts),
        "feature_correlations": correlations,
        "prompt_char_averages": prompt_stats,
        "instances": rows,
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Handoff & Outcome Analysis ({report['agent']})")
    lines.append("")
    lines.append(
        f"- Total instances: **{report['total_instances']}** | "
        f"Resolved: **{report['verdict_counts'].get('resolved', 0)}** ({report['resolve_rate'] * 100:.1f}%) | "
        f"P0 Snapshot Coverage: **{report['snapshot_coverage']}/{report['total_instances']}**"
    )
    lines.append(f"- Verdict counts: `{json.dumps(report['verdict_counts'], ensure_ascii=False)}`")
    lines.append(f"- Failure attribution: `{json.dumps(report['failure_attribution_counts'], ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Per-Instance Summary")
    lines.append("| Instance | Verdict | Attribution | Route | Repro | Esc | Calls | Tokens |")
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for r in report["instances"]:
        f = r["features"]
        lines.append(
            f"| {r['instance_id']} | {r['verdict']} | {r['failure_attribution']} | "
            f"{f['route']} | {f['reproduction_status']} | {f['escalations_count']} | "
            f"{r['api_calls'] if r['api_calls'] is not None else '-'} | "
            f"{r['total_tokens'] if r['total_tokens'] is not None else '-'} |"
        )
    lines.append("")

    lines.append("## Feature Correlations")
    lines.append("| Feature | Value | N | Resolved | Unresolved | Other | Resolve Rate | Avg Calls | Avg Tokens |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for fkey, buckets in report["feature_correlations"].items():
        for bval, st in buckets.items():
            lines.append(
                f"| {fkey} | {bval} | {st['total']} | {st['resolved']} | {st['unresolved']} | {st['other']} | "
                f"{st['resolve_rate'] * 100:.1f}% | {st['avg_api_calls'] if st['avg_api_calls'] is not None else '-'} | "
                f"{st['avg_total_tokens'] if st['avg_total_tokens'] is not None else '-'} |"
            )

    if report.get("prompt_char_averages"):
        lines.append("")
        lines.append("## Prompt Char Breakdown Averages")
        for pkey, avgs in report["prompt_char_averages"].items():
            lines.append(f"- **{pkey}**: `{json.dumps(avgs, ensure_ascii=False)}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory (e.g. validation/runs/<run_id>)")
    parser.add_argument("--agent", default="test", help="Agent name in summary.json (default: test)")
    parser.add_argument("--harness-logs", help="Optional path to Rust-bench/logs/run_evaluation")
    parser.add_argument("--out-json", help="Optional output path for structured JSON report")
    parser.add_argument("--out-md", help="Optional output path for Markdown summary")
    args = parser.parse_args(argv)

    run_dirs = [Path(p).resolve() for p in args.run_dir]
    hlogs = Path(args.harness_logs).resolve() if args.harness_logs else None
    report = analyze_runs(run_dirs, agent=args.agent, harness_logs=hlogs)
    md = format_markdown_report(report)
    if args.out_json:
        out_j = Path(args.out_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_md:
        out_m = Path(args.out_md)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
