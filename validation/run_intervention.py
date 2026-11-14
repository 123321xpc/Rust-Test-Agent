#!/usr/bin/env python3
"""Run offline causal interventions on pre_fix checkpoints and evaluate flip rates (CAM P1)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path.home() / "xpc"
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / "envs/rustbench/bin/python"
HARNESS = ROOT / "Rust-bench"


def _safe_load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_intervention_task(
    *,
    repo: Path,
    instance_id: str,
    issue_file: Path,
    checkpoint_dir: Path,
    intervention: dict[str, Any] | None,
    output_dir: Path,
    model: str,
    api_key: str,
    base_url: str | None = None,
    max_api_calls: int = 70,
    rust_toolchain: str | None = None,
    dataset: Path | None = None,
) -> dict[str, Any]:
    """Execute one replay pass from a pre_fix checkpoint with optional causal intervention."""
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    trace_path = traces_dir / f"{uuid.uuid4().hex}.jsonl"

    sys.path.insert(0, str(HERE.parent / "baselines" / "test"))
    import agent as agent_mod
    from agent_trace import TraceRecorder

    trace = TraceRecorder(trace_path, secrets=[api_key])
    issue = issue_file.read_text(encoding="utf-8")
    trace.record(
        "run_start",
        instance_id=instance_id,
        model=model,
        repo=str(repo),
        issue=issue,
        mode="causal_intervention_replay",
        checkpoint_dir=str(checkpoint_dir),
        intervention=intervention,
    )

    started = dt.datetime.now(dt.timezone.utc)
    patch = ""
    error = None
    try:
        patch = agent_mod.run_agent(
            repo,
            issue,
            model,
            api_key,
            base_url=base_url,
            trace=trace,
            max_api_calls=max_api_calls,
            rust_version=rust_toolchain,
            require_reproduction=True,
            instance_id=instance_id,
            multi_agent=True,
            replay_pre_fix_dir=checkpoint_dir,
            intervention=intervention,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        trace.record("run_error", stage="agent", error=error)
    finally:
        metrics = trace.metrics()
        trace.record(
            "agent_metrics",
            status="completed" if patch else "failed",
            elapsed_seconds=(dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
            **metrics,
        )
        trace.record("run_end", status="completed" if patch else "failed")

    pred_record = {
        "instance_id": instance_id,
        "model_name_or_path": f"test_intervention__{model}",
        "model_patch": patch.replace("\r\n", "\n") if patch else "",
    }
    pred_path = output_dir / "prediction.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(pred_record, ensure_ascii=False) + "\n")

    summary = {
        "instance_id": instance_id,
        "checkpoint_dir": str(checkpoint_dir),
        "intervention": intervention,
        "has_patch": bool(patch),
        "patch_bytes": len(patch.encode("utf-8")),
        "error": error,
        "trace_file": str(trace_path),
        "api_calls": metrics.get("api_calls", 0),
        "total_tokens": metrics.get("total_tokens", 0),
    }
    (output_dir / "intervention_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def compare_outcomes(
    baseline_summary: dict[str, Any],
    intervened_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare baseline run against post-intervention run to evaluate causal effect / flip."""
    b_verdict = baseline_summary.get("verdict", "unknown")
    i_verdict = intervened_summary.get("verdict", "unknown")
    flipped = (b_verdict != i_verdict)
    positive_flip = (b_verdict != "resolved" and i_verdict == "resolved")
    negative_flip = (b_verdict == "resolved" and i_verdict != "resolved")

    b_calls = baseline_summary.get("api_calls") or 0
    i_calls = intervened_summary.get("api_calls") or 0
    b_toks = baseline_summary.get("total_tokens") or 0
    i_toks = intervened_summary.get("total_tokens") or 0

    return {
        "instance_id": baseline_summary.get("instance_id") or intervened_summary.get("instance_id"),
        "intervention": intervened_summary.get("intervention"),
        "baseline_verdict": b_verdict,
        "intervened_verdict": i_verdict,
        "flipped": flipped,
        "positive_flip": positive_flip,
        "negative_flip": negative_flip,
        "api_calls_diff": i_calls - b_calls,
        "tokens_diff": i_toks - b_toks,
    }


def format_intervention_report(results: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("# CAM Causal Intervention Report (P1)")
    lines.append("")
    lines.append(f"Total intervention runs: **{len(results)}**")
    pos_flips = sum(1 for r in results if r.get("positive_flip"))
    neg_flips = sum(1 for r in results if r.get("negative_flip"))
    lines.append(f"Positive flips (Unresolved -> Resolved): **{pos_flips}**")
    lines.append(f"Negative flips (Resolved -> Unresolved): **{neg_flips}**")
    lines.append("")
    lines.append("| Instance | Target Feature | Action | Baseline | Intervened | Flip | Calls Diff | Tokens Diff |")
    lines.append("|---|---|---|---|---|---|---:|---:|")
    for r in results:
        inv = r.get("intervention") or {}
        f = inv.get("feature", "none")
        act = inv.get("action", "none")
        flip_sym = "▲ WIN" if r.get("positive_flip") else ("▼ LOSS" if r.get("negative_flip") else ("NEUTRAL" if not r.get("flipped") else "FLIP"))
        lines.append(
            f"| {r.get('instance_id')} | {f} | {act} | {r.get('baseline_verdict')} | "
            f"{r.get('intervened_verdict')} | {flip_sym} | {r.get('api_calls_diff')} | {r.get('tokens_diff')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to pre_fix checkpoint bundle directory")
    parser.add_argument("--repo", required=True, help="Repository root checked out at base commit")
    parser.add_argument("--issue-file", required=True, help="File containing issue text")
    parser.add_argument("--output-dir", required=True, help="Output directory for predictions and traces")
    parser.add_argument("--feature", help="Target feature name for causal intervention")
    parser.add_argument("--action", choices=("drop", "weaken", "corrupt", "override", "truncate"), default="drop")
    parser.add_argument("--value", help="Value for intervention")
    parser.add_argument("--intervention-file", help="JSON file specifying intervention")
    parser.add_argument("--rust-toolchain", default="1.72.0")
    parser.add_argument("--max-api-calls", type=int, default=70)
    parser.add_argument("--dry-run", action="store_true", help="Print intervention plan and exit without calling model")
    args = parser.parse_args(argv)

    ckpt_dir = Path(args.checkpoint).resolve()
    meta = _safe_load_json(ckpt_dir / "meta.json")
    if not meta:
        parser.error(f"Invalid checkpoint directory (missing meta.json): {ckpt_dir}")

    inv_spec = None
    if args.intervention_file:
        inv_spec = json.loads(Path(args.intervention_file).read_text(encoding="utf-8"))
    elif args.feature:
        inv_spec = {"feature": args.feature, "action": args.action, "value": args.value}

    instance_id = meta.get("instance_id") or ckpt_dir.name
    if args.dry_run:
        print(f"Checkpoint valid: {instance_id}")
        print(f"Intervention plan: {json.dumps(inv_spec, indent=2, ensure_ascii=False)}")
        return 0

    from dotenv import load_dotenv
    load_dotenv(HERE.parent / ".env")
    model = os.getenv("MODEL")
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    if not model or not api_key:
        parser.error("MODEL and API_KEY must be configured in .env")

    summary = run_intervention_task(
        repo=Path(args.repo).resolve(),
        instance_id=instance_id,
        issue_file=Path(args.issue_file).resolve(),
        checkpoint_dir=ckpt_dir,
        intervention=inv_spec,
        output_dir=Path(args.output_dir).resolve(),
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_api_calls=args.max_api_calls,
        rust_toolchain=args.rust_toolchain,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
