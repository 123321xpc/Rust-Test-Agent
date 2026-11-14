#!/usr/bin/env python3
"""Capture and evaluate paired post-Architect replays for bytes-377 on Jifang."""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import run as validation

INSTANCE = "tokio-rs__bytes-377"
RATIONALE = (
    "Re-check the issue's exact contract and the Vec<u8> implementations of remaining_mut, "
    "advance_mut, and bytes_mut before editing. The issue permits panic but also describes "
    "a non-panicking contract; do not assume panic or spare-capacity clamping without "
    "reconciling those semantics. Keep cnt == 0 a no-op, prevent undocumented implicit "
    "growth, and ensure the proposed edits and safety rationale match the visible contract "
    "and tests. Preserve legacy behavior only when it is compatible with that contract."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "prepare", "replay", "report"))
    parser.add_argument("--dir", required=True)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--plan-trace")
    parser.add_argument("--repeats", type=int, default=3, help="Number of matched control/intervention pairs")
    args = parser.parse_args()
    root = Path(args.dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "report":
        provenance = json.loads((root / "checkpoint/meta.json").read_text(encoding="utf-8"))
        rows = []
        for pair_id in range(1, args.repeats + 1):
            pair_dir = root / f"pair_{pair_id:02d}"
            arms = {}
            for arm in ("control", "intervention"):
                path = pair_dir / arm / "status.json"
                arms[arm] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"verdict": "missing"}
            rows.append({"pair": pair_id, "control": arms["control"], "intervention": arms["intervention"]})
        positive = sum(r["control"].get("verdict") != "resolved" and r["intervention"].get("verdict") == "resolved" for r in rows)
        negative = sum(r["control"].get("verdict") == "resolved" and r["intervention"].get("verdict") != "resolved" for r in rows)
        report = ["# bytes-377 配对干预实验", "", f"完成配对：{sum(r['control'].get('verdict') != 'missing' and r['intervention'].get('verdict') != 'missing' for r in rows)}/{args.repeats}",
                  f"正向翻转：{positive}", f"负向翻转：{negative}", "",
                  "所有组从同一个新采集的 post-Architect 断点独立重放；每组仅干预 arbitration_rationale。",
                  "干预内容要求核对可见契约和相关实现，没有把 panic 当作唯一正确行为。",
                  "该定向任务已有历史评测记录，实验属于开发阶段的事后分析。", "",
                  "| Pair | Control | Intervention | Applied | Control audit | Intervention audit | Consistency blocked | Positive flip |",
                  "|---:|---|---|---|---|---|---|---|"]
        for row in rows:
            c, i = row["control"], row["intervention"]
            blocked = "yes" if c.get("consistency_blocked") or i.get("consistency_blocked") else "no"
            report.append(f"| {row['pair']} | {c.get('verdict', 'missing')} | {i.get('verdict', 'missing')} | {i.get('intervention_applied', 'unknown')} | {c.get('plan_consistency', 'unknown')} | {i.get('plan_consistency', 'unknown')} | {blocked} | {'yes' if c.get('verdict') != 'resolved' and i.get('verdict') == 'resolved' else 'no'} |")
        report.extend(["", "## 完整状态", "", "```json", json.dumps(rows, indent=2, ensure_ascii=False), "```", "",
                       "## 断点来源", "", "```json", json.dumps({k: v for k, v in provenance.items() if k != "blackboard"}, indent=2, ensure_ascii=False), "```", ""])
        (root / "report.md").write_text("\n".join(report), encoding="utf-8")
        print("\n".join(report[:7]), flush=True)
        return
    if args.phase == "prepare":
        if not args.source_checkpoint or not args.plan_trace:
            parser.error("prepare requires --source-checkpoint and --plan-trace")
        source = Path(args.source_checkpoint).resolve()
        trace = Path(args.plan_trace).resolve()
        entries = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
        plans = [entry["input"] for entry in entries if entry.get("event") == "tool_start"
                 and entry.get("tool") == "submit_repair_plan" and isinstance(entry.get("input"), dict)]
        if not plans:
            raise RuntimeError("Original trace contains no submitted Architect plan")
        starts = [entry for entry in entries if entry.get("event") == "run_start"]
        meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
        if (not starts or starts[0].get("instance_id") != INSTANCE or
                meta.get("instance_id") not in (None, INSTANCE) or
                not meta.get("issue", "").endswith(starts[0].get("issue", ""))):
            raise RuntimeError("Checkpoint and original trace do not match the same instance and issue")
        dest = root / "checkpoint"
        shutil.copytree(source, dest)
        meta_path = dest / "meta.json"
        meta["instance_id"] = INSTANCE
        meta["blackboard"]["repair_plan"] = {"action": "plan", **plans[0]}
        meta["experiment_provenance"] = {"checkpoint": str(source), "plan_trace": str(trace),
                                         "note": "Original plan combined with newly captured Navigator state; not an exact reconstruction."}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Prepared hybrid checkpoint: {dest}", flush=True)
        return
    env = validation.child_env("direct")
    tasks = json.loads((validation.HERE / "manifest.json").read_text(encoding="utf-8"))["tasks"]
    task = next(t for t in tasks if t["instance_id"] == INSTANCE)
    row = next(r for r in validation.read_jsonl(validation.ROOT / "data/rustbench-all.jsonl") if r["instance_id"] == INSTANCE)
    validation.preflight([task], {INSTANCE: row}, ["test"], env)
    (root / "issue.txt").write_text(row["problem_statement"], encoding="utf-8")
    (root / "dataset.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.phase == "capture":
        (root / "experiment.json").write_text(json.dumps({
            "instance_id": INSTANCE, "phase": "matched_post_architect_replay",
            "repeats": args.repeats, "network": "direct",
            "intervention": {"feature": "arbitration_rationale", "action": "override", "value": RATIONALE,
                             "value_sha256": hashlib.sha256(RATIONALE.encode("utf-8")).hexdigest()},
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Experiment directory: {root}", flush=True)
    if args.phase == "replay" and args.repeats < 1:
        parser.error("--repeats must be positive")
    arms = [("capture", root / "capture")] if args.phase == "capture" else [
        (arm, root / f"pair_{pair_id:02d}" / arm)
        for pair_id in range(1, args.repeats + 1)
        for arm in ("control", "intervention")
    ]
    for arm, folder in arms:
        folder.mkdir(parents=True, exist_ok=False)
        task_subdir = folder.relative_to(root).as_posix()
        cmd = validation.docker_base(task, env)
        agent_path = validation.ROOT / "baselines/test"
        cmd += ["-v", f"{agent_path}:{agent_path}:ro", "-v", f"{root}:/run/task:rw"]
        if arm == "capture":
            cmd += ["-e", "RUSTBENCH_DUMP_POST_PLAN=/run/task/checkpoint",
                    "-e", "RUSTBENCH_FORCE_PLAN_CAPTURE=1"]
        script = 'rm -rf /testbed/.git /home/"$1"/.git; rm -f /home/fix.patch /home/test.patch; cd /testbed; shift; exec "$@"'
        agent_cmd = [str(validation.PYTHON), str(agent_path / "agent.py"), "--repo", "/testbed",
                     "--instance-id", INSTANCE, "--issue-file", "/run/task/issue.txt", "--rust-toolchain",
                     task["rust_toolchain"], "--output", f"/run/task/{task_subdir}/prediction.jsonl",
                     "--trace-dir", f"/run/task/{task_subdir}/traces"]
        if arm != "capture":
            agent_cmd += ["--replay-pre-fix", "/run/task/checkpoint"]
        if arm == "intervention":
            agent_cmd += ["--intervene-feature", "arbitration_rationale", "--intervene-action", "override", "--intervene-value", RATIONALE]
        cmd += ["--entrypoint", "/bin/bash", task["image"], "-c", script, "runner", task["repo"].split("/")[1], *agent_cmd]
        print(f"{dt.datetime.now().isoformat()} starting {arm}", flush=True)
        rc, elapsed = validation.run_logged(cmd, folder / "agent.log", validation.ROOT, env)
        pred = folder / "prediction.jsonl"
        status = {"arm": arm, "agent_exit_code": rc, "seconds": elapsed}
        if pred.exists():
            records = validation.read_jsonl(pred)
            status["patch_bytes"] = len(records[0].get("model_patch", "").encode("utf-8")) if records else 0
        (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status), flush=True)
        if arm == "capture":
            checkpoint = root / "checkpoint/meta.json"
            if not checkpoint.exists():
                raise RuntimeError("No Architect plan was captured; inspect capture/agent.log")
            plan = json.loads(checkpoint.read_text(encoding="utf-8"))["blackboard"].get("repair_plan")
            print("Captured rationale: " + str((plan or {}).get("arbitration_rationale")), flush=True)
        if arm == "intervention":
            trace_files = list((folder / "traces").glob("*.jsonl"))
            events = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines() if line.strip()] if trace_files else []
            intervention_events = [e for e in events if e.get("event") == "causal_intervention"]
            status["intervention_applied"] = any(e.get("applied") and e.get("target_feature") == "arbitration_rationale" for e in intervention_events)
            audit_events = [e for e in events if e.get("event") == "plan_consistency_audit"]
            blocked_events = [e for e in events if e.get("event") == "consistency_blocked"]
            status["plan_consistency"] = audit_events[-1].get("status", "not_checked") if audit_events else "not_checked"
            status["plan_conflicts"] = audit_events[-1].get("conflicts", []) if audit_events else []
            status["consistency_blocked"] = bool(blocked_events)
            status["consistency_block_reasons"] = [e.get("reason", "") for e in blocked_events]
            status["audit_rounds"] = [{"status": e.get("status", "not_checked"), "conflicts": e.get("conflicts", []),
                                        "outcome": e.get("outcome")} for e in audit_events]
            status["intervention"] = {"feature": "arbitration_rationale", "action": "override",
                                      "value_sha256": hashlib.sha256(RATIONALE.encode("utf-8")).hexdigest()}
            (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(status), flush=True)
        elif arm == "control":
            trace_files = list((folder / "traces").glob("*.jsonl"))
            events = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines() if line.strip()] if trace_files else []
            audit_events = [e for e in events if e.get("event") == "plan_consistency_audit"]
            blocked_events = [e for e in events if e.get("event") == "consistency_blocked"]
            status["plan_consistency"] = audit_events[-1].get("status", "not_checked") if audit_events else "not_checked"
            status["plan_conflicts"] = audit_events[-1].get("conflicts", []) if audit_events else []
            status["consistency_blocked"] = bool(blocked_events)
            status["consistency_block_reasons"] = [e.get("reason", "") for e in blocked_events]
            status["audit_rounds"] = [{"status": e.get("status", "not_checked"), "conflicts": e.get("conflicts", []),
                                        "outcome": e.get("outcome")} for e in audit_events]
            status["intervention_applied"] = False
            (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if arm != "capture" and pred.exists() and status.get("patch_bytes"):
            (folder / "dataset.jsonl").write_text((root / "dataset.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
            records = validation.read_jsonl(pred)
            records[0]["model_name_or_path"] = "test"
            validation.write_jsonl(folder / "predictions.jsonl", records)
            eval_id = root.name + "_" + arm
            print(f"{dt.datetime.now().isoformat()} evaluating {arm}", flush=True)
            eval_rc = validation.evaluate("test", [task], folder, eval_id, env, 1, 1800)
            report = validation.HARNESS / "logs/run_evaluation" / eval_id / "test" / INSTANCE / "report.json"
            output = report.parent / "test_output.txt"
            if report.exists() and output.exists():
                result = json.loads(report.read_text(encoding="utf-8"))[INSTANCE]
                status["verdict"] = "error" if result.get("error") else ("resolved" if result.get("resolved") else "unresolved")
            else:
                status["verdict"] = "evaluation_error"
            status.update(evaluation_exit_code=eval_rc, report=str(report), test_output=str(output))
            (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
