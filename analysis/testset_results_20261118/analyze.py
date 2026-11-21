import json
from pathlib import Path
from collections import defaultdict

DIR = Path("analysis/testset_results_20261118")
manifest = json.loads(Path("suites/testset/manifest.json").read_text(encoding="utf-8"))
tasks = manifest["tasks"]
task_map = {t["instance_id"]: t for t in tasks}

agents = ["normal", "ACI-impr", "graph", "graph-2", "test"]
summaries = {}
for a in agents:
    with open(DIR / f"summary_{a}.json", encoding="utf-8") as f:
        summaries[a] = json.load(f)[a]

print("=" * 80)
print("1. OVERALL VERDICT COUNTS")
print("=" * 80)
header = f"{'Agent':<12} | {'Resolved':<10} | {'Unresolved':<11} | {'Agent Err':<10} | {'Eval Err/Err':<12} | {'Resolve %':<10}"
print(header)
print("-" * len(header))
for a in agents:
    counts = summaries[a]["counts"]
    res = counts.get("resolved", 0)
    unres = counts.get("unresolved", 0)
    a_err = counts.get("agent_error", 0)
    ev_err = counts.get("evaluation_error", 0) + counts.get("error", 0)
    pct = (res / 50.0) * 100
    print(f"{a:<12} | {res:<10} | {unres:<11} | {a_err:<10} | {ev_err:<12} | {pct:.1f}%")

print("\n" + "=" * 80)
print("2. BREAKDOWN BY PATCH SIZE (Small: 25, Medium: 25)")
print("=" * 80)
print(f"{'Agent':<12} | {'Small (25)':<12} | {'Small %':<10} | {'Medium (25)':<12} | {'Medium %':<10}")
print("-" * 65)
for a in agents:
    instances = summaries[a]["instances"]
    small_res = sum(1 for inst in instances if task_map[inst["instance_id"]]["size"] == "small" and inst["verdict"] == "resolved")
    med_res = sum(1 for inst in instances if task_map[inst["instance_id"]]["size"] == "medium" and inst["verdict"] == "resolved")
    print(f"{a:<12} | {small_res:<12} | {small_res/25*100:.1f}%     | {med_res:<12} | {med_res/25*100:.1f}%")

print("\n" + "=" * 80)
print("3. EXECUTION TIME & PATCH STATS")
print("=" * 80)
print(f"{'Agent':<12} | {'Total Time':<12} | {'Avg Time/Task':<14} | {'Patches Gen':<12} | {'Avg Patch Bytes':<15}")
print("-" * 75)
for a in agents:
    instances = summaries[a]["instances"]
    total_sec = sum(inst.get("seconds", 0) for inst in instances)
    avg_sec = total_sec / len(instances)
    patches = sum(1 for inst in instances if inst.get("patch_bytes", 0) > 0)
    avg_bytes = sum(inst.get("patch_bytes", 0) for inst in instances) / max(patches, 1)
    
    # Format total sec to h:m:s
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    print(f"{a:<12} | {h}h {m}m {s}s      | {avg_sec:.1f}s        | {patches}/50 ({patches/50*100:.0f}%)   | {avg_bytes:.0f} bytes")

print("\n" + "=" * 80)
print("4. INSTANCE LEVEL BREAKDOWN ACROSS AGENTS")
print("=" * 80)

resolved_sets = {}
for a in agents:
    resolved_sets[a] = {inst["instance_id"] for inst in summaries[a]["instances"] if inst["verdict"] == "resolved"}

union_solved = set()
for a in agents:
    union_solved |= resolved_sets[a]

print(f"Total Unique Instances Resolved by at least ONE agent: {len(union_solved)} / 50 ({len(union_solved)/50*100:.1f}%)")
all_solved = set(tasks[0]["instance_id"] for _ in [0])
all_solved = set.intersection(*[resolved_sets[a] for a in agents])
print(f"Instances Resolved by ALL 5 agents: {len(all_solved)} / 50 ({len(all_solved)/50*100:.1f}%)")

print("\nUnique solves (instances solved ONLY by this agent):")
for a in agents:
    other_agents = [other for other in agents if other != a]
    other_solved = set.union(*[resolved_sets[other] for other in other_agents])
    unique = resolved_sets[a] - other_solved
    print(f"  {a:<12}: {len(unique)} tasks: {sorted(list(unique))}")

print("\n" + "=" * 80)
print("5. REPO-BY-REPO RESOLUTION")
print("=" * 80)
repo_tasks = defaultdict(list)
for t in tasks:
    repo_tasks[t["repo"]].append(t["instance_id"])

print(f"{'Repository':<32} | {'Tasks':<6} | " + " | ".join(f"{a:^8}" for a in agents))
print("-" * 85)
for repo in sorted(repo_tasks.keys()):
    t_ids = repo_tasks[repo]
    counts_str = []
    for a in agents:
        c = sum(1 for tid in t_ids if tid in resolved_sets[a])
        counts_str.append(f"{c}/{len(t_ids):<6}")
    print(f"{repo:<32} | {len(t_ids):<6} | " + " | ".join(f"{s:^8}" for s in counts_str))
