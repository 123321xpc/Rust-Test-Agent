import json
from pathlib import Path

DIR = Path("testset_results_20261118")
manifest = json.loads(Path("testset/manifest.json").read_text(encoding="utf-8"))
tasks = manifest["tasks"]
task_map = {t["instance_id"]: t for t in tasks}

agents = ["normal", "ACI-impr", "graph", "graph-2", "test"]
summaries = {a: json.load(open(DIR / f"summary_{a}.json", encoding="utf-8"))[a] for a in agents}

print("=== AGENT ERRORS ===")
for a in agents:
    errs = [inst for inst in summaries[a]["instances"] if inst.get("agent_exit_code") != 0 or inst.get("error")]
    print(f"\n{a} errors ({len(errs)}):")
    for e in errs[:5]:
        print(f"  {e['instance_id']}: exit={e.get('agent_exit_code')}, err={e.get('error')}")

print("\n=== MATRIX OF ALL 50 INSTANCES ===")
header = f"{'Instance ID':<35} | {'Size':<6} | " + " | ".join(f"{a:^8}" for a in agents)
print(header)
print("-" * len(header))
for t in tasks:
    iid = t["instance_id"]
    size = t["size"]
    row = [f"{iid:<35}", f"{size:<6}"]
    for a in agents:
        inst = next(i for i in summaries[a]["instances"] if i["instance_id"] == iid)
        v = inst["verdict"]
        symbol = " PASS " if v == "resolved" else (" FAIL " if v == "unresolved" else (" ERR  " if "error" in v else " NONE "))
        row.append(f"{symbol:^8}")
    print(" | ".join(row))
