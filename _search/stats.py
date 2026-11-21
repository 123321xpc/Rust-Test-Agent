import json,glob
for f in sorted(glob.glob("testset_results_20261118/summary_*.json")):
    d=json.load(open(f,encoding="utf-8"))
    k=list(d.keys())[0]; v=d[k]; c=v["counts"]
    print("%-10s resolved=%3d unresolved=%3d no_patch=%2d agent_error=%3d eval_err=%2d error=%2d total=%d" % (k,c["resolved"],c["unresolved"],c["no_patch"],c["agent_error"],c["evaluation_error"],c["error"],v["total"]))
print()
for f in sorted(glob.glob("testset_results_20261118/meta_*.json")):
    d=json.load(open(f,encoding="utf-8"))
    print("%-10s model=%s api=%s limit=%s sha=%s" % (list(d.keys())[0], d.get("model"), d.get("api"), d.get("api_limit"), str(list(d.get("agent_source_sha256",{}).values())[0])[:12]))
