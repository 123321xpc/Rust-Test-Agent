import json
papers=json.load(open("_search/merged_ranked.json",encoding="utf-8"))
# Top-venue + highest-overlap subset for the report table
keep=[]
for p in papers:
    v=(p.get("venue") or "")
    t=p.get("title") or ""
    if any(s in v for s in ["International Conference on Software Engineering","Neural Information","Empirical Software Engineering","International Conference on Automated Software Engineering","International Symposium on Software Testing and Analysis","Transactions on Software Engineering","International Conference on Software Testing","ACM/IEEE International Conference","Design Automation Conference","Joint European Software Engineering","International Conference on Software Maintenance","International Conference on Program Comprehension","Annual International Computer Software"]):
        keep.append(p)
keep.sort(key=lambda p:-(p.get("citation_count") or 0))
print("TOP-VENUE papers:",len(keep))
for p in keep[:45]:
    print("%-4s | %-58s | %s | %s" % (p.get("year"), (p.get("venue") or "")[:58], (p.get("title") or "")[:78], p.get("citation_count")))
