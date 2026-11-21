import json,re
papers=json.load(open("_search/merged_ranked.json",encoding="utf-8"))
kw=["rust","multi-agent","multi agent","repository-level","test lock","regression test","trace","dynamic","knowledge graph","code property graph","rollback","unit","compiler","diagnostic","debate","orchestrat","fault propagation"]
sel=[]
for p in papers:
    t=(p.get("title") or "")
    a=(p.get("abstract") or "")
    txt=(t+" "+a).lower()
    score=sum(1 for k in kw if k in txt)
    if score>=3:
        sel.append((score,p))
sel.sort(key=lambda x:-x[0])
out=[]
for s,p in sel[:45]:
    out.append("### %s  (%s %s) cites=%s src=%s" % (p.get("title"),p.get("year"),(p.get("venue") or "")[:40],p.get("citation_count"),",".join(p.get("found_in") or [])))
    out.append("URL: %s" % (p.get("url") or ""))
    out.append("ABS: %s" % ((p.get("abstract") or "")[:900]))
    out.append("")
open("_search/close.txt","w",encoding="utf-8").write("\n".join(out))
print("selected",len(sel))
