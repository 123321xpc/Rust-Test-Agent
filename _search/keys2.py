import json
papers=json.load(open("_search/merged_ranked.json",encoding="utf-8"))
keys=["test tampering","cheat","weak test","oracle","lock","tamper","reward hack","sabotage","benchmark contamination","solution leakage","sabotage test"]
out=[]
for p in papers:
    txt=((p.get("title") or "")+" "+(p.get("abstract") or "")).lower()
    if any(k in txt for k in keys):
        out.append("### %s\nY:%s V:%s cites=%s src=%s\nURL:%s\nABS:%s\n" % (p.get("title"),p.get("year"),(p.get("venue") or "")[:40],p.get("citation_count"),",".join(p.get("found_in") or []),p.get("url"),(p.get("abstract") or "")[:700]))
open("_search/keys2.txt","w",encoding="utf-8").write("\n".join(out[:30]))
print(len(out))
