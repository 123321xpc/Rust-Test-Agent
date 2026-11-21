import json
papers=json.load(open("_search/merged_ranked.json",encoding="utf-8"))
# Final relevance screen against the user's actual question:
# "multi-agent solving Rust repository-level repair, top venues, overlap, and forward-looking advice"
must=[]; borderline=[]; drop=0
for p in papers:
    t=((p.get("title") or "")+" "+(p.get("abstract") or "")).lower()
    rust = "rust" in t
    repo = any(k in t for k in ["repository-level","repository level","swe-bench","repo-level","multi-swe","github issue"])
    ma = any(k in t for k in ["multi-agent","multi agent","agentic","multi-role","debate","collaborative","ensemble","orchestrat"])
    apr = any(k in t for k in ["repair","issue resolution","issue resolving","bug fix","fixing","patch"])
    if rust and repo and apr: must.append(p)
    elif (repo and ma and apr) or (rust and ma) or (repo and ma): borderline.append(p)
    else: drop+=1
print("MUST:",len(must),"BORDERLINE:",len(borderline),"DROP:",drop)
def fmt(ps):
    o=[]
    for p in ps[:45]:
        o.append("%s | %s | %s | cites=%s | %s" % (p.get("year"),(p.get("venue") or "?")[:38],(p.get("title") or "")[:95],p.get("citation_count"),p.get("url")))
    return "\n".join(o)
open("_search/must.txt","w",encoding="utf-8").write(fmt(must))
open("_search/border.txt","w",encoding="utf-8").write(fmt(borderline))
print("--- MUST ---"); print(fmt(must))
