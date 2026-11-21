import json
papers=json.load(open("_search/merged_ranked.json",encoding="utf-8"))
n=len(papers)
rust=[p for p in papers if "rust" in ((p.get("title") or "")+" "+(p.get("abstract") or "")).lower()]
ma=[p for p in papers if any(k in (p.get("title") or "").lower() for k in ["multi-agent","multi agent","agentic","debate","collaborative","ensemble"])]
print("total unique:",n)
print("rust-mentioning:",len(rust))
print("multi-agent-ish titles:",len(ma))
print("2026:",len([p for p in papers if p.get("year")==2026]),"2025:",len([p for p in papers if p.get("year")==2025]),"2024:",len([p for p in papers if p.get("year")==2024]),"2023:",len([p for p in papers if p.get("year")==2023]))
src={}
for p in papers:
    for s in (p.get("found_in") or []): src[s]=src.get(s,0)+1
print("source coverage:",src)
