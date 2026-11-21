import json
papers=json.load(open("_search/merged_ranked.json",encoding="utf-8"))
print("--- RUST-RELATED ---")
for p in papers:
    t=(p.get("title") or "")
    a=(p.get("abstract") or "")
    if "rust" in (t+" "+a).lower() and "repair" in (t+" "+a).lower():
        print("*",p.get("year"),"|",t,"|",(p.get("venue") or "")[:40],"| cites",p.get("citation_count"),"|",p.get("url"))
print()
print("--- MULTI-SWE / MULTILINGUAL BENCH ---")
for p in papers:
    t=(p.get("title") or "")
    if "multi-swe" in t.lower() or "multilingual" in t.lower() or "polyglot" in t.lower():
        print("*",p.get("year"),"|",t,"|",(p.get("venue") or "")[:45],"| cites",p.get("citation_count"),"|",p.get("url"))
        print("   ",(p.get("abstract") or "")[:420])
