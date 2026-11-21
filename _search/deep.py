import json
papers=json.load(open("_search/merged_ranked.json",encoding="utf-8"))
want=["Beyond Fixed Tests","SGAgent","SWE-Debate","Agent-Based Ensemble","MASAI","CodeR","HGARena","ARISE","SWT-Bench","SWE-ABS","CodeTeam","Beyond Fixed"]
for p in papers:
    t=p.get("title") or ""
    if any(w.lower() in t.lower() for w in want):
        print("="*70)
        print("TITLE:",t)
        print("YEAR:",p.get("year"),"| VENUE:",(p.get("venue") or "")[:60],"| CITES:",p.get("citation_count"))
        print("URL:",p.get("url"),"| DOI:",p.get("doi"),"| ARXIV:",p.get("arxiv_id"))
        print("ABSTRACT:",(p.get("abstract") or "NO ABSTRACT")[:1400])
