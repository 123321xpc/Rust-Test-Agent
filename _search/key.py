import json
papers=json.load(open("_search/merged_ranked.json",encoding="utf-8"))
want=["SGAgent","ARISE","HGARena","SWE-Debate","AkiraRust","SWE-Adept","MASAI","CodeR","Multi-SWE-bench","ARISE","Agentic Harness","SemAgent","SGAgent","Rust"]
out=[]
for p in papers:
    t=p.get("title") or ""
    if any(w.lower() in t.lower() for w in want):
        out.append("### %s\nYEAR: %s | VENUE: %s | CITES: %s | SRC: %s\nURL: %s\nDOI: %s | ARXIV: %s\nABS: %s\n" % (t,p.get("year"),p.get("venue"),p.get("citation_count"),",".join(p.get("found_in") or []),p.get("url"),p.get("doi"),p.get("arxiv_id"),(p.get("abstract") or "NO ABSTRACT")))
open("_search/key.txt","w",encoding="utf-8").write("\n".join(out))
print(len(out))
