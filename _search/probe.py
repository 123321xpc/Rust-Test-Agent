import json,glob,sys
sys.path.insert(0,r"C:\Users\vickey\.codex\skills\paper_search\scripts")
from search_papers import search_papers
from postprocess import dedup, rank

merged={}
for f in sorted(glob.glob("_search/r*.json")):
    d=json.load(open(f,encoding="utf-8"))
    for s,p in d.items(): merged.setdefault(s,[]).extend(p)
print("raw:", {k:len(v) for k,v in merged.items()}, sum(len(v) for v in merged.values()))
u=dedup(merged)
print("dedup type:", type(u), "len:", len(u))
print(json.dumps(u, ensure_ascii=False)[:600])
