import json,urllib.request
for aid in ["2602.23647","2602.21681"]:
    try:
        j=json.load(urllib.request.urlopen("http://export.arxiv.org/api/query?id_list="+aid,timeout=40))["feed"]
        e=j.get("entry")
        if not e: print(aid,"NOT FOUND"); continue
        if isinstance(e,list): e=e[0]
        print("arXiv",aid,":",e.get("title","").strip().replace("\n"," "))
        print("  published:",e.get("published"),"updated:",e.get("updated"))
        print("  authors:",", ".join(a.find("name").text for a in e.findall("author"))[:200])
        print("  comment:",(e.find("{http://arxiv.org/schemas/atom}comment").text if e.find("{http://arxiv.org/schemas/atom}comment") is not None else None))
        print("  abs:",e.find("summary").text.strip().replace("\n"," ")[:1100])
        print()
    except Exception as ex:
        print(aid,"ERR",ex)
