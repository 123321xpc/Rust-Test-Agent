import json,urllib.request,time
dois=["10.1145/3744916.3773108","10.1145/3770743.3804397","10.1145/3818617","10.48550/arxiv.2602.23647"]
for d in dois:
    try:
        req=urllib.request.Request("https://api.crossref.org/works/"+d,headers={"User-Agent":"research/1.0 (mailto:v@example.com)"})
        j=json.load(urllib.request.urlopen(req,timeout=30))["message"]
        print("DOI:",d)
        print("  title:",(j.get("title") or [""])[0])
        print("  container:",(j.get("container-title") or [""])[0])
        print("  type:",j.get("type"),"| event:",(j.get("event") or {}).get("name"))
        print("  year:",(j.get("published") or {}).get("date-parts"))
        print("  authors:",", ".join((a.get("given","")+" "+a.get("family","")).strip() for a in (j.get("author") or [])[:5]))
        print("  url:",j.get("URL"))
    except Exception as e:
        print("DOI:",d,"ERR",e)
    time.sleep(0.5)
