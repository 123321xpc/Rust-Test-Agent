import json,urllib.request,re
ids=["2602.22764","2602.21681","2602.23647","2605.03117","2607.12605"]
for aid in ids:
    try:
        req=urllib.request.Request("https://export.arxiv.org/api/query?id_list="+aid,headers={"User-Agent":"Mozilla/5.0"})
        raw=urllib.request.urlopen(req,timeout=45).read().decode("utf-8","replace")
        m=re.search(r"<entry>(.*?)</entry>",raw,re.S)
        if not m: print(aid,"NO ENTRY"); continue
        e=m.group(1)
        def g(tag):
            mm=re.search(r"<%s[^>]*>(.*?)</%s>"%(tag,tag),e,re.S)
            return re.sub(r"\s+"," ",mm.group(1)).strip() if mm else ""
        cm=re.search(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>",e,re.S)
        print("arXiv",aid,":",g("title"))
        print("   pub:",g("published"),"| comment:",re.sub(r"\s+"," ",cm.group(1)).strip() if cm else None)
        print("   authors:",", ".join(re.findall(r"<name>(.*?)</name>",e))[:180])
        print("   ABS:",g("summary")[:1200])
        print()
    except Exception as ex:
        print(aid,"ERR",ex)
