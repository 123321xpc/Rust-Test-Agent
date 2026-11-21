import json,urllib.request,re
for aid in ["2406.01304","2406.11638"]:
    try:
        req=urllib.request.Request("https://export.arxiv.org/api/query?id_list="+aid,headers={"User-Agent":"Mozilla/5.0"})
        raw=urllib.request.urlopen(req,timeout=45).read().decode("utf-8","replace")
        m=re.search(r"<entry>(.*?)</entry>",raw,re.S); e=m.group(1)
        def g(t):
            mm=re.search(r"<%s[^>]*>(.*?)</%s>"%(t,t),e,re.S); return re.sub(r"\s+"," ",mm.group(1)).strip() if mm else ""
        cm=re.search(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>",e,re.S)
        jr=re.search(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>",e,re.S)
        print("### arXiv",aid,":",g("title"))
        print("  pub:",g("published"),"| comment:",cm.group(1).strip() if cm else None,"| journal_ref:",jr.group(1).strip() if jr else None)
        print("  ABS:",g("summary")[:900]); print()
    except Exception as ex: print(aid,"ERR",ex)
