import json,urllib.request,re
ids=["2607.26591","2610.33853","2609.34752","2509.16299","2607.18213"]
for aid in ids:
    try:
        req=urllib.request.Request("https://export.arxiv.org/api/query?id_list="+aid,headers={"User-Agent":"Mozilla/5.0"})
        raw=urllib.request.urlopen(req,timeout=45).read().decode("utf-8","replace")
        m=re.search(r"<entry>(.*?)</entry>",raw,re.S)
        if not m: print(aid,"NO ENTRY"); continue
        e=m.group(1)
        def g(t):
            mm=re.search(r"<%s[^>]*>(.*?)</%s>"%(t,t),e,re.S); return re.sub(r"\s+"," ",mm.group(1)).strip() if mm else ""
        cm=re.search(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>",e,re.S)
        jr=re.search(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>",e,re.S)
        print("### arXiv",aid,":",g("title"))
        print("  pub:",g("published"),"| comment:",re.sub(r"\s+"," ",cm.group(1)).strip() if cm else None,"| jref:",jr.group(1).strip() if jr else None)
        print("  authors:",", ".join(re.findall(r"<name>(.*?)</name>",e))[:160])
        print("  ABS:",g("summary")[:950]); print()
    except Exception as ex: print(aid,"ERR",ex)
