import json,urllib.request,re
req=urllib.request.Request("https://export.arxiv.org/api/query?id_list=2504.02605",headers={"User-Agent":"Mozilla/5.0"})
raw=urllib.request.urlopen(req,timeout=45).read().decode("utf-8","replace")
m=re.search(r"<entry>(.*?)</entry>",raw,re.S); e=m.group(1)
def g(t):
    mm=re.search(r"<%s[^>]*>(.*?)</%s>"%(t,t),e,re.S); return re.sub(r"\s+"," ",mm.group(1)).strip() if mm else ""
cm=re.search(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>",e,re.S)
print("TITLE:",g("title")); print("pub:",g("published")); print("comment:",cm.group(1) if cm else None)
print("authors:",", ".join(re.findall(r"<name>(.*?)</name>",e)))
print("ABS:",g("summary"))
