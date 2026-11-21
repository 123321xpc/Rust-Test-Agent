import json,urllib.request,re
try:
    req=urllib.request.Request("https://arxiv.org/html/2504.02605v1",headers={"User-Agent":"Mozilla/5.0"})
    raw=urllib.request.urlopen(req,timeout=60).read().decode("utf-8","replace")
    txt=re.sub(r"<[^>]+>"," ",raw)
    txt=" ".join(txt.split())
    # find per-language counts near Rust
    for m in re.finditer(r".{0,200}Rust.{0,200}", txt):
        s=m.group(0)
        if re.search(r"\b(239|66|126|47)\b", s):
            print("*",s[:330]); print()
    # also look for the table with language distribution
    i=txt.find("Rust")
    print("--- context ---")
    print(txt[max(0,i-600):i+600])
except Exception as e: print("ERR",e)
