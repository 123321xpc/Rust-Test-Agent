import json,urllib.request
# Final fact check: Multi-SWE-bench Rust subset size from the paper HTML table
try:
    req=urllib.request.Request("https://arxiv.org/abs/2504.02605",headers={"User-Agent":"Mozilla/5.0"})
    raw=urllib.request.urlopen(req,timeout=45).read().decode("utf-8","replace")
    import re
    for m in re.finditer(r"[^.]{0,160}\bRust\b[^.]{0,160}", re.sub(r"<[^>]+>"," ",raw)):
        s=" ".join(m.group(0).split())
        if any(k in s for k in ["239","instances","Rust","subset"]):
            print("-",s[:250])
except Exception as e: print("ERR",e)
