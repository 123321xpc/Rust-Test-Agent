import json,urllib.request
# Multi-SWE-bench per-language instance counts
try:
    req=urllib.request.Request("https://datasets-server.huggingface.co/size?dataset=Daoguang%2FMulti-SWE-bench",headers={"User-Agent":"Mozilla/5.0"})
    j=json.load(urllib.request.urlopen(req,timeout=40))
    for s in j.get("size",{}).get("splits",[]): print(s.get("config"),s.get("split"),s.get("num_rows"))
except Exception as e: print("size ERR",e)
for cfg in ["rust_verified","rust","rust_full","default"]:
    try:
        req=urllib.request.Request("https://datasets-server.huggingface.co/size?dataset=Daoguang%2FMulti-SWE-bench&config="+cfg,headers={"User-Agent":"Mozilla/5.0"})
        j=json.load(urllib.request.urlopen(req,timeout=40))
        print(cfg,"->",[(s.get("config"),s.get("split"),s.get("num_rows")) for s in j.get("size",{}).get("splits",[])])
    except Exception as e: print(cfg,"ERR",e)
