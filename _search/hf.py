import json,urllib.request
req=urllib.request.Request("https://datasets-server.huggingface.co/splits?dataset=Daoguang%2FMulti-SWE-bench",headers={"User-Agent":"Mozilla/5.0"})
try:
    j=json.load(urllib.request.urlopen(req,timeout=40))
    for s in j.get("splits",[]): print(s)
except Exception as e: print("ERR",e)
