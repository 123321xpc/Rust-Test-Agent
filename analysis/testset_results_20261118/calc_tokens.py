import json
import glob

runs = [
    ("normal", "testset_20261117_152359_c2af99"),
    ("ACI-impr", "testset_20261117_152405_41a770"),
    ("graph", "testset_20261117_152410_d586c5"),
    ("graph-2", "testset_20261117_152416_73bceb"),
    ("test", "testset_20261117_152421_8f28d8")
]

for a, r in runs:
    trace_files = glob.glob(f"/home/xupeicai/xpc/testset/runs/{r}/{a}/*/traces/*.jsonl")
    in_tok, out_tok, calls = 0, 0, 0
    for tf in trace_files:
        try:
            with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                        if ev.get("event") == "api_response":
                            calls += 1
                            usage = ev.get("body", {}).get("usage", {})
                            in_tok += usage.get("input_tokens", 0)
                            out_tok += usage.get("output_tokens", 0)
                    except Exception:
                        pass
        except Exception:
            pass
    print(f"{a:10}: traces={len(trace_files)}, api_calls={calls}, input_tok={in_tok:,}, output_tok={out_tok:,}, total_tok={in_tok+out_tok:,}")
