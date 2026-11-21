"""Merge, dedup and rank search results across the three query batches."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\vickey\.codex\skills\paper_search\scripts")
from postprocess import dedup, rank  # noqa: E402

SEARCH_DIR = Path(r"C:\Users\vickey\Desktop\Rust-SWE-Bench-main\_search")


def merge() -> dict:
    merged: dict[str, list] = {}
    for f in sorted(SEARCH_DIR.glob("r*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for src, papers in data.items():
            merged.setdefault(src, []).extend(papers)
    return merged


def main() -> None:
    merged = merge()
    queries = [
        "multi-agent repository-level issue resolution LLM agent",
        "Rust program repair large language model agent",
        "SWE-bench Rust benchmark repository issue",
        "agent graph automated program repair multi-agent",
        "dynamic tracing runtime instrumentation program repair agent",
        "compiler feedback loop LLM agent code repair Rust",
        "test-driven automated program repair agent regression test generation",
        "multi-agent LLM collaboration software engineering issue fixing",
        "SWE-bench Java Multi-SWE-bench Rust repository-level benchmark",
        "code knowledge graph repository understanding agent bug localization",
        "failure localization repository-level fault propagation LLM",
        "agentic software engineering survey design paradigm",
    ]
    unique = dedup(merged)
    ranked, n_dropped = rank(unique, queries)
    out = SEARCH_DIR / "merged_ranked.json"
    out.write_text(json.dumps(ranked, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {k: len(v) for k, v in merged.items()}
    print("per-source:", counts, "raw total:", sum(counts.values()))
    print("deduped:", len(ranked), "dropped:", n_dropped)


if __name__ == "__main__":
    main()
