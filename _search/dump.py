"""Dump search_papers() results (with abstracts) to JSON for relevance filtering."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\vickey\.codex\skills\paper_search\scripts")

from search_papers import search_papers  # noqa: E402


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    results = search_papers(
        query=payload["queries"],
        start_year=payload["start_year"],
        end_year=payload["end_year"],
        max_results=payload.get("max_papers", 10),
        sources=payload.get("sources"),
        parallel=True,
        start_date=payload.get("start_date"),
        end_date=payload.get("end_date"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    counts = {k: len(v) for k, v in results.items()}
    print("per-source:", counts, "total:", sum(counts.values()))


if __name__ == "__main__":
    main()
