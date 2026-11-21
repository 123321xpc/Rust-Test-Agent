"""Render the ranked corpus into a compact table + keyword stats."""
import json
import re
from collections import Counter
from pathlib import Path

SEARCH_DIR = Path(r"C:\Users\vickey\Desktop\Rust-SWE-Bench-main\_search")


def main() -> None:
    papers = json.loads((SEARCH_DIR / "merged_ranked.json").read_text(encoding="utf-8"))
    for p in papers:
        p["found_in"] = p.get("found_in") or []
    top = papers[:70]
    lines = []
    for i, p in enumerate(top, 1):
        src = ",".join(s[:4] for s in p["found_in"])
        survey = " [survey]" if p.get("is_survey") else ""
        lines.append(
            "%3d | %s%s | %s | %s | cites=%s | score=%s | %s"
            % (
                i,
                p.get("url") or "",
                "",
                p.get("year"),
                (p.get("venue") or "")[:34],
                p.get("citation_count"),
                p.get("relevance_score"),
                src,
            )
        )
        lines.append("      TITLE: " + (p.get("title") or ""))
    (SEARCH_DIR / "top70.txt").write_text("\n".join(lines), encoding="utf-8")

    # keyword frequency over titles
    stop = {
        "agent", "agents", "agentic", "llm", "llms", "large", "language", "model",
        "models", "based", "using", "with", "for", "and", "the", "towards", "via",
        "multi", "automated", "repair", "program", "software", "issue", "issues",
        "resolving", "repository", "level", "code", "coding", "bench", "benchmark",
        "swe", "generation", "generation", "framework", "system", "approach",
        "task", "tasks", "generation", "repair", "code", "llm", "agent", "agents",
    }
    freq = Counter()
    for p in papers:
        title = (p.get("title") or "").lower()
        for w in set(re.split(r"[^a-z0-9]+", title)):
            if len(w) > 3 and w not in stop:
                freq[w] += 1
    (SEARCH_DIR / "keywords.txt").write_text(
        "\n".join(f"{w}: {c}" for w, c in freq.most_common(40)), encoding="utf-8"
    )

    venues = Counter((p.get("venue") or "?").strip() for p in papers)
    (SEARCH_DIR / "venues.txt").write_text(
        "\n".join(f"{v}: {c}" for v, c in venues.most_common(30)), encoding="utf-8"
    )
    print("papers:", len(papers))
    print("top keywords:", freq.most_common(25))


if __name__ == "__main__":
    main()
