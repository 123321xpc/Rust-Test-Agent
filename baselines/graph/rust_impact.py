"""Derive producer->consumer impact surfaces from the Rust Code Knowledge Graph.

The repair agents need a deterministic, machine-readable blast-radius view whenever
production code changes: which symbols were modified, which other files consume
them, and which trait impls / match arms may now be stale. This module turns the
parsed CKG into that compact view so agents stop re-discovering it via ad-hoc grep.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _changed_symbols(ckg: Any, rel_path: str) -> list[dict[str, Any]]:
    """Return the non-test items defined in a changed file."""
    try:
        parsed = ckg.get_parsed_file((ckg.root / rel_path).resolve())
    except Exception:
        parsed = None
    if parsed is None:
        return []
    out: list[dict[str, Any]] = []
    for item in parsed.items:
        if item.get("is_test"):
            continue
        if item.get("kind") not in ("fn", "struct", "enum", "trait", "type", "const", "static"):
            continue
        out.append({
            "name": item.get("name", ""),
            "kind": item.get("kind", ""),
            "line": item.get("line", 0),
            "vis": item.get("vis", ""),
            "enclosing": item.get("enclosing", ""),
        })
    return out


def _changed_symbols_for_lines(ckg: Any, rel_path: str, line_numbers: list[int]) -> list[dict[str, Any]]:
    """Return the non-test items in a changed file that enclose any of the changed lines.

    When the changed line numbers are known, only the enclosing symbols matter; the
    file's first N top-level items are usually unrelated to the edit.
    """
    try:
        parsed = ckg.get_parsed_file((ckg.root / rel_path).resolve())
    except Exception:
        parsed = None
    if parsed is None:
        return []
    out: list[dict[str, Any]] = []
    for item in parsed.items:
        if item.get("is_test"):
            continue
        if item.get("kind") not in ("fn", "struct", "enum", "trait", "type", "const", "static"):
            continue
        start = int(item.get("line", 0) or 0)
        end = int(item.get("end_line", start) or start)
        if any(start <= number <= end for number in line_numbers):
            out.append({
                "name": item.get("name", ""),
                "kind": item.get("kind", ""),
                "line": item.get("line", 0),
                "vis": item.get("vis", ""),
                "enclosing": item.get("enclosing", ""),
            })
    return out


def _consumers_of(ckg: Any, symbol: str, exclude_paths: set[str], limit: int = 12) -> list[dict[str, Any]]:
    """Find non-test call sites that consume `symbol`."""
    try:
        all_files = ckg._all_parsed(exclude_tests=True)
    except Exception:
        return []
    consumers: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for parsed in all_files:
        if parsed.rel_path in exclude_paths:
            continue
        for edge in parsed.call_edges:
            if edge.get("is_test"):
                continue
            callee = str(edge.get("callee", ""))
            base = callee.split("::")[-1]
            if base != symbol and callee != symbol:
                continue
            key = (edge.get("caller_path", ""), int(edge.get("caller_line", 0)))
            if key in seen:
                continue
            seen.add(key)
            consumers.append({
                "path": edge.get("caller_path", ""),
                "line": edge.get("caller_line", 0),
                "caller": edge.get("caller", ""),
                "enclosing": edge.get("caller_enclosing", ""),
                "call": callee,
            })
            if len(consumers) >= limit:
                return consumers
    return consumers


def _consumer_symbols(ckg: Any, symbols: list[str], exclude_paths: set[str], limit: int = 12) -> list[dict[str, Any]]:
    """Find non-test call sites consuming any of `symbols` in one repository scan."""
    try:
        all_files = ckg._all_parsed(exclude_tests=True)
    except Exception:
        return []
    wanted = {s for s in symbols if s}
    if not wanted:
        return []
    consumers: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for parsed in all_files:
        if parsed.rel_path in exclude_paths:
            continue
        for edge in parsed.call_edges:
            if edge.get("is_test"):
                continue
            callee = str(edge.get("callee", ""))
            base = callee.split("::")[-1]
            if base not in wanted and callee not in wanted:
                continue
            key = (edge.get("caller_path", ""), int(edge.get("caller_line", 0)))
            if key in seen:
                continue
            seen.add(key)
            consumers.append({
                "path": edge.get("caller_path", ""),
                "line": edge.get("caller_line", 0),
                "caller": edge.get("caller", ""),
                "enclosing": edge.get("caller_enclosing", ""),
                "call": callee,
            })
            if len(consumers) >= limit:
                return consumers
    return consumers


def _trait_impl_surfaces(ckg: Any, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return trait/impl blocks that mention the symbol (likely match-arm or impl sites)."""
    try:
        all_files = ckg._all_parsed(exclude_tests=True)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for parsed in all_files:
        for block in parsed.impl_blocks:
            if block.get("is_test"):
                continue
            blob = " ".join(str(block.get(k, "")) for k in ("trait", "for_type", "context", "header"))
            if symbol and symbol in blob:
                out.append({
                    "path": parsed.rel_path,
                    "line": block.get("line", 0),
                    "trait": block.get("trait", ""),
                    "for_type": block.get("for_type", ""),
                    "context": str(block.get("context", ""))[:200],
                })
                if len(out) >= limit:
                    return out
    return out


def _trait_impl_surfaces_many(ckg: Any, symbols: list[str], limit: int = 10) -> list[dict[str, Any]]:
    """Return trait/impl blocks referencing any of `symbols` in one repository scan."""
    try:
        all_files = ckg._all_parsed(exclude_tests=True)
    except Exception:
        return []
    wanted = {s for s in symbols if s}
    if not wanted:
        return []
    out: list[dict[str, Any]] = []
    for parsed in all_files:
        for block in parsed.impl_blocks:
            if block.get("is_test"):
                continue
            blob = " ".join(str(block.get(k, "")) for k in ("trait", "for_type", "context", "header"))
            hit = next((s for s in wanted if s and s in blob), None)
            if hit is None:
                continue
            out.append({
                "path": parsed.rel_path,
                "line": block.get("line", 0),
                "trait": block.get("trait", ""),
                "for_type": block.get("for_type", ""),
                "context": str(block.get("context", ""))[:200],
            })
            if len(out) >= limit:
                return out
    return out


def build_impact_surface(
    ckg: Any,
    changed_paths: list[str],
    *,
    max_consumers: int = 12,
    max_symbols_per_file: int = 8,
    changed_lines: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Build the producer->consumer impact surface for a set of changed production files."""
    if ckg is None or not changed_paths:
        return {"files": [], "consumers": [], "trait_impls": [], "cross_file": False}
    rels = [p for p in changed_paths if p and p.endswith(".rs")]
    files: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    impls: list[dict[str, Any]] = []
    for rel in rels[:12]:
        line_numbers = (changed_lines or {}).get(rel) or []
        symbols = (
            _changed_symbols_for_lines(ckg, rel, line_numbers)
            if line_numbers else _changed_symbols(ckg, rel)
        )
        top = symbols[:max_symbols_per_file]
        files.append({
            "path": rel,
            "symbols": top,
            "symbol_count": len(symbols),
        })
    names = [
        sym.get("name", "")
        for entry in files
        for sym in entry.get("symbols", [])
        if sym.get("name") and not str(sym.get("name")).startswith("__")
    ]
    if names:
        for consumer in _consumer_symbols(ckg, names, exclude_paths=set(rels), limit=max_consumers):
            consumer["producer_symbol"] = consumer.get("call", "").split("::")[-1]
            consumer["producer_path"] = consumer.get("path", "")
            consumers.append(consumer)
        for impl in _trait_impl_surfaces_many(ckg, names):
            impl["producer_symbol"] = next(
                (s for s in names if s and s in " ".join(str(impl.get(k, "")) for k in ("trait", "for_type", "context", "header"))),
                "",
            )
            impls.append(impl)
    seen_c: set[tuple[str, int, str]] = set()
    unique_consumers = []
    for item in consumers:
        key = (item.get("path", ""), int(item.get("line", 0)), item.get("producer_symbol", ""))
        if key in seen_c:
            continue
        seen_c.add(key)
        unique_consumers.append(item)
    seen_i: set[tuple[str, int, str]] = set()
    unique_impls = []
    for item in impls:
        key = (item.get("path", ""), int(item.get("line", 0)), item.get("producer_symbol", ""))
        if key in seen_i:
            continue
        seen_i.add(key)
        unique_impls.append(item)
    consumer_paths = sorted({c.get("path", "") for c in unique_consumers if c.get("path")})
    return {
        "files": files,
        "consumers": unique_consumers[: max_consumers * 2],
        "trait_impls": unique_impls[: max_consumers],
        "consumer_paths": consumer_paths,
        "cross_file": bool(consumer_paths),
    }


def format_impact_surface(surface: dict[str, Any], *, max_chars: int = 4000) -> str:
    """Render the impact surface as a compact agent-readable block."""
    if not surface or not surface.get("files"):
        return ""
    lines: list[str] = ["CKG Impact Surface (producer -> consumer, derived from the parsed code graph):"]
    for entry in surface.get("files", []):
        syms = entry.get("symbols", [])
        names = ", ".join(
            f"{s.get('name')}({s.get('kind')}, line {s.get('line')})" for s in syms[:6] if s.get("name")
        )
        if names:
            lines.append(f"- Modified {entry.get('path')}: {names}")
    consumers = surface.get("consumers") or []
    if consumers:
        lines.append("Consumers that may need synchronized updates:")
        for item in consumers[:12]:
            lines.append(
                f"  * {item.get('path')}:{item.get('line')} {item.get('caller')}() "
                f"calls {item.get('producer_symbol')} (producer {item.get('producer_path')})"
            )
    impls = surface.get("trait_impls") or []
    if impls:
        lines.append("Trait/impl or match-arm sites referencing modified symbols:")
        for item in impls[:8]:
            lines.append(
                f"  * {item.get('path')}:{item.get('line')} "
                f"impl {item.get('trait') or '?'} for {item.get('for_type') or '?'} "
                f"(references {item.get('producer_symbol')})"
            )
    if surface.get("cross_file"):
        lines.append(
            "Cross-file impact detected: verify every listed consumer handles the new producer output "
            "before submitting."
        )
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (impact surface truncated)"
    return text


def detect_impact_triggers(surface: dict[str, Any]) -> list[str]:
    """Return human-readable triggers that justify cross-file verification."""
    triggers: list[str] = []
    consumers = surface.get("consumers") or []
    impls = surface.get("trait_impls") or []
    if consumers:
        paths = sorted({c.get("path", "") for c in consumers if c.get("path")})
        triggers.append(
            "producer->consumer cross-file impact on: " + ", ".join(paths[:6])
        )
    if impls:
        triggers.append(
            "trait/impl or match-arm sites may be stale: "
            + ", ".join(f"{i.get('path')}:{i.get('line')}" for i in impls[:6])
        )
    return triggers
