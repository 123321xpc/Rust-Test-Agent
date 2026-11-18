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
        if item.get("variants"):
            out[-1]["variants"] = list(item["variants"])
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
            entry = {
                "name": item.get("name", ""),
                "kind": item.get("kind", ""),
                "line": item.get("line", 0),
                "vis": item.get("vis", ""),
                "enclosing": item.get("enclosing", ""),
            }
            if item.get("variants"):
                entry["variants"] = list(item["variants"])
            out.append(entry)
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


def _wildcard_match_surfaces(
    ckg: Any,
    enum_names: set[str],
    variant_names: set[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find non-test `match ... { ..., _ => ... }` sites referencing modified enums or variants."""
    if not enum_names and not variant_names:
        return []
    try:
        all_files = ckg._all_parsed(exclude_tests=True)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for parsed in all_files:
        for wm in getattr(parsed, "wildcard_matches", ()):
            if wm.get("is_test"):
                continue
            w_types = set(wm.get("types") or ())
            w_vars = set(wm.get("variants") or ())
            matched_types = sorted(w_types & enum_names)
            matched_vars = sorted(w_vars & variant_names)
            if not matched_types and not matched_vars:
                continue
            key = (parsed.rel_path, int(wm.get("line") or 0))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "path": parsed.rel_path,
                "line": wm.get("line", 0),
                "fn": wm.get("fn", ""),
                "enclosing": wm.get("enclosing") or "",
                "matched_types": matched_types or matched_vars[:3],
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
        return {"files": [], "consumers": [], "trait_impls": [], "wildcard_matches": [], "cfg_features": [], "cross_file": False}
    rels = [p for p in changed_paths if p and p.endswith(".rs")]
    files: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    impls: list[dict[str, Any]] = []
    cfg_features: list[str] = []
    enum_names: set[str] = set()
    variant_names: set[str] = set()
    for rel in rels[:12]:
        line_numbers = (changed_lines or {}).get(rel) or []
        symbols = (
            _changed_symbols_for_lines(ckg, rel, line_numbers)
            if line_numbers else _changed_symbols(ckg, rel)
        )
        if hasattr(ckg, "features_for_file_lines"):
            for feat in ckg.features_for_file_lines(rel, line_numbers or None):
                if feat not in cfg_features:
                    cfg_features.append(feat)
        top = symbols[:max_symbols_per_file]
        for sym in top:
            if sym.get("kind") == "enum":
                if sym.get("name"):
                    enum_names.add(str(sym["name"]))
                for v in sym.get("variants") or ():
                    if len(str(v)) >= 3:
                        variant_names.add(str(v))
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
    wildcard_matches = _wildcard_match_surfaces(ckg, enum_names, variant_names, limit=10)
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
    consumer_paths = sorted(
        {c.get("path", "") for c in unique_consumers if c.get("path")}
        | {w.get("path", "") for w in wildcard_matches if w.get("path") and w.get("path") not in rels}
    )
    return {
        "files": files,
        "consumers": unique_consumers[: max_consumers * 2],
        "trait_impls": unique_impls[: max_consumers],
        "wildcard_matches": wildcard_matches,
        "cfg_features": cfg_features,
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
    wildcards = surface.get("wildcard_matches") or []
    if wildcards:
        lines.append("Wildcard `_ =>` match sites on modified enums (rustc will NOT warn on new/changed variants):")
        for wm in wildcards[:8]:
            fn_label = f"{wm.get('fn')}()" if wm.get("fn") else "match"
            lines.append(
                f"  * {wm.get('path')}:{wm.get('line')} {fn_label} "
                f"(wildcard match on {', '.join(wm.get('matched_types') or [])})"
            )
    cfgs = surface.get("cfg_features") or []
    if cfgs:
        lines.append(f"#[cfg(feature = ...)] gates enclosing modified items: {', '.join(cfgs)}")
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


_CONTRACT_PANIC_RE = re.compile(
    r'(?:panic!\s*\(|assert!\s*\(|assert_eq!\s*\(|assert_ne!\s*\(|\bexpect\s*\(|#\[should_panic[^]]*\])'
)


_TRAIT_IMPL_HDR_RE = re.compile(r"\bimpl\b(?:<[^>]*>)?\s+([A-Za-z_][A-Za-z0-9_:<>]*)\s+for\s+([A-Za-z_][A-Za-z0-9_:<>]*)")


def _extract_trait_from_enclosing(enclosing: str) -> str:
    m = _TRAIT_IMPL_HDR_RE.search(enclosing or "")
    if not m:
        return ""
    return m.group(1).split("<")[0].split("::")[-1].strip()


def find_dead_enum_variant_producers(
    ckg: Any,
    patch_text: str,
    *,
    requested_symbols: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Detect newly added enum variants in `patch_text` that have zero non-test references while wildcard `_ =>` matches exist."""
    if ckg is None or not patch_text:
        return []
    req_set = {str(s).split("::")[-1].removeprefix("r#") for s in requested_symbols if s}
    try:
        all_files = ckg._all_parsed(exclude_tests=True)
    except Exception:
        return []
    variants_by_file: dict[str, dict[str, str]] = {}
    for pf in all_files:
        for it in pf.items:
            if not it.get("is_test") and it.get("kind") == "enum":
                vmap = variants_by_file.setdefault(pf.rel_path.replace("\\", "/"), {})
                for v in it.get("variants") or ():
                    vmap[str(v)] = str(it.get("name") or "")

    added_variants: list[tuple[str, str, str]] = []
    cur_file = ""
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:].strip()
            continue
        if cur_file not in variants_by_file:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            body = line[1:].strip()
            vm = re.match(r"^([A-Z][A-Za-z0-9_]*)\s*(?:\(|{|,|=|$)", body)
            if vm:
                vname = vm.group(1)
                owner = variants_by_file[cur_file].get(vname)
                if owner and vname not in req_set:
                    added_variants.append((cur_file, vname, owner))
    if not added_variants:
        return []

    findings: list[dict[str, Any]] = []
    for prod_file, vname, owner in added_variants[:6]:
        has_consumer_ref = False
        for pf in all_files:
            for it in pf.items:
                if it.get("is_test"):
                    continue
                if vname in (it.get("type_refs") or ()):
                    has_consumer_ref = True
                    break
            if has_consumer_ref:
                break
        if has_consumer_ref:
            continue
        wms = _wildcard_match_surfaces(ckg, {owner}, {vname}, limit=4)
        if wms:
            findings.append({
                "file": prod_file,
                "variant": vname,
                "enums": [owner],
                "wildcard_matches": [f"{w['path']}:{w['line']}" for w in wms],
            })
    return findings


def collect_sibling_contracts(
    ckg: Any,
    target_symbols: list[str],
    *,
    exclude_paths: set[str] | None = None,
    max_entries: int = 5,
) -> dict[str, Any]:
    """Extract panic/assert contract patterns, trait siblings, and paired methods from CKG."""
    if ckg is None or not target_symbols:
        return {"sibling_impls": [], "test_contracts": [], "trait_docs": [], "paired_methods": []}
    wanted = {
        str(s).split("::")[-1].removeprefix("r#")
        for s in target_symbols
        if str(s).strip() and len(str(s).split("::")[-1]) >= 3
    }
    if not wanted:
        return {"sibling_impls": [], "test_contracts": [], "trait_docs": [], "paired_methods": []}
    try:
        all_files = ckg._all_parsed(exclude_tests=False)
    except Exception:
        return {"sibling_impls": [], "test_contracts": [], "trait_docs": [], "paired_methods": []}

    excluded = {p.replace("\\", "/") for p in (exclude_paths or ())}
    file_lines_cache: dict[str, list[str]] = {}

    def _get_lines(rel_path: str) -> list[str]:
        if rel_path not in file_lines_cache:
            try:
                fpath = (ckg.root / rel_path).resolve()
                file_lines_cache[rel_path] = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                file_lines_cache[rel_path] = []
        return file_lines_cache[rel_path]

    # Identify Trait names associated with wanted methods (e.g., impl BufMut for ...)
    target_traits: set[str] = set()
    for parsed in all_files:
        for item in parsed.items:
            if item.get("is_test") or item.get("kind") != "fn":
                continue
            if str(item.get("name") or "").removeprefix("r#") in wanted:
                tr = _extract_trait_from_enclosing(str(item.get("enclosing") or ""))
                if tr:
                    target_traits.add(tr)

    sibling_impls: list[dict[str, Any]] = []
    test_contracts: list[dict[str, Any]] = []
    trait_docs: list[dict[str, Any]] = []
    paired_methods: list[dict[str, Any]] = []
    seen_sib: set[tuple[str, int]] = set()
    seen_tst: set[tuple[str, int]] = set()
    seen_tdoc: set[tuple[str, int]] = set()
    seen_pair: set[tuple[str, str]] = set()

    for parsed in all_files:
        rel = parsed.rel_path.replace("\\", "/")
        is_test_file = rel.startswith("tests/") or "/tests/" in rel or rel.endswith(("_test.rs", "tests.rs"))
        for item in parsed.items:
            if item.get("kind") != "fn":
                continue
            name = str(item.get("name") or "").removeprefix("r#")
            start = int(item.get("line") or 0)
            end = min(int(item.get("end_line") or start), start + 80)
            if start <= 0:
                continue

            if not item.get("is_test") and not is_test_file:
                enc = str(item.get("enclosing") or "")
                enc_trait = _extract_trait_from_enclosing(enc)
                # Extract trait definition doc/panics for wanted methods on target_traits
                if (
                    enc.startswith(("pub trait ", "trait "))
                    and name in wanted
                    and len(trait_docs) < max_entries
                    and (rel, start) not in seen_tdoc
                ):
                    lines = _get_lines(rel)
                    doc_lines = []
                    for back in range(max(0, start - 20), start - 1):
                        ln_s = lines[back].strip()
                        if ln_s.startswith("///"):
                            doc_lines.append(ln_s[3:].strip())
                    if any(k in "\n".join(doc_lines).lower() for k in ("panic", "error", "safety")):
                        seen_tdoc.add((rel, start))
                        trait_docs.append({
                            "path": rel,
                            "line": start,
                            "symbol": name,
                            "enclosing": enc[:80],
                            "doc": " ".join(doc_lines[-4:])[:240],
                        })

                # Detect companion methods in same impl/file (e.g. foo_opt <-> foo, len <-> is_empty)
                for w in wanted:
                    companion = None
                    if name == f"{w}_opt" or (w.endswith("_opt") and name == w[:-4]):
                        companion = name
                    elif w == "len" and name == "is_empty":
                        companion = name
                    if companion and (rel, companion) not in seen_pair and len(paired_methods) < max_entries:
                        seen_pair.add((rel, companion))
                        paired_methods.append({
                            "path": rel,
                            "line": start,
                            "symbol": companion,
                            "enclosing": enc[:80],
                            "context": str(item.get("context") or "")[:160],
                        })

                if name not in wanted or rel in excluded:
                    continue
                key = (rel, start)
                if key in seen_sib or len(sibling_impls) >= max_entries:
                    continue
                lines = _get_lines(rel)
                if not lines:
                    continue
                snippet_lines = lines[max(0, start - 1):min(len(lines), end)]
                clauses: list[str] = []
                for idx, ln in enumerate(snippet_lines):
                    if _CONTRACT_PANIC_RE.search(ln):
                        block = " ".join(x.strip() for x in snippet_lines[idx:idx + 2])
                        if '"' in block:
                            clauses.append(block[:180])
                    if len(clauses) >= 2:
                        break
                # Also capture compact body summary for trait siblings even without explicit panic string
                if not clauses and enc_trait and enc_trait in target_traits and item.get("has_body"):
                    body_summary = " ".join(x.strip() for x in snippet_lines[:5])[:180]
                    if body_summary:
                        clauses.append(body_summary)
                if clauses:
                    seen_sib.add(key)
                    sibling_impls.append({
                        "path": rel,
                        "line": start,
                        "symbol": name,
                        "enclosing": enc[:100],
                        "clauses": clauses,
                    })
            else:
                if len(test_contracts) >= max_entries:
                    continue
                callees = {str(c).split("::")[-1].removeprefix("r#") for c in (item.get("callees") or ())}
                lines = _get_lines(rel)
                if not lines:
                    continue
                attr_start = max(0, start - 4)
                body_text = "\n".join(lines[attr_start:min(len(lines), end)])
                if not (callees & wanted) and not any(re.search(rf"\b{re.escape(w)}\s*\(", body_text) for w in wanted):
                    continue
                key = (rel, start)
                if key in seen_tst:
                    continue
                hints: list[str] = []
                for ln in lines[attr_start:min(len(lines), end)]:
                    st = ln.strip()
                    if "should_panic" in st or (any(w in st for w in wanted) and ("assert" in st or "panic" in st)):
                        hints.append(st[:180])
                    elif "should_panic" in body_text and '"' in st and ("panic!" in st or "expected" in st or "assert" in st):
                        hints.append(st[:180])
                    if len(hints) >= 2:
                        break
                if hints:
                    seen_tst.add(key)
                    test_contracts.append({
                        "path": rel,
                        "line": start,
                        "test_name": name,
                        "hints": hints,
                    })

    # Surface sibling std-wrapper trait impls (e.g. impl Serialize/Deserialize for Wrapping<T>)
    wrapper_types = {
        "Wrapping", "Reverse", "ManuallyDrop", "Cell", "RefCell", "OnceCell", "Pin", "PhantomData",
        "NonZeroU8", "NonZeroU16", "NonZeroU32", "NonZeroU64", "NonZeroU128", "NonZeroUsize",
        "NonZeroI8", "NonZeroI16", "NonZeroI32", "NonZeroI64", "NonZeroI128", "NonZeroIsize",
    }
    trait_candidates = target_traits | {w for w in wanted if w[:1].isupper()}
    if trait_candidates and len(sibling_impls) < max_entries:
        for parsed in all_files:
            rel = parsed.rel_path.replace("\\", "/")
            if rel in excluded or rel.startswith("tests/") or "/tests/" in rel or rel.endswith(("_test.rs", "tests.rs")):
                continue
            for ib in parsed.impl_blocks:
                if ib.get("is_test"):
                    continue
                tr = str(ib.get("trait") or "").split("::")[-1].removeprefix("r#")
                ft = str(ib.get("for_type") or "").split("::")[-1].removeprefix("r#")
                ib_line = int(ib.get("line") or 0)
                if tr in trait_candidates and ft in wrapper_types and ib_line > 0 and (rel, ib_line) not in seen_sib:
                    lines = _get_lines(rel)
                    excerpt = " ".join(x.strip() for x in lines[max(0, ib_line - 3):min(len(lines), ib_line + 3)] if x.strip())[:180]
                    if excerpt:
                        seen_sib.add((rel, ib_line))
                        sibling_impls.append({
                            "path": rel,
                            "line": ib_line,
                            "symbol": f"{tr} for {ft}",
                            "enclosing": str(ib.get("header") or "")[:100],
                            "clauses": [excerpt],
                        })
                        if len(sibling_impls) >= max_entries:
                            break
            if len(sibling_impls) >= max_entries:
                break

    return {
        "sibling_impls": sibling_impls,
        "test_contracts": test_contracts,
        "trait_docs": trait_docs,
        "paired_methods": paired_methods,
    }


def format_sibling_contracts(contracts: dict[str, Any] | None, *, max_chars: int = 1600) -> str:
    """Format sibling implementation panic/assert wording, trait docs, paired methods, and test contracts."""
    if not contracts:
        return ""
    sibs = contracts.get("sibling_impls") or []
    tests = contracts.get("test_contracts") or []
    tdocs = contracts.get("trait_docs") or []
    paired = contracts.get("paired_methods") or []
    if not sibs and not tests and not tdocs and not paired:
        return ""
    lines: list[str] = ["Sibling & Existing Test Contracts (match established error/panic wording and invariants):"]
    for td in tdocs:
        lines.append(f"- {td['path']}:{td['line']} trait `{td['symbol']}` ({td['enclosing']}): {td['doc']}")
    for s in sibs:
        enc = f" in `{s['enclosing']}`" if s.get("enclosing") else ""
        lines.append(f"- {s['path']}:{s['line']} `{s['symbol']}`{enc}: {' | '.join(s['clauses'])}")
    for pm in paired:
        enc = f" in `{pm['enclosing']}`" if pm.get("enclosing") else ""
        lines.append(f"- Companion API {pm['path']}:{pm['line']} `{pm['symbol']}`{enc}: {pm['context']}")
    for t in tests:
        lines.append(f"- {t['path']}:{t['line']} test `{t['test_name']}`: {' | '.join(t['hints'])}")
    out = "\n".join(lines)
    return out[:max_chars] + ("\n... (truncated)" if len(out) > max_chars else "")
