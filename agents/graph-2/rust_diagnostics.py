import importlib.util
"""Run Cargo and classify structured rustc diagnostics for the test agent."""

import json
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool

# Error codes take precedence over wording, which can change across compiler versions.
_CODES = {
    "ownership_move": {"E0382", "E0507", "E0508", "E0509"},
    "borrow_conflict": {"E0499", "E0500", "E0501", "E0502", "E0503", "E0505", "E0506", "E0596"},
    "lifetime": {"E0106", "E0310", "E0311", "E0495", "E0515", "E0521", "E0597", "E0621"},
    "trait_bound": {"E0277", "E0276", "E0271"},
    "type_mismatch": {"E0308"},
}


def classify_diagnostic(message):
    """Return a category for a rustc diagnostic; leave unrelated errors as other."""
    code = (message.get("code") or {}).get("code")
    for category, codes in _CODES.items():
        if code in codes:
            return category
    text = message.get("message", "").lower()
    if "lifetime" in text or "does not live long enough" in text:
        return "lifetime"
    if "trait bound" in text or "trait is not implemented" in text:
        return "trait_bound"
    if "mismatched types" in text or "type mismatch" in text:
        return "type_mismatch"
    if "borrow" in text and ("more than once" in text or "also borrowed" in text or "already borrowed" in text):
        return "borrow_conflict"
    if "moved value" in text or "value moved" in text or "move out of" in text:
        return "ownership_move"
    return "other"


_REMEDIATION_HINTS = {
    "ownership_move": "Value was moved. Consider borrowing (&/&mut), cloning (.clone()) if Clone is implemented, or restructuring usage before the move.",
    "borrow_conflict": "Conflicting mutable/immutable borrow. Try narrowing borrow scope, separating read/write operations, or using interior mutability if needed.",
    "lifetime": "Lifetime mismatch or reference outlives referent. Check if an owned value should be stored, or adjust lifetime annotations on the struct/function.",
    "trait_bound": "Required trait bound is unsatisfied. Check if the type needs deriving/implementing the trait, an in-scope import (`use ...`), or a where clause.",
    "type_mismatch": "Types do not match. Check expected vs found types, Result/Option unwrapping (.unwrap()/?), or conversions (.into(), as, deref).",
}

_CODE_HINTS = {
    "E0599": "No method found on type. Check if the required trait is in scope (`use ...`), or if a deref/conversion is missing.",
    "E0425": "Unresolved name. Check `use` imports, typo, or feature flags.",
    "E0433": "Unresolved module or crate. Check `use` path, extern crate, or Cargo.toml dependencies.",
    "E0432": "Unresolved import. Verify the crate name, module path, or re-exports in src/lib.rs, and check if a feature flag is needed in Cargo.toml.",
    "E0412": "Type name not in scope. Check if `use ...` is missing or if the type requires a generic parameter.",
    "E0405": "Trait not in scope. Check if `use ...` is needed to bring the trait into scope before using or implementing it.",
    "E0753": "Inner doc comment `//!` or attribute `#![...]` used in an invalid context. Inner attributes must appear at the top of the file before other items.",
    "E0282": "Type cannot be inferred. Add an explicit type annotation (e.g. `let x: TargetType = ...` or `::<Type>`).",
    "E0061": "Mismatched number of function arguments. Inspect the function signature with rust_symbols(name=..., kind='fn').",
    "E0407": "Method definition not in trait. Verify method name spelling and check the trait definition.",
    "E0616": "Field is private. Use public accessor methods or add visibility in the defining module.",
    "E0624": "Associated item is private. Check item visibility or use public constructor/methods.",
}


def get_remediation_hint(category: str, code: str | None, message: str | None) -> str | None:
    if code in _CODE_HINTS:
        return _CODE_HINTS[code]
    return _REMEDIATION_HINTS.get(category)


def _compact_span(span, keys):
    out = {}
    for k in keys:
        val = span.get(k)
        if val is None or (k == "is_primary" and not val) or (k == "label" and val == ""):
            continue
        out[k] = val
    return out


def _is_redundant_child(child_msg: str) -> bool:
    low = (child_msg or "").lower()
    return (
        "for more information about this error, try `rustc --explain" in low
        or "some errors have detailed explanations:" in low
    )


def _simplify(message):
    spans = [
        _compact_span(
            span,
            ("file_name", "line_start", "line_end", "column_start", "column_end", "is_primary", "label", "suggested_replacement", "suggestion_applicability"),
        )
        for span in message.get("spans", [])
    ]

    category = classify_diagnostic(message)
    code = (message.get("code") or {}).get("code")
    level = message.get("level")
    hint = get_remediation_hint(category, code, message.get("message")) if level == "error" else None

    suggested_fix = None
    machine_applicable_fix = None
    for span in message.get("spans", []):
        rep = span.get("suggested_replacement")
        if rep is not None and rep != "":
            if span.get("is_primary") and suggested_fix is None:
                suggested_fix = rep
            if span.get("suggestion_applicability") == "MachineApplicable" and machine_applicable_fix is None:
                machine_applicable_fix = {
                    "file": span.get("file_name"),
                    "line_start": span.get("line_start"),
                    "line_end": span.get("line_end"),
                    "column_start": span.get("column_start"),
                    "column_end": span.get("column_end"),
                    "replacement": rep,
                }
    if suggested_fix is None or machine_applicable_fix is None:
        for child in message.get("children", []):
            for span in child.get("spans", []):
                rep = span.get("suggested_replacement")
                if rep is not None and rep != "":
                    if suggested_fix is None:
                        suggested_fix = rep
                    if span.get("suggestion_applicability") == "MachineApplicable" and machine_applicable_fix is None:
                        machine_applicable_fix = {
                            "file": span.get("file_name"),
                            "line_start": span.get("line_start"),
                            "line_end": span.get("line_end"),
                            "column_start": span.get("column_start"),
                            "column_end": span.get("column_end"),
                            "replacement": rep,
                        }

    children = []
    for child in message.get("children", []):
        c_msg = child.get("message") or ""
        if _is_redundant_child(c_msg) and not child.get("spans"):
            continue
        c_spans = [
            _compact_span(s, ("file_name", "line_start", "column_start", "suggested_replacement", "suggestion_applicability"))
            for s in child.get("spans", [])
        ]
        c_entry = {"level": child.get("level"), "message": child.get("message")}
        if c_spans:
            c_entry["spans"] = c_spans
        children.append(c_entry)

    payload = {
        "category": category,
        "level": level,
        "code": code,
        "message": message.get("message"),
        "spans": spans,
        "children": children,
    }
    if hint:
        payload["remediation_hint"] = hint
    if suggested_fix:
        payload["suggested_fix"] = suggested_fix
    if machine_applicable_fix:
        payload["machine_applicable_fix"] = machine_applicable_fix
    return payload


def _is_summary_diagnostic(diag):
    if diag.get("code") or diag.get("spans"):
        return False
    text = (diag.get("message") or "").lower()
    return any(phrase in text for phrase in ("aborting due to", "warning emitted", "warnings emitted", "previous error"))


def compact_diagnostics(diagnostics, max_warnings=10, max_errors=12):
    """Prioritize error diagnostics, deduplicate exact copies, cluster cascading errors,
    and cap repetitive warnings to save context."""
    non_summary = [d for d in diagnostics if not _is_summary_diagnostic(d)]
    items = non_summary if non_summary else list(diagnostics)
    has_errors = any(d.get("level") == "error" for d in items)
    warning_budget = min(max_warnings, 5) if has_errors else max_warnings
    error_count = 0
    warning_count = 0
    kept = []
    omitted = 0

    seen_exact = set()
    cluster_counts = {}
    cluster_first = {}
    seen_hints = set()

    for diag in items:
        level = diag.get("level")
        code = diag.get("code")
        msg_text = diag.get("message") or ""
        spans = diag.get("spans") or []
        first_span = spans[0] if spans else {}
        file_name = first_span.get("file_name") or ""
        line_start = first_span.get("line_start")
        col_start = first_span.get("column_start")

        exact_key = (level, code, msg_text, file_name, line_start, col_start)
        if exact_key in seen_exact:
            omitted += 1
            continue
        seen_exact.add(exact_key)

        if level == "error":
            cluster_key = (code, msg_text) if (code or msg_text) else None
            if cluster_key is not None:
                cur_count = cluster_counts.get(cluster_key, 0)
                if cur_count >= 2:
                    rep = cluster_first.get(cluster_key)
                    if rep is not None and file_name and line_start:
                        loc = f"{file_name}:{line_start}"
                        also_at = rep.setdefault("also_at", [])
                        if len(also_at) < 8 and loc not in also_at:
                            also_at.append(loc)
                    omitted += 1
                    cluster_counts[cluster_key] = cur_count + 1
                    continue
                cluster_counts[cluster_key] = cur_count + 1

            if error_count < max_errors:
                entry = dict(diag)
                hint_key = (entry.get("category"), entry.get("code"))
                if hint_key in seen_hints:
                    entry.pop("remediation_hint", None)
                elif entry.get("remediation_hint"):
                    seen_hints.add(hint_key)
                kept.append(entry)
                error_count += 1
                if cluster_key is not None and cluster_key not in cluster_first:
                    cluster_first[cluster_key] = entry
            else:
                omitted += 1
        else:
            if warning_count < warning_budget:
                kept.append(dict(diag))
                warning_count += 1
            else:
                omitted += 1
    return kept, omitted


def clean_cargo_stderr(stderr: str, max_chars: int = 2000) -> str:
    """Strip repetitive Cargo compilation progress lines while preserving actual errors/warnings."""
    if not stderr:
        return ""
    lines = stderr.splitlines()
    filtered = [
        line for line in lines
        if not re.match(r"^\s*(?:Compiling|Checking|Downloading|Downloaded|Updating|Fresh|Locking)\s+", line)
    ]
    cleaned = "\n".join(filtered).strip()
    if not cleaned:
        cleaned = "\n".join(lines[-6:]).strip()
    return cleaned[-max_chars:]


def attach_diagnostic_snippets(diagnostics, root, manifest_dir=None, context_lines=2, max_total_chars=4000):
    """Inline source code snippets around primary and child spans for error diagnostics."""
    root_path = Path(root).resolve()
    file_cache = {}
    total_chars = 0

    def get_file_lines(file_path: Path):
        if file_path not in file_cache:
            try:
                file_cache[file_path] = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                file_cache[file_path] = None
        return file_cache[file_path]

    def resolve_span_file(fname: str) -> Path | None:
        if not fname or fname.startswith("<"):
            return None
        p = Path(fname)
        candidates = []
        if p.is_absolute():
            candidates.append(p.resolve())
        else:
            candidates.append((root_path / p).resolve())
            if manifest_dir is not None:
                candidates.append((Path(manifest_dir).resolve() / p).resolve())
        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.is_relative_to(root_path):
                    if not any(part.lower() == ".git" for part in candidate.relative_to(root_path).parts):
                        return candidate
            except OSError:
                continue
        return None

    for diag in diagnostics:
        if diag.get("level") != "error" or total_chars >= max_total_chars:
            continue

        seen_windows = set()
        spans_to_enrich = []
        # Primary spans first, then other direct spans, then up to 2 child spans
        primary = [s for s in diag.get("spans", []) if s.get("is_primary")]
        secondary = [s for s in diag.get("spans", []) if not s.get("is_primary")]
        spans_to_enrich.extend(primary + secondary)
        child_added = 0
        for child in diag.get("children", []):
            for cs in child.get("spans", []):
                if cs.get("file_name") and cs.get("line_start"):
                    spans_to_enrich.append(cs)
                    child_added += 1
                    break
            if child_added >= 2:
                break

        for span in spans_to_enrich:
            if total_chars >= max_total_chars or "snippet" in span:
                continue
            fname = span.get("file_name")
            l_start = span.get("line_start")
            l_end = span.get("line_end") or l_start
            if not fname or not isinstance(l_start, int) or l_start < 1:
                continue
            target_file = resolve_span_file(fname)
            if target_file is None:
                continue
            lines = get_file_lines(target_file)
            if not lines:
                continue
            total_lines = len(lines)
            if l_start > total_lines:
                continue
            w_start = max(1, l_start - context_lines)
            w_end = min(total_lines, l_end + context_lines)
            if w_end - w_start >= 12:
                w_end = min(total_lines, w_start + 11)

            win_key = (target_file, w_start, w_end)
            if win_key in seen_windows:
                continue
            seen_windows.add(win_key)

            snippet_lines = []
            for ln in range(w_start, w_end + 1):
                marker = ">" if (l_start <= ln <= l_end) else " "
                text = lines[ln - 1]
                if len(text) > 180:
                    text = text[:177] + "..."
                snippet_lines.append(f"{marker} {ln:4d}: {text}")

            snippet = "\n".join(snippet_lines)
            if snippet:
                span["snippet"] = snippet
                total_chars += len(snippet)

        maf = diag.get("machine_applicable_fix")
        if isinstance(maf, dict) and "old_text" not in maf:
            mfname = maf.get("file")
            mls = maf.get("line_start")
            mle = maf.get("line_end") or mls
            mcs = maf.get("column_start")
            mce = maf.get("column_end")
            rep = maf.get("replacement")
            mfile = resolve_span_file(mfname) if mfname else None
            if mfile is not None and isinstance(mls, int) and isinstance(mle, int) and 1 <= mls <= mle and rep is not None:
                mlines = get_file_lines(mfile) or []
                if mle <= len(mlines):
                    old_slice = mlines[mls - 1:mle]
                    old_text = "\n".join(old_slice)
                    if mls == mle and isinstance(mcs, int) and isinstance(mce, int) and 1 <= mcs <= mce:
                        line_txt = old_slice[0]
                        if mce - 1 <= len(line_txt):
                            new_text = line_txt[:mcs - 1] + str(rep) + line_txt[mce - 1:]
                            if old_text != new_text:
                                maf["old_text"] = old_text
                                maf["new_text"] = new_text
                    elif isinstance(rep, str) and old_text and len(old_slice) <= 6:
                        maf["old_text"] = old_text
                        maf["new_text"] = rep



_SHARED_DIAG_MASK_CACHE: dict[Path, dict] = {}


def enrich_diagnostic_symbol_lookup(
    diagnostics: list[dict],
    root: Path,
    symbols_module=None,
    ckg=None,
    manifest_dir: Path | None = None,
    mask_cache: dict | None = None,
    time_budget_sec: float = 5.0,
    max_diags: int = 5,
):
    """Enrich compiler error diagnostics with cached, deadline-bounded symbol lookup and guidance."""
    import time as _time

    if not diagnostics or not root:
        return
    if time_budget_sec is not None and time_budget_sec <= 0:
        return
    if symbols_module is None:
        try:
            spec = importlib.util.spec_from_file_location("rust_symbols_lookup", Path(__file__).with_name("rust_symbols.py"))
            symbols_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(symbols_module)
        except Exception:
            return

    resolved_root = Path(root).resolve()
    if mask_cache is None:
        mask_cache = _SHARED_DIAG_MASK_CACHE.setdefault(resolved_root, {})
    deadline = _time.monotonic() + time_budget_sec if time_budget_sec and time_budget_sec > 0 else None
    enriched_count = 0
    seen_queries: set[tuple[str, str]] = set()

    for diag in diagnostics:
        if diag.get("level") != "error":
            continue
        if enriched_count >= max_diags:
            break
        if deadline is not None and _time.monotonic() >= deadline:
            break
        code = diag.get("code")
        msg = diag.get("message") or ""
        lookup_info = []
        sym = None
        fn_name = None

        # 1. E0599: No method found on type
        if code == "E0599":
            m = re.search(r"(?:no method|no function or associated item) named `([^`]+)` found for (?:type|struct|enum|trait|reference|union) `([^`]+)`", msg)
            if m:
                fn_name, ty_name = m.group(1).strip(), m.group(2).strip()
                if (code, fn_name) in seen_queries:
                    continue
                seen_queries.add((code, fn_name))
                enriched_count += 1
                matches = symbols_module.lookup_symbol_details(
                    root, name=fn_name, kind="fn", exact=True, max_results=3,
                    mask_cache=mask_cache, priority_dir=manifest_dir, deadline=deadline,
                )
                inferred_trait = None
                if matches:
                    top = matches[0]
                    enclosing = top.get("enclosing")
                    vis = top.get("vis", "private")
                    loc = f"{top['path']}:{top['line']}"
                    if enclosing:
                        m_tr = re.search(r"\btrait\s+([A-Za-z_]\w*)|\bimpl\b(?:<[^>]*>)?\s+([A-Za-z_]\w*)\s+for\b", enclosing)
                        if m_tr:
                            inferred_trait = m_tr.group(1) or m_tr.group(2)
                        lookup_info.append(
                            f"[Auto-Lookup: Method '{fn_name}' is defined in '{enclosing}' at {loc} (vis: {vis}). "
                            f"Ensure the enclosing trait/type is imported or check receiver type.]"
                        )
                    else:
                        lookup_info.append(
                            f"[Auto-Lookup: Function/method '{fn_name}' is defined at {loc} (vis: {vis}). "
                            f"Check if receiver type '{ty_name}' matches the definition.]"
                        )
                if inferred_trait and ckg is not None and (deadline is None or _time.monotonic() < deadline):
                    try:
                        nb_tr = ckg.query_neighborhood(inferred_trait, kind="trait", exclude_tests=True)
                        tr_paths = [
                            pp
                            for d_item in nb_tr.get("definitions", [])[:2]
                            for pp in d_item.get("public_paths", [])
                        ]
                        if tr_paths:
                            lookup_info.append(
                                f"[Auto-Lookup CKG: Trait import path(s) for '{inferred_trait}': "
                                + ", ".join(f"use {p};" for p in list(dict.fromkeys(tr_paths))[:3])
                                + "]"
                            )
                    except Exception:
                        pass
                if ty_name.startswith(("Option<", "Result<", "Box<", "Arc<", "Rc<", "&mut ")):
                    lookup_info.append(
                        f"[Auto-Lookup: Receiver '{ty_name}' may need unwrapping (?, .as_ref(), .as_deref()) or dereferencing before calling '{fn_name}'.]"
                    )

        # 2. E0425, E0412, E0405, E0432, E0433: Unresolved names, types, traits, imports, modules
        elif code in ("E0425", "E0412", "E0405", "E0432", "E0433"):
            sym = None
            if code == "E0425":
                m = re.search(r"cannot find (?:value|function|macro) `([^`]+)` in this scope", msg)
                if m: sym = m.group(1).strip()
            elif code == "E0412":
                m = re.search(r"cannot find type `([^`]+)` in this scope", msg)
                if m: sym = m.group(1).strip()
            elif code == "E0405":
                m = re.search(r"cannot find trait `([^`]+)` in this scope", msg)
                if m: sym = m.group(1).strip()
            elif code == "E0432":
                m = re.search(r"unresolved import `([^`]+)`", msg)
                if m:
                    parts = m.group(1).strip().split("::")
                    sym = parts[-1].strip("{}")
            elif code == "E0433":
                m = re.search(r"failed to resolve: use of undeclared (?:crate or module|type) `([^`]+)`", msg)
                if m: sym = m.group(1).strip()

            if sym and re.match(r"^[A-Za-z_]\w*$", sym):
                if (code, sym) in seen_queries:
                    continue
                seen_queries.add((code, sym))
                enriched_count += 1
                matches = symbols_module.lookup_symbol_details(
                    root, name=sym, exact=True, max_results=3,
                    mask_cache=mask_cache, priority_dir=manifest_dir, deadline=deadline,
                )
                if matches:
                    top = matches[0]
                    vis = top.get("vis", "private")
                    kind = top.get("kind", "item")
                    loc = f"{top['path']}:{top['line']}"
                    vis_hint = " Note: pub(crate) items cannot be accessed from external tests/ (use an in-module test or re-export)." if vis == "pub(crate)" else ""
                    lookup_info.append(
                        f"[Auto-Lookup: '{sym}' ({kind}) is defined at {loc} (vis: {vis}).{vis_hint}]"
                    )
                reexports = symbols_module.find_reexports(
                    root, symbol_name=sym, max_results=1,
                    mask_cache=mask_cache, priority_dir=manifest_dir, deadline=deadline,
                )
                if reexports:
                    r_top = reexports[0]
                    lookup_info.append(
                        f"[Auto-Lookup: '{sym}' is re-exported at {r_top['path']}:{r_top['line']} ('{r_top['reexport']}').]"
                    )

        # 3. E0277: Trait bound not satisfied
        elif code == "E0277":
            m = re.search(r"the trait bound `([^:]+): ([^`]+)` is not satisfied", msg)
            if m:
                ty_name, trait_name = m.group(1).strip(), m.group(2).strip()
                base_trait = trait_name.split("<")[0].split("::")[-1].strip()
                if re.match(r"^[A-Za-z_]\w*$", base_trait):
                    if (code, base_trait) in seen_queries:
                        continue
                    seen_queries.add((code, base_trait))
                    enriched_count += 1
                    impls = symbols_module.find_trait_implementations(
                        root, trait_name=base_trait, max_results=2,
                        mask_cache=mask_cache, priority_dir=manifest_dir, deadline=deadline,
                    )
                    if impls:
                        top = impls[0]
                        lookup_info.append(
                            f"[Auto-Lookup: Trait '{base_trait}' has existing implementation at {top['path']}:{top['line']} ('{top['header']}'). "
                            f"Check if '{ty_name}' needs implementing '{base_trait}' or deriving it.]"
                        )

        if ckg is not None and code in ("E0425", "E0412", "E0405", "E0432", "E0433", "E0599") and (deadline is None or _time.monotonic() < deadline):
            target_sym = sym or fn_name
            if target_sym and re.match(r"^[A-Za-z_]\w*$", target_sym):
                try:
                    nb = ckg.query_neighborhood(target_sym, exclude_tests=True)
                    pub_paths = []
                    for d_item in nb.get("definitions", [])[:2]:
                        for pp in d_item.get("public_paths", []):
                            if pp not in pub_paths:
                                pub_paths.append(pp)
                    if pub_paths:
                        use_hint = f"[Auto-Lookup CKG: Public import path(s) for '{target_sym}': " + ", ".join(f"use {p};" for p in pub_paths[:3]) + "]"
                        if use_hint not in lookup_info:
                            lookup_info.append(use_hint)
                except Exception:
                    pass

        if lookup_info:
            full_lookup = chr(10).join(lookup_info)
            diag["auto_lookup"] = full_lookup
            existing_hint = diag.get("remediation_hint")
            if existing_hint:
                diag["remediation_hint"] = f"{existing_hint}\n{full_lookup}"
            else:
                diag["remediation_hint"] = full_lookup


def attach_snippets_and_lookup(diagnostics, root, manifest_dir=None, ckg=None, mask_cache=None):
    """Inline source snippets and run automated symbol/CKG lookup for compiler error diagnostics."""
    attach_diagnostic_snippets(diagnostics, root, manifest_dir=manifest_dir)
    if any(d.get("level") == "error" for d in diagnostics):
        enrich_diagnostic_symbol_lookup(
            diagnostics, Path(root), ckg=ckg, manifest_dir=manifest_dir, mask_cache=mask_cache,
        )


def make_cargo_check_tool(root, resolve_path, env=None, on_result=None, ckg=None, mask_cache=None):
    """Build a Cargo check tool for the agent's isolated working copy."""
    @tool
    def cargo_check(manifest_path: str = "Cargo.toml", package: str = "",
                    features: list[str] | None = None, all_features: bool = False,
                    no_default_features: bool = False, target: str = "",
                    tests: bool = False, all_targets: bool = False) -> str:
        """Check Rust code and classify compiler diagnostics.

        manifest_path is repository-relative; package selects a workspace package.
        features is a list of feature names; all_features enables every feature,
        and no_default_features disables defaults. target selects a Rust target
        triple. tests checks test targets; all_targets checks all targets. Select
        at most one of tests and all_targets.
        """
        try:
            if tests and all_targets:
                return "Error: tests and all_targets cannot both be selected."
            if features is not None and any(not feature.strip() or feature.startswith("-") for feature in features):
                return "Error: features must contain non-empty feature names."
            manifest = resolve_path(manifest_path)
            if not manifest.is_file() or manifest.name != "Cargo.toml":
                return "Error: manifest_path must point to a Cargo.toml file in the working copy."
            command = ["cargo", "check", "--message-format=json", "--manifest-path", str(manifest)]
            if (root / "Cargo.lock").is_file() or (manifest.parent / "Cargo.lock").is_file():
                command.append("--locked")
            if package:
                command.append(f"--package={package}")
            if features:
                command.append("--features=" + ",".join(features))
            if all_features:
                command.append("--all-features")
            if no_default_features:
                command.append("--no-default-features")
            if target:
                command.append(f"--target={target}")
            if tests:
                command.append("--tests")
            if all_targets:
                command.append("--all-targets")
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, env=env)
            diagnostics = []
            for line in result.stdout.splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("reason") == "compiler-message" and isinstance(item.get("message"), dict):
                    diagnostics.append(_simplify(item["message"]))
            kept, omitted = compact_diagnostics(diagnostics)
            attach_snippets_and_lookup(kept, root, manifest_dir=manifest.parent, ckg=ckg, mask_cache=mask_cache)

            has_errors = any(d.get("level") == "error" for d in kept)
            if result.returncode == 0:
                cargo_output = clean_cargo_stderr(result.stderr, max_chars=1000)
            elif has_errors:
                cargo_output = clean_cargo_stderr(result.stderr, max_chars=2000)
            else:
                cargo_output = result.stderr[-8000:]

            payload = {"exit_code": result.returncode, "diagnostics": kept,
                       "cargo_output": cargo_output}
            if omitted:
                payload["diagnostics_omitted"] = omitted
            if on_result:
                strategy_warning = on_result(result.returncode, kept)
                if strategy_warning:
                    payload["strategy_warning"] = strategy_warning
            return json.dumps(payload, ensure_ascii=False)
        except subprocess.TimeoutExpired:
            return "Error: cargo check timed out after 120 seconds."
        except (OSError, ValueError) as exc:
            return f"Error running cargo check: {exc}"

    return cargo_check
