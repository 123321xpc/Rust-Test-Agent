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
            ("file_name", "line_start", "line_end", "column_start", "column_end", "is_primary", "label", "suggested_replacement"),
        )
        for span in message.get("spans", [])
    ]

    category = classify_diagnostic(message)
    code = (message.get("code") or {}).get("code")
    level = message.get("level")
    hint = get_remediation_hint(category, code, message.get("message")) if level == "error" else None

    suggested_fix = None
    for span in message.get("spans", []):
        if span.get("suggested_replacement") and span.get("is_primary"):
            suggested_fix = span.get("suggested_replacement")
            break
    if suggested_fix is None:
        for child in message.get("children", []):
            for span in child.get("spans", []):
                if span.get("suggested_replacement"):
                    suggested_fix = span.get("suggested_replacement")
                    break
            if suggested_fix is not None:
                break

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


def make_cargo_check_tool(root, resolve_path, env=None, on_result=None):
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
            attach_diagnostic_snippets(kept, root, manifest_dir=manifest.parent)

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
