"""File, Rust investigation, and patch tools for the test baseline."""

import importlib.util
import json
from functools import wraps
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from pathlib import Path

from langchain_core.tools import tool


def _manifest_has_targets(manifest_file: Path) -> bool:
    if not manifest_file.is_file():
        return False
    pkg_dir = manifest_file.parent
    if (pkg_dir / "src").is_dir():
        return True
    try:
        content = manifest_file.read_text(encoding="utf-8", errors="replace")
        if "[workspace]" in content or "[lib]" in content or "[[bin]]" in content:
            return True
    except Exception:
        pass
    return False


def _append_cargo_feature_flags(command: list[str], config: dict | None = None) -> None:
    config = config or {}
    if config.get("features"):
        command.append("--features=" + ",".join(config["features"]))
    use_all_features = os.getenv("RUSTBENCH_ALL_FEATURES", "1").lower() not in {"0", "false", "off"}
    if config.get("all_features") or (use_all_features and not config.get("no_default_features")):
        command.append("--all-features")
    if config.get("no_default_features"):
        command.append("--no-default-features")
    if config.get("target"):
        command.append(f"--target={config['target']}")


def inline_test_line_ranges(orig_code: str) -> list[tuple[int, int]]:
    """Return 1-based (start, end) line ranges of #[cfg(test)] or mod tests blocks in orig_code."""
    lines = orig_code.splitlines()
    ranges: list[tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        is_cfg_test = stripped.startswith("#[cfg(test)]")
        is_mod_tests = bool(re.match(r"^(?:pub(?:\([^)]*\))?\s+)?mod\s+(?:tests?|[a-z0-9_]*_tests)\b", stripped))
        if is_cfg_test or is_mod_tests:
            start_ln = i + 1
            depth = 0
            opened = False
            j = i
            while j < n:
                ln_txt = lines[j]
                if "{" in ln_txt:
                    depth += ln_txt.count("{")
                    opened = True
                if "}" in ln_txt:
                    depth -= ln_txt.count("}")
                if opened and depth <= 0:
                    ranges.append((start_ln, j + 1))
                    i = j
                    break
                if not opened and ";" in ln_txt and j > i:
                    break
                j += 1
            else:
                if opened:
                    ranges.append((start_ln, n))
        i += 1
    return ranges


def touched_line_hits_inline_tests(touched: tuple[tuple[str, int], ...], ranges: list[tuple[int, int]]) -> bool:
    """Return True when a hunk's real edits touch an inline #[cfg(test)]/mod tests region."""
    for kind, number in touched:
        for start, end in ranges:
            if kind == "line":
                if start <= number <= end:
                    return True
            elif start <= number <= end - 1:
                return True
    return False


def _is_non_test_rs_path(path_str: str) -> bool:
    norm = path_str.replace("\\", "/").lower()
    if not norm.endswith(".rs"):
        return False
    if norm.startswith(("tests/", "benches/", "examples/")):
        return False
    if "/tests/" in norm or "/benches/" in norm or "/examples/" in norm:
        return False
    return not norm.endswith(("_test.rs", "tests.rs"))


def _edited_inline_test_warning(orig_code: str, updated_code: str, path_str: str) -> str:
    """Warn immediately when an edit touches a pre-existing inline #[cfg(test)]/mod tests region."""
    if not _is_non_test_rs_path(path_str):
        return ""
    ranges = inline_test_line_ranges(orig_code)
    if not ranges:
        return ""
    import difflib
    clean_updated = _strip_in_module_tests(updated_code)
    touched: list[tuple[str, int]] = []
    matcher = difflib.SequenceMatcher(None, orig_code.splitlines(), clean_updated.splitlines(), autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            touched.append(("gap", i1))
        else:
            for ln in range(i1 + 1, i2 + 1):
                touched.append(("line", ln))
    if touched and touched_line_hits_inline_tests(tuple(touched), ranges):
        span_str = ", ".join(f"{s}-{e}" for s, e in ranges[:3])
        return (
            f"\n[Inline-Test Isolation Warning in {path_str}]: edit touches pre-existing #[cfg(test)]/mod tests lines ({span_str}). "
            "Patch submission blocks edits to pre-existing inline tests; leave existing inline tests untouched "
            "and put regression tests in write_regression_test."
        )
    return ""


def _detect_inline_test_patch_conflicts(patch_text: str, originals: dict) -> list[str]:
    """Flag hunks in production .rs files that overlap pre-existing inline #[cfg(test)]/mod tests regions."""
    if not patch_text:
        return []
    findings: list[str] = []
    current = ""
    old_path = ""
    old_line = 0
    touched: list[tuple[str, int]] = []
    has_change = False

    def _flush():
        if not current or not has_change or not touched or not _is_non_test_rs_path(current):
            return
        orig = originals.get(Path(current))
        if not orig or orig[0] is None:
            return
        ranges = inline_test_line_ranges(orig[0].decode("utf-8", errors="replace"))
        if ranges and touched_line_hits_inline_tests(tuple(touched), ranges):
            tag = f"{current}: hunk overlaps a pre-existing inline #[cfg(test)]/mod tests region"
            if tag not in findings:
                findings.append(tag)

    for line in patch_text.splitlines():
        if line.startswith("--- "):
            old_path = line[4:].split("\t", 1)[0].strip()
        elif line.startswith("+++ "):
            _flush()
            touched = []
            has_change = False
            new_path = line[4:].split("\t", 1)[0].strip()
            raw = new_path if new_path != "/dev/null" else old_path
            if raw.startswith(("a/", "b/")):
                raw = raw[2:]
            current = raw
        elif line.startswith("@@ "):
            _flush()
            touched = []
            has_change = False
            m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+", line)
            old_line = int(m.group(1)) if m else 0
        elif line.startswith("-") and not line.startswith("---"):
            has_change = True
            touched.append(("line", old_line))
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            has_change = True
            gap = old_line - 1
            if ("gap", gap) not in touched:
                touched.append(("gap", gap))
        elif line.startswith(" "):
            old_line += 1
    _flush()
    return findings[:6]


def _format_line_window(text: str, center_line: int, radius: int = 3) -> str:
    lines = text.splitlines()
    if not lines:
        return "(empty)"
    start = max(1, center_line - radius)
    end = min(len(lines), center_line + radius)
    return "\n".join(
        f"{'>' if ln == center_line else ' '} {ln}: {lines[ln - 1]}"
        for ln in range(start, end + 1)
    )


def _find_unclosed_noncode(source: str) -> tuple[str, int] | None:
    """Return (description, 1-based line) if a comment or string/raw-string literal is unclosed at EOF."""
    n, i = len(source), 0
    while i < n:
        if source.startswith("//", i):
            end = source.find("\n", i)
            i = n if end < 0 else end + 1
        elif source.startswith("/*", i):
            start_line = source.count("\n", 0, i) + 1
            i += 2
            depth = 1
            while i < n and depth:
                if source.startswith("/*", i):
                    depth += 1
                    i += 2
                elif source.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth > 0:
                return ("unclosed block comment '/*'", start_line)
        else:
            raw = (
                re.match(r'(?:br|cr|r)(#*)"', source[i:i + 260])
                if source[i] in "bcr" and (i == 0 or not (source[i - 1].isalnum() or source[i - 1] == "_"))
                else None
            )
            if raw:
                start_line = source.count("\n", 0, i) + 1
                closing = '"' + raw.group(1)
                end = source.find(closing, i + raw.end())
                if end < 0:
                    return (f"unclosed raw string literal '{raw.group(0)}'", start_line)
                i = end + len(closing)
            elif source[i] == '"':
                start_line = source.count("\n", 0, i) + 1
                i += 1
                closed = False
                while i < n:
                    if source[i] == "\\":
                        i += 2
                    elif source[i] == '"':
                        i += 1
                        closed = True
                        break
                    else:
                        i += 1
                if not closed:
                    return ("unclosed string literal '\"'", start_line)
            elif source[i] == "'":
                char = re.match(r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\r\n])'", source[i:i + 32])
                i += char.end() if char else 1
            else:
                i += 1
    return None


def _syntax_rejection_error(path_str: str, line: int, reason: str, content: str, original_content: str | None) -> ValueError:
    parts = [
        f"Syntax guardrail error in {path_str}:{line}: {reason}.",
        f"Proposed code around line {line}:\n{_format_line_window(content, line)}",
    ]
    if original_content is not None and original_content != content:
        parts.append(f"Original code around line {line}:\n{_format_line_window(original_content, line)}")
    parts.append("Your changes have NOT been applied. Fix the edit and try again.")
    return ValueError("\n".join(parts))


def _check_rust_syntax(path_str: str, content: str, original_content: str | None = None) -> None:
    """Validate delimiter balance, unclosed comments/strings, and E0753 before writing .rs files."""
    if not path_str.endswith(".rs") or not content.strip():
        return
    unclosed = _find_unclosed_noncode(content)
    if unclosed is not None:
        desc, err_line = unclosed
        raise _syntax_rejection_error(
            path_str, err_line, f"{desc} reaches end of file", content, original_content,
        )

    spec = importlib.util.spec_from_file_location("symbols_mask", Path(__file__).with_name("rust_symbols.py"))
    sym_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sym_mod)
    try:
        masked = sym_mod._mask_noncode(content)
    except Exception:
        return

    pairs = {"}": "{", ")": "(", "]": "["}
    stack = []
    lines = content.splitlines()
    for offset, ch in enumerate(masked):
        if ch in "{([":
            line = content.count("\n", 0, offset) + 1
            stack.append((ch, line, offset))
        elif ch in "})]":
            line = content.count("\n", 0, offset) + 1
            if not stack:
                raise _syntax_rejection_error(
                    path_str, line,
                    f"unexpected closing delimiter '{ch}' with no matching opening delimiter",
                    content, original_content,
                )
            top, top_line, _ = stack.pop()
            if top != pairs[ch]:
                raise _syntax_rejection_error(
                    path_str, line,
                    f"mismatched delimiter '{ch}', expected closing for '{top}' from line {top_line}",
                    content, original_content,
                )
    if stack:
        top, top_line, _ = stack[-1]
        raise _syntax_rejection_error(
            path_str, top_line,
            f"unclosed opening delimiter '{top}' (reaches end of file without matching closing bracket)",
            content, original_content,
        )

    orig_inner_lines = set()
    if original_content is not None:
        orig_inner_lines = {
            ln.strip() for ln in original_content.splitlines()
            if ln.strip().startswith(("#![", "//!"))
        }

    masked_lines = masked.splitlines()
    scope_stack = [[False, None]]
    attr_bracket_depth = 0
    for lineno, line in enumerate(lines, 1):
        s = line.strip()
        mline = masked_lines[lineno - 1] if lineno - 1 < len(masked_lines) else ""
        ms = mline.strip()
        if attr_bracket_depth > 0:
            attr_bracket_depth = max(0, attr_bracket_depth + mline.count("[") - mline.count("]"))
            continue
        if s.startswith("//!") or s.startswith("#!["):
            has_item, first_item_line = scope_stack[-1]
            if has_item and s not in orig_inner_lines:
                kind_label = "inner attribute or doc comment" if s.startswith("//!") else "inner attribute"
                raise _syntax_rejection_error(
                    path_str, lineno,
                    f"{kind_label} ('{s[:25]}') found after code declarations at line {first_item_line} (Rust compiler error E0753); inner attributes/docs must be at the top of the file/module",
                    content, original_content,
                )
            if s.startswith("#!["):
                attr_bracket_depth = max(0, attr_bracket_depth + mline.count("[") - mline.count("]"))
            continue
        if s.startswith("#["):
            attr_bracket_depth = max(0, attr_bracket_depth + mline.count("[") - mline.count("]"))
            continue
        if not ms:
            continue
        if not scope_stack[-1][0]:
            scope_stack[-1][0] = True
            scope_stack[-1][1] = lineno
        for ch in mline:
            if ch == "{":
                if not scope_stack[-1][0]:
                    scope_stack[-1][0] = True
                    scope_stack[-1][1] = lineno
                scope_stack.append([False, None])
            elif ch == "}" and len(scope_stack) > 1:
                scope_stack.pop()


def _strip_in_module_tests(text: str) -> str:
    """Strip temporary in-module regression test blocks from Rust source code."""
    pattern = r"(?ms)\r?\n// __TEST_REGRESSION_START: [^\n]+\r?\n.*?// __TEST_REGRESSION_END: [^\n]+(?:\r?\n)?"
    return re.sub(pattern, "", text)


def _extract_in_module_test_block(text: str) -> str:
    """Return the normalized temporary in-module regression block(s) embedded in text."""
    pattern = r"(?ms)// __TEST_REGRESSION_START: [^\n]+\r?\n.*?// __TEST_REGRESSION_END: [^\n]+"
    return "\n".join(m.group(0).replace("\r\n", "\n") for m in re.finditer(pattern, text))


def _detect_doctest_line_shift(old_text: str, new_text: str, path_str: str) -> str:
    """Detect if an edit shifted downstream Rust doctests (e.g. 'file.rs - (line N)')."""
    if not path_str.endswith(".rs"):
        return ""
    old_doc_starts = [m.start() for m in re.finditer(r"(?m)^\s*(?:#\[doc\s*=\s*|///\s*)```", old_text)]
    new_doc_starts = [m.start() for m in re.finditer(r"(?m)^\s*(?:#\[doc\s*=\s*|///\s*)```", new_text)]
    if not old_doc_starts or not new_doc_starts:
        return ""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    line_delta = len(new_lines) - len(old_lines)
    if line_delta == 0:
        return ""
    first_diff = None
    for idx, (ol, nl) in enumerate(zip(old_lines, new_lines)):
        if ol != nl:
            first_diff = idx + 1
            break
    if first_diff is None:
        first_diff = min(len(old_lines), len(new_lines)) + 1
    affected_docs = [old_text[:pos].count("\n") + 1 for pos in old_doc_starts if old_text[:pos].count("\n") + 1 >= first_diff]
    if affected_docs:
        sign = f"+{line_delta}" if line_delta > 0 else f"{line_delta}"
        return (
            f"\n[Doctest line-shift notice in {path_str}: edit changed line count by {sign} from line {first_diff}+, "
            f"shifting {len(affected_docs)} downstream doctest(s) (first at line {affected_docs[0]} -> {affected_docs[0] + line_delta}). "
            "Avoid unnecessary line additions/deletions above existing doctests unless required.]"
        )
    return ""


def _capture_file_entry(fpath: Path) -> tuple[bytes, int, int] | None:
    if not fpath.is_file():
        return None
    st = fpath.stat()
    return (fpath.read_bytes(), st.st_mode, st.st_mtime_ns)


def _restore_file_entry(fpath: Path, data, preserve_mtime: bool = False) -> None:
    if data is None:
        if fpath.is_file() or fpath.is_symlink():
            fpath.unlink()
        return
    raw_bytes = data[0]
    mode = data[1]
    mtime_ns = data[2] if (preserve_mtime and len(data) > 2) else None
    fpath.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(raw_bytes, str):
        fpath.write_text(raw_bytes, encoding="utf-8")
    else:
        fpath.write_bytes(raw_bytes)
    fpath.chmod(mode)
    if mtime_ns is not None:
        try:
            st = fpath.stat()
            os.utime(fpath, ns=(st.st_atime_ns, mtime_ns))
        except OSError:
            pass


def _resolve_crate_test_target(manifest_file: Path) -> tuple[str, str | None]:
    """Infer whether an in-module test should run via --lib or --bin=<name>."""
    pkg_dir = manifest_file.parent
    if (pkg_dir / "src" / "lib.rs").is_file():
        return ("lib", None)
    try:
        content = manifest_file.read_text(encoding="utf-8", errors="replace")
        if "[lib]" in content:
            return ("lib", None)
        m_bin = re.search(r'(?ms)^\[\[bin\]\](.*?)(?:^\[|\Z)', content)
        if m_bin:
            m_name = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', m_bin.group(1))
            if m_name:
                return ("bin", m_name.group(1))
        if (pkg_dir / "src" / "main.rs").is_file():
            m_sec = re.search(r'(?ms)^\[package\]\b(.*?)(?:^\[|\Z)', content)
            if m_sec:
                m_name = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', m_sec.group(1))
                if m_name:
                    return ("bin", m_name.group(1))
    except OSError:
        pass
    return ("lib", None)


def _format_edit_snippet(content: str, line_ranges: list[tuple[int, int]], context: int = 6) -> str:
    lines = content.splitlines()
    if not lines or not line_ranges:
        return ""
    total = len(lines)
    windows = []
    for start, end in sorted(line_ranges):
        w_start = max(1, start - context)
        w_end = min(total, max(start, end) + context)
        if windows and w_start <= windows[-1][1] + 1:
            windows[-1][1] = max(windows[-1][1], w_end)
        else:
            windows.append([w_start, w_end])
    blocks = []
    for w_start, w_end in windows:
        blocks.append("\n".join(f"{lineno}: {lines[lineno - 1]}" for lineno in range(w_start, w_end + 1)))
    return "\n--\n".join(blocks)


def _apply_one_replacement(
    content: str,
    old_text: str,
    new_text: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> tuple[str, tuple[int, int]]:
    if not isinstance(old_text, str) or not old_text:
        raise ValueError("old_text must not be empty")
    if not isinstance(new_text, str):
        raise ValueError("new_text must be a string")

    lines = content.splitlines(keepends=True)
    total_lines = len(lines)
    if start_line is not None or end_line is not None:
        s_line = 1 if start_line is None else max(1, start_line)
        e_line = total_lines if end_line is None else min(total_lines, max(s_line, end_line))
        if s_line > total_lines:
            raise ValueError(f"start_line ({s_line}) exceeds line count ({total_lines})")
        prefix = "".join(lines[:s_line - 1])
        target_region = "".join(lines[s_line - 1:e_line])
        suffix = "".join(lines[e_line:])
        line_offset = s_line - 1
    else:
        prefix = ""
        target_region = content
        suffix = ""
        line_offset = 0

    actual_old = old_text
    actual_new = new_text
    position = target_region.find(actual_old)
    if position < 0 and "\r\n" in target_region and "\r\n" not in old_text and "\n" in old_text:
        crlf_old = old_text.replace("\n", "\r\n")
        if target_region.find(crlf_old) >= 0:
            actual_old = crlf_old
            actual_new = new_text.replace("\r\n", "\n").replace("\n", "\r\n")
            position = target_region.find(actual_old)

    if position >= 0:
        if target_region.find(actual_old, position + 1) >= 0:
            raise ValueError("old_text matches more than once; provide more context or start_line/end_line")
        start_line_out = line_offset + target_region.count("\n", 0, position) + 1
        end_line_out = start_line_out + actual_new.count("\n")
        updated_region = target_region[:position] + actual_new + target_region[position + len(actual_old):]
        return prefix + updated_region + suffix, (start_line_out, end_line_out)

    target_lines = target_region.splitlines(keepends=True)
    old_lines = old_text.splitlines()
    if old_lines and len(target_lines) >= len(old_lines):
        stripped_old = [l.strip() for l in old_lines if l.strip()]
        matches = []
        for i in range(len(target_lines) - len(old_lines) + 1):
            window = target_lines[i:i + len(old_lines)]
            stripped_win = [l.strip() for l in window if l.strip()]
            if stripped_win == stripped_old:
                matches.append(i)
        if len(matches) > 1:
            raise ValueError("old_text matches more than once (whitespace-fuzzy match); provide more context or start_line/end_line")
        if len(matches) == 1:
            idx = matches[0]
            matched_slice = target_lines[idx:idx + len(old_lines)]
            first_matched_indent = matched_slice[0][:len(matched_slice[0]) - len(matched_slice[0].lstrip(" \t"))]
            first_old_indent = old_lines[0][:len(old_lines[0]) - len(old_lines[0].lstrip(" \t"))]
            newline_char = "\r\n" if any("\r\n" in l for l in matched_slice) else "\n"
            replacement_lines = []
            for nl in new_text.splitlines():
                if nl.startswith(first_old_indent):
                    replacement_lines.append(first_matched_indent + nl[len(first_old_indent):] + newline_char)
                else:
                    replacement_lines.append(first_matched_indent + nl.lstrip(" \t") + newline_char)
            start_line_out = line_offset + idx + 1
            end_line_out = start_line_out + len(replacement_lines) - 1
            new_target_lines = target_lines[:idx] + replacement_lines + target_lines[idx + len(old_lines):]
            return prefix + "".join(new_target_lines) + suffix, (start_line_out, max(start_line_out, end_line_out))

    extra_hint = ""
    if start_line is not None or end_line is not None:
        formatted = "\n".join(f"{s_line + idx}: {l.rstrip()}" for idx, l in enumerate(lines[s_line - 1:e_line]))
        extra_hint = f"\n[Region hint: lines {s_line}-{e_line} currently contain:\n{formatted}]"
    else:
        first_line = next((l.strip() for l in old_lines if l.strip()), "")
        if first_line:
            near_matches = [idx + 1 for idx, l in enumerate(target_lines) if first_line in l]
            if near_matches:
                locs = ", ".join(str(m) for m in near_matches[:4])
                extra_hint = f"\n[Region hint: the first line of old_text ('{first_line[:40]}...') was found near line(s): {locs}. Use read_lines to verify exact line contents]"
    raise ValueError(f"old_text was not found.{extra_hint}")


_FN_SIG_RE = re.compile(
    r"(?<![\w#])(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+|unsafe\s+|const\s+)*"
    r"fn\s+(?P<name>(?:r#)?[A-Za-z_]\w*)\s*(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^\{;]+)?",
    re.DOTALL,
)


_TYPE_SIG_RE = re.compile(
    r"(?<![\w#])(?:pub(?:\s*\([^)]*\))?\s+)?(?P<kind>trait|struct|enum)\s+(?P<name>(?:r#)?[A-Za-z_]\w*)",
    re.UNICODE,
)


def _detect_modified_fn_names(replacements: list[tuple]) -> tuple[list[str], list[tuple[str, str]]]:
    changed_fns = []
    changed_types = []
    for old_item, new_item, _, _ in replacements:
        old_sigs = {}
        for m in _FN_SIG_RE.finditer(old_item or ""):
            old_sigs.setdefault(m.group("name"), []).append(" ".join(m.group(0).split()))
        new_sigs = {}
        for m in _FN_SIG_RE.finditer(new_item or ""):
            new_sigs.setdefault(m.group("name"), []).append(" ".join(m.group(0).split()))
        for name, o_list in old_sigs.items():
            if (name not in new_sigs or o_list != new_sigs[name]) and name not in changed_fns:
                changed_fns.append(name)

        old_types = {m.group("name"): m.group("kind") for m in _TYPE_SIG_RE.finditer(old_item or "")}
        new_types = {m.group("name"): m.group("kind") for m in _TYPE_SIG_RE.finditer(new_item or "")}
        for name, kind in old_types.items():
            if name in new_types and (old_item != new_item) and (name, kind) not in changed_types:
                changed_types.append((name, kind))
    return changed_fns, changed_types


def _remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


class PatchState:
    def __init__(self, repo, require_reproduction: bool = False, post_strip_check: bool = True, instance_id: str | None = None):
        self._temp = None
        self._custom_workspace = False
        if instance_id and instance_id.strip():
            sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id.strip()).strip("_")
            custom_root = Path(os.environ.get("RUSTBENCH_WORKSPACE_ROOT") or (Path(tempfile.gettempdir()) / "test-agent-workspaces"))
            base_dir = custom_root / sanitized
            try:
                if base_dir.exists():
                    shutil.rmtree(base_dir, onerror=_remove_readonly)
                base_dir.mkdir(parents=True, exist_ok=True)
                self._workspace_dir = base_dir
                self.root = base_dir / "checkout"
                self._custom_workspace = True
            except Exception:
                self._temp = tempfile.TemporaryDirectory(prefix="test-agent-")
                self.root = Path(self._temp.name) / "checkout"
                self._custom_workspace = False
        else:
            self._temp = tempfile.TemporaryDirectory(prefix="test-agent-")
            self.root = Path(self._temp.name) / "checkout"
        self._repo = Path(repo).resolve()
        self._is_worktree = False
        self._is_git = False

        if (self._repo / ".git").exists() and shutil.which("git"):
            try:
                subprocess.run(
                    ["git", "rev-parse", "--verify", "HEAD"],
                    cwd=self._repo, capture_output=True, check=True
                )
                wt_res = subprocess.run(
                    ["git", "worktree", "add", "--detach", str(self.root), "HEAD"],
                    cwd=self._repo, capture_output=True, text=True
                )
                if wt_res.returncode == 0:
                    self._is_worktree = True
                    self._is_git = True
                else:
                    clone_res = subprocess.run(
                        ["git", "-c", "core.autocrlf=false", "clone", "--local", "--no-hardlinks", str(self._repo), str(self.root)],
                        capture_output=True, text=True
                    )
                    if clone_res.returncode == 0:
                        self._is_git = True
            except Exception:
                pass

        if not self._is_git:
            try:
                shutil.copytree(repo, self.root, symlinks=True,
                                ignore=shutil.ignore_patterns(".git", "target", ".cargo"))
                cargo_dir = self._repo / ".cargo"
                if cargo_dir.is_dir():
                    dest_cargo = self.root / ".cargo"
                    dest_cargo.mkdir(parents=True, exist_ok=True)
                    for item in cargo_dir.iterdir():
                        dest_item = dest_cargo / item.name
                        if not dest_item.exists() and not dest_item.is_symlink():
                            if item.is_dir():
                                try:
                                    dest_item.symlink_to(item.resolve())
                                except OSError:
                                    pass
                            else:
                                try:
                                    dest_item.write_bytes(item.read_bytes())
                                except OSError:
                                    pass
            except Exception:
                if self._temp is not None:
                    self._temp.cleanup()
                raise
        self.originals = {}
        self.regression_tests = set()
        self.in_module_regressions = {}
        self.patch = ""
        self._lock = threading.RLock()
        self._revision = 0
        self._test_generation = 0
        self._verified_revision = None
        self._passed_regressions = set()
        self._regression_test_names = {}
        self._passed_suites = set()
        self._reproduced_regressions = {}
        self._passed_regression_configs = {}
        self._reproduction_notes = {}
        self._read_only_calls = 0
        self._last_submit_error = ""
        self._last_check_sig = None
        self._last_check_rev = -1
        self._consecutive_same_check_errors = 0
        self._last_test_fail_sig = None
        self._last_test_fail_rev = -1
        self._consecutive_same_test_failures = 0
        self._consecutive_edit_failures = {}
        self._checkpoints = {}
        self._best_verified_snapshot = None
        self._last_compiling_snapshot = None
        self._consecutive_compile_failures = 0
        self._workspace_deps_cache = None
        self.require_reproduction = require_reproduction
        self.post_strip_check = post_strip_check
        self.rust_env = None
        self.rust_info = None
        self._budget_client = None
        self._budget_warning_threshold = 10
        self._package_manifests_cache = None
        self._compile_failed_regressions = set()
        self._paired_repro_counts = {}
        self._env_lockfile_blocked = False
        self._existing_suite_checked_rev = None

    _API_COMPILE_ERROR_CODES = {
        "E0599", "E0308", "E0061", "E0425", "E0432", "E0433",
        "E0277", "E0609", "E0616", "E0624", "E0560",
    }

    def _scan_package_manifests(self) -> dict[str, Path]:
        mapping = {}
        root = self.root.resolve()
        for manifest in root.glob("**/Cargo.toml"):
            if any(part in (".git", "target") for part in manifest.parts):
                continue
            try:
                content = manifest.read_text(encoding="utf-8", errors="replace")
                m_sec = re.search(r'(?ms)^\[package\](.*?)(?=^\[|\Z)', content)
                if m_sec:
                    m = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', m_sec.group(1))
                    if m:
                        mapping[m.group(1)] = manifest.resolve()
            except Exception:
                continue
        return mapping

    def _get_package_manifest(self, package: str) -> Path | None:
        if self._package_manifests_cache is None:
            self._package_manifests_cache = self._scan_package_manifests()
        return self._package_manifests_cache.get(package)

    def _resolve_manifest_target(self, manifest_path: str = "Cargo.toml", package: str = "") -> Path:
        manifest = (self.root / manifest_path).resolve()
        if package:
            pkg_manifest = self._get_package_manifest(package)
            if pkg_manifest is not None:
                return pkg_manifest
            return (self.root / f"__unknown_pkg_{package}__").resolve()
        return manifest

    def _regression_targets(self) -> list[tuple[Path, str, bool, Path]]:
        targets = [
            (path.parent.parent / "Cargo.toml", path.stem, False, path)
            for path in sorted(self.regression_tests)
        ]
        for src_rel in sorted(self.in_module_regressions):
            info = self.in_module_regressions[src_rel]
            targets.append((info["manifest"], info["name"], True, src_rel))
        return targets

    def _is_target_reproduced(self, key) -> bool:
        info = self._reproduced_regressions.get(key)
        if not info:
            return False
        return bool(info.get("runtime_failure") or info.get("compile_reproduction") or info.get("soft_fallback"))

    def reproduced(self) -> bool:
        with self._lock:
            targets = {(manifest, name) for manifest, name, _, _ in self._regression_targets()}
            if not targets:
                return False
            return all(self._is_target_reproduced(t) for t in targets)

    def reproduced_cleanly(self) -> bool:
        with self._lock:
            targets = {(manifest, name) for manifest, name, _, _ in self._regression_targets()}
            if not targets:
                return False
            return all(
                self._is_target_reproduced(t) and self._reproduced_regressions[t].get("clean_red", False)
                for t in targets
            )

    def set_budget(self, client, warning_threshold: int | None = None):
        with self._lock:
            self._budget_client = client
            if warning_threshold is not None:
                self._budget_warning_threshold = warning_threshold
            elif getattr(client, "max_calls", None) is not None:
                max_calls = client.max_calls
                self._budget_warning_threshold = min(10, max(2, max_calls // 4)) if max_calls < 20 else 10

    def remaining_api_calls(self) -> int | None:
        client = self._budget_client
        if client is None or getattr(client, "max_calls", None) is None:
            return None
        remaining = getattr(client, "remaining_calls", None)
        if isinstance(remaining, int):
            return remaining
        calls = getattr(client, "_calls", 0)
        return max(0, client.max_calls - calls)

    def _in_module_cmd_str(self, manifest: Path, name: str) -> str:
        t_kind, bin_name = _resolve_crate_test_target(self.root / manifest)
        if t_kind == "bin" and bin_name:
            return f"cargo_test(manifest_path='{manifest.as_posix()}', bin_target='{bin_name}', test_filter='{name}')"
        return f"cargo_test(manifest_path='{manifest.as_posix()}', lib=True, test_filter='{name}')"

    def missing_verification_steps(self) -> str:
        with self._lock:
            if self._env_lockfile_blocked:
                return ""
            targets = self._regression_targets()
            if not targets:
                return "write a temporary regression test with write_regression_test"
            parts = []
            if self.require_reproduction:
                missing_repro = []
                for manifest, name, is_in_mod, _ in targets:
                    key = (manifest, name)
                    if not self._is_target_reproduced(key):
                        cmd_str = (
                            self._in_module_cmd_str(manifest, name)
                            if is_in_mod
                            else f"cargo_test(manifest_path='{manifest.as_posix()}', test_target='{name}')"
                        )
                        note = self._reproduction_notes.get(key)
                        missing_repro.append(f"{cmd_str} ({note})" if note else cmd_str)
                if missing_repro:
                    parts.append("confirm failure (Red phase): " + ", ".join(missing_repro))

            missing_reg = [
                (
                    self._in_module_cmd_str(manifest, name)
                    if is_in_mod
                    else f"cargo_test(manifest_path='{manifest.as_posix()}', test_target='{name}')"
                )
                for manifest, name, is_in_mod, _ in targets
                if (manifest, name) not in self._passed_regressions
            ]
            if missing_reg:
                parts.append("pass regression tests: " + ", ".join(missing_reg))
            summary = "; ".join(parts)
            if len(targets) > 1 and summary:
                summary += " (Tip: call revert_file on any obsolete temporary regression test you no longer need)"
            return summary

    def budget_notice(self) -> str | None:
        remaining = self.remaining_api_calls()
        if remaining is not None and remaining <= self._budget_warning_threshold:
            with self._lock:
                source_changes = [p.as_posix() for p in self.changed_paths() if p not in self.regression_tests]
                if self.verified() and source_changes:
                    files = ", ".join(source_changes)
                    return (f"[Budget notice: {remaining} API call(s) remaining. "
                            f"Verification is complete—call generate_patch with paths=[{files}] now.]")
                missing = self.missing_verification_steps()
                detail = f" Pending: {missing}." if missing else ""
                return (f"[Budget notice: {remaining} API call(s) remaining.{detail} "
                        "Finish edits, run required cargo_test checks, and submit the patch soon.]")
        with self._lock:
            if self._read_only_calls >= 15 and not self.changed_paths():
                return (f"[Exploration notice: {self._read_only_calls} read/search calls made without a test or edit yet. "
                        "Synthesize findings and create a temporary regression test with write_regression_test soon.]")
        return None

    def changed_paths(self):
        with self._lock:
            root = self.root.resolve()
            changed = []
            for relative, original in self.originals.items():
                source = root / relative
                if source.is_file():
                    if relative in self.in_module_regressions:
                        stripped = _strip_in_module_tests(
                            source.read_bytes().decode("utf-8", errors="replace")
                        ).encode("utf-8")
                        if original is None or stripped != original[0]:
                            changed.append(relative)
                    elif original is None or source.read_bytes() != original[0]:
                        changed.append(relative)
                elif original is not None:
                    changed.append(relative)
            return changed

    def build_patch(self, paths):
        root = self.root.resolve()
        has_in_mod = any(p in self.in_module_regressions for p in paths)
        if self._is_git and not has_in_mod:
            try:
                posix_paths = [p.as_posix() for p in paths]
                for relative in paths:
                    source = root / relative
                    if self.originals.get(relative) is None and source.is_file():
                        subprocess.run(
                            ["git", "-c", "core.autocrlf=false", "add", "-N", "--force", "--", relative.as_posix()],
                            cwd=root, capture_output=True, check=True
                        )
                diff = subprocess.run(
                    ["git", "-c", "core.autocrlf=false", "-c", "core.quotepath=false",
                     "diff", "--no-ext-diff", "--no-color", "--binary", "HEAD", "--", *posix_paths],
                    cwd=root, capture_output=True, check=True
                ).stdout.decode("utf-8")
                for relative in paths:
                    if self.originals.get(relative) is None and (root / relative).is_file():
                        subprocess.run(
                            ["git", "reset", "-q", "--", relative.as_posix()],
                            cwd=root, capture_output=True
                        )
                return diff
            except Exception:
                pass

        def git(directory, *args):
            return subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.quotepath=false", *args],
                                  cwd=directory, capture_output=True, check=True).stdout
        with tempfile.TemporaryDirectory(prefix="test-patch-") as directory:
            work = Path(directory)
            git(work, "init", "-q")
            for relative in paths:
                original = self.originals[relative]
                if original is not None:
                    destination = work / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(original[0])
                    destination.chmod(original[1])
            git(work, "add", "--force", "--all")
            baseline = git(work, "write-tree").decode("ascii").strip()
            for relative in paths:
                original = self.originals[relative]
                source = root / relative
                destination = work / relative
                if not source.is_file():
                    if destination.exists():
                        destination.unlink()
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    raw_bytes = source.read_bytes()
                    if relative in self.in_module_regressions:
                        raw_bytes = _strip_in_module_tests(raw_bytes.decode("utf-8", errors="replace")).encode("utf-8")
                    destination.write_bytes(raw_bytes)
                    if original is not None:
                        destination.chmod(original[1])
            git(work, "add", "--force", "--all")
            return git(work, "diff", "--cached", "--no-ext-diff", "--no-color", "--binary", baseline).decode("utf-8")

    def record_edit_outcome(self, path_key: str, success: bool) -> str:
        with self._lock:
            if success:
                self._consecutive_edit_failures.pop(path_key, None)
                return ""
            count = self._consecutive_edit_failures.get(path_key, 0) + 1
            self._consecutive_edit_failures[path_key] = count
            if count >= 2:
                return (
                    f"\n[Recovery notice: {count} consecutive edit failures on '{path_key}'. "
                    f"Call read_lines('{path_key}', ...) to verify exact lines, "
                    "or use revert_file / checkpoint(action='restore').]"
                )
            return ""

    def record_check_result(self, exit_code: int, diagnostics: list[dict]) -> str | None:
        with self._lock:
            errors = [d for d in diagnostics if d.get("level") == "error"]
            if exit_code != 0 and errors:
                self._consecutive_compile_failures += 1
                sig = tuple(sorted(
                    (
                        d.get("code") or "",
                        ((d.get("spans") or [{}])[0].get("file_name") or ""),
                        (d.get("message") or "")[:80],
                    )
                    for d in errors[:3]
                ))
                if sig == self._last_check_sig:
                    if self._revision != self._last_check_rev:
                        self._consecutive_same_check_errors += 1
                else:
                    self._consecutive_same_check_errors = 1
                self._last_check_sig = sig
                self._last_check_rev = self._revision
                if self._consecutive_same_check_errors >= 2:
                    cat = errors[0].get("category", "other")
                    if cat in ("trait_bound", "type_mismatch", "lifetime", "borrow_conflict"):
                        return (
                            f"Strategy warning: the same compiler error persisted across {self._consecutive_same_check_errors} edit revisions. "
                            "Stop applying local fixes at the error line; use rust_symbols(callers=True/references=True) "
                            "to inspect callers, trait bounds, and type definitions, or restore a checkpoint and change approach."
                        )
                    return (
                        f"Strategy warning: the same compiler error persisted across {self._consecutive_same_check_errors} edit revisions. "
                        "Re-examine your root cause hypothesis or restore a checkpoint before making further edits."
                    )
                if self._consecutive_compile_failures >= 3 and (
                    self._checkpoints or self._last_compiling_snapshot is not None
                ):
                    return (
                        f"Strategy warning: {self._consecutive_compile_failures} consecutive cargo checks failed to compile. "
                        "Consider calling checkpoint(action='restore') or reverting recent edits before trying a different approach."
                    )
            elif exit_code == 0:
                self._consecutive_same_check_errors = 0
                self._consecutive_compile_failures = 0
                self._last_check_sig = None
                if any(p not in self.regression_tests for p in self.changed_paths()):
                    self._last_compiling_snapshot = self._capture_snapshot()
            return None

    def record_test_result(self, manifest_path: str, test_target: str, exit_code: int, failures: list[str], executed: int, compile_errors: list[dict]) -> str | None:
        with self._lock:
            if exit_code == 0 and executed > 0:
                self._consecutive_same_test_failures = 0
                self._last_test_fail_sig = None
                return None
            if exit_code != 0 and failures:
                sig = (manifest_path, test_target, tuple(sorted(failures[:10])))
                if sig == self._last_test_fail_sig:
                    if self._revision != self._last_test_fail_rev:
                        self._consecutive_same_test_failures += 1
                else:
                    self._last_test_fail_sig = sig
                    self._consecutive_same_test_failures = 1
                self._last_test_fail_rev = self._revision
                if self._consecutive_same_test_failures >= 2:
                    return (
                        f"Strategy warning: the same test failure(s) {list(failures[:5])} persisted across "
                        f"{self._consecutive_same_test_failures} edit revisions. Re-check the issue assumptions, "
                        "inspect runtime behavior or callers, or restore a checkpoint to try a different fix."
                    )
            return None

    def _capture_snapshot(self) -> dict:
        with self._lock:
            root = self.root.resolve()
            files = {rel: _capture_file_entry(root / rel) for rel in self.originals}
            return {
                "files": files,
                "originals": dict(self.originals),
                "regression_tests": set(self.regression_tests),
                "in_module_regressions": {k: dict(v, fn_names=set(v.get("fn_names", ()))) for k, v in self.in_module_regressions.items()},
                "reproduced_regressions": {k: dict(v) for k, v in self._reproduced_regressions.items()},
                "passed_regressions": set(self._passed_regressions),
                "passed_regression_configs": {k: dict(v) for k, v in self._passed_regression_configs.items()},
                "regression_test_names": {k: set(v) for k, v in self._regression_test_names.items()},
                "reproduction_notes": dict(self._reproduction_notes),
                "verified": self.verified(),
            }

    def _restore_snapshot(self, snap: dict):
        with self._lock:
            root = self.root.resolve()
            for rel in list(self.originals):
                if rel not in snap["originals"]:
                    _restore_file_entry(root / rel, self.originals[rel])
            for rel, data in snap["files"].items():
                _restore_file_entry(root / rel, data)

            self.originals = dict(snap["originals"])
            self.regression_tests = set(snap["regression_tests"])
            self.in_module_regressions = {
                k: dict(v, fn_names=set(v.get("fn_names", ())))
                for k, v in snap.get("in_module_regressions", {}).items()
            }
            self._reproduced_regressions = {k: dict(v) for k, v in snap["reproduced_regressions"].items()}
            self._passed_regressions = set(snap["passed_regressions"])
            self._passed_regression_configs = {k: dict(v) for k, v in snap["passed_regression_configs"].items()}
            self._regression_test_names = {k: set(v) for k, v in snap["regression_test_names"].items()}
            self._reproduction_notes = dict(snap["reproduction_notes"])
            self._passed_suites.clear()
            self._consecutive_edit_failures.clear()
            self._revision += 1
            self._verified_revision = self._revision if snap["verified"] else None
            self._last_submit_error = ""
            self.patch = ""

    def save_checkpoint(self, name: str = "default") -> str:
        with self._lock:
            clean_name = name.strip() or "default"
            snap = self._capture_snapshot()
            self._checkpoints[clean_name] = snap
            changed = [p.as_posix() for p in self.changed_paths()]
            return (
                f"Saved checkpoint '{clean_name}' "
                f"({len(changed)} changed file(s), verified={snap['verified']})."
            )

    def restore_checkpoint(self, name: str = "default") -> str:
        with self._lock:
            clean_name = name.strip() or "default"
            if clean_name not in self._checkpoints:
                available = sorted(self._checkpoints)
                raise ValueError(f"Unknown checkpoint '{clean_name}'. Available: {available or 'none'}")
            self._restore_snapshot(self._checkpoints[clean_name])
            changed = [p.as_posix() for p in self.changed_paths()]
            return (
                f"Restored checkpoint '{clean_name}' "
                f"({len(changed)} changed file(s), verified={self.verified()})."
            )

    def list_checkpoints(self) -> str:
        with self._lock:
            if not self._checkpoints:
                return "No checkpoints saved."
            rows = []
            for cname, snap in self._checkpoints.items():
                rows.append({
                    "name": cname,
                    "files": sorted(p.as_posix() for p in snap["files"]),
                    "verified": snap["verified"],
                })
            return json.dumps(rows, ensure_ascii=False)

    def _find_downstream_workspace_packages(self, modified_manifests: list[Path]) -> list[str]:
        root = self.root.resolve()
        ws_manifest = root / "Cargo.toml"
        if not ws_manifest.is_file():
            return []
        try:
            ws_text = ws_manifest.read_text(encoding="utf-8", errors="replace")
            if "[workspace]" not in ws_text:
                return []
        except OSError:
            return []

        if self._workspace_deps_cache is None:
            try:
                res = subprocess.run(
                    ["cargo", "metadata", "--format-version", "1", "--no-deps", "--offline", "--manifest-path", str(ws_manifest)],
                    cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=self.rust_env,
                )
                if res.returncode != 0:
                    return []
                meta = json.loads(res.stdout)
                self._workspace_deps_cache = meta
            except Exception:
                return []

        meta = self._workspace_deps_cache
        members = set(meta.get("workspace_members", []))
        pkgs = [p for p in meta.get("packages", []) if p.get("id") in members]
        modified_resolved = {m.resolve() for m in modified_manifests}
        modified_names = {
            p["name"] for p in pkgs
            if Path(p["manifest_path"]).resolve() in modified_resolved
        }
        if not modified_names:
            return []

        downstream = []
        for p in pkgs:
            if p["name"] in modified_names:
                continue
            dep_names = {d.get("name") for d in p.get("dependencies", []) if d.get("kind") in (None, "normal")}
            if dep_names & modified_names:
                downstream.append(p["name"])
        return downstream[:5]

    def check_existing_tests_compile_warning(self, valid_manifests: list[Path]) -> str:
        """Non-blocking check of existing test targets to warn if signature changes broke callers."""
        if not shutil.which("cargo") or not valid_manifests:
            return ""
        root = self.root.resolve()
        backup = {}
        for reg_rel in self.regression_tests:
            reg_file = root / reg_rel
            if reg_file.is_file():
                backup[reg_rel] = _capture_file_entry(reg_file)
                reg_file.unlink()
        for src_rel in self.in_module_regressions:
            src_file = root / src_rel
            if src_file.is_file():
                entry = _capture_file_entry(src_file)
                backup[src_rel] = entry
                src_file.write_text(_strip_in_module_tests(entry[0].decode("utf-8", errors="replace")), encoding="utf-8")
        warnings = []
        try:
            for manifest in sorted(valid_manifests)[:2]:
                cmd = ["cargo", "check", "--tests", "--manifest-path", str(manifest)]
                matching_cfgs = [
                    cfg for (m, _name), cfg in self._passed_regression_configs.items()
                    if (root / m).resolve() == manifest.resolve()
                ]
                if matching_cfgs:
                    cfg = matching_cfgs[0]
                    if cfg.get("package"):
                        cmd.append(f"--package={cfg['package']}")
                    if cfg.get("features"):
                        cmd.append("--features=" + ",".join(cfg["features"]))
                    if cfg.get("all_features"):
                        cmd.append("--all-features")
                    if cfg.get("no_default_features"):
                        cmd.append("--no-default-features")
                    if cfg.get("target"):
                        cmd.append(f"--target={cfg['target']}")
                try:
                    res = subprocess.run(
                        cmd, cwd=root, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=90, env=self.rust_env,
                    )
                except subprocess.TimeoutExpired:
                    continue
                if res.returncode != 0:
                    rel_m = manifest.relative_to(root).as_posix() if manifest.is_relative_to(root) else str(manifest)
                    err_line = (res.stderr.strip() or res.stdout.strip()).splitlines()[-10:]
                    warnings.append(f"{rel_m}: " + " | ".join(err_line)[-400:])
        finally:
            for reg_rel, data in backup.items():
                _restore_file_entry(root / reg_rel, data, preserve_mtime=True)
        if not warnings:
            return ""
        return (
            "\nWarning: existing test target(s) failed 'cargo check --tests' (non-blocking). "
            "Check whether your edits unintentionally broke existing callers:\n" + "\n".join(warnings) + "\n"
        )

    def check_existing_tests_regression(self, selected: list[Path]) -> None:
        """Run targeted existing crate tests once per revision to catch regressions before patch submission."""
        if (
            not self.post_strip_check
            or not selected
            or self._passed_suites
            or self._env_lockfile_blocked
            or self._existing_suite_checked_rev == self._revision
            or not shutil.which("cargo")
        ):
            return
        remaining = self.remaining_api_calls()
        if remaining is not None and remaining <= 15:
            return
        self._existing_suite_checked_rev = self._revision
        root = self.root.resolve()
        manifests = set()
        for rel in selected:
            cur = (root / rel).parent
            while cur != root.parent:
                if (cur / "Cargo.toml").is_file():
                    manifests.add((cur / "Cargo.toml").resolve())
                    break
                cur = cur.parent
        valid_manifests = sorted(m for m in manifests if _manifest_has_targets(m))[:2]
        if not valid_manifests:
            return

        backup = {}
        for reg_rel in self.regression_tests:
            reg_file = root / reg_rel
            if reg_file.is_file():
                backup[reg_rel] = _capture_file_entry(reg_file)
                reg_file.unlink()
        for src_rel in self.in_module_regressions:
            src_file = root / src_rel
            if src_file.is_file():
                entry = _capture_file_entry(src_file)
                backup[src_rel] = entry
                src_file.write_text(_strip_in_module_tests(entry[0].decode("utf-8", errors="replace")), encoding="utf-8")

        tokens = {
            tok for p in selected
            for tok in re.split(r"[_/.-]+", p.as_posix().lower())
            if len(tok) >= 3 and tok not in ("src", "lib", "mod", "main", "tests", "rs")
        }
        new_failures: list[str] = []
        try:
            for manifest in valid_manifests:
                cfgs = [
                    cfg for (m, _name), cfg in self._passed_regression_configs.items()
                    if (root / m).resolve() == manifest.resolve()
                ]
                cfg = cfgs[0] if cfgs else {}
                base_cmd = ["cargo", "test", "--manifest-path", str(manifest)]
                if cfg.get("package"):
                    base_cmd.append(f"--package={cfg['package']}")
                _append_cargo_feature_flags(base_cmd, cfg)

                cmds: list[tuple[list[str], str]] = []
                if (manifest.parent / "src" / "lib.rs").is_file():
                    cmds.append((base_cmd + ["--lib"], "lib"))
                tests_dir = manifest.parent / "tests"
                if tests_dir.is_dir():
                    t_stems = sorted(p.stem for p in tests_dir.glob("*.rs") if not p.stem.startswith(("test_regression_", "normal_regression_")))
                    matched_t = [t for t in t_stems if any(tok in t.lower() for tok in tokens) or "integration" in t.lower()]
                    for t_name in (matched_t or t_stems)[:2]:
                        cmds.append((base_cmd + ["--test", t_name], f"test:{t_name}"))

                for cmd, label in cmds:
                    try:
                        res = subprocess.run(
                            cmd, cwd=root, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60, env=self.rust_env,
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        continue
                    counts = re.findall(r"test result: (?:ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed", res.stdout)
                    executed = sum(int(p) + int(f) for p, f in counts)
                    if res.returncode == 0 or executed == 0:
                        continue
                    failed_names = sorted(set(re.findall(r"^test (.+?) \.\.\. FAILED\s*$", res.stdout, re.MULTILINE)))
                    # Check whether those failing tests passed on unpatched baseline
                    prod_paths = [p for p in self.originals if p not in self.regression_tests]
                    if not prod_paths:
                        continue
                    prod_backup = {rel: _capture_file_entry(root / rel) for rel in prod_paths}
                    baseline_failed = set()
                    baseline_ok = False
                    try:
                        for rel in prod_paths:
                            _restore_file_entry(root / rel, self.originals[rel])
                        base_res = subprocess.run(
                            cmd, cwd=root, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60, env=self.rust_env,
                        )
                        baseline_ok = base_res.returncode == 0
                        base_counts = re.findall(r"test result: (?:ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed", base_res.stdout)
                        base_executed = sum(int(p) + int(f) for p, f in base_counts)
                        if not baseline_ok and base_executed == 0:
                            continue
                        baseline_failed = set(re.findall(r"^test (.+?) \.\.\. FAILED\s*$", base_res.stdout, re.MULTILINE))
                    except (OSError, subprocess.TimeoutExpired):
                        continue
                    finally:
                        for rel, data in prod_backup.items():
                            _restore_file_entry(root / rel, data)
                    introduced = failed_names if baseline_ok else [n for n in failed_names if n not in baseline_failed]
                    if introduced:
                        rel_m = manifest.relative_to(root).as_posix() if manifest.is_relative_to(root) else str(manifest)
                        new_failures.append(f"{rel_m} ({label}): {', '.join(introduced[:5])}")
        finally:
            for reg_rel, data in backup.items():
                _restore_file_entry(root / reg_rel, data, preserve_mtime=True)

        if new_failures:
            raise ValueError(
                "Existing crate tests regressed after your edits (did not fail on original code): "
                + "; ".join(new_failures)
                + ". Fix the regression and rerun cargo_test (or resubmit if an obsolete test asserts the buggy behavior removed by the issue)."
            )

    def verify_pre_fix_reproduction(self, targets=None):
        if not shutil.which("cargo"):
            return
        root = self.root.resolve()
        all_target_rows = self._regression_targets()
        if targets is None:
            targets = {(m, name) for m, name, _, _ in all_target_rows}
        target_meta = {(m, name): (is_in_mod, reg_rel) for m, name, is_in_mod, reg_rel in all_target_rows}
        needed = [
            t for t in targets
            if not (self._is_target_reproduced(t) and self._reproduced_regressions[t].get("clean_red", False))
        ]
        if not needed:
            return
        valid_needed = [t for t in needed if _manifest_has_targets(root / t[0])]
        if not valid_needed:
            return

        prod_changed = [p for p in self.changed_paths() if p not in self.regression_tests]
        backup = {}
        for rel in prod_changed:
            backup[rel] = _capture_file_entry(root / rel)

        try:
            for rel in prod_changed:
                orig = self.originals.get(rel)
                fpath = root / rel
                in_mod_block = ""
                if rel in self.in_module_regressions and fpath.is_file():
                    in_mod_block = _extract_in_module_test_block(fpath.read_bytes().decode("utf-8", errors="replace"))
                _restore_file_entry(fpath, orig)
                if in_mod_block and fpath.is_file():
                    orig_text = fpath.read_bytes().decode("utf-8", errors="replace")
                    fpath.write_text(orig_text.rstrip() + "\n\n" + in_mod_block + "\n", encoding="utf-8")

            for rel_manifest, name in valid_needed:
                manifest_file = root / rel_manifest
                cfg = self._passed_regression_configs.get((rel_manifest, name), {})
                is_in_mod, _ = target_meta.get((rel_manifest, name), (False, None))
                cmd = [
                    "cargo", "test", "--message-format=json",
                    "--manifest-path", str(manifest_file),
                ]
                if is_in_mod:
                    t_kind, bin_name = _resolve_crate_test_target(manifest_file)
                    if t_kind == "bin" and bin_name:
                        cmd.append(f"--bin={bin_name}")
                    else:
                        cmd.append("--lib")
                else:
                    cmd.extend([f"--test={name}", "--no-fail-fast"])
                if cfg.get("package"):
                    cmd.append(f"--package={cfg['package']}")
                if cfg.get("features"):
                    cmd.append("--features=" + ",".join(cfg["features"]))
                if cfg.get("all_features"):
                    cmd.append("--all-features")
                if cfg.get("no_default_features"):
                    cmd.append("--no-default-features")
                if cfg.get("target"):
                    cmd.append(f"--target={cfg['target']}")
                if is_in_mod:
                    cmd.append(name)

                try:
                    res = subprocess.run(
                        cmd, cwd=root, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=180, env=self.rust_env,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ValueError(
                        f"Paired reproduction check timed out while running '{name}' on the original code."
                    ) from exc

                counts = re.findall(
                    r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;",
                    res.stdout, re.MULTILINE,
                )
                executed = sum(int(p) + int(f) for p, f, _ in counts)
                failures = re.findall(r"^test (.+?) \.\.\. FAILED\s*$", res.stdout, re.MULTILINE)

                if res.returncode == 0 and executed > 0:
                    key = (rel_manifest, name)
                    self._reproduced_regressions.pop(key, None)
                    self._reproduction_notes[key] = "passed on unpatched original code"
                    attempts = self._paired_repro_counts.get(key, 0) + 1
                    self._paired_repro_counts[key] = attempts
                    remaining = self.remaining_api_calls()
                    if (remaining is not None and remaining <= 20) or attempts >= 2:
                        self._reproduced_regressions[key] = {
                            "clean_red": False,
                            "runtime_failure": False,
                            "compile_reproduction": False,
                            "executed": executed,
                            "failures": [],
                            "returncode": 0,
                            "soft_fallback": True,
                        }
                        continue
                    raise ValueError(
                        f"Paired reproduction check failed: temporary regression test '{name}' "
                        f"PASSED ({executed} tests passed) on the unpatched original code, so it does "
                        "not reproduce the bug. Update the test assertions to fail on the buggy behavior."
                    )
                if res.returncode != 0:
                    self._reproduced_regressions[(rel_manifest, name)] = {
                        "clean_red": True,
                        "runtime_failure": executed > 0,
                        "compile_reproduction": executed == 0,
                        "executed": executed,
                        "failures": failures,
                        "returncode": res.returncode,
                    }
                    self._reproduction_notes.pop((rel_manifest, name), None)
            all_targets = {(m, name) for m, name, _, _ in self._regression_targets()}
            if all_targets and all_targets <= self._passed_regressions and all(self._is_target_reproduced(t) for t in all_targets):
                self._verified_revision = self._revision
                if not self.require_reproduction or self.reproduced_cleanly():
                    self._best_verified_snapshot = self._capture_snapshot()
        finally:
            for rel, data in backup.items():
                _restore_file_entry(root / rel, data)

    def _compile_fails_on_baseline(self, cmd: list[str], manifest: Path) -> bool:
        root = self.root.resolve()
        try:
            rel_manifest = manifest.resolve().relative_to(root)
        except ValueError:
            rel_manifest = None
        if rel_manifest is not None and rel_manifest in self.originals and self.originals[rel_manifest] is None:
            return False
        prod_paths = [p for p in self.originals if p not in self.regression_tests]
        if not prod_paths:
            return False
        src_backup = {rel: _capture_file_entry(root / rel) for rel in prod_paths}
        try:
            for rel in prod_paths:
                _restore_file_entry(root / rel, self.originals[rel])
            base_res = subprocess.run(
                cmd,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                env=self.rust_env,
            )
            return base_res.returncode != 0
        except (OSError, subprocess.TimeoutExpired):
            return False
        finally:
            for rel, data in src_backup.items():
                _restore_file_entry(root / rel, data, preserve_mtime=True)

    def run_post_strip_check(self, selected_paths: list[Path]):
        if not shutil.which("cargo"):
            return
        root = self.root.resolve()
        affected_manifests = set()
        for manifest_rel, _, _, _ in self._regression_targets():
            manifest_file = root / manifest_rel
            if manifest_file.is_file():
                affected_manifests.add(manifest_file.resolve())
        for sel_rel in selected_paths:
            cur = (root / sel_rel).parent
            while cur != root.parent:
                if (cur / "Cargo.toml").is_file():
                    affected_manifests.add((cur / "Cargo.toml").resolve())
                    break
                cur = cur.parent
        if (root / "Cargo.toml").is_file() and not affected_manifests:
            affected_manifests.add((root / "Cargo.toml").resolve())

        valid_manifests = [m for m in affected_manifests if _manifest_has_targets(m)]
        if not valid_manifests:
            return

        backup = {}
        for reg_rel in self.regression_tests:
            reg_file = root / reg_rel
            if reg_file.is_file():
                backup[reg_rel] = _capture_file_entry(reg_file)
                reg_file.unlink()
        for src_rel in self.in_module_regressions:
            src_file = root / src_rel
            if src_file.is_file():
                entry = _capture_file_entry(src_file)
                backup[src_rel] = entry
                src_file.write_text(_strip_in_module_tests(entry[0].decode("utf-8", errors="replace")), encoding="utf-8")
        env_error_markers = (
            "requires rustc",
            "failed to select a version",
            "failed to download",
            "network is unreachable",
            "no matching package named",
            "failed to parse lock file",
            "lock file version",
            "is not supported by winit",
        )
        try:
            for manifest in sorted(valid_manifests):
                cmd = ["cargo", "check", "--tests", "--manifest-path", str(manifest)]
                matching_cfgs = [
                    cfg for (m, _name), cfg in self._passed_regression_configs.items()
                    if (root / m).resolve() == manifest.resolve()
                ]
                if matching_cfgs:
                    first_cfg = matching_cfgs[0]
                    if first_cfg.get("package"):
                        cmd.append(f"--package={first_cfg['package']}")
                    _append_cargo_feature_flags(cmd, first_cfg)
                else:
                    _append_cargo_feature_flags(cmd)
                res_tests = subprocess.run(
                    cmd,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=120,
                    env=self.rust_env,
                )
                if res_tests.returncode != 0:
                    rel_manifest = manifest.relative_to(root).as_posix() if manifest.is_relative_to(root) else str(manifest)
                    err_snippet = (res_tests.stderr.strip() or res_tests.stdout.strip())[-3000:]
                    low_err = err_snippet.lower()
                    if any(marker in low_err for marker in env_error_markers) or self._compile_fails_on_baseline(cmd, manifest):
                        lib_cmd = [arg for arg in cmd if arg != "--tests"]
                        try:
                            res_lib = subprocess.run(
                                lib_cmd,
                                cwd=root,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                timeout=120,
                                env=self.rust_env,
                            )
                        except subprocess.TimeoutExpired:
                            continue
                        if res_lib.returncode == 0:
                            continue
                        lib_err = (res_lib.stderr.strip() or res_lib.stdout.strip())[-3000:]
                        if any(marker in lib_err.lower() for marker in env_error_markers) or self._compile_fails_on_baseline(lib_cmd, manifest):
                            continue
                        raise ValueError(
                            f"Post-strip verification failed for {rel_manifest} under 'cargo check'. "
                            "When temporary regression tests were excluded, the remaining changes failed compilation:\n"
                            f"{lib_err}"
                        )
                    raise ValueError(
                        f"Post-strip verification failed for {rel_manifest} under 'cargo check --tests'. "
                        "When temporary regression tests were excluded, the remaining changes failed compilation "
                        f"or broke existing test targets:\n{err_snippet}"
                    )
            downstream_pkgs = self._find_downstream_workspace_packages(valid_manifests)
            ws_manifest = root / "Cargo.toml"
            for dep_pkg in downstream_pkgs:
                dep_cmd = ["cargo", "check", "--tests", "--manifest-path", str(ws_manifest), f"--package={dep_pkg}"]
                _append_cargo_feature_flags(dep_cmd)
                dep_res = subprocess.run(
                    dep_cmd,
                    cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=120, env=self.rust_env,
                )
                if dep_res.returncode != 0:
                    err_snippet = (dep_res.stderr.strip() or dep_res.stdout.strip())[-3000:]
                    low_err = err_snippet.lower()
                    if any(marker in low_err for marker in env_error_markers) or self._compile_fails_on_baseline(dep_cmd, ws_manifest):
                        continue
                    raise ValueError(
                        f"Post-strip downstream workspace check failed for dependent package '{dep_pkg}'. "
                        f"Your changes broke downstream crate compilation:\n{err_snippet}"
                    )
            if ws_manifest.is_file():
                ws_text = ws_manifest.read_text(encoding="utf-8", errors="replace")
                ws_cmd = ["cargo", "check", "--all-targets", "--manifest-path", str(ws_manifest)]
                _append_cargo_feature_flags(ws_cmd)
                if "[workspace]" in ws_text:
                    ws_cmd.append("--workspace")
                if (root / "Cargo.lock").is_file():
                    ws_cmd.append("--locked")
                try:
                    ws_res = subprocess.run(
                        ws_cmd,
                        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
                        timeout=300, env=self.rust_env,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    ws_res = None
                if ws_res is not None and ws_res.returncode != 0:
                    err_snippet = (ws_res.stderr.strip() or ws_res.stdout.strip())[-3000:]
                    low_err = err_snippet.lower()
                    if not (any(marker in low_err for marker in env_error_markers) or self._compile_fails_on_baseline(ws_cmd, ws_manifest)):
                        raise ValueError(
                            "Workspace compilation regressed against the unpatched checkout:\n"
                            f"{err_snippet}"
                        )
        finally:
            for reg_rel, data in backup.items():
                _restore_file_entry(root / reg_rel, data, preserve_mtime=True)

    def auto_submit_verified_patch(self, allow_snapshot_fallback: bool = True, allow_unverified_fallback: bool = False) -> str:
        with self._lock:
            if self.patch:
                return self.patch

            def _try_current():
                selected = [path for path in self.changed_paths() if path not in self.regression_tests]
                targets = {(m, name) for m, name, _, _ in self._regression_targets()}
                if selected and ((targets and targets <= self._passed_regressions) or self._env_lockfile_blocked):
                    if not self._env_lockfile_blocked and self.require_reproduction and not self.reproduced_cleanly():
                        try:
                            self.verify_pre_fix_reproduction(targets)
                        except Exception as exc:
                            self._last_submit_error = str(exc)
                            if allow_snapshot_fallback:
                                for t in targets:
                                    self._reproduced_regressions[t] = {
                                        "clean_red": False, "runtime_failure": False,
                                        "compile_reproduction": False, "executed": 0,
                                        "failures": [], "returncode": 0, "soft_fallback": True,
                                    }
                                self._verified_revision = self._revision
                            else:
                                return ""
                    if self.verified():
                        try:
                            if self.post_strip_check:
                                self.run_post_strip_check(selected)
                            self.patch = self.build_patch(selected)
                            return self.patch
                        except Exception as exc:
                            self._last_submit_error = str(exc)
                            return ""
                return ""

            patch = _try_current()
            if patch:
                return patch

            if allow_snapshot_fallback and self._best_verified_snapshot is not None:
                cur_snap = self._capture_snapshot()
                try:
                    self._restore_snapshot(self._best_verified_snapshot)
                    patch = _try_current()
                    if patch:
                        return patch
                finally:
                    if not self.patch:
                        self._restore_snapshot(cur_snap)

            if allow_unverified_fallback:
                selected = [path for path in self.changed_paths() if path not in self.regression_tests]
                has_test_attempt = bool(
                    self.regression_tests
                    or self.in_module_regressions
                    or self._test_generation > 0
                    or self._last_check_rev >= 0
                )
                if not selected and has_test_attempt and self._last_compiling_snapshot is not None:
                    cur_snap = self._capture_snapshot()
                    try:
                        self._restore_snapshot(self._last_compiling_snapshot)
                        snap_selected = [p for p in self.changed_paths() if p not in self.regression_tests]
                        if snap_selected:
                            if self.post_strip_check and shutil.which("cargo"):
                                self.run_post_strip_check(snap_selected)
                            self.patch = self.build_patch(snap_selected)
                            if self.patch:
                                return self.patch
                    except Exception:
                        pass
                    finally:
                        if not self.patch:
                            self._restore_snapshot(cur_snap)
                if selected and has_test_attempt:
                    if self._last_compiling_snapshot is not None and self.post_strip_check and shutil.which("cargo"):
                        current_compiles = True
                        try:
                            self.run_post_strip_check(selected)
                        except Exception:
                            current_compiles = False
                        if not current_compiles:
                            cur_snap = self._capture_snapshot()
                            try:
                                self._restore_snapshot(self._last_compiling_snapshot)
                                snap_selected = [p for p in self.changed_paths() if p not in self.regression_tests]
                                if snap_selected:
                                    self.run_post_strip_check(snap_selected)
                                    self.patch = self.build_patch(snap_selected)
                                    if self.patch:
                                        return self.patch
                            except Exception:
                                pass
                            finally:
                                if not self.patch:
                                    self._restore_snapshot(cur_snap)
                    try:
                        self.patch = self.build_patch(selected)
                        return self.patch
                    except Exception:
                        return ""
            return ""


    def invalidate_write(self):
        with self._lock:
            self._revision += 1
            self._verified_revision = None
            self._passed_regressions.clear()
            self._passed_regression_configs.clear()
            self._regression_test_names.clear()
            self._passed_suites.clear()
            self._package_manifests_cache = None
            self._last_submit_error = ""
            self.patch = ""

    def start_test(self):
        with self._lock:
            self._test_generation += 1
            self.patch = ""
            return self._revision, self._test_generation

    def pass_test(self, token, manifest_path="Cargo.toml", package="", test_target="",
                  test_filter="", lib=False, bin_target="", tests=False, passed_tests=None, ignored_tests=0,
                  features=None, all_features=False, no_default_features=False, target=""):
        with self._lock:
            if token == (self._revision, self._test_generation):
                effective_manifest = self._resolve_manifest_target(manifest_path, package)
                target_rows = self._regression_targets()
                targets = {(m, name) for m, name, _, _ in target_rows}
                for relative_manifest, name, is_in_mod, _ in target_rows:
                    target_manifest = (self.root / relative_manifest).resolve()
                    manifest_match = (
                        target_manifest == effective_manifest
                        or (not package and (self.root / manifest_path).resolve() == target_manifest)
                    )
                    matched = (
                        manifest_match and (lib or bool(bin_target)) and test_filter == name
                        if is_in_mod
                        else manifest_match and test_target == name and not test_filter
                    )
                    if matched:
                        key = (relative_manifest, name)
                        if ignored_tests:
                            self._passed_regressions.discard(key)
                            self._passed_regression_configs.pop(key, None)
                            self._regression_test_names.pop(key, None)
                            self._verified_revision = None
                        else:
                            self._passed_regressions.add(key)
                            self._compile_failed_regressions.discard(key)
                            self._passed_regression_configs[key] = {
                                "package": package,
                                "features": list(features) if features else None,
                                "all_features": bool(all_features),
                                "no_default_features": bool(no_default_features),
                                "target": target,
                            }
                            self._regression_test_names[key] = set(passed_tests or [])
                is_suite = (not test_target and not test_filter and not lib and not bin_target)
                if is_suite:
                    self._passed_suites.add(effective_manifest)
                self._consecutive_compile_failures = 0
                if any(p not in self.regression_tests for p in self.changed_paths()):
                    self._last_compiling_snapshot = self._capture_snapshot()
                reproduction_ok = (not self.require_reproduction) or all(self._is_target_reproduced(t) for t in targets)
                if targets and targets <= self._passed_regressions and reproduction_ok:
                    self._verified_revision = self._revision
                    if not self.require_reproduction or self.reproduced_cleanly():
                        self._best_verified_snapshot = self._capture_snapshot()

    _MACRO_OR_API_PHRASES = (
        "panicked", "unexpected end of macro", "ambiguous",
        "unresolved", "not found", "no method", "mismatched types", "trait bound",
    )

    def fail_test(self, token, manifest_path="Cargo.toml", package="", test_target="",
                  test_filter="", lib=False, bin_target="", tests=False,
                  executed=0, failures=None, returncode=None, failed_target="",
                  compile_errors=None, cargo_output: str = ""):
        with self._lock:
            if token == (self._revision, self._test_generation):
                self._passed_suites.clear()
                low_cargo = (cargo_output or "").lower()
                if executed == 0 and returncode not in (None, 0) and (
                    "failed to parse lock file" in low_cargo or "lock file version" in low_cargo
                ):
                    self._env_lockfile_blocked = True
                effective_manifest = self._resolve_manifest_target(manifest_path, package)
                for relative_manifest, name, is_in_mod, reg_rel in self._regression_targets():
                    target_manifest = (self.root / relative_manifest).resolve()
                    key = (relative_manifest, name)
                    same_package = target_manifest == effective_manifest
                    fn_names = self.in_module_regressions.get(reg_rel, {}).get("fn_names", set()) if is_in_mod else set()
                    known_names = self._regression_test_names.get(key, set()) | fn_names
                    failed_match = (
                        (same_package and (((lib or bool(bin_target)) and test_filter == name)
                                           or (not test_target and bool(set(failures or []) & known_names))))
                        if is_in_mod
                        else ((same_package and test_target == name)
                              or (not test_target and (failed_target == name
                                  or bool(set(failures or []) & known_names))))
                    )
                    if failed_match:
                        self._passed_regressions.discard(key)
                        self._passed_regression_configs.pop(key, None)
                        self._regression_test_names.pop(key, None)
                        self._verified_revision = None
                    repro_match = (
                        (target_manifest == effective_manifest or not package) and (lib or bool(bin_target)) and test_filter == name
                        if is_in_mod
                        else (target_manifest == effective_manifest or not package) and test_target == name and not test_filter
                    )
                    if repro_match:
                        clean_red = not any(p not in self.regression_tests for p in self.changed_paths())
                        if executed == 0 and returncode is None:
                            self._reproduced_regressions.pop(key, None)
                            self._reproduction_notes[key] = "test failed to run or timed out"
                            continue
                        if executed == 0 and returncode is not None and returncode != 0:
                            self._compile_failed_regressions.add(key)
                        else:
                            self._compile_failed_regressions.discard(key)
                        runtime_failure = executed > 0 and (returncode != 0 or returncode is None or bool(failures))
                        compile_repro = False
                        if executed == 0 and returncode is not None and returncode != 0:
                            err_codes = {d.get("code") for d in (compile_errors or []) if d.get("code")}
                            err_msgs = " ".join((d.get("message") or "").lower() for d in (compile_errors or []))
                            if clean_red and (
                                bool(err_codes & self._API_COMPILE_ERROR_CODES)
                                or any(phrase in err_msgs for phrase in self._MACRO_OR_API_PHRASES)
                            ):
                                compile_repro = True
                        if runtime_failure or compile_repro:
                            self._reproduced_regressions[key] = {
                                "clean_red": clean_red,
                                "runtime_failure": runtime_failure,
                                "compile_reproduction": compile_repro,
                                "executed": executed,
                                "failures": list(failures or []),
                                "returncode": returncode,
                            }
                            self._reproduction_notes.pop(key, None)
                        else:
                            self._reproduced_regressions.pop(key, None)
                            self._reproduction_notes[key] = "test did not execute failing assertions or expose an API compile mismatch"

    def verified(self):
        with self._lock:
            if self._env_lockfile_blocked:
                return any(p not in self.regression_tests for p in self.changed_paths())
            return self._verified_revision == self._revision

    def close(self):
        if self._is_worktree:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(self.root)],
                    cwd=self._repo, capture_output=True, timeout=30
                )
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=self._repo, capture_output=True, timeout=30
                )
            except Exception:
                pass
        if getattr(self, "_custom_workspace", False) and hasattr(self, "_workspace_dir"):
            try:
                shutil.rmtree(self._workspace_dir, onerror=_remove_readonly)
            except Exception:
                pass
        elif getattr(self, "_temp", None):
            try:
                self._temp.cleanup()
            except Exception:
                try:
                    shutil.rmtree(self._temp.name, onerror=_remove_readonly)
                except Exception:
                    pass


def make_tools(repo_path, benchmark_version=None, require_reproduction=False, post_strip_check=True, instance_id=None):
    """Return tools and state; call state.close() after the agent run."""
    repo = Path(repo_path).resolve(strict=True)
    if not repo.is_dir():
        raise ValueError(f"Not a directory: {repo}")
    state = PatchState(repo, require_reproduction=require_reproduction, post_strip_check=post_strip_check, instance_id=instance_id)
    root = state.root.resolve()
    rust_env = None
    state.rust_info = None
    spec = importlib.util.spec_from_file_location("rust_environment", Path(__file__).with_name("rust_environment.py"))
    environment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(environment)
    if benchmark_version is not None:
        try:
            rust_env, state.rust_info = environment.prepare_rust_environment(root, benchmark_version, repo_path=repo, instance_id=instance_id)
            state.rust_env = rust_env
        except Exception:
            state.close()
            raise
    else:
        try:
            base_env = os.environ.copy()
            cache_info = environment.configure_cargo_cache(base_env, repo_path=repo, instance_id=instance_id)
            state.rust_env = base_env
            rust_env = base_env
            state.rust_info = cache_info
        except Exception:
            pass

    def synchronized(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with state._lock:
                return function(*args, **kwargs)
        return wrapper

    changed_paths = state.changed_paths
    build_patch = state.build_patch

    def resolve_path(path):
        if not isinstance(path, str) or not path:
            raise ValueError("Path must be a non-empty string")
        target = (root / path).resolve()
        if not target.is_relative_to(root) or any(p.lower() == ".git" for p in target.relative_to(root).parts):
            raise ValueError("Path must be inside the repository and outside .git")
        return target

    @tool
    def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """List a directory or read UTF-8 files (files over 250 lines are previewed with line numbers; pass optional 1-based start_line/end_line or use read_lines); paths are repository-relative."""
        try:
            target = resolve_path(path)
            if target.is_dir():
                return "\n".join(p.name + ("/" if p.is_dir() else "") for p in sorted(target.iterdir()) if p.name != ".git") or "(empty directory)"
            content = target.read_bytes().decode("utf-8")
            lines = content.splitlines()
            if start_line is not None or end_line is not None:
                s_line = 1 if start_line is None else start_line
                e_line = (s_line + 99) if end_line is None else end_line
                if s_line < 1:
                    raise ValueError("start_line must be at least 1")
                if e_line < s_line:
                    raise ValueError("end_line must be greater than or equal to start_line")
                clamped_notice = ""
                if e_line - s_line >= 500:
                    e_line = s_line + 499
                    clamped_notice = f"\n[Notice: requested range exceeded 500 lines; clamped to {s_line}-{e_line}]"
                total_lines = len(lines)
                rel_posix = target.relative_to(root).as_posix()
                parts = [f"[File: {rel_posix} ({total_lines} lines total)]"]
                if total_lines == 0:
                    parts.append("(empty file)")
                elif s_line > total_lines:
                    parts.append("(no lines in range)")
                else:
                    actual_end = min(e_line, total_lines)
                    if s_line > 1:
                        parts.append(f"({min(s_line - 1, total_lines)} lines above)")
                    parts.extend(f"{idx}: {lines[idx - 1]}" for idx in range(s_line, actual_end + 1))
                    if actual_end < total_lines:
                        parts.append(f"({total_lines - actual_end} lines below)")
                return "\n".join(parts) + clamped_notice
            if len(lines) > 250 or len(content) > 15000:
                total_lines = len(lines)
                head = "\n".join(f"{idx}: {ln}" for idx, ln in enumerate(lines[:120], 1))
                tail_start = max(121, total_lines - 29)
                tail = "\n".join(f"{idx}: {ln}" for idx, ln in enumerate(lines[tail_start - 1:], tail_start))
                return (
                    f"[File: {target.relative_to(root).as_posix()} ({total_lines} lines total)]\n"
                    f"{head}\n"
                    f"... [File has {total_lines} lines ({len(content)} chars); displaying lines 1-120 and {tail_start}-{total_lines}. "
                    f"Use read_lines('{path}', start_line, end_line) to inspect specific ranges, "
                    f"or search_code to locate definitions] ...\n"
                    f"{tail}"
                )
            return content
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error reading {path}: {exc}"

    def _write_file_internal(path: str, content: str | None, *, from_edit: bool = False) -> str:
        try:
            target = resolve_path(path)
            relative = target.relative_to(root)
            if relative in state.regression_tests:
                raise ValueError("Use write_regression_test to modify temporary regression tests")
            if relative.name == "Cargo.lock":
                raise ValueError("Modifying Cargo.lock is prohibited to preserve offline dependency resolution.")
            forbidden_dirs = [part for part in relative.parts[:-1] if part in ("tests", "benches")]
            if forbidden_dirs:
                raise ValueError(
                    f"Editing or creating files in {forbidden_dirs[0]}/ is prohibited to avoid benchmark "
                    "evaluation conflicts. Use write_regression_test to create temporary tests at "
                    "tests/test_regression_<name>.rs, or modify production code in src/."
                )
            if not relative.parts or (content is not None and not isinstance(content, str)):
                raise ValueError("Provide a file path and string or null content")
            if target.exists() and not target.is_file():
                raise ValueError("Path is not a regular file")
            if content is None and not target.is_file():
                raise ValueError("Cannot delete a missing file")
            original_text = None
            if target.is_file():
                try:
                    original_text = target.read_bytes().decode("utf-8", errors="replace")
                except OSError:
                    original_text = None
            if not from_edit and relative.suffix == ".rs":
                orig_entry = state.originals.get(relative) if relative in state.originals else None
                orig_bytes = orig_entry[0] if orig_entry is not None else (target.read_bytes() if target.is_file() else None)
                if orig_bytes is not None:
                    orig_line_count = len(orig_bytes.decode("utf-8", errors="replace").splitlines())
                    if orig_line_count >= 20:
                        if content is None:
                            raise ValueError(
                                f"Refusing to delete existing Rust source file '{relative.as_posix()}' ({orig_line_count} lines). "
                                "Use edit_file for targeted changes or revert_file to restore."
                            )
                        new_line_count = len(content.splitlines())
                        if new_line_count < max(5, orig_line_count // 3):
                            raise ValueError(
                                f"Refusing to overwrite existing Rust source file '{relative.as_posix()}' ({orig_line_count} lines) "
                                f"with {new_line_count} lines. Use edit_file for targeted replacements or revert_file to restore."
                            )
            if content is not None:
                _check_rust_syntax(relative.as_posix(), content, original_content=original_text)
            if relative not in state.originals:
                state.originals[relative] = _capture_file_entry(target)
            if relative in state.in_module_regressions and (content is None or "// __TEST_REGRESSION_START:" not in content):
                old_info = state.in_module_regressions.pop(relative)
                old_key = (old_info["manifest"], old_info["name"])
                state._reproduced_regressions.pop(old_key, None)
                state._passed_regression_configs.pop(old_key, None)
                state._reproduction_notes.pop(old_key, None)
                state._regression_test_names.pop(old_key, None)
            state.invalidate_write()
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
            return f"Updated {relative.as_posix()} in the working copy."
        except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as exc:
            return f"Error writing {path}: {exc}"

    @tool
    @synchronized
    def write_file(path: str, content: str | None) -> str:
        """Write UTF-8 contents in the working copy; null deletes a file.

        Any write invalidates the submitted patch. Call generate_patch after
        all edits, selecting only files that belong in the final fix.
        """
        return _write_file_internal(path, content, from_edit=False)

    def _prune_compile_failed_standalone_tests(exclude: Path | None = None) -> None:
        for old_reg in list(state.regression_tests):
            if exclude is not None and old_reg == exclude:
                continue
            old_key = (old_reg.parent.parent / "Cargo.toml", old_reg.stem)
            if old_key in state._compile_failed_regressions and old_key not in state._passed_regressions:
                old_file = root / old_reg
                if old_file.is_file():
                    old_file.unlink()
                state.regression_tests.discard(old_reg)
                state.originals.pop(old_reg, None)
                state._reproduced_regressions.pop(old_key, None)
                state._passed_regression_configs.pop(old_key, None)
                state._reproduction_notes.pop(old_key, None)
                state._regression_test_names.pop(old_key, None)
                state._compile_failed_regressions.discard(old_key)

    @tool
    @synchronized
    def write_regression_test(path: str, content: str, run_now: bool = False, target_src: str = "") -> str:
        """Create or replace a temporary Cargo regression test.

        By default, creates a standalone integration test under a package's tests/ directory
        named test_regression_<name>.rs. To test private or pub(crate) items directly, pass
        target_src='src/...' (e.g. 'src/lib.rs') to mount a temporary #[cfg(test)] module at
        the end of that source file. Both are excluded from the submitted patch.
        """
        try:
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Provide non-empty Rust test content")
            if target_src and target_src.strip():
                target = resolve_path(target_src.strip())
                relative = target.relative_to(root)
                forbidden_dirs = [part for part in relative.parts[:-1] if part in ("tests", "benches")]
                if forbidden_dirs or target.suffix != ".rs" or not target.is_file():
                    raise ValueError("target_src must be an existing .rs source file outside tests/ and benches/")
                clean_stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(path).stem).strip("_") or "case"
                mod_name = clean_stem if clean_stem.startswith("__test_regression_") else f"__test_regression_{clean_stem}"
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", mod_name):
                    raise ValueError("Invalid regression test name")

                cur = target.parent
                manifest_abs = None
                while cur != root.parent:
                    if (cur / "Cargo.toml").is_file():
                        manifest_abs = (cur / "Cargo.toml").resolve()
                        break
                    cur = cur.parent
                if manifest_abs is None:
                    raise ValueError(f"Could not locate Cargo.toml for target_src '{relative.as_posix()}'")
                manifest_rel = manifest_abs.relative_to(root)

                orig_bytes = target.read_bytes()
                orig_text = orig_bytes.decode("utf-8", errors="replace")
                newline = "\r\n" if "\r\n" in orig_text else "\n"
                clean_base = _strip_in_module_tests(orig_text)
                norm_content = content.replace("\r\n", "\n").replace("\n", newline)
                test_block = (
                    f"{newline}// __TEST_REGRESSION_START: {mod_name}__{newline}"
                    f"#[cfg(test)]{newline}"
                    f"mod {mod_name} {{{newline}"
                    f"    use super::*;{newline}"
                    f"{norm_content}{newline}"
                    f"}}{newline}"
                    f"// __TEST_REGRESSION_END: {mod_name}__{newline}"
                )
                updated_text = clean_base + test_block
                _check_rust_syntax(relative.as_posix(), updated_text, original_content=orig_text)

                st = target.stat()
                state.originals.setdefault(relative, (orig_bytes, st.st_mode, st.st_mtime_ns))
                state.invalidate_write()
                fn_names = set(re.findall(r"(?m)^\s*(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(", content))
                state.in_module_regressions[relative] = {
                    "name": mod_name,
                    "manifest": manifest_rel,
                    "content": content,
                    "fn_names": fn_names,
                }
                reg_key = (manifest_rel, mod_name)
                state._reproduced_regressions.pop(reg_key, None)
                state._passed_regression_configs.pop(reg_key, None)
                state._reproduction_notes.pop(reg_key, None)
                state._compile_failed_regressions.discard(reg_key)
                target.write_bytes(updated_text.encode("utf-8"))
                _prune_compile_failed_standalone_tests()
                msg = f"Mounted temporary in-module regression test '{mod_name}' in {relative.as_posix()}."
                if run_now:
                    t_kind, bin_name = _resolve_crate_test_target(manifest_abs)
                    if t_kind == "bin" and bin_name:
                        run_res = cargo_test.func(manifest_path=manifest_rel.as_posix(), bin_target=bin_name, test_filter=mod_name)
                    else:
                        run_res = cargo_test.func(manifest_path=manifest_rel.as_posix(), lib=True, test_filter=mod_name)
                    msg += f"\n\n[Auto-run result for {mod_name}]:\n{run_res}"
                return msg

            target = resolve_path(path)
            relative = target.relative_to(root)
            if (target.parent.name != "tests" or not (target.name.startswith("test_regression_") or target.name.startswith("normal_regression_"))
                    or target.suffix != ".rs"):
                raise ValueError("Use a nonempty test at tests/test_regression_<name>.rs, or set target_src='src/...'")
            if relative not in state.regression_tests and target.exists():
                raise ValueError("Regression test must be a new file")
            existing_text = target.read_bytes().decode("utf-8", errors="replace") if target.is_file() else None
            _check_rust_syntax(relative.as_posix(), content, original_content=existing_text)
            state.invalidate_write()
            state.originals.setdefault(relative, None)
            state.regression_tests.add(relative)
            reg_key = (relative.parent.parent / "Cargo.toml", relative.stem)
            state._reproduced_regressions.pop(reg_key, None)
            state._passed_regression_configs.pop(reg_key, None)
            state._reproduction_notes.pop(reg_key, None)
            state._compile_failed_regressions.discard(reg_key)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
            _prune_compile_failed_standalone_tests(exclude=relative)
            msg = f"Updated temporary regression test {relative.as_posix()} in the working copy."
            if run_now:
                manifest_rel = (relative.parent.parent / "Cargo.toml").as_posix()
                run_res = cargo_test.func(manifest_path=manifest_rel, test_target=relative.stem)
                msg += f"\n\n[Auto-run result for {relative.stem}]:\n{run_res}"
            return msg
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error writing regression test {path}: {exc}"

    @tool
    @synchronized
    def edit_file(
        path: str,
        old_text: str = "",
        new_text: str = "",
        edits: list[dict] | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Replace exact occurrence(s) of text in an existing UTF-8 file and return surrounding numbered lines.

        Paths are repository-relative. Pass either (old_text, new_text) for a single replacement
        or edits=[{"old_text": "...", "new_text": "..."}, ...] for multiple replacements.
        Optional start_line and end_line can anchor the search to a specific line range.
        If exact matching fails, leading whitespace is matched flexibly to handle indentation differences.
        """
        rel_key = path
        try:
            target = resolve_path(path)
            relative = target.relative_to(root)
            rel_key = relative.as_posix()
            if edits is not None:
                if old_text or new_text:
                    raise ValueError("Provide either old_text/new_text or edits, not both")
                if not isinstance(edits, list) or not edits:
                    raise ValueError("edits must be a non-empty list of {'old_text', 'new_text'} objects")
                replacements = []
                for item in edits:
                    if not isinstance(item, dict) or "old_text" not in item or "new_text" not in item:
                        raise ValueError("Each edit must contain 'old_text' and 'new_text'")
                    replacements.append((
                        item["old_text"],
                        item["new_text"],
                        item.get("start_line"),
                        item.get("end_line"),
                    ))
            else:
                replacements = [(old_text, new_text, start_line, end_line)]
            content = target.read_bytes().decode("utf-8")
            updated = content
            line_ranges = []
            for old_item, new_item, s_line, e_line in replacements:
                updated, line_range = _apply_one_replacement(updated, old_item, new_item, s_line, e_line)
                line_ranges.append(line_range)
            write_result = _write_file_internal(path, updated, from_edit=True)
            if not write_result.startswith("Updated "):
                return write_result + state.record_edit_outcome(rel_key, False)
            state.record_edit_outcome(rel_key, True)
            snippet = _format_edit_snippet(updated, line_ranges, context=6)
            out_text = f"{write_result}\n{snippet}" if snippet else write_result
            if target.suffix == ".rs" and relative not in state.regression_tests:
                inline_warn = _edited_inline_test_warning(content, updated, rel_key)
                if inline_warn:
                    out_text += inline_warn
                shift_notice = _detect_doctest_line_shift(content, updated, rel_key)
                if shift_notice:
                    out_text += shift_notice
                mod_fns, mod_types = _detect_modified_fn_names(replacements)
                impact_lines = []
                if mod_fns:
                    for fn_name in mod_fns[:3]:
                        callers_list, _ = symbols_mod.find_rust_references(
                            root, fn_name, max_results=6, callers_only=True, mask_cache=impact_mask_cache
                        )
                        filtered_locs = [
                            f"{c['path']}:{c['line']}"
                            for c in callers_list
                            if not (c["path"] == rel_key and any(s <= c["line"] <= e for s, e in line_ranges))
                        ]
                        if filtered_locs:
                            impact_lines.append(
                                f"signature modified for `{fn_name}` -> call sites: {', '.join(filtered_locs[:5])}. "
                                f"Use rust_symbols(name='{fn_name}', callers=True) or cargo_check(tests=True) to verify callers."
                            )
                if mod_types:
                    for t_name, t_kind in mod_types[:2]:
                        ref_list, _ = symbols_mod.find_rust_references(
                            root, t_name, max_results=6, callers_only=False, mask_cache=impact_mask_cache
                        )
                        other_locs = [
                            f"{c['path']}:{c['line']}"
                            for c in ref_list
                            if not (c["path"] == rel_key and any(s <= c["line"] <= e for s, e in line_ranges))
                        ]
                        if other_locs:
                            impact_lines.append(
                                f"{t_kind} modified for `{t_name}` -> usages/impls at: {', '.join(other_locs[:5])}. "
                                f"Verify trait bounds and pattern matches across callers."
                            )
                if impact_lines:
                    out_text += "\n[Impact notice: " + " | ".join(impact_lines) + "]"
            return out_text
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error editing {path}: {exc}" + state.record_edit_outcome(rel_key, False)

    @tool
    @synchronized
    def revert_file(path: str) -> str:
        """Revert a modified, created, or deleted file in the working copy back to its original state.

        Also removes temporary regression tests created with write_regression_test.
        Reverting invalidates prior test verification and submitted patches.
        """
        try:
            target = resolve_path(path)
            relative = target.relative_to(root)
            if not relative.parts:
                raise ValueError("Provide a file path")
            if relative not in state.originals and relative not in state.regression_tests and relative not in state.in_module_regressions:
                if not target.exists():
                    raise ValueError(f"File does not exist: {path}")
                if not target.is_file():
                    raise ValueError("Path is not a regular file")
                return f"No changes to revert for {relative.as_posix()}."
            original = state.originals.pop(relative, None)
            state.regression_tests.discard(relative)
            in_mod_info = state.in_module_regressions.pop(relative, None)
            reg_key = (
                (in_mod_info["manifest"], in_mod_info["name"])
                if in_mod_info
                else (relative.parent.parent / "Cargo.toml", relative.stem)
            )
            state._reproduced_regressions.pop(reg_key, None)
            state._passed_regression_configs.pop(reg_key, None)
            state._reproduction_notes.pop(reg_key, None)
            state._regression_test_names.pop(reg_key, None)
            state._compile_failed_regressions.discard(reg_key)
            state._paired_repro_counts.pop(reg_key, None)
            state._consecutive_edit_failures.pop(relative.as_posix(), None)
            state.invalidate_write()
            _restore_file_entry(target, original)
            if original is None:
                return f"Reverted {relative.as_posix()} (removed newly created file)."
            return f"Reverted {relative.as_posix()} to its original state."
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error reverting {path}: {exc}"

    @tool
    @synchronized
    def generate_patch(paths: list[str] | None = None, submit: bool = False) -> str:
        """Preview changes or submit a patch for explicitly selected files.

        With no paths, preview all changed files and their diff without submitting.
        To submit, pass the repository-relative paths of every file needed for the
        fix, including new files and deletions. Temporary regression tests created
        with write_regression_test are automatically excluded from submission.
        Call again after any write, which invalidates the previous submission.
        """
        try:
            changed = changed_paths()
            targets = {(m, name) for m, name, _, _ in state._regression_targets()}
            if not state._env_lockfile_blocked and state.require_reproduction and targets and targets <= state._passed_regressions and not state.reproduced_cleanly():
                state.verify_pre_fix_reproduction(targets)
            if changed and not state.verified():
                missing = state.missing_verification_steps()
                detail = f" Missing: {missing}." if missing else ""
                return ("After the last write, pass each temporary regression test with cargo_test "
                        f"with at least one test executed.{detail}")
            if paths is None and submit:
                source_candidates = [p.as_posix() for p in changed if p not in state.regression_tests]
                if not source_candidates:
                    return "No changes to submit."
                paths = source_candidates

            if paths is None:
                if not changed:
                    return "No changes to submit."
                names = "\n".join(path.as_posix() for path in changed)
                repro_notice = ""
                unrepro = [name for manifest, name in targets if (manifest, name) not in state._reproduced_regressions]
                if unrepro:
                    repro_notice = f"\nNotice: The following temporary regression test(s) have not been confirmed failing before repair: {', '.join(unrepro)}.\n"
                source_candidates = [p.as_posix() for p in changed if p not in state.regression_tests]
                test_compile_warn = ""
                if source_candidates:
                    affected_manifests = set()
                    for src_rel in source_candidates:
                        cur = (root / src_rel).parent
                        while cur != root.parent:
                            if (cur / "Cargo.toml").is_file():
                                affected_manifests.add((cur / "Cargo.toml").resolve())
                                break
                            cur = cur.parent
                    valid_m = [m for m in affected_manifests if _manifest_has_targets(m)]
                    if valid_m:
                        test_compile_warn = state.check_existing_tests_compile_warning(valid_m)
                hint = (f"Call generate_patch(paths={json.dumps(source_candidates)}) to submit."
                        if source_candidates else "Call generate_patch with paths for the final fix to submit.")
                return (f"Unsubmitted preview. Changed files:\n{names}\n{repro_notice}{test_compile_warn}"
                        f"{hint}\n\n"
                        + build_patch(changed))
            state.patch = ""
            if not paths:
                raise ValueError("Provide at least one changed file path")
            selected = []
            for path in paths:
                relative = resolve_path(path).relative_to(root)
                if relative in state.regression_tests:
                    raise ValueError(f"Temporary regression test must not be submitted: {path}")
                if relative not in changed:
                    raise ValueError(f"Not a changed file: {path}")
                if relative not in selected:
                    selected.append(relative)
            omitted = set(changed) - set(selected) - state.regression_tests
            if omitted:
                names = ", ".join(sorted(path.as_posix() for path in omitted))
                raise ValueError(f"Include all other changed files or revert them and rerun tests: {names}")
            if state.post_strip_check:
                state.run_post_strip_check(selected)
                state.check_existing_tests_regression(selected)
            patch = build_patch(selected)
            inline_conflicts = _detect_inline_test_patch_conflicts(patch, state.originals)
            if inline_conflicts:
                raise ValueError(
                    "Refusing to submit patch modifying pre-existing inline #[cfg(test)]/mod tests region(s): "
                    + "; ".join(inline_conflicts)
                    + ". Revert inline test edits and use write_regression_test instead."
                )
            state.patch = patch
            return patch or "No changes to submit."
        except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as exc:
            return f"Error generating patch: {exc}"

    def load_local(name):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @tool
    @synchronized
    def checkpoint(action: str = "save", name: str = "default") -> str:
        """Save, restore, or list working-copy checkpoints for safe experimentation.

        action='save' snapshots all modified files and test state under `name`.
        action='restore' atomically rolls back all edits/new files made since `name` was saved.
        action='list' returns saved checkpoints.
        """
        try:
            act = action.strip().lower()
            if act == "save":
                return state.save_checkpoint(name)
            if act == "restore":
                return state.restore_checkpoint(name)
            if act == "list":
                return state.list_checkpoints()
            raise ValueError("action must be 'save', 'restore', or 'list'")
        except ValueError as exc:
            return f"Error in checkpoint: {exc}"

    rust_tools = load_local("rust_tools").make_rust_tools(root, resolve_path, rust_env)
    diagnostics = load_local("rust_diagnostics")
    cargo_check = diagnostics.make_cargo_check_tool(
        root, resolve_path, rust_env, on_result=state.record_check_result
    )
    cargo_test = load_local("cargo_testing").make_cargo_test_tool(
        root,
        resolve_path,
        diagnostics._simplify,
        state.start_test,
        state.pass_test,
        synchronized,
        state.fail_test,
        rust_env,
        compact_fn=diagnostics.compact_diagnostics,
        enrich_diagnostics=diagnostics.attach_snippets_and_lookup,
        on_test_feedback=state.record_test_result,
    )
    explain = load_local("rust_error_explain").make_rust_error_explain_tool(root, rust_env)
    symbols_mod = load_local("rust_symbols")
    impact_mask_cache = {}
    symbols = symbols_mod.make_rust_symbols_tool(root, resolve_path)
    all_tools = [read_file, write_file, write_regression_test, edit_file, revert_file, checkpoint, generate_patch,
                 *rust_tools, cargo_check, cargo_test, symbols, explain]

    def attach_budget_notice(tool_obj):
        if tool_obj.name == "read_file":
            return tool_obj
        original_func = tool_obj.func

        @wraps(original_func)
        def wrapped(*args, **kwargs):
            if tool_obj.name in ("read_lines", "search_code", "rust_symbols", "cargo_workspace"):
                with state._lock:
                    state._read_only_calls += 1
            elif tool_obj.name in ("write_file", "write_regression_test", "edit_file"):
                with state._lock:
                    state._read_only_calls = 0
            output = original_func(*args, **kwargs)
            notice = state.budget_notice()
            if not notice or not isinstance(output, str):
                return output
            if tool_obj.name == "generate_patch" and state.patch:
                return output
            if output.startswith("{"):
                try:
                    data = json.loads(output)
                    if isinstance(data, dict):
                        data["budget_notice"] = notice
                        return json.dumps(data, ensure_ascii=False)
                except ValueError:
                    pass
            return f"{output}\n\n{notice}"

        tool_obj.func = wrapped
        return tool_obj

    return [attach_budget_notice(t) for t in all_tools], state
