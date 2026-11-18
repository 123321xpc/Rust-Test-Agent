"""File, Rust investigation, and patch tools for the test baseline."""

import hashlib
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
import time
from pathlib import Path
from typing import Any

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


def _is_virtual_workspace_manifest(manifest_file: Path) -> bool:
    """Return True if manifest_file defines a Cargo [workspace] without a [package] or src/lib/bin target."""
    if not manifest_file.is_file():
        return False
    pkg_dir = manifest_file.parent
    if (pkg_dir / "src").is_dir():
        return False
    try:
        content = manifest_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    has_workspace = bool(re.search(r"(?m)^\[workspace\]", content))
    has_package = bool(re.search(r"(?m)^\[package\]", content))
    has_explicit_target = bool(re.search(r"(?m)^(?:\[lib\]|\[\[bin\]\]|\[\[test\]\])", content))
    return has_workspace and not has_package and not has_explicit_target


def _manifest_package_name(manifest_file: Path) -> str:
    """Return the [package] name declared by a manifest, or ""."""
    try:
        content = manifest_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    section = re.search(r"(?ms)^\[package\](.*?)(?:^\[|\Z)", content)
    if not section:
        return ""
    name = re.search(r"(?m)^name\s*=\s*[\"']([^\"']+)[\"']", section.group(1))
    return name.group(1) if name else ""


def discover_integration_test_targets(manifest_file: Path) -> list[str]:
    """Return integration test target names from tests/*.rs and [[test]] blocks."""
    names: set[str] = set()
    test_dir = manifest_file.parent / "tests"
    if test_dir.is_dir():
        for p in test_dir.glob("*.rs"):
            if p.is_file() and not p.name.startswith(("test_regression_", "normal_regression_", "test_verifier_probe_tmp")):
                names.add(p.stem)
    if manifest_file.is_file():
        try:
            content = manifest_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        for blk in re.finditer(r"(?ms)^\[\[test\]\](.*?)(?:^\[|\Z)", content):
            m = re.search(r"(?m)^name\s*=\s*[\"']([^\"']+)[\"']", blk.group(1))
            if m and not m.group(1).startswith(("test_regression_", "normal_regression_", "test_verifier_probe_tmp")):
                names.add(m.group(1))
    return sorted(names)


def _snapshot_smoke_filter(manifest_file: Path, selected_paths: list[Path]) -> str:
    """Return a targeted test filter matching edited templates/files to tests/snapshots/*.snap."""
    snap_dir = manifest_file.parent / "tests" / "snapshots"
    if not snap_dir.is_dir():
        return ""
    snaps = sorted(p.stem.lower() for p in snap_dir.glob("*.snap") if p.is_file())
    if not snaps:
        return ""
    tokens: list[str] = []
    for p in selected_paths:
        for part in re.split(r"[^a-z0-9]+", p.stem.lower()):
            if len(part) >= 5:
                for cand in (part, part[:8], part[:7], part[:6]):
                    if len(cand) >= 5 and cand not in tokens:
                        tokens.append(cand)
    for tok in tokens:
        if any(tok in s for s in snaps):
            return tok
    return ""


def _workspace_member_manifests(state: Any, workspace_manifest: Path) -> list[Path]:
    """Return member crate manifests of a virtual workspace, preferring crates that own tests.

    `cargo test --test=<name>` cannot run against a virtual workspace root, so a probe
    must target a member crate. Members that ship an integration test directory are
    preferred because they can host a temporary assertion module.
    """
    root = state.root.resolve()
    workspace_dir = workspace_manifest.parent
    members: list[Path] = []
    try:
        content = workspace_manifest.read_text(encoding="utf-8", errors="replace")
    except OSError:
        content = ""
    patterns = re.findall(r"(?m)^\s*members\s*=\s*\[(.*?)\]", content, re.DOTALL)
    globs: list[str] = []
    for block in patterns:
        globs.extend(re.findall(r"[\"']([^\"']+)[\"']", block))
    for pattern in globs:
        for candidate in sorted(workspace_dir.glob(pattern)):
            manifest = candidate / "Cargo.toml" if candidate.is_dir() else candidate
            if manifest.is_file():
                members.append(manifest.resolve())
    if not members:
        for manifest in sorted(root.glob("**/Cargo.toml")):
            if any(part in (".git", "target") for part in manifest.parts):
                continue
            if manifest.resolve() == workspace_manifest.resolve():
                continue
            if _is_virtual_workspace_manifest(manifest):
                continue
            members.append(manifest.resolve())
    with_tests = [m for m in members if (m.parent / "tests").is_dir()]
    return with_tests or members


def _touch_crate_entry_mtime(root: Path, rel_path: Path) -> None:
    """Refresh mtime on nearest crate entry (.rs) without changing bytes so include_dir!/include_str! rebuilds."""
    if rel_path.suffix == ".rs" or any(p in ("tests", "benches", "examples", "target", ".git") for p in rel_path.parts):
        return
    cur = (root / rel_path).parent
    while cur != root.parent:
        if (cur / "Cargo.toml").is_file():
            for candidate in (cur / "src" / "lib.rs", cur / "src" / "main.rs", cur / "build.rs"):
                if candidate.is_file():
                    try:
                        os.utime(candidate, None)
                    except OSError:
                        pass
                    return
            break
        cur = cur.parent


def _strip_in_module_tests(text: str) -> str:
    """Strip temporary in-module regression test blocks from Rust source code."""
    pattern = r"(?ms)\r?\n// __TEST_REGRESSION_START: [^\n]+\r?\n.*?// __TEST_REGRESSION_END: [^\n]+(?:\r?\n)?"
    return re.sub(pattern, "", text)


def _extract_in_module_test_block(text: str) -> str:
    """Return the normalized temporary in-module regression block(s) embedded in text."""
    pattern = r"(?ms)// __TEST_REGRESSION_START: [^\n]+\r?\n.*?// __TEST_REGRESSION_END: [^\n]+"
    return "\n".join(m.group(0).replace("\r\n", "\n") for m in re.finditer(pattern, text))


def _contains_compression_placeholder(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return (
        "[earlier write omitted" in text
        or "[earlier read of '" in text
        or "... [omitted " in text
    )


def _rust_has_substantive_code(content: str) -> bool:
    """Return whether Rust text contains code beyond comments and attributes."""
    if not isinstance(content, str) or not content.strip():
        return False
    try:
        spec = importlib.util.spec_from_file_location("symbols_mask", Path(__file__).with_name("rust_symbols.py"))
        sym_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sym_mod)
        masked = sym_mod._mask_noncode(content)
    except Exception:
        masked = content
    for line in masked.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(r"[A-Za-z_]\w*|[{}();=<>+\-*/&|!?]", stripped):
            return True
    return False


def _strip_snapshot_diff_noise(text: str) -> str:
    """Remove insta snapshot diff payload lines so template strings inside snapshots are not misclassified."""
    out = []
    in_snapshot_block = False
    for line in text.splitlines():
        low = line.strip().lower()
        if "snapshot summary" in low:
            in_snapshot_block = True
            continue
        if in_snapshot_block:
            if (
                low.startswith("to update snapshots")
                or low.startswith("thread '")
                or low.startswith("failures:")
                or low.startswith("test result:")
                or low.startswith("---- ")
            ):
                in_snapshot_block = False
            else:
                continue
        if low.startswith("stored new snapshot "):
            continue
        out.append(line)
    return "\n".join(out)


def _extract_smoke_failure_details(stdout: str, stderr: str, limit: int = 2400) -> str:
    """Preserve panic assertions and snapshot diffs instead of trailing .snap.new paths."""
    cleaned_stdout_lines = []
    snap_new_count = 0
    for line in (stdout or "").splitlines():
        if line.lstrip().startswith("stored new snapshot "):
            snap_new_count += 1
            continue
        cleaned_stdout_lines.append(line)
    cleaned_stdout = "\n".join(cleaned_stdout_lines).strip()
    cleaned_stderr = "\n".join(
        line for line in (stderr or "").splitlines()
        if not re.match(r"^\s*(?:Compiling|Checking|Downloading|Downloaded|Updating|Fresh|Locking)\s+", line)
    ).strip()

    excerpts = []
    blocks = list(re.finditer(r"(?ms)^---- .+? stdout ----\s*.*?(?=^---- .+? stdout ----|^failures:|^test result:|\Z)", cleaned_stdout))
    if blocks:
        for match in blocks[:2]:
            blk = match.group(0).strip()
            if blk:
                excerpts.append(blk[:1000])
    failures_m = re.search(r"(?ms)^failures:\s*\n.*?(?=^test result:|\Z)", cleaned_stdout)
    if failures_m:
        excerpts.append(failures_m.group(0).strip()[:500])
    result_lines = [
        line for line in cleaned_stdout.splitlines()
        if line.startswith("test result:") or "panicked at" in line or "FAILED" in line
    ]
    if not excerpts and result_lines:
        excerpts.append("\n".join(result_lines[:15]))
    if snap_new_count:
        excerpts.append(f"[{snap_new_count} .snap.new snapshot mismatch file(s) generated]")
    if cleaned_stderr:
        excerpts.append(cleaned_stderr[-800:])
    combined = "\n\n".join(part for part in excerpts if part).strip()
    if not combined:
        combined = f"{cleaned_stdout}\n{cleaned_stderr}".strip()
    return combined[:limit] if len(combined) > limit else combined


def _ckg_cross_file_impact_notice(root: Path, state: Any, rel_path: str, changed_lines: list[int] | None = None) -> str:
    """Return a cross-file producer->consumer notice for an edited Rust source file."""
    ckg = getattr(state, "ckg", None)
    if ckg is None or not rel_path.endswith(".rs"):
        return ""
    try:
        impact_spec = importlib.util.spec_from_file_location(
            "test_rust_impact_tools", Path(__file__).with_name("rust_impact.py")
        )
        impact_mod = importlib.util.module_from_spec(impact_spec)
        impact_spec.loader.exec_module(impact_mod)
        surface = impact_mod.build_impact_surface(
            ckg, [rel_path], changed_lines={rel_path: changed_lines} if changed_lines else None,
        )
        text = impact_mod.format_impact_surface(surface, max_chars=2200)
    except Exception:
        return ""
    if not text:
        return ""
    return (
        "\n[CKG cross-file impact: this file's symbols are consumed elsewhere. "
 "Verify each listed consumer handles the new producer output before submitting.]\n"
        f"{text}"
    )


def _extract_snapshot_diff_hunks(stdout: str, limit: int = 4000) -> list[str]:
    """Extract insta snapshot diff hunks from cargo test output."""
    text = stdout or ""
    blocks: list[str] = []
    pattern = re.compile(r"(?ms)^-{3,} Snapshot Summary.*?(?=^test result:|\Z)")
    for match in pattern.finditer(text):
        block = match.group(0).strip()
        if block:
            blocks.append(block)
    if not blocks:
        pattern = re.compile(r"(?ms)^Snapshot Summary:.*?(?=^test result:|\Z)")
        for match in pattern.finditer(text):
            block = match.group(0).strip()
            if block:
                blocks.append(block)
    if not blocks:
        for line in text.splitlines():
            if re.match(r"^\s*[-+]\s", line) or "snapshot assertion" in line.lower():
                blocks.append(line.strip())
    joined = "\n\n".join(blocks)
    return [joined[:limit]] if joined else []


def _snapshot_hunks_differ(patched_details: str, baseline_stdout: str) -> bool:
    """Return True when the patched run and the unpatched baseline produce different snapshots.

    A snapshot test keeps its test name across revisions, so name comparison alone
    cannot tell a pre-existing failure from a regression the patch introduced.
    """
    patched_hunks = _extract_snapshot_diff_hunks(patched_details or "")
    baseline_hunks = _extract_snapshot_diff_hunks(baseline_stdout or "")
    if not patched_hunks and not baseline_hunks:
        return False
    patched_norm = re.sub(r"\s+", " ", "\n".join(patched_hunks)).strip()
    baseline_norm = re.sub(r"\s+", " ", "\n".join(baseline_hunks)).strip()
    return patched_norm != baseline_norm


def _smoke_environment_failure(stdout: str, stderr: str) -> bool:
    """Identify failures that clearly originate outside the code under test."""
    filtered_stdout = _strip_snapshot_diff_noise(stdout or "")
    text = f"{filtered_stdout}\n{stderr or ''}".lower()
    environment_markers = (
        "failed to download", "failed to fetch", "requires rustc", "network is unreachable",
        "could not resolve host", "connection reset", "connection refused", "connection timed out",
        "failed to connect to", "gnutls recv error", "failed to select a version",
        "failed to update", "failed to load source",
    )
    return any(marker in text for marker in environment_markers)


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


def _resolve_in_module_test_target(manifest_file: Path) -> tuple[str, str]:
    """Return ('lib', '') or ('bin', '<name>') for running an in-module test under manifest_file."""
    manifest_dir = manifest_file.parent
    manifest_txt = ""
    if manifest_file.is_file():
        try:
            manifest_txt = manifest_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            manifest_txt = ""
    if (manifest_dir / "src" / "lib.rs").is_file() or bool(re.search(r"(?m)^\[lib\]", manifest_txt)):
        return "lib", ""
    for m_sec in re.finditer(r"(?ms)^\[\[bin\]\](.*?)(?:^\[|\Z)", manifest_txt):
        m_name = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', m_sec.group(1))
        if m_name:
            return "bin", m_name.group(1)
    m_pkg = re.search(r"(?ms)^\[package\](.*?)(?:^\[|\Z)", manifest_txt)
    pkg_name = ""
    if m_pkg:
        m_name = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', m_pkg.group(1))
        if m_name:
            pkg_name = m_name.group(1)
    if (manifest_dir / "src" / "main.rs").is_file() and pkg_name:
        return "bin", pkg_name
    bin_dir = manifest_dir / "src" / "bin"
    if bin_dir.is_dir():
        for rs in sorted(bin_dir.glob("*.rs")):
            return "bin", rs.stem
    if pkg_name and re.search(r"(?m)^\[package\]", manifest_txt):
        return "bin", pkg_name
    return "lib", ""


def _is_static_manifest_only_test(content: str) -> bool:
    """Detect tests that statically inspect Cargo.toml text without calling crate code or running cargo."""
    if not isinstance(content, str) or "Cargo.toml" not in content:
        return False
    if not ("include_str!" in content or "read_to_string" in content or "fs::read" in content):
        return False
    non_manifest = re.sub(r"Cargo\.toml", "", content)
    if re.search(r"\b(?:Command|cargo_bin|cargo)\b", non_manifest, re.IGNORECASE):
        return False
    cleaned_uses = re.sub(
        r"(?m)^\s*use\s+(?:std|core|alloc|tempfile|toml|serde|anyhow|regex|assert_cmd|predicates)\b[^;]*;",
        "",
        content,
    )
    if re.search(r"(?m)^\s*(?:use|extern\s+crate)\s+[A-Za-z_]\w*", cleaned_uses):
        return False
    return True

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

    has_item = False
    first_item_line = None
    for lineno, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            if (s.startswith("//!") or s.startswith("#![")) and has_item:
                raise _syntax_rejection_error(
                    path_str, lineno,
                    f"inner attribute or doc comment ('{s[:25]}') found after code declarations at line {first_item_line} (Rust compiler error E0753); inner attributes/docs must be at the top of the file/module",
                    content, original_content,
                )
            continue
        if s.startswith("#!["):
            if has_item:
                raise _syntax_rejection_error(
                    path_str, lineno,
                    f"inner attribute ('{s[:25]}') found after code declarations at line {first_item_line} (Rust compiler error E0753); inner attributes/docs must be at the top of the file/module",
                    content, original_content,
                )
            continue
        if s.startswith("#["):
            continue
        if not has_item:
            has_item = True
            first_item_line = lineno


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


def _changed_line_numbers(old_text: str, new_text: str) -> list[int]:
    """Return 1-based line numbers in new_text that differ from old_text."""
    import difflib

    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    changed: list[int] = []
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed.extend(range(j1 + 1, j2 + 1))
    return changed


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


def _manifest_declared_features(manifest_file: Path) -> set[str]:
    """Parse declared `[features]` keys from a Cargo.toml manifest."""
    try:
        content = manifest_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    sec = re.search(r"(?ms)^\[features\](.*?)(?:^\[|\Z)", content)
    if not sec:
        return set()
    return set(re.findall(r"(?m)^\s*([A-Za-z0-9_-]+)\s*=", sec.group(1)))


def _nearest_manifest_for_file(root: Path, fpath: Path) -> Path | None:
    cur = fpath.parent
    while cur != root.parent:
        mf = cur / "Cargo.toml"
        if mf.is_file():
            return mf.resolve()
        if cur == root:
            break
        cur = cur.parent
    return None


def _detect_doctest_line_shift(old_text: str, new_text: str, path_str: str) -> str:
    """Detect if an edit shifted downstream Rust doctests (e.g. 'file.rs - (line X)')."""
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
        last_new_doc_line = max(new_text[:pos].count("\n") + 1 for pos in new_doc_starts)
        placement_hint = (
            f" Place newly added helper functions/types below existing doctests (right after their caller/impl block after line {last_new_doc_line}, before any #[cfg(test)] — never at raw EOF after the last item) or match existing line count."
            if line_delta > 0
            else " Keep doc-comment/formatting edits line-neutral (delta=0) whenever possible."
        )
        return (
            f"[Doctest Line-Shift Notice in {path_str}]: edit changed line count by {sign} (from line {first_diff}+), "
            f"shifting {len(affected_docs)} downstream Rust doc-test(s) (first shifted doc at line {affected_docs[0]} -> {affected_docs[0] + line_delta}). "
            "Benchmark evaluation matches doctests by exact `(line N)` suffix, so shifting lines can break PASS_TO_PASS even if `cargo test --doc` passes;"
            + placement_hint
        )
    return ""


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


def _edited_inline_test_warning(orig_code: str, updated_code: str, path_str: str) -> str:
    """Warn immediately when an edit touches a pre-existing inline #[cfg(test)]/mod tests region."""
    norm = path_str.replace("\\", "/").lower()
    if (
        not norm.endswith(".rs")
        or norm.startswith(("tests/", "benches/", "examples/"))
        or "/tests/" in norm or "/benches/" in norm or "/examples/" in norm
        or norm.endswith(("_test.rs", "tests.rs"))
    ):
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
            f"[Inline-Test Isolation Warning in {path_str}]: edit touches pre-existing #[cfg(test)]/mod tests lines ({span_str}). "
            "Submission gate BLOCKS patches that modify pre-existing inline tests (which also risk test_patch merge conflicts). "
            "Leave existing inline tests and test-helper constructors untouched: keep existing struct/fn signatures compatible "
            "(e.g., derive auxiliary/default state or add a new helper method instead of altering mod tests)."
        )
    return ""


def _eof_append_warning(orig_code: str, updated_code: str, path_str: str) -> str:
    """Warn when appending new top-level items at raw EOF of a non-test .rs file without #[cfg(test)]."""
    norm = path_str.replace("\\", "/").lower()
    if (
        not norm.endswith(".rs")
        or norm.startswith(("tests/", "benches/", "examples/"))
        or "/tests/" in norm or "/benches/" in norm or "/examples/" in norm
        or norm.endswith(("_test.rs", "tests.rs"))
        or inline_test_line_ranges(orig_code)
    ):
        return ""
    orig_lines = orig_code.rstrip().splitlines()
    new_lines = updated_code.rstrip().splitlines()
    if len(orig_lines) < 5 or len(new_lines) <= len(orig_lines):
        return ""
    tail = [ln.rstrip() for ln in orig_lines[-2:]]
    if [ln.rstrip() for ln in new_lines[len(orig_lines) - len(tail):len(orig_lines)]] != tail:
        return ""
    appended = "\n".join(new_lines[len(orig_lines):])
    if re.search(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+|unsafe\s+)?(?:fn|struct|enum|impl|trait|const|static|mod)\b", appended):
        return (
            f"[EOF-Append Conflict Warning in {path_str}]: new item(s) appended at raw EOF of a file without #[cfg(test)]. "
            "Evaluation test_patch often appends tests at EOF, causing patch merge conflicts. "
            "Place new private helpers/types immediately after their caller or impl block before the file's final item instead."
        )
    return ""


def _same_file_parallel_pattern_warning(orig_code: str, updated_code: str, path_str: str) -> str:
    """Flag remaining identical macro/literal statements elsewhere in the same production .rs file."""
    norm = path_str.replace("\\", "/").lower()
    if (
        not norm.endswith(".rs")
        or norm.startswith(("tests/", "benches/", "examples/"))
        or "/tests/" in norm or "/benches/" in norm or "/examples/" in norm
        or norm.endswith(("_test.rs", "tests.rs"))
    ):
        return ""
    import difflib
    orig_lines = orig_code.splitlines()
    upd_lines = updated_code.splitlines()
    orig_test_ranges = inline_test_line_ranges(orig_code)
    upd_test_ranges = inline_test_line_ranges(updated_code)
    in_ranges = lambda ln_no, rs: any(s <= ln_no <= e for s, e in rs)

    replaced: list[str] = []
    changed_upd_lines: set[int] = set()
    upd_added_texts: set[str] = set()
    matcher = difflib.SequenceMatcher(None, orig_lines, upd_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for j in range(j1, j2):
            changed_upd_lines.add(j + 1)
            upd_added_texts.add(upd_lines[j].strip())
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                if in_ranges(i + 1, orig_test_ranges):
                    continue
                st = orig_lines[i].strip()
                if len(st) >= 12 and not st.startswith(("//", "/*", "*")) and ("!(" in st or '"' in st):
                    if st not in upd_added_texts and st not in replaced:
                        replaced.append(st)
    if not replaced:
        return ""

    target_set = {s for s in replaced if s not in upd_added_texts}
    if not target_set:
        return ""
    hits: dict[str, list[int]] = {}
    for idx, ln in enumerate(upd_lines, 1):
        if idx in changed_upd_lines or in_ranges(idx, upd_test_ranges):
            continue
        st = ln.strip()
        if st in target_set:
            hits.setdefault(st, []).append(idx)

    notices = [
        f"line(s) {', '.join(str(x) for x in lns[:4])} still contain `{pat[:90]}`"
        for pat, lns in hits.items()
        if 1 <= len(lns) <= 8
    ]
    if not notices:
        return ""
    return (
        f"[Same-File Pattern Notice in {path_str}]: {'; '.join(notices[:2])}. "
        "Verify whether parallel branches in this file need the same update."
    )


def _unimported_crate_path_hint(prev_code: str, updated_code: str, path_str: str) -> str:
    """Suggest importing externally qualified types at module top when inline tests use `use super::*;`."""
    if not path_str.endswith(".rs") or "use super::*;" not in updated_code:
        return ""
    brace_uses: dict[str, str] = {}
    for m in re.finditer(r"(?m)^use\s+([A-Za-z_]\w*)::\{([^}]+)\}\s*;", updated_code):
        brace_uses[m.group(1)] = m.group(2)
    if not brace_uses:
        return ""
    import difflib
    matcher = difflib.SequenceMatcher(None, prev_code.splitlines(), updated_code.splitlines(), autojunk=False)
    added_lines = [
        ln for tag, _i1, _i2, j1, j2 in matcher.get_opcodes() if tag in ("insert", "replace")
        for ln in updated_code.splitlines()[j1:j2]
    ]
    hints: list[str] = []
    for ln in added_lines:
        for m in re.finditer(r"\b([a-z_][a-z0-9_]*)::([A-Z][A-Za-z0-9_]*)\b", ln):
            crate_name, ident = m.group(1), m.group(2)
            if crate_name in brace_uses and not re.search(rf"\b{re.escape(ident)}\b", brace_uses[crate_name]):
                if not re.search(rf"(?m)^use\s+{re.escape(crate_name)}::{re.escape(ident)}\b", updated_code):
                    hints.append(f"`{crate_name}::{ident}` -> `use {crate_name}::{{..., {ident}}}`")
    if not hints:
        return ""
    uniq = list(dict.fromkeys(hints))[:3]
    return (
        f"[Rust Test-Scope Hint in {path_str}]: file has `use super::*;` in inline tests; consider adding top-level imports ({'; '.join(uniq)}) "
        "so test suites importing `super::*` can resolve the type directly."
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
        self.instance_id = instance_id
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
        self._checkpoints = {}
        self._active_cfg_features = {}
        self._best_verified_snapshot = None
        self._workspace_deps_cache = None
        self.require_reproduction = require_reproduction
        self.post_strip_check = post_strip_check
        self.rust_env = None
        self.rust_info = None
        self._budget_client = None
        self._budget_warning_threshold = 10
        self._package_manifests_cache = None
        self.ckg = None
        self._submit_hint_override = None
        self._last_check_diagnostics = []
        self._last_test_failure_info = {}
        self._existing_test_failures = []
        self._executed_test_targets = []
        self._last_compiling_prod_snapshot = None
        self._existing_test_smoke_cache = None
        self._locked_regressions: dict[tuple[str, str], dict[str, Any]] = {}
        self._red_lock_violations: list[dict[str, Any]] = []
        self._slow_smoke_targets: set[str] = set()
        self._unlocked_regressions: dict[tuple[str, str], dict[str, Any]] = {}
        self._consecutive_edit_failures: dict[str, int] = {}

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

    _API_COMPILE_ERROR_CODES = {
        "E0599", "E0308", "E0061", "E0425", "E0432", "E0433",
        "E0277", "E0609", "E0616", "E0624", "E0560",
    }

    def get_regression_targets(self):
        """Return dict mapping (manifest_rel, target_name) to target metadata."""
        res = {}
        for path in self.regression_tests:
            if path in self.in_module_regressions:
                info = self.in_module_regressions[path]
                res[(info["manifest"], info["name"])] = {"is_in_module": True, "path": path, "name": info["name"], "content": info.get("content", "")}
            else:
                manifest_rel = path.parent.parent / "Cargo.toml"
                res[(manifest_rel, path.stem)] = {"is_in_module": False, "path": path, "name": path.stem, "content": ""}
        return res

    def _scan_package_manifests(self) -> dict[str, Path]:
        mapping = {}
        root = self.root.resolve()
        for manifest in root.glob("**/Cargo.toml"):
            if any(part in (".git", "target") for part in manifest.parts):
                continue
            try:
                content = manifest.read_text(encoding="utf-8", errors="replace")
                m_sec = re.search(r'(?ms)^\[package\](.*?)(?:^\[|\Z)', content)
                if m_sec:
                    m_name = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', m_sec.group(1))
                    if m_name:
                        mapping[m_name.group(1)] = manifest.resolve()
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
            if manifest.is_file():
                return manifest
            return (self.root / f"__unknown_pkg_{package}__").resolve()
        return manifest

    def _is_target_reproduced(self, key) -> bool:
        info = self._reproduced_regressions.get(key)
        if not info:
            return False
        return bool(info.get("runtime_failure") or info.get("compile_reproduction") or info.get("soft_fallback"))

    def reproduced(self) -> bool:
        with self._lock:
            targets = set(self.get_regression_targets().keys())
            if not targets:
                return False
            return all(self._is_target_reproduced(t) for t in targets)

    def reproduced_cleanly(self) -> bool:
        with self._lock:
            targets = set(self.get_regression_targets().keys())
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

    def _target_sha256(self, key) -> str:
        t_info = self.get_regression_targets().get(key)
        if not t_info:
            return ""
        if t_info.get("is_in_module"):
            return hashlib.sha256((t_info.get("content") or "").encode("utf-8", errors="replace")).hexdigest()
        fpath = self.root.resolve() / t_info["path"]
        if fpath.is_file():
            try:
                return hashlib.sha256(fpath.read_bytes()).hexdigest()
            except OSError:
                return ""
        return ""

    def lock_regressions(self, keys: list[tuple[str, str]] | None = None) -> dict[str, Any]:
        """Freeze the current Red-phase regression tests so the Fixer cannot weaken them."""
        with self._lock:
            targets = self.get_regression_targets()
            selected = keys if keys is not None else list(targets.keys())
            locked: list[str] = []
            for key in selected:
                t_info = targets.get(key)
                if not t_info:
                    continue
                sha = self._target_sha256(key)
                if not sha:
                    continue
                reg_path = t_info.get("path", "")
                lock_key = (str(key[0]), str(key[1]))
                # The Architect already released this oracle; do not re-lock it.
                if lock_key in self._unlocked_regressions:
                    continue
                self._locked_regressions[lock_key] = {
                    "sha256": sha,
                    "path": reg_path.as_posix() if isinstance(reg_path, Path) else str(reg_path),
                    "name": t_info["name"],
                    "is_in_module": bool(t_info.get("is_in_module")),
                    "manifest": str(key[0]),
                    "locked_at_revision": self._revision,
                }
                locked.append(f"{lock_key[0]}::{lock_key[1]}")
            return {
                "locked": locked,
                "count": len(locked),
                "violations": list(self._red_lock_violations),
            }

    def _match_regression_lock_key(self, mapping: dict[tuple[str, str], dict[str, Any]], query: str) -> tuple[str, str] | None:
        raw = str(query or "").strip()
        if not raw:
            return None
        norm = raw.replace("\\", "/")
        for key, meta in mapping.items():
            mname = str(meta.get("name") or "")
            mpath = str(meta.get("path") or "").replace("\\", "/")
            if raw == mname or raw == key[1] or norm == mpath or (mpath and norm == Path(mpath).name) or (mpath and norm == Path(mpath).stem):
                return key
        fn_pat = re.compile(rf"\bfn\s+{re.escape(raw)}\s*(?:<|\()")
        for key, meta in mapping.items():
            mpath = str(meta.get("path") or "")
            content = ""
            if meta.get("is_in_module"):
                info = self.in_module_regressions.get(Path(mpath))
                if info:
                    content = str(info.get("content") or "")
            if not content and mpath:
                fpath = self.root.resolve() / mpath
                if fpath.is_file():
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        content = ""
            if content and fn_pat.search(content):
                return key
        return None

    def unlock_regression(
        self,
        name: str,
        reason: str,
        *,
        approved_by: str = "",
    ) -> dict[str, Any]:
        """Release the Red-phase lock on one regression test after an explicit approval.

        The lock exists so a Fixer cannot weaken a confirmed Red oracle.  When the
        Architect confirms the locked test is factually wrong (wrong crate path,
        impossible assertion, wrong expected value), the lock must be released or
        the repair is stuck.  Each regression can only be unlocked once.
        """
        with self._lock:
            reason_text = str(reason or "").strip()
            if len(reason_text) < 20:
                return {
                    "ok": False,
                    "error": "Provide a concrete reason (>= 20 chars) explaining why the locked test is factually wrong.",
                }
            target_key = self._match_regression_lock_key(self._locked_regressions, name)
            if target_key is None:
                if self._match_regression_lock_key(self._unlocked_regressions, name) is not None:
                    return {
                        "ok": False,
                        "error": f"Regression '{name}' was already unlocked once; it cannot be unlocked again.",
                    }
                return {"ok": False, "error": f"No locked regression test named '{name}'."}
            meta = dict(self._locked_regressions.pop(target_key))
            self._unlocked_regressions[target_key] = {
                "name": meta.get("name", name),
                "path": meta.get("path", ""),
                "reason": reason_text[:600],
                "approved_by": str(approved_by or "").strip()[:120],
                "unlocked_at_revision": self._revision,
            }
            for reg_key in [
                k for k in list(self._reproduced_regressions)
                if str(k[1]) == meta.get("name", "")
            ]:
                self._reproduced_regressions.pop(reg_key, None)
                self._passed_regressions.discard(reg_key)
                self._passed_regression_configs.pop(reg_key, None)
                self._reproduction_notes.pop(reg_key, None)
                self._regression_test_names.pop(reg_key, None)
            self._verified_revision = None
            return {
                "ok": True,
                "unlocked": meta.get("name", name),
                "path": meta.get("path", ""),
                "reason": reason_text[:600],
                "approved_by": str(approved_by or "").strip()[:120],
            }

    def locked_regression_report(self) -> dict[str, Any]:
        """Report whether locked Red-phase tests are still byte-identical to their locked form."""
        with self._lock:
            if not self._locked_regressions:
                return {"locked": [], "violations": list(self._red_lock_violations), "tampered": False}
            locked: list[dict[str, Any]] = []
            tampered: list[dict[str, Any]] = []
            for key, meta in self._locked_regressions.items():
                current = self._locked_target_sha256(meta)
                entry = {
                    "manifest": key[0],
                    "name": key[1],
                    "path": meta["path"],
                    "locked_sha256": meta["sha256"],
                    "current_sha256": current,
                }
                if not current:
                    entry["status"] = "removed"
                    tampered.append(entry)
                elif current != meta["sha256"]:
                    entry["status"] = "modified"
                    tampered.append(entry)
                else:
                    entry["status"] = "intact"
                    locked.append(entry)
            return {
                "locked": locked,
                "violations": list(self._red_lock_violations),
                "tampered": bool(tampered),
                "tampered_entries": tampered,
            }

    def _locked_target_sha256(self, meta: dict[str, Any]) -> str:
        """Compute the current hash of a locked regression test from its recorded path."""
        rel = meta.get("path") or ""
        if not rel:
            return ""
        if meta.get("is_in_module"):
            in_mod = self.in_module_regressions.get(rel) or {}
            content = in_mod.get("content")
            if content is not None:
                return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
            fpath = self.root.resolve() / rel
            if not fpath.is_file():
                return ""
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
            block = ""
            name = meta.get("name", "")
            start = f"// __TEST_REGRESSION_START: {name}__"
            end = f"// __TEST_REGRESSION_END: {name}__"
            if start in text and end in text:
                block = text.split(start, 1)[1].split(end, 1)[0]
            return hashlib.sha256(block.encode("utf-8", errors="replace")).hexdigest()
        fpath = self.root.resolve() / rel
        if not fpath.is_file():
            return ""
        try:
            return hashlib.sha256(fpath.read_bytes()).hexdigest()
        except OSError:
            return ""

    def _record_red_lock_violation(self, action: str, path: str, detail: str = "") -> None:
        with self._lock:
            for entry in self._red_lock_violations:
                if entry.get("action") == action and entry.get("path") == path:
                    entry["count"] = int(entry.get("count", 1)) + 1
                    return
            self._red_lock_violations.append({
                "action": action,
                "path": path,
                "detail": detail,
                "count": 1,
                "revision": self._revision,
            })

    def _format_regression_cmd(self, manifest: Path, name: str, t_info: dict) -> str:
        if t_info.get("is_in_module"):
            kind, bin_name = _resolve_in_module_test_target(self.root.resolve() / manifest)
            if kind == "bin" and bin_name:
                return f"cargo_test(manifest_path='{manifest.as_posix()}', bin_target='{bin_name}', test_filter='{name}')"
            return f"cargo_test(manifest_path='{manifest.as_posix()}', lib=True, test_filter='{name}')"
        return f"cargo_test(manifest_path='{manifest.as_posix()}', test_target='{name}')"

    def missing_verification_steps(self) -> str:
        with self._lock:
            all_targets_info = self.get_regression_targets()
            targets = sorted(all_targets_info.keys())
            if not targets:
                return "write a temporary regression test with write_regression_test"
            parts = []
            if self.require_reproduction:
                missing_repro = []
                for manifest, name in targets:
                    key = (manifest, name)
                    if not self._is_target_reproduced(key):
                        t_info = all_targets_info.get(key, {})
                        cmd_str = self._format_regression_cmd(manifest, name, t_info)
                        note = self._reproduction_notes.get(key)
                        missing_repro.append(f"{cmd_str} ({note})" if note else cmd_str)
                if missing_repro:
                    parts.append("confirm failure (Red phase): " + ", ".join(missing_repro))

            missing_reg = []
            for manifest, name in targets:
                if (manifest, name) not in self._passed_regressions:
                    t_info = all_targets_info.get((manifest, name), {})
                    missing_reg.append(self._format_regression_cmd(manifest, name, t_info))
            if missing_reg:
                parts.append("pass regression tests: " + ", ".join(missing_reg))
            return "; ".join(parts)

    def budget_notice(self) -> str | None:
        remaining = self.remaining_api_calls()
        if remaining is not None and remaining <= self._budget_warning_threshold:
            with self._lock:
                if self._submit_hint_override:
                    if callable(self._submit_hint_override):
                        hint_text = self._submit_hint_override(self, remaining)
                        if hint_text:
                            return f"[Budget notice: {remaining} API call(s) remaining. {hint_text}]"
                    else:
                        return f"[Budget notice: {remaining} API call(s) remaining. {self._submit_hint_override}]"
                source_changes = [
                    p.as_posix() for p in self.changed_paths()
                    if p not in self.regression_tests or p in self.in_module_regressions
                ]
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
                    cur_bytes = source.read_bytes()
                    if relative in self.in_module_regressions:
                        cur_text = cur_bytes.decode("utf-8", errors="replace")
                        stripped_bytes = _strip_in_module_tests(cur_text).encode("utf-8")
                        if original is None or stripped_bytes != original[0]:
                            changed.append(relative)
                    else:
                        if original is None or cur_bytes != original[0]:
                            changed.append(relative)
                elif original is not None:
                    changed.append(relative)
            return changed

    def _substantive_changed_paths(self):
        """Return changed production paths that still contain meaningful work."""
        root = self.root.resolve()
        result = []
        for relative in self.changed_paths():
            if relative in self.regression_tests and relative not in self.in_module_regressions:
                continue
            target = root / relative
            if not target.is_file():
                continue
            original = self.originals.get(relative)
            if relative.suffix == ".rs":
                if _rust_has_substantive_code(target.read_text(encoding="utf-8", errors="replace")):
                    result.append(relative)
            elif relative.name.lower() in {"readme.md", "changelog.md"} or relative.suffix.lower() in {".md", ".rst", ".txt"}:
                continue
            elif original is None or target.read_bytes() != original[0]:
                result.append(relative)
        return result

    def _snapshot_substantive_paths(self, snap):
        """Return meaningful production paths represented by a snapshot."""
        result = []
        for relative, original in snap.get("originals", {}).items():
            if relative in snap.get("regression_tests", set()) and relative not in snap.get("in_module_regressions", {}):
                continue
            data = snap.get("files", {}).get(relative)
            current = data[0] if data else None
            original_bytes = original[0] if original else None
            if current == original_bytes or current is None:
                continue
            if relative.suffix == ".rs":
                if _rust_has_substantive_code(current.decode("utf-8", errors="replace")):
                    result.append(relative)
            elif relative.name.lower() in {"readme.md", "changelog.md"} or relative.suffix.lower() in {".md", ".rst", ".txt"}:
                continue
            else:
                result.append(relative)
        return result

    def _revert_would_drop_last_substantive_change(self, relative):
        """Guard an explicit revert when it would discard the only compiling work."""
        current = set(self._substantive_changed_paths())
        if relative not in current:
            return False
        return not any(path != relative for path in current)

    def _restore_would_drop_last_substantive_change(self, snap):
        """Guard checkpoint restores that would discard the only known good work."""
        current = set(self._substantive_changed_paths())
        if not current:
            return False
        return not set(self._snapshot_substantive_paths(snap))

    def build_patch(self, paths):
        root = self.root.resolve()
        in_mod_backup = {}
        for rel in paths:
            if rel in self.in_module_regressions:
                tf = root / rel
                if tf.is_file():
                    content_str = tf.read_text(encoding="utf-8")
                    in_mod_backup[tf] = content_str
                    tf.write_text(_strip_in_module_tests(content_str), encoding="utf-8")
        try:
            return self._build_patch_internal(paths, root)
        finally:
            for tf, content_str in in_mod_backup.items():
                tf.write_text(content_str, encoding="utf-8")

    def _build_patch_internal(self, paths, root):
        if self._is_git:
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
                    destination.write_bytes(source.read_bytes())
                    if original is not None:
                        destination.chmod(original[1])
            git(work, "add", "--force", "--all")
            return git(work, "diff", "--cached", "--no-ext-diff", "--no-color", "--binary", baseline).decode("utf-8")

    def reset_stuck_counters(self):
        with self._lock:
            self._consecutive_same_check_errors = 0
            self._last_check_sig = None
            self._last_check_rev = -1
            self._consecutive_same_test_failures = 0
            self._last_test_fail_sig = None
            self._last_test_fail_rev = -1

    def stuck_status(self) -> dict:
        with self._lock:
            return {
                "compile_repeats": self._consecutive_same_check_errors,
                "test_repeats": self._consecutive_same_test_failures,
                "last_check_errors": list(self._last_check_diagnostics[:5]),
                "last_test_failure": dict(self._last_test_failure_info),
                "current_revision": self._revision,
            }

    def diff_hygiene_report(self, selected_paths=None) -> dict:
        with self._lock:
            root = self.root.resolve()
            changed = self.changed_paths()
            if selected_paths is None:
                source_candidates = [
                    p for p in changed
                    if (p not in self.regression_tests or p in self.in_module_regressions)
                ]
            else:
                source_candidates = list(selected_paths)
            doctest_shifts = []
            for p_obj in source_candidates:
                cur_file = root / p_obj
                if p_obj.suffix == ".rs" and cur_file.is_file():
                    cur_c = cur_file.read_text(encoding="utf-8", errors="replace")
                    orig_raw = self.originals.get(p_obj)
                    if orig_raw and orig_raw[0] is not None:
                        orig_c = orig_raw[0].decode("utf-8", errors="replace")
                        w = _detect_doctest_line_shift(orig_c, cur_c, p_obj.as_posix())
                        if w:
                            doctest_shifts.append(w)
            scratch_files = []
            for p_obj in source_candidates:
                if self.originals.get(p_obj) is None:
                    name = p_obj.name
                    low_name = name.lower()
                    if (
                        (len(p_obj.parts) == 1 and p_obj.suffix in (".py", ".sh", ".ps1", ".bat") and name.startswith("_"))
                        or (p_obj.suffix in (".py", ".ps1", ".bat") and (name.startswith("_") or "splice" in low_name or "probe" in low_name or "tmp" in low_name))
                        or low_name.endswith(("_probe.rs", "_tmp.rs"))
                    ):
                        scratch_files.append(p_obj.as_posix())
            patch_text = self.build_patch(source_candidates) if source_candidates else ""
            added_dbg = [
                line[1:].strip() for line in patch_text.splitlines()
                if line.startswith("+") and not line.startswith("+++") and "dbg!(" in line
            ]
            trace_leftovers = []
            for p_obj in source_candidates:
                fpath = root / p_obj
                if p_obj.suffix == ".rs" and fpath.is_file():
                    txt = fpath.read_text(encoding="utf-8", errors="replace")
                    if "__TraceGuard" in txt or "[RUST_TRACE_" in txt:
                        trace_leftovers.append(p_obj.as_posix())
            return {
                "source_files": [p.as_posix() for p in source_candidates],
                "doctest_shifts": doctest_shifts,
                "scratch_files": scratch_files,
                "added_dbg_lines": added_dbg[:5],
                "trace_leftovers": trace_leftovers,
                "patch": patch_text,
            }

    def record_check_result(self, exit_code: int, diagnostics: list[dict]) -> str | None:
        with self._lock:
            errors = [d for d in diagnostics if d.get("level") == "error"]
            self._last_check_diagnostics = errors[:5] if exit_code != 0 else []
            if exit_code != 0 and errors:
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
            elif exit_code == 0:
                self._consecutive_same_check_errors = 0
                self._last_check_sig = None
                self._maybe_capture_compiling_snapshot()
            return None

    def record_test_result(self, manifest_path: str, test_target: str, exit_code: int, failures: list[str], executed: int, compile_errors: list[dict], failure_details=None, selection=None) -> str | None:
        with self._lock:
            if exit_code == 0 and executed > 0:
                self._consecutive_same_test_failures = 0
                self._last_test_fail_sig = None
                previous = self._last_test_failure_info
                if (previous.get("manifest_path") == manifest_path and previous.get("test_target") == test_target
                        and previous.get("selection", {}) == (selection or {})):
                    self._last_test_failure_info = {}
                return None
            if exit_code != 0:
                self._last_test_failure_info = {
                    "manifest_path": manifest_path,
                    "test_target": test_target,
                    "failures": list(failures[:10]),
                    "compile_errors": list((compile_errors or [])[:4]),
                    "failure_details": list((failure_details or [])[:10]),
                    "selection": dict(selection or {}),
                    "observed_revision": self._revision,
                }
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
            files = {}
            for rel in self.originals:
                fpath = root / rel
                if fpath.is_file():
                    files[rel] = (fpath.read_bytes(), fpath.stat().st_mode)
                else:
                    files[rel] = None
            return {
                "files": files,
                "originals": dict(self.originals),
                "regression_tests": set(self.regression_tests),
            "in_module_regressions": dict(self.in_module_regressions),
                "reproduced_regressions": {k: dict(v) for k, v in self._reproduced_regressions.items()},
                "passed_regressions": set(self._passed_regressions),
                "passed_regression_configs": {k: dict(v) for k, v in self._passed_regression_configs.items()},
                "regression_test_names": {k: set(v) for k, v in self._regression_test_names.items()},
                "reproduction_notes": dict(self._reproduction_notes),
                "existing_test_failures": [dict(x) for x in self._existing_test_failures],
                "last_test_failure": dict(self._last_test_failure_info),
                "executed_test_targets": [dict(x) for x in self._executed_test_targets],
                "verified": self.verified(),
            }

    def _maybe_capture_compiling_snapshot(self):
        with self._lock:
            selected = [
                p for p in self.changed_paths()
                if (p not in self.regression_tests or p in self.in_module_regressions)
            ]
            if not selected:
                return
            root = self.root.resolve()
            for p_obj in selected:
                if self.originals.get(p_obj) is None:
                    low = p_obj.name.lower()
                    if (
                        (len(p_obj.parts) == 1 and p_obj.suffix in (".py", ".sh", ".ps1", ".bat") and p_obj.name.startswith("_"))
                        or (p_obj.suffix in (".py", ".ps1", ".bat") and (p_obj.name.startswith("_") or "splice" in low or "probe" in low or "tmp" in low))
                        or low.endswith(("_probe.rs", "_tmp.rs"))
                    ):
                        return
                fpath = root / p_obj
                if p_obj.suffix == ".rs" and fpath.is_file():
                    cur_c = fpath.read_text(encoding="utf-8", errors="replace")
                    orig_raw = self.originals.get(p_obj)
                    if "dbg!(" in cur_c and (not orig_raw or orig_raw[0] is None or "dbg!(" not in orig_raw[0].decode("utf-8", errors="replace")):
                        return
            self._last_compiling_prod_snapshot = self._capture_snapshot()

    def _restore_snapshot(self, snap: dict):
        with self._lock:
            prior_clean_repro = {
                k: dict(v)
                for k, v in self._reproduced_regressions.items()
                if v.get("clean_red") and v.get("test_sha256")
            }
            root = self.root.resolve()
            # Revert any files touched after snapshot back to their original baseline state
            for rel in list(self.originals):
                if rel not in snap["originals"]:
                    orig = self.originals[rel]
                    fpath = root / rel
                    if orig is None:
                        if fpath.is_file() or fpath.is_symlink():
                            fpath.unlink()
                    else:
                        fpath.parent.mkdir(parents=True, exist_ok=True)
                        fpath.write_bytes(orig[0])
                        fpath.chmod(orig[1])
            # Restore files to their snapshot contents
            for rel, data in snap["files"].items():
                fpath = root / rel
                if data is None:
                    if fpath.is_file() or fpath.is_symlink():
                        fpath.unlink()
                else:
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_bytes(data[0])
                    fpath.chmod(data[1])
                _touch_crate_entry_mtime(root, rel)

            self.originals = dict(snap["originals"])
            self.regression_tests = set(snap["regression_tests"])
            self.in_module_regressions = dict(snap.get("in_module_regressions", {}))
            self._reproduced_regressions = {k: dict(v) for k, v in snap["reproduced_regressions"].items()}
            self._passed_regressions = set(snap["passed_regressions"])
            self._passed_regression_configs = {k: dict(v) for k, v in snap["passed_regression_configs"].items()}
            self._regression_test_names = {k: set(v) for k, v in snap["regression_test_names"].items()}
            self._reproduction_notes = dict(snap["reproduction_notes"])
            self._existing_test_failures = [dict(x) for x in snap.get("existing_test_failures", [])]
            self._last_test_failure_info = dict(snap.get("last_test_failure", {}))
            self._executed_test_targets = [dict(x) for x in snap.get("executed_test_targets", [])]
            self._passed_suites.clear()
            self._revision += 1
            restored_targets = set(self.get_regression_targets().keys())
            for r_key in restored_targets:
                prev_info = prior_clean_repro.get(r_key)
                if prev_info and prev_info.get("test_sha256") == self._target_sha256(r_key):
                    cur_r = self._reproduced_regressions.get(r_key)
                    if not cur_r or not cur_r.get("clean_red"):
                        self._reproduced_regressions[r_key] = dict(prev_info)
                        self._reproduction_notes.pop(r_key, None)
            all_passed = bool(restored_targets) and restored_targets <= self._passed_regressions
            all_repro = (not self.require_reproduction) or all(self._is_target_reproduced(t) for t in restored_targets)
            self._verified_revision = self._revision if (snap["verified"] or (all_passed and all_repro)) else None
            self._consecutive_edit_failures.clear()
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
            snap = self._checkpoints[clean_name]
            recovery_name = None
            if self._restore_would_drop_last_substantive_change(snap):
                recovery_name = f"__recovery_before_restore_{clean_name}_{self._revision}"
                self._checkpoints[recovery_name] = self._capture_snapshot()
            self._restore_snapshot(snap)
            changed = [p.as_posix() for p in self.changed_paths()]
            message = (
                f"Restored checkpoint '{clean_name}' "
                f"({len(changed)} changed file(s), verified={self.verified()})."
            )
            if recovery_name:
                message += f" Preserved the prior working copy as checkpoint '{recovery_name}' because this restore removed its last substantive change."
            return message

    def restore_partial_snapshot(self, baseline: dict, paths) -> None:
        """Restore only a failed repair unit's files; invalidate global Green evidence.

        Unrelated source edits and clean Red evidence survive. A previously passing
        whole-crate suite is not evidence for a newly combined partial patch.
        """
        with self._lock:
            snap = self._capture_snapshot()
            affected = {Path(p) for p in paths}
            for path in affected:
                snap["files"][path] = baseline["files"].get(path, self.originals.get(path))
                if path in baseline["regression_tests"]:
                    snap["regression_tests"].add(path)
                else:
                    snap["regression_tests"].discard(path)
                if path in baseline.get("in_module_regressions", {}):
                    snap["in_module_regressions"][path] = baseline["in_module_regressions"][path]
                else:
                    snap["in_module_regressions"].pop(path, None)
            # Recover the baseline test identity/oracle when this unit rewrote a
            # temporary test; unrelated targets keep their own clean Red evidence.
            for path in affected & baseline["regression_tests"]:
                info = baseline.get("in_module_regressions", {}).get(path)
                key = (info["manifest"], info["name"]) if info else (path.parent.parent / "Cargo.toml", path.stem)
                for field in ("reproduced_regressions", "regression_test_names", "reproduction_notes"):
                    if key in baseline[field]:
                        snap[field][key] = baseline[field][key]
            snap["verified"] = False
            snap["passed_regressions"] = set()
            snap["passed_regression_configs"] = {}
            snap["executed_test_targets"] = []
            self._restore_snapshot(snap)

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

    def dump_checkpoint_dir(self, target_dir: Path | str, snap: dict | None = None) -> dict:
        """Export working copy overlay and checkpoint metadata to an inspectable directory."""
        with self._lock:
            target_dir = Path(target_dir).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            files_dir = target_dir / "files"
            orig_dir = target_dir / "originals"
            files_dir.mkdir(parents=True, exist_ok=True)
            orig_dir.mkdir(parents=True, exist_ok=True)

            if snap is None:
                snap = self._capture_snapshot()

            files_meta = []
            for rel, data in snap["files"].items():
                if data is None:
                    files_meta.append({"path": rel.as_posix(), "exists": False, "mode": None})
                else:
                    dest = files_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data[0])
                    files_meta.append({"path": rel.as_posix(), "exists": True, "mode": data[1]})

            orig_meta = []
            for rel, orig in snap["originals"].items():
                if orig is None:
                    orig_meta.append({"path": rel.as_posix(), "exists": False, "mode": None, "mtime_ns": None})
                else:
                    dest = orig_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(orig[0])
                    mtime_ns = orig[2] if len(orig) > 2 else None
                    orig_meta.append({"path": rel.as_posix(), "exists": True, "mode": orig[1], "mtime_ns": mtime_ns})

            meta = {
                "format_version": 1,
                "files": files_meta,
                "originals": orig_meta,
                "regression_tests": [p.as_posix() for p in snap["regression_tests"]],
                "in_module_regressions": {p.as_posix(): info for p, info in snap.get("in_module_regressions", {}).items()},
                "reproduced_regressions": [
                    {"manifest": k[0].as_posix(), "name": k[1], "info": v}
                    for k, v in snap.get("reproduced_regressions", {}).items()
                ],
                "passed_regressions": [
                    {"manifest": k[0].as_posix(), "name": k[1]}
                    for k in snap.get("passed_regressions", [])
                ],
                "passed_regression_configs": [
                    {"manifest": k[0].as_posix(), "name": k[1], "config": v}
                    for k, v in snap.get("passed_regression_configs", {}).items()
                ],
                "regression_test_names": [
                    {"manifest": k[0].as_posix(), "name": k[1], "test_names": sorted(v)}
                    for k, v in snap.get("regression_test_names", {}).items()
                ],
                "reproduction_notes": [
                    {"manifest": k[0].as_posix(), "name": k[1], "note": v}
                    for k, v in snap.get("reproduction_notes", {}).items()
                ],
                "existing_test_failures": snap.get("existing_test_failures", []),
                "last_test_failure": snap.get("last_test_failure", {}),
                "executed_test_targets": snap.get("executed_test_targets", []),
                "verified": bool(snap.get("verified", False)),
            }
            (target_dir / "state.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return meta

    def load_checkpoint_dir(self, source_dir: Path | str) -> dict:
        """Load working copy overlay and restore PatchState from an exported checkpoint directory."""
        with self._lock:
            source_dir = Path(source_dir).resolve()
            state_json = source_dir / "state.json"
            if not state_json.is_file():
                raise FileNotFoundError(f"Missing state.json in {source_dir}")
            meta = json.loads(state_json.read_text(encoding="utf-8"))
            files_dir = source_dir / "files"
            orig_dir = source_dir / "originals"

            files_dict = {}
            for item in meta.get("files", []):
                rel = Path(item["path"])
                if item["exists"]:
                    src_f = files_dir / rel
                    files_dict[rel] = (src_f.read_bytes(), item["mode"])
                else:
                    files_dict[rel] = None

            orig_dict = {}
            for item in meta.get("originals", []):
                rel = Path(item["path"])
                if item["exists"]:
                    src_f = orig_dir / rel
                    mtime = item.get("mtime_ns")
                    entry = (src_f.read_bytes(), item["mode"]) if mtime is None else (src_f.read_bytes(), item["mode"], mtime)
                    orig_dict[rel] = entry
                else:
                    orig_dict[rel] = None

            snap = {
                "files": files_dict,
                "originals": orig_dict,
                "regression_tests": {Path(p) for p in meta.get("regression_tests", [])},
                "in_module_regressions": {Path(p): info for p, info in meta.get("in_module_regressions", {}).items()},
                "reproduced_regressions": {
                    (Path(x["manifest"]), x["name"]): x["info"]
                    for x in meta.get("reproduced_regressions", [])
                },
                "passed_regressions": {
                    (Path(x["manifest"]), x["name"])
                    for x in meta.get("passed_regressions", [])
                },
                "passed_regression_configs": {
                    (Path(x["manifest"]), x["name"]): x["config"]
                    for x in meta.get("passed_regression_configs", [])
                },
                "regression_test_names": {
                    (Path(x["manifest"]), x["name"]): set(x.get("test_names", []))
                    for x in meta.get("regression_test_names", [])
                },
                "reproduction_notes": {
                    (Path(x["manifest"]), x["name"]): x.get("note", "")
                    for x in meta.get("reproduction_notes", [])
                },
                "existing_test_failures": meta.get("existing_test_failures", []),
                "last_test_failure": meta.get("last_test_failure", {}),
                "executed_test_targets": meta.get("executed_test_targets", []),
                "verified": bool(meta.get("verified", False)),
            }
            self._restore_snapshot(snap)
            return snap

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
                backup[reg_rel] = (reg_file.read_bytes(), reg_file.stat().st_mode)
                reg_file.unlink()
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
                    _append_cargo_feature_flags(cmd, cfg)
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
            for reg_rel, (content, mode) in backup.items():
                reg_file = root / reg_rel
                reg_file.parent.mkdir(parents=True, exist_ok=True)
                reg_file.write_bytes(content)
                reg_file.chmod(mode)
        if not warnings:
            return ""
        return (
            "\nWarning: existing test target(s) failed 'cargo check --tests' (non-blocking). "
            "Check whether your edits unintentionally broke existing callers:\n" + "\n".join(warnings) + "\n"
        )

    def run_existing_test_smoke(self, selected_paths: list[Path], timeout: int = 90) -> list[dict]:
        """Run a small set of existing test targets after temporary tests are stripped.

        This is intentionally a bounded signal for the Verifier. It records failures
        for review and never replaces the paired regression test or post-strip check.
        """
        if not shutil.which("cargo") or not selected_paths:
            return []
        configurations = sorted(
            (str(manifest), name, json.dumps(config, sort_keys=True))
            for (manifest, name), config in self._passed_regression_configs.items()
        )
        cache_key = (self._revision, tuple(sorted(str(p) for p in selected_paths)), tuple(configurations), timeout)
        if self._existing_test_smoke_cache and self._existing_test_smoke_cache[0] == cache_key:
            return [dict(item) for item in self._existing_test_smoke_cache[1]]
        root = self.root.resolve()
        manifests = set()
        for rel in selected_paths:
            cur = (root / rel).parent
            while cur != root.parent:
                if (cur / "Cargo.toml").is_file():
                    manifests.add((cur / "Cargo.toml").resolve())
                    break
                cur = cur.parent
        manifests = [m for m in manifests if _manifest_has_targets(m)]
        if not manifests:
            return []

        backup = {}
        for reg_rel in self.regression_tests:
            fpath = root / reg_rel
            if reg_rel in self.in_module_regressions:
                if fpath.is_file():
                    stat = fpath.stat()
                    backup[reg_rel] = (fpath.read_bytes(), stat.st_mode, stat.st_mtime_ns)
                    fpath.write_text(_strip_in_module_tests(fpath.read_text(encoding="utf-8")), encoding="utf-8")
            elif fpath.is_file():
                stat = fpath.stat()
                backup[reg_rel] = (fpath.read_bytes(), stat.st_mode, stat.st_mtime_ns)
                fpath.unlink()

        results = []
        baseline_candidates = []
        try:
            for manifest in sorted(manifests)[:2]:
                cfgs = [
                    cfg for (m, _name), cfg in self._passed_regression_configs.items()
                    if (root / m).resolve() == manifest.resolve()
                ]
                cfg = cfgs[0] if cfgs else {}
                common = ["cargo", "test", "--manifest-path", str(manifest)]
                if cfg.get("package"):
                    common.append(f"--package={cfg['package']}")
                _append_cargo_feature_flags(common, cfg)

                commands = []
                if (manifest.parent / "src/lib.rs").is_file():
                    commands.append((common + ["--lib"], "lib"))
                targets = discover_integration_test_targets(manifest)
                if targets:
                    stems = {p.stem.lower() for p in selected_paths}
                    related = {
                        name for name in targets
                        if any(stem in name.lower() or name.lower() in stem for stem in stems if len(stem) >= 3)
                    }
                    integration_targets = {name for name in targets if "integration" in name.lower()}
                    related.update(integration_targets)
                    selected_text = " ".join(str(path).lower() for path in selected_paths)
                    if not related and any(token in selected_text for token in ("template", "build.rs", "installer", "snapshot")):
                        related.update(targets[:3])
                    if not related:
                        related.update(targets if len(targets) <= 6 else targets[:3])
                    related_non_integration = related - integration_targets
                    if not related_non_integration:
                        related_non_integration = set(targets[:3]) - integration_targets
                    selected_targets = sorted(integration_targets) + sorted(related_non_integration)[:3]
                    snap_filter = _snapshot_smoke_filter(manifest, selected_paths)
                    if snap_filter:
                        for target_name in (sorted(integration_targets) or selected_targets[:1]):
                            commands.append((common + ["--test", target_name, "--", snap_filter], f"test:{target_name}[{snap_filter}]"))
                    for target_name in selected_targets:
                        commands.append((common + ["--test", target_name], f"test:{target_name}"))
                for command, target in commands:
                    if target in self._slow_smoke_targets:
                        results.append({
                            "manifest": str(manifest.relative_to(root)),
                            "target": target,
                            "status": "timeout",
                            "classification": "unknown",
                            "baseline_status": "unknown",
                            "tests_executed": 0,
                            "details": (
                                f"skipped: this target already exceeded {timeout}s in an earlier "
                                "smoke run during this session"
                            ),
                            "observed_revision": self._revision,
                        })
                        continue
                    try:
                        res = subprocess.run(
                            command, cwd=root, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout, env=self.rust_env,
                        )
                        summaries = re.findall(r"test result: (?:ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed", res.stdout)
                        executed = sum(int(passed) + int(failed) for passed, failed in summaries)
                        environment_failure = _smoke_environment_failure(res.stdout, res.stderr) if res.returncode else False
                        result = {
                            "manifest": str(manifest.relative_to(root)),
                            "target": target,
                            "status": (
                                "environment_error" if res.returncode and environment_failure
                                else "failed" if res.returncode
                                else "passed" if executed else "no_tests"
                            ),
                            "classification": "environment" if environment_failure else "unknown",
                            "baseline_status": "unknown" if res.returncode else None,
                            "tests_executed": executed,
                            "failed_tests": sorted(set(re.findall(
                                r"^test (.+?) \.\.\. FAILED\s*$", res.stdout, re.MULTILINE
                            ))),
                            "details": _extract_smoke_failure_details(res.stdout, res.stderr) if res.returncode else "",
                            "observed_revision": self._revision,
                        }
                        results.append(result)
                        if result["status"] == "failed":
                            baseline_candidates.append((result, command, manifest))
                    except subprocess.TimeoutExpired:
                        self._slow_smoke_targets.add(target)
                        results.append({
                            "manifest": str(manifest.relative_to(root)),
                            "target": target,
                            "status": "timeout",
                            "classification": "unknown",
                            "baseline_status": "unknown",
                            "tests_executed": 0,
                            "details": f"cargo test exceeded {timeout}s; completion is unknown",
                            "observed_revision": self._revision,
                        })
                    except OSError as exc:
                        results.append({
                            "manifest": str(manifest.relative_to(root)),
                            "target": target,
                            "status": "environment_error",
                            "classification": "environment",
                            "baseline_status": "unknown",
                            "tests_executed": 0,
                            "details": f"unable to execute cargo: {exc}",
                            "observed_revision": self._revision,
                        })
            manifest_deadlines = {}
            for result, command, manifest in baseline_candidates:
                deadline = manifest_deadlines.setdefault(manifest, time.monotonic() + max(1, timeout))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    continue
                current_files = {}
                try:
                    for rel, original in self.originals.items():
                        if rel in self.regression_tests and rel not in self.in_module_regressions:
                            continue
                        path = root / rel
                        if path.is_file():
                            current_stat = path.stat()
                            current_files[rel] = (path.read_bytes(), current_stat.st_mode, current_stat.st_mtime_ns)
                        else:
                            current_files[rel] = None
                    for rel, original in self.originals.items():
                        if rel in self.regression_tests and rel not in self.in_module_regressions:
                            continue
                        path = root / rel
                        if original is None:
                            if path.exists() or path.is_symlink():
                                path.unlink()
                        else:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(original[0])
                            path.chmod(original[1])
                            os.utime(path, None)
                            _touch_crate_entry_mtime(root, rel)
                    baseline_res = subprocess.run(
                        command, cwd=root, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=remaining, env=self.rust_env,
                    )
                    baseline_summaries = re.findall(
                        r"test result: (?:ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed",
                        baseline_res.stdout,
                    )
                    baseline_executed = sum(int(passed) + int(failed) for passed, failed in baseline_summaries)
                    if baseline_res.returncode == 0 and baseline_executed:
                        result["baseline_status"] = "passed"
                        result["classification"] = "code"
                    elif baseline_res.returncode != 0 and not _smoke_environment_failure(baseline_res.stdout, baseline_res.stderr):
                        result["baseline_status"] = "failed"
                        baseline_failures = sorted(set(re.findall(
                            r"^test (.+?) \.\.\. FAILED\s*$", baseline_res.stdout, re.MULTILINE
                        )))
                        result["baseline_failed_tests"] = baseline_failures
                        if result["failed_tests"] and baseline_failures:
                            result["new_failed_tests"] = sorted(set(result["failed_tests"]) - set(baseline_failures))
                            # Identical failing test names do not prove the failure is
                            # pre-existing: a snapshot test can keep its name while its
                            # rendered output changes. Compare the snapshot hunks too.
                            if result["new_failed_tests"]:
                                result["classification"] = "code"
                            elif _snapshot_hunks_differ(result.get("details", ""), baseline_res.stdout):
                                result["classification"] = "code"
                                result["new_failed_tests"] = list(result["failed_tests"])
                                result["details"] += (
                                    "\nThe same tests fail on the unpatched baseline, but their "
                                    "snapshot output differs, so the patch changed the rendered result."
                                )
                            else:
                                result["classification"] = "unknown"
                                result["details"] += (
                                    "\nThe unpatched baseline fails the same tests with the same output, so this "
                                    "failure is NOT confirmed pre-existing: it is unconfirmed, and the patch "
                                    "cannot be cleared by it alone."
                                )
                            result["details"] += "\nCompared failed test names with the unpatched baseline."
                        else:
                            result["classification"] = "unknown"
                            result["details"] += "\nBaseline also failed; individual failures could not be compared."
                        if result["classification"] == "unknown":
                            result["baseline_label"] = "BASELINE_UNCONFIRMED_NOT_PREEXISTING"
                    else:
                        result["baseline_status"] = "unknown"
                        result["classification"] = "unknown"
                        result["details"] += "\nBaseline rerun did not produce a reliable comparison."
                except subprocess.TimeoutExpired:
                    result["baseline_status"] = "unknown"
                    result["classification"] = "unknown"
                    result["details"] += "\nBaseline rerun timed out; this result is not attributed to the patch."
                except OSError as exc:
                    result["baseline_status"] = "unknown"
                    result["classification"] = "unknown"
                    result["details"] += f"\nBaseline rerun could not execute: {exc}"
                finally:
                    for rel, data in current_files.items():
                        path = root / rel
                        if data is None:
                            if path.exists() or path.is_symlink():
                                path.unlink()
                        else:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(data[0])
                            path.chmod(data[1])
                            os.utime(path, None)
                            _touch_crate_entry_mtime(root, rel)
        finally:
            for reg_rel, (content, mode, mtime_ns) in backup.items():
                fpath = root / reg_rel
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_bytes(content)
                fpath.chmod(mode)
                os.utime(fpath, None)
        self._existing_test_smoke_cache = (cache_key, [dict(item) for item in results])
        return results

    def verify_pre_fix_reproduction(self, targets=None):
        if not shutil.which("cargo"):
            return
        root = self.root.resolve()
        if targets is None:
            targets = set(self.get_regression_targets().keys())
        needed = [
            t for t in targets
            if not (self._is_target_reproduced(t) and self._reproduced_regressions[t].get("clean_red", False))
        ]
        if not needed:
            return
        valid_needed = [t for t in needed if _manifest_has_targets(root / t[0])]
        if not valid_needed:
            return

        prod_changed = [
            p for p in self.changed_paths()
            if p not in self.regression_tests or p in self.in_module_regressions
        ]
        backup = {}
        for rel in prod_changed:
            fpath = root / rel
            if fpath.is_file():
                st = fpath.stat()
                backup[rel] = (fpath.read_bytes(), st.st_mode, st.st_mtime_ns)
            else:
                backup[rel] = None

        try:
            for rel in prod_changed:
                orig = self.originals.get(rel)
                fpath = root / rel
                if orig is None:
                    if fpath.is_file() or fpath.is_symlink():
                        fpath.unlink()
                else:
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_bytes(orig[0])
                    fpath.chmod(orig[1])
                    _touch_crate_entry_mtime(root, rel)
                    # If this file has an in-module test, re-attach the test block to the reverted original file
                    if rel in self.in_module_regressions:
                        test_info = self.in_module_regressions[rel]
                        cur_text = orig[0].decode("utf-8", errors="replace")
                        cur_text = _strip_in_module_tests(cur_text)
                        target_code = (
                            f"{cur_text}\n"
                            f"// __TEST_REGRESSION_START: {test_info['name']}__\n"
                            f"#[cfg(test)]\n"
                            f"mod {test_info['name']} {{\n"
                            f"    use super::*;\n"
                            f"{test_info['content']}\n"
                            f"}}\n"
                            f"// __TEST_REGRESSION_END: {test_info['name']}__\n"
                        )
                        fpath.write_text(target_code, encoding="utf-8")

            all_target_info = self.get_regression_targets()
            for rel_manifest, name in valid_needed:
                manifest_file = root / rel_manifest
                t_info = all_target_info.get((rel_manifest, name), {})
                cfg = self._passed_regression_configs.get((rel_manifest, name), {})
                cmd = [
                    "cargo", "test", "--message-format=json",
                    "--manifest-path", str(manifest_file),
                    "--no-fail-fast",
                ]
                if t_info.get("is_in_module"):
                    kind, bin_name = _resolve_in_module_test_target(manifest_file)
                    if kind == "bin" and bin_name:
                        cmd.append(f"--bin={bin_name}")
                    else:
                        cmd.append("--lib")
                    cmd.append(name)
                else:
                    cmd.append(f"--test={name}")

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
                        encoding="utf-8", errors="replace", timeout=180, env=self.rust_env,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ValueError(
                        f"Paired reproduction check timed out while running '{name}' on the original code."
                    ) from exc

                started_counts = re.findall(r"^running (\d+) tests?\s*$", res.stdout, re.MULTILINE)
                tests_started = sum(int(c) for c in started_counts)
                counts = re.findall(
                    r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;",
                    res.stdout, re.MULTILINE,
                )
                executed = sum(int(p) + int(f) for p, f, _ in counts)
                failures = re.findall(r"^test (.+?) \.\.\. FAILED\s*$", res.stdout, re.MULTILINE)

                if res.returncode == 0 and executed > 0:
                    self._reproduced_regressions.pop((rel_manifest, name), None)
                    self._reproduction_notes[(rel_manifest, name)] = "passed on unpatched original code"
                    remaining = self.remaining_api_calls()
                    if remaining is not None and remaining <= 20:
                        self._reproduced_regressions[(rel_manifest, name)] = {
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
                    test_sha = self._target_sha256((rel_manifest, name))
                    self._reproduced_regressions[(rel_manifest, name)] = {
                        "clean_red": True,
                        "runtime_failure": executed > 0 or tests_started > 0,
                        "compile_reproduction": executed == 0 and tests_started == 0,
                        "executed": executed,
                        "failures": failures,
                        "returncode": res.returncode,
                        "test_sha256": test_sha,
                    }
                    self._reproduction_notes.pop((rel_manifest, name), None)
                    if test_sha and (rel_manifest, name) not in self._locked_regressions:
                        # The Architect already released this oracle; do not re-lock it.
                        if (str(rel_manifest), name) not in self._unlocked_regressions:
                            t_info = self.get_regression_targets().get((rel_manifest, name), {})
                            reg_path = t_info.get("path", "")
                            self._locked_regressions[(str(rel_manifest), name)] = {
                                "sha256": test_sha,
                                "path": reg_path.as_posix() if isinstance(reg_path, Path) else str(reg_path),
                                "name": name,
                                "is_in_module": bool(t_info.get("is_in_module")),
                                "manifest": str(rel_manifest),
                                "locked_at_revision": self._revision,
                            }
            all_targets = set(self.get_regression_targets().keys())
            if all_targets and all_targets <= self._passed_regressions and all(self._is_target_reproduced(t) for t in all_targets):
                self._verified_revision = self._revision
                if not self.require_reproduction or self.reproduced_cleanly():
                    self._best_verified_snapshot = self._capture_snapshot()
        finally:
            for rel, data in backup.items():
                fpath = root / rel
                if data is None:
                    if fpath.is_file() or fpath.is_symlink():
                        fpath.unlink()
                else:
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_bytes(data[0])
                    fpath.chmod(data[1])
                _touch_crate_entry_mtime(root, rel)

    def run_post_strip_check(self, selected_paths: list[Path]):
        if not shutil.which("cargo"):
            return {"status": "unavailable", "warnings": ["Cargo is not available."]}
        root = self.root.resolve()
        affected_manifests = set()
        for reg_rel in self.regression_tests:
            if reg_rel in self.in_module_regressions:
                manifest_rel = self.in_module_regressions[reg_rel]["manifest"]
            else:
                manifest_rel = reg_rel.parent.parent / "Cargo.toml"
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
            return {"status": "unavailable", "warnings": ["No affected Cargo manifest was found."]}

        backup = {}
        for reg_rel in self.regression_tests:
            if reg_rel in self.in_module_regressions:
                fpath = root / reg_rel
                if fpath.is_file():
                    content_str = fpath.read_text(encoding="utf-8")
                    st = fpath.stat()
                    backup[reg_rel] = (content_str, st.st_mode, st.st_mtime_ns)
                    fpath.write_text(_strip_in_module_tests(content_str), encoding="utf-8")
            else:
                reg_file = root / reg_rel
                if reg_file.is_file():
                    st = reg_file.stat()
                    backup[reg_rel] = (reg_file.read_bytes(), st.st_mode, st.st_mtime_ns)
                    reg_file.unlink()
        warnings = []
        try:
            has_lock = (root / "Cargo.lock").is_file()
            for manifest in sorted(valid_manifests):
                cmd = ["cargo", "check", "--tests", "--manifest-path", str(manifest)]
                if has_lock or (manifest.parent / "Cargo.lock").is_file():
                    cmd.append("--locked")
                matching_cfgs = [
                    cfg for (m, _name), cfg in self._passed_regression_configs.items()
                    if (root / m).resolve() == manifest.resolve()
                ]
                active_cfg = list(self._active_cfg_features.get(manifest.resolve()) or ())
                if matching_cfgs:
                    first_cfg = dict(matching_cfgs[0])
                    if active_cfg and not first_cfg.get("all_features") and not first_cfg.get("no_default_features"):
                        merged_feats = list(dict.fromkeys([*(first_cfg.get("features") or ()), *active_cfg]))
                        first_cfg["features"] = merged_feats
                    if first_cfg.get("package"):
                        cmd.append(f"--package={first_cfg['package']}")
                    _append_cargo_feature_flags(cmd, first_cfg)
                elif active_cfg:
                    _append_cargo_feature_flags(cmd, {"features": active_cfg})
                else:
                    _append_cargo_feature_flags(cmd)

                result = self._compare_workspace_compile_with_baseline(cmd, timeout=120)
                if result["status"] == "regression":
                    rel_manifest = manifest.relative_to(root).as_posix() if manifest.is_relative_to(root) else str(manifest)
                    raise ValueError(
                        f"Post-strip verification failed for {rel_manifest} under 'cargo check'; "
                        f"the unpatched checkout passed:\n{result['details']}"
                    )
                if result["status"] != "passed":
                    warnings.append(result["details"])
            downstream_pkgs = self._find_downstream_workspace_packages(valid_manifests)
            ws_manifest = root / "Cargo.toml"
            for dep_pkg in downstream_pkgs:
                dep_cmd = ["cargo", "check", "--tests", "--manifest-path", str(ws_manifest), f"--package={dep_pkg}"]
                _append_cargo_feature_flags(dep_cmd)
                if has_lock:
                    dep_cmd.append("--locked")
                result = self._compare_workspace_compile_with_baseline(dep_cmd, timeout=120)
                if result["status"] == "regression":
                    raise ValueError(
                        f"Post-strip downstream workspace check failed for dependent package '{dep_pkg}'. "
                        f"The unpatched checkout passed:\n{result['details']}"
                    )
                if result["status"] != "passed":
                    warnings.append(result["details"])
            workspace_manifest = root / "Cargo.toml"
            if workspace_manifest.is_file():
                workspace_text = workspace_manifest.read_text(encoding="utf-8", errors="replace")
                workspace_cmd = ["cargo", "check", "--all-targets", "--manifest-path", str(workspace_manifest)]
                _append_cargo_feature_flags(workspace_cmd)
                if "[workspace]" in workspace_text:
                    workspace_cmd.append("--workspace")
                if has_lock:
                    workspace_cmd.append("--locked")
                workspace_result = self._compare_workspace_compile_with_baseline(workspace_cmd)
                if workspace_result["status"] == "regression":
                    raise ValueError("Workspace compilation regressed against the unpatched checkout:\n"
                                     + workspace_result["details"])
                if workspace_result["status"] != "passed":
                    warnings.append(workspace_result["details"])
        finally:
            for reg_rel, data in backup.items():
                if reg_rel in self.in_module_regressions:
                    fpath = root / reg_rel
                    fpath.write_text(data[0], encoding="utf-8")
                    fpath.chmod(data[1])
                    if len(data) >= 3 and data[2] is not None:
                        try:
                            os.utime(fpath, ns=(data[2], data[2]))
                        except Exception:
                            pass
                else:
                    reg_file = root / reg_rel
                    reg_file.parent.mkdir(parents=True, exist_ok=True)
                    reg_file.write_bytes(data[0])
                    reg_file.chmod(data[1])
                    if len(data) >= 3 and data[2] is not None:
                        try:
                            os.utime(reg_file, ns=(data[2], data[2]))
                        except Exception:
                            pass
        return {"status": "inconclusive" if warnings else "passed", "warnings": warnings}

    def _compare_workspace_compile_with_baseline(self, command: list[str], timeout: int = 300) -> dict:
        root = self.root.resolve()
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout, env=self.rust_env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"status": "inconclusive", "details": f"Workspace compile check could not complete: {exc}"}
        if result.returncode == 0:
            return {"status": "passed", "details": ""}
        failure = (result.stderr.strip() or result.stdout.strip())[-3000:]
        if _smoke_environment_failure(result.stdout, result.stderr):
            return {"status": "inconclusive", "details": f"Workspace compile check encountered an environment failure: {failure}"}

        current_files = {}
        try:
            for rel, original in self.originals.items():
                if rel in self.regression_tests and rel not in self.in_module_regressions:
                    continue
                path = root / rel
                if path.is_file():
                    current_stat = path.stat()
                    current_files[rel] = (path.read_bytes(), current_stat.st_mode, current_stat.st_mtime_ns)
                else:
                    current_files[rel] = None
                if original is None:
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(original[0])
                    path.chmod(original[1])
                    _touch_crate_entry_mtime(root, rel)
            try:
                baseline = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                          encoding="utf-8", errors="replace", timeout=timeout, env=self.rust_env)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return {"status": "inconclusive", "details": f"Workspace baseline compile could not complete: {exc}; candidate: {failure}"}
            if baseline.returncode == 0:
                return {"status": "regression", "details": failure}
            baseline_failure = (baseline.stderr.strip() or baseline.stdout.strip())[-1000:]
            if "manifest path" in baseline_failure.lower() and "does not exist" in baseline_failure.lower():
                return {"status": "regression", "details": failure}
            return {"status": "inconclusive", "details": f"Workspace baseline also failed: {baseline_failure}; candidate: {failure}"}
        finally:
            for rel, current in current_files.items():
                path = root / rel
                if current is None:
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(current[0])
                    path.chmod(current[1])
                    os.utime(path, ns=(current[2], current[2]))
                    _touch_crate_entry_mtime(root, rel)

    def auto_submit_verified_patch(self, allow_snapshot_fallback: bool = True) -> str:
        with self._lock:
            if self.patch:
                return self.patch

            def _try_current():
                selected = [path for path in self.changed_paths() if (path not in self.regression_tests or path in self.in_module_regressions)]
                targets = set(self.get_regression_targets().keys())
                if selected and targets and targets <= self._passed_regressions:
                    if self.require_reproduction and not self.reproduced_cleanly():
                        try:
                            self.verify_pre_fix_reproduction(targets)
                        except Exception as exc:
                            self._last_submit_error = str(exc)
                            return ""
                        if not self.reproduced_cleanly():
                            self._last_submit_error = "Regression test did not reproduce cleanly on the original code"
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
            return ""


    def invalidate_write(self):
        with self._lock:
            self._revision += 1
            self._verified_revision = None
            self._passed_regressions.clear()
            self._passed_regression_configs.clear()
            self._regression_test_names.clear()
            self._passed_suites.clear()
            self._executed_test_targets.clear()
            self._last_submit_error = ""
            self.patch = ""

    def start_test(self):
        with self._lock:
            self._test_generation += 1
            self.patch = ""
            return self._revision, self._test_generation

    def _record_executed_target(self, manifest_path: str, test_target: str, lib: bool, bin_target: str, tests: bool, doc: bool, test_filter: str, status: str):
        label = test_target or bin_target or ("doc" if doc else ("lib" if lib else ("tests" if tests else "default")))
        entry = {
            "manifest": manifest_path,
            "target": label,
            "filter": test_filter,
            "status": status,
        }
        self._executed_test_targets = [
            x for x in self._executed_test_targets
            if not (x.get("manifest") == manifest_path and x.get("target") == label and x.get("filter") == test_filter)
        ] + [entry]

    def pass_test(self, token, manifest_path="Cargo.toml", package="", test_target="",
                  test_filter="", lib=False, bin_target="", tests=False, passed_tests=None, ignored_tests=0,
                  features=None, all_features=False, no_default_features=False, target="",
                  doc=False, executed=0, filtered_tests=None):
        with self._lock:
            if token == (self._revision, self._test_generation):
                effective_manifest = self._resolve_manifest_target(manifest_path, package)
                all_targets_info = self.get_regression_targets()
                reg_matched = False
                for (relative_manifest, name), tinfo in all_targets_info.items():
                    target_manifest = (self.root / relative_manifest).resolve()
                    same_manifest = (target_manifest == effective_manifest or (not package and (self.root / manifest_path).resolve() == target_manifest))
                    if not same_manifest:
                        continue
                    matched = False
                    if not tinfo["is_in_module"]:
                        complete_target = not test_filter or filtered_tests == 0
                        matched = test_target == name and complete_target
                    else:
                        in_mod_passed = any(name in pt for pt in (passed_tests or []))
                        matched = (
                            not test_target
                            and (lib or bool(bin_target) or tests)
                            and ((bool(test_filter) and (test_filter == name or test_filter in name)) or in_mod_passed)
                        )
                    if matched:
                        reg_matched = True
                        key = (relative_manifest, name)
                        if ignored_tests:
                            self._passed_regressions.discard(key)
                            self._passed_regression_configs.pop(key, None)
                            self._regression_test_names.pop(key, None)
                            self._verified_revision = None
                        else:
                            self._passed_regressions.add(key)
                            self._passed_regression_configs[key] = {
                                "package": package,
                                "features": list(features) if features else None,
                                "all_features": bool(all_features),
                                "no_default_features": bool(no_default_features),
                                "target": target,
                            }
                            self._regression_test_names[key] = set(passed_tests or [])
                target_label = test_target or bin_target or ("lib" if lib else "tests")
                if self._existing_test_failures and not test_filter:
                    self._existing_test_failures = [
                        item for item in self._existing_test_failures
                        if item.get("target") != target_label
                    ]
                is_suite = (not test_target and not test_filter and not lib and not bin_target)
                if is_suite:
                    self._passed_suites.add(effective_manifest)
                targets = set(self.get_regression_targets().keys())
                reproduction_ok = (not self.require_reproduction) or all(self._is_target_reproduced(t) for t in targets)
                if not reg_matched or tests or doc or (not test_target and not test_filter):
                    self._record_executed_target(manifest_path, test_target, lib, bin_target, tests, doc, test_filter, "passed")
                self._maybe_capture_compiling_snapshot()
                if targets and targets <= self._passed_regressions and reproduction_ok:
                    self._verified_revision = self._revision
                    if not self.require_reproduction or self.reproduced_cleanly():
                        self._best_verified_snapshot = self._capture_snapshot()

    def fail_test(self, token, manifest_path="Cargo.toml", package="", test_target="",
                  test_filter="", lib=False, bin_target="", tests=False,
                  executed=0, failures=None, returncode=None, failed_target="",
                  compile_errors=None, tests_started=0, doc=False, failure_details=None,
                  filtered_tests=None):
        with self._lock:
            if token == (self._revision, self._test_generation):
                self._passed_suites.clear()
                effective_manifest = self._resolve_manifest_target(manifest_path, package)
                all_targets_info = self.get_regression_targets()
                any_regression_matched = False
                for (relative_manifest, name), tinfo in all_targets_info.items():
                    target_manifest = (self.root / relative_manifest).resolve()
                    key = (relative_manifest, name)
                    same_package = target_manifest == effective_manifest
                    matched = False
                    target_failed = False
                    if not tinfo["is_in_module"]:
                        target_failed = same_package and test_target == name
                        complete_target = not test_filter or filtered_tests == 0
                        matched = target_failed and complete_target
                    else:
                        in_mod_failed = any(name in ft for ft in (failures or []))
                        in_mod_info = self.in_module_regressions.get(tinfo["path"], {})
                        inner_fns = in_mod_info.get("fn_names") or set()
                        filter_matches_inner = bool(test_filter) and (
                            test_filter == name
                            or test_filter in name
                            or any(test_filter == fn or test_filter in fn for fn in inner_fns)
                        )
                        matched = (
                            same_package
                            and not test_target
                            and (lib or bool(bin_target) or tests)
                            and (filter_matches_inner or in_mod_failed)
                        )
                    if target_failed or matched or (not test_target and not test_filter and (failed_target == name or set(failures or []) & self._regression_test_names.get(key, set()))):
                        any_regression_matched = True
                        self._passed_regressions.discard(key)
                        self._passed_regression_configs.pop(key, None)
                        self._regression_test_names.pop(key, None)
                        self._verified_revision = None
                    if (target_manifest == effective_manifest or not package) and matched:
                        any_regression_matched = True
                        clean_red = not any(
                            p not in self.regression_tests or p in self.in_module_regressions
                            for p in self.changed_paths()
                        )
                        prior_clean = bool((self._reproduced_regressions.get(key) or {}).get("clean_red"))
                        if executed == 0 and tests_started == 0 and returncode is None:
                            if not (not clean_red and prior_clean):
                                self._reproduced_regressions.pop(key, None)
                                self._reproduction_notes[key] = "test failed to run or timed out"
                            continue
                        aborted_after_start = (tests_started > 0 and not compile_errors and returncode not in (0, None))
                        runtime_failure = (executed > 0 or aborted_after_start) and (returncode != 0 or returncode is None or bool(failures))
                        compile_repro = False
                        if executed == 0 and not aborted_after_start and returncode is not None and returncode != 0:
                            err_codes = {d.get("code") for d in (compile_errors or []) if d.get("code")}
                            if clean_red and bool(err_codes & self._API_COMPILE_ERROR_CODES):
                                compile_repro = True
                        if runtime_failure or compile_repro:
                            suspect_setup = False
                            if runtime_failure and failure_details and all(
                                d.get("kind") == "unexpected_panic" for d in failure_details
                            ):
                                detail_blob = " ".join(str(d.get("output") or "").lower() for d in failure_details)
                                if (
                                    any(m in detail_blob for m in ("no such file or directory", "notpresent", "cannot find binary", "permission denied"))
                                    and "src/" not in detail_blob
                                ):
                                    suspect_setup = True
                            self._reproduced_regressions[key] = {
                                "clean_red": clean_red,
                                "runtime_failure": runtime_failure,
                                "compile_reproduction": compile_repro,
                                "executed": executed,
                                "failures": list(failures or []),
                                "returncode": returncode,
                                "test_sha256": self._target_sha256(key),
                                "suspect_setup_panic": suspect_setup,
                            }
                            self._reproduction_notes.pop(key, None)
                            # Lock only runtime-confirmed Red oracles; compile-only missing-API tests stay editable
                            # so Fixer can fix import/type typos in the test without triggering replan_red_lock loops.
                            if clean_red and runtime_failure and not suspect_setup and key not in self._locked_regressions:
                                sha = self._target_sha256(key)
                                # The Architect already released this oracle; do not re-lock it.
                                if sha and key not in self._unlocked_regressions:
                                    reg_info = self.get_regression_targets().get(key, {})
                                    reg_path = reg_info.get("path", "")
                                    lock_key = (str(key[0]), key[1])
                                    self._locked_regressions[lock_key] = {
                                        "sha256": sha,
                                        "path": reg_path.as_posix() if isinstance(reg_path, Path) else str(reg_path),
                                        "name": key[1],
                                        "is_in_module": bool(reg_info.get("is_in_module")),
                                        "manifest": str(key[0]),
                                        "locked_at_revision": self._revision,
                                    }
                        else:
                            if not clean_red and prior_clean:
                                # Do not let a post-edit Cargo resolution / compile error erase an already confirmed Red proof.
                                continue
                            self._reproduced_regressions.pop(key, None)
                            self._reproduction_notes[key] = "test did not execute failing assertions or expose an API compile mismatch"
                if not any_regression_matched and (executed > 0 or tests_started > 0 or doc):
                    self._record_executed_target(manifest_path, test_target, lib, bin_target, tests, doc, test_filter, "failed")
                if not any_regression_matched and failures and any(p not in self.regression_tests for p in self.changed_paths()):
                    target_label = test_target or failed_target or bin_target or ("lib" if lib else "tests")
                    entry = {
                        "target": target_label,
                        "manifest": manifest_path,
                        "failures": list(failures[:10]),
                        "failure_details": list((failure_details or [])[:10]),
                        "observed_revision": self._revision,
                    }
                    self._existing_test_failures = [
                        item for item in self._existing_test_failures
                        if item.get("target") != target_label
                    ] + [entry]

    def verified(self):
        with self._lock:
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

    def cargo_lock_update_warning(manifest: Path) -> str:
        if not shutil.which("cargo"):
            return ""
        current = manifest.parent
        has_lockfile = False
        while current == root or current.is_relative_to(root):
            if (current / "Cargo.lock").is_file():
                has_lockfile = True
                break
            if current == root:
                break
            current = current.parent
        if not has_lockfile:
            return ""
        try:
            result = subprocess.run(
                [
                    "cargo", "metadata", "--offline", "--locked", "--format-version", "1",
                    "--manifest-path", str(manifest),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                env=rust_env,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if result.returncode == 0:
            return ""
        diagnostics = result.stderr.strip() or result.stdout.strip()
        lock_update_error = re.search(
            r"(?:cannot update (?:the )?(?:lock file|Cargo\.lock)|"
            r"(?:lock file|Cargo\.lock).{0,300}(?:needs? to be updated|must be updated|--locked was passed))",
            diagnostics,
            re.IGNORECASE | re.DOTALL,
        )
        if not lock_update_error:
            return ""
        return (
            "\n[Cargo.lock constraint] Cargo reports that this Cargo.toml change needs a lockfile update, "
            "but validation uses --locked and Cargo.lock is read-only. Revert the dependency/feature change "
            "or solve the issue using dependencies already in Cargo.lock."
        )

    def _record_and_format_cfg_notice(target: Path, relative: Path, changed_lines: list[int] | None = None) -> str:
        if relative.suffix != ".rs" or not target.is_file():
            return ""
        raw_feats = ckg_graph.features_for_file_lines(relative.as_posix(), changed_lines or None)
        if not raw_feats:
            return ""
        manifest_file = _nearest_manifest_for_file(root, target)
        if manifest_file is None:
            return ""
        declared = _manifest_declared_features(manifest_file)
        active = [f for f in raw_feats if f in declared and f != "default"]
        if not active:
            return ""
        merged = list(dict.fromkeys([*(state._active_cfg_features.get(manifest_file) or ()), *active]))
        state._active_cfg_features[manifest_file] = merged
        return (
            f"\n[Cfg Feature Notice: edited code is gated by #[cfg(feature = ...)]: {active}. "
            f"Verify with features={active} so feature-gated code is actually compiled.]"
        )

    @tool
    def read_file(path: str) -> str:
        """List a directory or read small UTF-8 files (files over 250 lines are previewed with line numbers; use read_lines for source files); paths are repository-relative."""
        try:
            target = resolve_path(path)
            if target.is_dir():
                return "\n".join(p.name + ("/" if p.is_dir() else "") for p in sorted(target.iterdir()) if p.name != ".git") or "(empty directory)"
            content = target.read_bytes().decode("utf-8")
            lines = content.splitlines()
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

    def _write_file_impl(path: str, content: str | None, force: bool = False, *, from_edit: bool = False) -> str:
        try:
            target = resolve_path(path)
            relative = target.relative_to(root)
            if relative in state.regression_tests and relative not in state.in_module_regressions:
                raise ValueError("Use write_regression_test to modify temporary regression tests")
            forbidden_dirs = [part for part in relative.parts[:-1] if part in ("tests", "benches")]
            if forbidden_dirs:
                raise ValueError(
                    f"Editing or creating files in {forbidden_dirs[0]}/ is prohibited to avoid benchmark "
                    "evaluation conflicts. Use write_regression_test to create temporary tests at "
                    "tests/test_regression_<name>.rs, or modify production code in src/."
                )
            if relative.name == "Cargo.lock":
                raise ValueError("Modifying Cargo.lock is prohibited to avoid breaking benchmark dependency environments")
            if not relative.parts or (content is not None and not isinstance(content, str)):
                raise ValueError("Provide a file path and string or null content")
            if target.exists() and not target.is_file():
                raise ValueError("Path is not a regular file")
            if content is None and not target.is_file():
                raise ValueError("Cannot delete a missing file")
            if not from_edit and not force and relative.suffix == ".rs":
                orig_entry = state.originals.get(relative) if relative in state.originals else None
                orig_bytes = orig_entry[0] if orig_entry is not None else (target.read_bytes() if target.is_file() else None)
                if orig_bytes is not None:
                    orig_line_count = len(orig_bytes.decode("utf-8", errors="replace").splitlines())
                    if orig_line_count >= 20:
                        if content is None:
                            raise ValueError(
                                f"Refusing to delete existing Rust source file '{relative.as_posix()}' ({orig_line_count} lines). "
                                "Use edit_file for targeted changes, revert_file to restore, or pass force=true."
                            )
                        new_line_count = len(content.splitlines())
                        if new_line_count < max(5, orig_line_count // 3):
                            raise ValueError(
                                f"Refusing to overwrite existing Rust source file '{relative.as_posix()}' ({orig_line_count} lines) "
                                f"with {new_line_count} lines. Use edit_file for targeted replacements, revert_file to restore, or pass force=true."
                            )
            if content is not None:
                if _contains_compression_placeholder(content):
                    raise ValueError(
                        "Input contains a history-compression omission placeholder; "
                        "use read_lines to inspect the actual file instead of copying omitted markers."
                    )
                existing_text = (
                    target.read_text(encoding="utf-8", errors="replace")
                    if target.is_file()
                    else None
                )
                if (
                    relative.suffix == ".rs"
                    and existing_text is not None
                    and _rust_has_substantive_code(existing_text)
                    and not _rust_has_substantive_code(content)
                    and not force
                ):
                    raise ValueError(
                        "Refusing to replace an existing Rust source file with comments or blank lines only. "
                        "Use read_lines to inspect the complete source, or pass force=true for an intentional rollback."
                    )
                _check_rust_syntax(relative.as_posix(), content, original_content=existing_text)
            check_cargo_lock = (
                relative.name == "Cargo.toml"
                and content is not None
                and (not target.is_file() or target.read_bytes() != content.encode("utf-8"))
            )
            if relative not in state.originals:
                st = target.stat() if target.is_file() else None
                state.originals[relative] = ((target.read_bytes(), st.st_mode, st.st_mtime_ns) if st else None)
            previous_text = ""
            if content is not None and target.is_file():
                try:
                    previous_text = target.read_text(encoding="utf-8", errors="replace")
                except (OSError, UnicodeError):
                    previous_text = ""
            state.invalidate_write()
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
            _touch_crate_entry_mtime(root, relative)
            result = f"Updated {relative.as_posix()} in the working copy."
            if check_cargo_lock:
                result += cargo_lock_update_warning(target)
            if relative.suffix == ".rs" and relative not in state.regression_tests:
                changed_lines = _changed_line_numbers(previous_text, content or "")
                result += _record_and_format_cfg_notice(target, relative, changed_lines or None)
                result += _ckg_cross_file_impact_notice(
                    root, state, relative.as_posix(), changed_lines=changed_lines or None
                )
            return result
        except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as exc:
            return f"Error writing {path}: {exc}"

    @tool
    @synchronized
    def write_file(path: str, content: str | None, force: bool = False) -> str:
        """Write UTF-8 contents in the working copy; null deletes a file.

        Pass force=true only for an intentional comments-only replacement or large shrink/delete of an existing Rust file.
        Any write invalidates the submitted patch. Call generate_patch after
        all edits, selecting only files that belong in the final fix.
        """
        return _write_file_impl(path, content, force=force, from_edit=False)

    @tool
    @synchronized
    def write_regression_test(path: str, content: str, target_src: str | None = None, run_now: bool = False) -> str:
        """Create or replace a temporary Cargo integration test or in-module unit test.

        Modes:
        1. Integration test (default): path='tests/test_regression_<name>.rs'.
        2. In-module ephemeral test: target_src='src/lib.rs' (or path in src/): mounts a temporary
           `#[cfg(test)] mod __test_regression_<name> { use super::*; ... }` block at the end of the
           target source file. Useful for testing internal logic, pub(crate), or private functions.
           Automatically stripped from final patches and post-strip checks.
        """
        try:
            if _contains_compression_placeholder(content):
                raise ValueError(
                    "Test content contains a history-compression omission placeholder; "
                    "write complete Rust test code instead of copying omitted markers."
                )
            is_in_module = False
            target_src_path = None
            if target_src:
                target_src_path = resolve_path(target_src)
                is_in_module = True
            elif not path.startswith("tests/") and (path.startswith("src/") or "/src/" in path):
                target_src_path = resolve_path(path)
                is_in_module = True

            if is_in_module:
                target = target_src_path
                relative = target.relative_to(root)
                if not target.is_file() or target.suffix != ".rs":
                    raise ValueError(f"target_src must point to an existing .rs source file: {target_src or path}")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Provide non-empty test content for in-module regression test")

                clean_name = path.replace(".rs", "").split("/")[-1].replace("-", "_")
                if clean_name.startswith("tests/"):
                    clean_name = clean_name.replace("tests/", "")
                if clean_name.startswith("test_regression_"):
                    clean_name = clean_name.replace("test_regression_", "", 1)
                elif clean_name.startswith("normal_regression_"):
                    clean_name = clean_name.replace("normal_regression_", "", 1)
                if not clean_name.startswith("__test_regression_"):
                    clean_name = f"__test_regression_{clean_name}"

                manifest_dir = target.parent
                while manifest_dir != root.parent and not (manifest_dir / "Cargo.toml").is_file():
                    manifest_dir = manifest_dir.parent
                if not (manifest_dir / "Cargo.toml").is_file():
                    manifest_dir = root
                manifest_rel = (manifest_dir / "Cargo.toml").relative_to(root)

                orig_bytes = target.read_bytes()
                orig_text = orig_bytes.decode("utf-8", errors="replace")
                newline = "\r\n" if "\r\n" in orig_text else "\n"
                clean_base_text = _strip_in_module_tests(orig_text)
                norm_content = content.replace("\r\n", "\n").replace("\n", newline)
                test_block = (
                    f"{newline}// __TEST_REGRESSION_START: {clean_name}__{newline}"
                    f"#[cfg(test)]{newline}"
                    f"mod {clean_name} {{{newline}"
                    f"    use super::*;{newline}"
                    f"{norm_content}{newline}"
                    f"}}{newline}"
                    f"// __TEST_REGRESSION_END: {clean_name}__{newline}"
                )
                updated_text = clean_base_text + test_block
                _check_rust_syntax(relative.as_posix(), updated_text, original_content=orig_text)

                locked_meta = None
                for prev_key, prev_lock in list(state._locked_regressions.items()):
                    if prev_lock.get("is_in_module") and prev_lock.get("path") == relative.as_posix():
                        locked_meta = prev_lock
                        break
                if locked_meta and locked_meta.get("name") != clean_name:
                    state._record_red_lock_violation(
                        "rewrite_locked_regression", relative.as_posix(),
                        f"Locked regression '{locked_meta.get('name')}' may not be replaced by '{clean_name}'.",
                    )
                    return (
                        f"Error: the Red-phase regression test '{locked_meta.get('name')}' in {relative.as_posix()} is locked "
                        "as the issue's contract oracle and cannot be replaced or renamed. "
                        "Fix the production code so the locked test passes instead of rewriting the test. "
                        "If the locked test is factually wrong, call escalate_for_replan with the concrete reason."
                    )
                if locked_meta and locked_meta.get("name") == clean_name:
                    proposed_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    if proposed_sha != locked_meta.get("sha256"):
                        state._record_red_lock_violation(
                            "rewrite_locked_regression", relative.as_posix(),
                            f"Locked regression '{clean_name}' content was modified.",
                        )
                        for reg_key_now in [
                            k for k in list(state._reproduced_regressions)
                            if str(k[1]) == clean_name
                        ]:
                            state._reproduced_regressions.pop(reg_key_now, None)
                            state._passed_regressions.discard(reg_key_now)
                            state._passed_regression_configs.pop(reg_key_now, None)
                            state._reproduction_notes.pop(reg_key_now, None)
                            state._regression_test_names.pop(reg_key_now, None)
                        state._verified_revision = None
                        return (
                            f"Error: the Red-phase regression test '{clean_name}' in {relative.as_posix()} is locked "
                            "and its assertions cannot be changed. Fix the production code so the locked test passes "
                            "instead of editing the test, or call escalate_for_replan with the concrete reason."
                        )

                for prev_rel, prev_info in list(state.in_module_regressions.items()):
                    if prev_rel != relative and prev_info.get("name") == clean_name:
                        prev_file = root / prev_rel
                        if prev_file.is_file():
                            stripped_bytes = _strip_in_module_tests(prev_file.read_bytes().decode("utf-8", errors="replace")).encode("utf-8")
                            prev_file.write_bytes(stripped_bytes)
                            orig_prev = state.originals.get(prev_rel)
                            if orig_prev is not None and stripped_bytes == orig_prev[0]:
                                state.originals.pop(prev_rel, None)
                        state.in_module_regressions.pop(prev_rel, None)
                        state.regression_tests.discard(prev_rel)
                        prev_key = (prev_info["manifest"], prev_info["name"])
                        state._reproduced_regressions.pop(prev_key, None)
                        state._passed_regressions.discard(prev_key)
                        state._passed_regression_configs.pop(prev_key, None)
                        state._regression_test_names.pop(prev_key, None)
                        state._reproduction_notes.pop(prev_key, None)

                st = target.stat()
                state.originals.setdefault(relative, (orig_bytes, st.st_mode, st.st_mtime_ns))
                state.invalidate_write()
                state.regression_tests.add(relative)
                fn_names = set(re.findall(r"(?m)^\s*(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(", content))
                state.in_module_regressions[relative] = {
                    "name": clean_name,
                    "manifest": manifest_rel,
                    "content": content,
                    "fn_names": fn_names,
                }
                reg_key = (manifest_rel, clean_name)
                state._reproduced_regressions.pop(reg_key, None)
                state._passed_regression_configs.pop(reg_key, None)
                state._reproduction_notes.pop(reg_key, None)

                target.write_bytes(updated_text.encode("utf-8"))
                msg = f"Mounted temporary in-module regression test '{clean_name}' at end of {relative.as_posix()}."
                if run_now:
                    manifest_file = manifest_dir / "Cargo.toml"
                    kind, bin_name = _resolve_in_module_test_target(manifest_file)
                    active_feats = list(state._active_cfg_features.get(manifest_file.resolve()) or ())
                    ct_kwargs = {"manifest_path": manifest_rel.as_posix(), "test_filter": clean_name}
                    if active_feats:
                        ct_kwargs["features"] = active_feats
                    if kind == "bin" and bin_name:
                        run_res = cargo_test.func(bin_target=bin_name, **ct_kwargs)
                    else:
                        run_res = cargo_test.func(lib=True, **ct_kwargs)
                    reg_meta = state._reproduced_regressions.get(reg_key) or {}
                    if reg_meta.get("suspect_setup_panic"):
                        msg += "\n[Warning: test panicked on a setup/IO error outside src/; ensure it actually exercises the target crate.]"
                    msg += f"\n\n[Auto-run result for {clean_name}]:\n{run_res}"
                return msg

            target = resolve_path(path)
            relative = target.relative_to(root)
            if (target.parent.name != "tests" or not (target.name.startswith("test_regression_") or target.name.startswith("normal_regression_"))
                    or target.suffix != ".rs" or not isinstance(content, str) or not content.strip()):
                raise ValueError("Use a nonempty test at tests/test_regression_<name>.rs, or use target_src='src/lib.rs' for in-module test")
            manifest_rel_path = relative.parent.parent / "Cargo.toml"
            manifest_abs = root / manifest_rel_path
            if _is_virtual_workspace_manifest(manifest_abs):
                pkgs = state._scan_package_manifests()
                member_hints = []
                for pkg_name, pkg_manifest in sorted(pkgs.items())[:8]:
                    rel_dir = pkg_manifest.parent.relative_to(root).as_posix()
                    prefix = "" if rel_dir == "." else rel_dir + "/"
                    member_hints.append(f"{pkg_name}: {prefix}tests/{target.name}")
                hint_str = f" Available member crate test paths: {', '.join(member_hints)}." if member_hints else ""
                raise ValueError(
                    f"'{manifest_rel_path.as_posix()}' is a virtual Cargo [workspace] manifest without a [package] section, "
                    f"so Cargo will not compile integration tests at '{relative.as_posix()}'.{hint_str} "
                    "Place the regression test under the target member crate's tests/ directory or use target_src='<crate>/src/...'."
                )
            target_kind, bin_name = _resolve_in_module_test_target(manifest_abs)
            if target_kind == "bin":
                pkg_ident = (bin_name or _manifest_package_name(manifest_abs)).replace("-", "_").strip()
                if pkg_ident and re.search(
                    rf"(?m)\b(?:use|extern\s+crate)\s+{re.escape(pkg_ident)}\b|\b{re.escape(pkg_ident)}::",
                    content,
                ):
                    raise ValueError(
                        f"'{manifest_rel_path.as_posix()}' defines only a binary target (no [lib] target), "
                        f"so standalone integration tests under tests/ cannot import `{pkg_ident}::...`. "
                        f"Use write_regression_test(path='{relative.stem}', target_src='<crate>/src/...', content=...) "
                        "for an in-module unit test, or test the compiled binary via std::process::Command."
                    )
            if relative not in state.regression_tests and target.exists():
                raise ValueError("Regression test must be a new file")
            locked_meta = None
            for prev_key, prev_lock in list(state._locked_regressions.items()):
                if not prev_lock.get("is_in_module") and prev_lock.get("path") == relative.as_posix():
                    locked_meta = prev_lock
                    break
            if locked_meta:
                proposed_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if proposed_sha != locked_meta.get("sha256"):
                    state._record_red_lock_violation(
                        "rewrite_locked_regression", relative.as_posix(),
                        f"Locked regression '{locked_meta.get('name')}' was rewritten.",
                    )
                    for reg_key_now in [
                        k for k in list(state._reproduced_regressions)
                        if str(k[1]) == locked_meta.get("name", "")
                    ]:
                        state._reproduced_regressions.pop(reg_key_now, None)
                        state._passed_regressions.discard(reg_key_now)
                        state._passed_regression_configs.pop(reg_key_now, None)
                        state._reproduction_notes.pop(reg_key_now, None)
                        state._regression_test_names.pop(reg_key_now, None)
                    state._verified_revision = None
                    return (
                        f"Error: '{relative.as_posix()}' is the locked Red-phase regression test for this issue "
                        "and cannot be rewritten. Fix the production code so the locked test passes instead of "
                        "changing the test, or call escalate_for_replan with the concrete reason the locked test is wrong."
                    )
            existing_text = target.read_bytes().decode("utf-8", errors="replace") if target.is_file() else None
            _check_rust_syntax(relative.as_posix(), content, original_content=existing_text)
            was_registered = relative in state.regression_tests
            state.invalidate_write()
            state.originals.setdefault(relative, None)
            state.regression_tests.add(relative)
            reg_key = (manifest_rel_path, relative.stem)
            state._reproduced_regressions.pop(reg_key, None)
            state._passed_regression_configs.pop(reg_key, None)
            state._reproduction_notes.pop(reg_key, None)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
            active_targets = state.get_regression_targets()
            msg = f"Updated temporary regression test {relative.as_posix()} in the working copy."
            if _is_static_manifest_only_test(content):
                msg += (
                    " [Warning: this regression test only reads Cargo.toml text statically without running cargo "
                    "or exercising crate code; write a real compilation/runtime test instead.]"
                )
            if len(active_targets) > 1:
                other_names = [k[1] for k in sorted(active_targets) if k != reg_key]
                msg += (
                    f" [Notice: {len(active_targets)} temporary regression tests are active ({', '.join(other_names)}, {relative.stem}). "
                    "Prefer reusing/overwriting one primary regression test or removing obsolete ones with revert_file instead of creating extra *_tmp files.]"
                )
            if run_now:
                manifest_rel = manifest_rel_path.as_posix()
                active_feats = list(state._active_cfg_features.get(manifest_abs.resolve()) or ())
                run_kwargs = {"manifest_path": manifest_rel, "test_target": relative.stem}
                if active_feats:
                    run_kwargs["features"] = active_feats
                run_res = cargo_test.func(**run_kwargs)
                if "no test target named" in run_res:
                    if not was_registered:
                        state.originals.pop(relative, None)
                        state.regression_tests.discard(relative)
                        state._reproduced_regressions.pop(reg_key, None)
                        state._passed_regression_configs.pop(reg_key, None)
                        state._reproduction_notes.pop(reg_key, None)
                        try:
                            if target.is_file():
                                target.unlink()
                        except OSError:
                            pass
                    return (
                        f"Error: Cargo reported no test target named '{relative.stem}' under '{manifest_rel}', "
                        f"so '{relative.as_posix()}' was not kept. Place the test under the target member crate's tests/ "
                        f"directory or use target_src='<crate>/src/...'.\n\n[Auto-run output]:\n{run_res}"
                    )
                if any(code in run_res for code in ('"E0432"', '"E0433"', '"E0412"', '"E0425"', '"E0599"')):
                    pkg_imp = _manifest_package_name(manifest_abs).replace("-", "_")
                    common_imports = rust_tools_mod.extract_common_test_imports(manifest_abs, pkg_imp)
                    if common_imports:
                        msg += "\n[Test Header Hint: sibling tests in this crate import: " + " | ".join(common_imports[:4]) + "]"
                reg_meta = state._reproduced_regressions.get(reg_key) or {}
                if reg_meta.get("suspect_setup_panic"):
                    msg += "\n[Warning: test panicked on a setup/IO error outside src/; ensure it actually exercises the target crate.]"
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
            for _old_item, new_item, _s_line, _e_line in replacements:
                if _contains_compression_placeholder(new_item):
                    raise ValueError(
                        "new_text contains a history-compression omission placeholder; "
                        "read the actual lines with read_lines instead of copying omitted markers."
                    )
            content = target.read_bytes().decode("utf-8")
            updated = content
            line_ranges = []
            for old_item, new_item, s_line, e_line in replacements:
                updated, line_range = _apply_one_replacement(updated, old_item, new_item, s_line, e_line)
                line_ranges.append(line_range)
            if relative in state.in_module_regressions:
                if _extract_in_module_test_block(updated) != _extract_in_module_test_block(content):
                    raise ValueError(
                        "Do not modify temporary '// __TEST_REGRESSION_START' blocks via edit_file; "
                        "use write_regression_test(target_src=...) to update the in-module regression test."
                    )
            write_result = _write_file_impl(path, updated, from_edit=True)
            if not write_result.startswith("Updated "):
                return write_result + state.record_edit_outcome(rel_key, False)
            state.record_edit_outcome(rel_key, True)
            snippet = _format_edit_snippet(updated, line_ranges, context=6)
            out_text = f"{write_result}\n{snippet}" if snippet else write_result
            is_prod_rs = target.suffix == ".rs" and (relative not in state.regression_tests or relative in state.in_module_regressions)
            if is_prod_rs:
                clean_updated = _strip_in_module_tests(updated) if relative in state.in_module_regressions else updated
                orig_raw = state.originals.get(relative)
                if orig_raw and orig_raw[0] is not None:
                    orig_code = orig_raw[0].decode("utf-8", errors="replace")
                    shift_warn = _detect_doctest_line_shift(orig_code, clean_updated, relative.as_posix())
                    if shift_warn:
                        out_text += f"\n{shift_warn}"
                    inline_warn = _edited_inline_test_warning(orig_code, clean_updated, relative.as_posix())
                    if inline_warn:
                        out_text += f"\n{inline_warn}"
                    eof_warn = _eof_append_warning(orig_code, clean_updated, relative.as_posix())
                    if eof_warn:
                        out_text += f"\n{eof_warn}"
                    pat_warn = _same_file_parallel_pattern_warning(orig_code, clean_updated, relative.as_posix())
                    if pat_warn:
                        out_text += f"\n{pat_warn}"
                clean_prev = _strip_in_module_tests(content) if relative in state.in_module_regressions else content
                scope_hint = _unimported_crate_path_hint(clean_prev, clean_updated, relative.as_posix())
                if scope_hint:
                    out_text += f"\n{scope_hint}"
            if is_prod_rs:
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
                            if not (c["path"] == relative.as_posix() and any(s <= c["line"] <= e for s, e in line_ranges))
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
                            if not (c["path"] == relative.as_posix() and any(s <= c["line"] <= e for s, e in line_ranges))
                        ]
                        if other_locs:
                            impact_lines.append(
                                f"{t_kind} modified for `{t_name}` -> usages/impls at: {', '.join(other_locs[:5])}. "
                                f"Verify trait bounds and pattern matches across callers."
                            )
                if impact_lines:
                    out_text += "\n[Impact notice: " + " | ".join(impact_lines) + "]"
                changed_lines = []
                for start, end in line_ranges:
                    changed_lines.extend(range(start, max(start, end) + 1))
                if relative in state.in_module_regressions:
                    out_text += _record_and_format_cfg_notice(target, relative, changed_lines or None)
                out_text += _ckg_cross_file_impact_notice(
                    root, state, relative.as_posix(), changed_lines=changed_lines or None
                )
            return out_text
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error editing {path}: {exc}" + state.record_edit_outcome(rel_key, False)

    @tool
    @synchronized
    def verifier_probe(
        probe_type: str,
        manifest_path: str = "Cargo.toml",
        test_filter: str = "",
        assertion_code: str = "",
        package: str = "",
        test_target: str = "",
        timeout: int = 120,
    ) -> str:
        """Run a read-only, sandboxed falsification probe to test the current patch.

        probe_type:
          - "existing_test": run an existing crate test target to observe its real behavior.
          - "assertion": mount a TEMPORARY assertion module compiled against the crate, run it, then
            remove it. Returns "no_tests" if no assertion test was executed.
          - "snapshot": re-run the relevant test target and extract insta snapshot diff hunks.

        manifest_path may point at a workspace root or a member crate. When it is a
        virtual workspace root, pass `package` to select the member crate. Use
        `test_target` to run one integration test target instead of the whole crate.

        The probe never modifies production source, never edits existing tests, and leaves no
        file behind. Use it to obtain concrete execution evidence before rejecting a patch.
        """
        try:
            probe = str(probe_type or "").strip().lower()
            if probe not in ("existing_test", "assertion", "snapshot"):
                return "Error: probe_type must be 'existing_test', 'assertion', or 'snapshot'."
            if not shutil.which("cargo"):
                return "Error: cargo is not available in this environment."
            root = state.root.resolve()
            manifest_abs = (root / manifest_path).resolve()
            if not manifest_abs.is_file():
                return f"Error: manifest not found: {manifest_path}"
            manifest_dir = manifest_abs.parent
            pkg = str(package or "").strip()
            target_name = str(test_target or "").strip()

            # A virtual workspace root has no test targets of its own, so `cargo test
            # --test=<name>` fails with "no test target named". Point the probe at the
            # member crate that owns the tests instead.
            if probe == "assertion" and not pkg and _is_virtual_workspace_manifest(manifest_abs):
                members = _workspace_member_manifests(state, manifest_abs)
                if members:
                    manifest_abs = members[0]
                    manifest_dir = manifest_abs.parent
                    pkg = _manifest_package_name(manifest_abs) or manifest_dir.name
            probe_dir = None
            probe_file = None
            probe_created = False
            try:
                if probe == "assertion":
                    if not assertion_code.strip():
                        return "Error: assertion_code is required for probe_type='assertion'."
                    body = assertion_code.strip()
                    if not re.search(r"#\s*\[\s*test\s*\]", body) and "fn main" not in body:
                        body = (
                            "#[test]\nfn __verifier_probe__() {\n" + body + "\n}\n"
                        )
                    probe_dir = manifest_dir / "tests"
                    probe_dir.mkdir(parents=True, exist_ok=True)
                    probe_file = probe_dir / "test_verifier_probe_tmp.rs"
                    if probe_file.exists():
                        return "Error: a verifier probe is already active; run it again after cleanup."
                    probe_file.write_text(body, encoding="utf-8")
                    probe_created = True

                cmd = ["cargo", "test", "--manifest-path", str(manifest_abs)]
                if (root / "Cargo.lock").is_file():
                    cmd.append("--locked")
                if pkg:
                    cmd.append(f"--package={pkg}")
                if probe == "existing_test" or probe == "snapshot":
                    if target_name:
                        cmd.append(f"--test={target_name}")
                    elif test_filter:
                        cmd.append(test_filter)
                else:
                    cmd.append(f"--test={probe_file.stem}")

                try:
                    res = subprocess.run(
                        cmd, cwd=root, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=max(20, min(int(timeout), 300)),
                        env=state.rust_env,
                    )
                except subprocess.TimeoutExpired:
                    return json.dumps({
                        "probe_type": probe,
                        "status": "timeout",
                        "exit_code": None,
                        "stdout_tail": "",
                        "stderr_tail": f"cargo test exceeded {timeout}s",
                    }, ensure_ascii=False)

                stdout = res.stdout or ""
                stderr = res.stderr or ""
                failed_tests = sorted(set(re.findall(
                    r"^test (.+?) \.\.\. FAILED\s*$", stdout, re.MULTILINE
                )))
                counts = re.findall(
                    r"^test result: (?:ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed", stdout, re.MULTILINE
                )
                tests_passed = sum(int(p) for p, _f in counts)
                tests_failed = sum(int(f) for _p, f in counts)
                probe_status = "failed" if res.returncode else "passed"
                if probe == "assertion" and probe_status == "passed" and not (tests_passed or tests_failed):
                    probe_status = "no_tests"
                payload = {
                    "probe_type": probe,
                    "status": probe_status,
                    "exit_code": res.returncode,
                    "tests_passed": tests_passed,
                    "tests_failed": tests_failed,
                    "failed_tests": failed_tests,
                    "stdout_tail": stdout[-3000:],
                    "stderr_tail": stderr[-2000:],
                }
                if probe == "snapshot":
                    payload["snapshot_hunks"] = _extract_snapshot_diff_hunks(stdout)
                return json.dumps(payload, ensure_ascii=False)
            finally:
                if probe_created and probe_file is not None and probe_file.exists():
                    try:
                        probe_file.unlink()
                    except OSError:
                        pass
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error running verifier probe: {exc}"

    @tool
    @synchronized
    def revert_file(path: str, force: bool = False) -> str:
        """Revert a modified, created, or deleted file in the working copy back to its original state.

        Also removes temporary regression tests created with write_regression_test.
        Reverting invalidates prior test verification and submitted patches.
        """
        try:
            target = resolve_path(path)
            relative = target.relative_to(root)
            if not relative.parts:
                raise ValueError("Provide a file path")
            locked_here = [
                (key, meta) for key, meta in state._locked_regressions.items()
                if meta.get("path") == relative.as_posix() and not meta.get("is_in_module")
            ]
            if locked_here:
                names = ", ".join(sorted(meta.get("name", "") for _key, meta in locked_here))
                state._record_red_lock_violation(
                    "revert_locked_regression", relative.as_posix(),
                    f"Attempted to revert locked regression '{names}'.",
                )
                for lock_key, lock_meta in locked_here:
                    target_name = lock_meta.get("name", "")
                    for reg_key in [
                        k for k in list(state._reproduced_regressions)
                        if str(k[1]) == target_name
                    ]:
                        state._reproduced_regressions.pop(reg_key, None)
                        state._passed_regressions.discard(reg_key)
                        state._passed_regression_configs.pop(reg_key, None)
                        state._reproduction_notes.pop(reg_key, None)
                        state._regression_test_names.pop(reg_key, None)
                    state._verified_revision = None
                return (
                    f"Error: '{relative.as_posix()}' holds the locked Red-phase regression test(s) '{names}'. "
                    "It cannot be reverted or deleted because it is the contract oracle for this issue. "
                    "Fix the production code so the locked test passes, or call escalate_for_replan explaining why the "
                    "locked test is factually wrong. Its Red-phase verification evidence was cleared, so the patch now "
                    "requires a fresh confirmed reproduction before submission."
                )
            recovery_name = None
            if not force and state._revert_would_drop_last_substantive_change(relative):
                recovery_name = f"__recovery_before_revert_{relative.stem}_{state._revision}"
                state._checkpoints[recovery_name] = state._capture_snapshot()
            if relative in state.in_module_regressions:
                info = state.in_module_regressions.pop(relative, None)
                state.regression_tests.discard(relative)
                had_passed_evidence = False
                if info:
                    reg_key = (info["manifest"], info["name"])
                    had_passed_evidence = reg_key in state._passed_regression_configs
                    state._reproduced_regressions.pop(reg_key, None)
                    state._passed_regression_configs.pop(reg_key, None)
                    state._reproduction_notes.pop(reg_key, None)
                if relative in state.originals:
                    original = state.originals.pop(relative, None)
                    if original is not None:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(original[0])
                        target.chmod(original[1])
                else:
                    if target.is_file():
                        c = target.read_text(encoding="utf-8")
                        target.write_text(_strip_in_module_tests(c), encoding="utf-8")
                state._consecutive_edit_failures.pop(relative.as_posix(), None)
                state.invalidate_write()
                warn_reg = " [Notice: Reverted active temporary regression test; its passing verification evidence was cleared. Temporary tests are stripped automatically on patch export.]" if had_passed_evidence else ""
                suffix = f" Preserved the prior working copy as checkpoint '{recovery_name}'." if recovery_name else ""
                return f"Reverted {relative.as_posix()} to its original state.{suffix}{warn_reg}"
            if relative not in state.originals and relative not in state.regression_tests:
                if not target.exists():
                    raise ValueError(f"File does not exist: {path}")
                if not target.is_file():
                    raise ValueError("Path is not a regular file")
                return f"No changes to revert for {relative.as_posix()}."
            original = state.originals.pop(relative, None)
            state.regression_tests.discard(relative)
            reg_key = (relative.parent.parent / "Cargo.toml", relative.stem)
            had_passed_evidence = reg_key in state._passed_regression_configs
            state._reproduced_regressions.pop(reg_key, None)
            state._passed_regression_configs.pop(reg_key, None)
            state._reproduction_notes.pop(reg_key, None)
            state._consecutive_edit_failures.pop(relative.as_posix(), None)
            state.invalidate_write()
            if original is None:
                if target.exists() or target.is_symlink():
                    target.unlink()
                warn_reg = " [Notice: Reverted active temporary regression test; its passing verification evidence was cleared. Temporary tests are stripped automatically on patch export.]" if had_passed_evidence else ""
                suffix = f" Preserved the prior working copy as checkpoint '{recovery_name}'." if recovery_name else ""
                return f"Reverted {relative.as_posix()} (removed newly created file).{suffix}{warn_reg}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original[0])
            target.chmod(original[1])
            if len(original) >= 3 and original[2] is not None:
                try:
                    os.utime(target, ns=(original[2], original[2]))
                except Exception:
                    pass
            _touch_crate_entry_mtime(root, relative)
            suffix = f" Preserved the prior working copy as checkpoint '{recovery_name}'." if recovery_name else ""
            return f"Reverted {relative.as_posix()} to its original state.{suffix}"
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
            targets = set(state.get_regression_targets().keys())
            if state.require_reproduction and targets and targets <= state._passed_regressions and not state.reproduced_cleanly():
                try:
                    state.verify_pre_fix_reproduction(targets)
                except Exception as exc:
                    state._last_submit_error = f"Relaxed reproduction warning: {exc}"
            if paths is None and submit:
                source_candidates = [p.as_posix() for p in changed if (p not in state.regression_tests or p in state.in_module_regressions)]
                if not source_candidates:
                    return "No changes to submit."
                paths = source_candidates

            if paths is None:
                if not changed:
                    return "No changes to submit."
                names = "\n".join(path.as_posix() for path in changed)
                repro_notice = ""
                targets = set(state.get_regression_targets().keys())
                unrepro = [name for manifest, name in targets if (manifest, name) not in state._reproduced_regressions]
                if unrepro:
                    repro_notice = f"\nNotice: The following temporary regression test(s) have not been confirmed failing before repair: {', '.join(unrepro)}.\n"
                verification_notice = ""
                pending = state.missing_verification_steps()
                if pending:
                    verification_notice += f"\nPending verification: {pending}\n"
                if state._last_submit_error:
                    verification_notice += f"\nVerification warning: {state._last_submit_error}\n"
                source_candidates = [p.as_posix() for p in changed if (p not in state.regression_tests or p in state.in_module_regressions)]
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
                doctest_warns = []
                for p_rel in source_candidates:
                    p_obj = Path(p_rel)
                    cur_file = root / p_obj
                    if p_obj.suffix == ".rs" and cur_file.is_file():
                        orig_raw = state.originals.get(p_obj)
                        if orig_raw and orig_raw[0] is not None:
                            orig_c = orig_raw[0].decode("utf-8", errors="replace")
                            cur_c = cur_file.read_text(encoding="utf-8", errors="replace")
                            w = _detect_doctest_line_shift(orig_c, cur_c, p_rel)
                            if w:
                                doctest_warns.append(w)
                doctest_notice = ("\n" + "\n".join(doctest_warns) + "\n") if doctest_warns else ""
                hint = (f"Call generate_patch(paths={json.dumps(source_candidates)}) to submit."
                        if source_candidates else "Call generate_patch with paths for the final fix to submit.")
                return (f"Unsubmitted preview. Changed files:\n{names}\n{repro_notice}{verification_notice}{test_compile_warn}{doctest_notice}"
                        f"{hint}\n\n"
                        + build_patch(changed))
            state.patch = ""
            if not paths:
                raise ValueError("Provide at least one changed file path")
            selected = []
            for path in paths:
                relative = resolve_path(path).relative_to(root)
                if relative in state.regression_tests and relative not in state.in_module_regressions:
                    raise ValueError(f"Temporary regression test must not be submitted: {path}")
                if relative not in changed:
                    raise ValueError(f"Not a changed file: {path}")
                if relative not in selected:
                    selected.append(relative)
            standalone_reg = {p for p in state.regression_tests if p not in state.in_module_regressions}
            omitted = set(changed) - set(selected) - standalone_reg
            if omitted:
                names = ", ".join(sorted(path.as_posix() for path in omitted))
                raise ValueError(f"Include all other changed files or revert them and rerun tests: {names}")
            if state.post_strip_check:
                try:
                    state.run_post_strip_check(selected)
                except Exception as exc:
                    state._last_submit_error = f"Relaxed post-strip warning: {exc}"
            patch = build_patch(selected)
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

    rust_tools_mod = load_local("rust_tools")
    rust_tools = rust_tools_mod.make_rust_tools(root, resolve_path, rust_env)
    ckg_mod = load_local("rust_ckg")
    ckg_tool, ckg_graph = ckg_mod.make_rust_ckg_tool(root)
    state.ckg = ckg_graph
    impact_mask_cache = {}
    diagnostics = load_local("rust_diagnostics")
    cargo_check = diagnostics.make_cargo_check_tool(
        root, resolve_path, rust_env, on_result=state.record_check_result, ckg=ckg_graph, mask_cache=impact_mask_cache
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
        enrich_diagnostics=lambda diags, r, manifest_dir=None: diagnostics.attach_snippets_and_lookup(
            diags, r, manifest_dir=manifest_dir, ckg=ckg_graph, mask_cache=impact_mask_cache
        ),
        on_test_feedback=state.record_test_result,
    )
    explain = load_local("rust_error_explain").make_rust_error_explain_tool(root, rust_env)
    symbols_mod = load_local("rust_symbols")
    trace_tool = load_local("rust_trace").make_rust_trace_tool(root, resolve_path, rust_env, ckg=ckg_graph)
    symbols = symbols_mod.make_rust_symbols_tool(root, resolve_path)
    all_tools = [read_file, write_file, write_regression_test, edit_file, revert_file, checkpoint, generate_patch,
                 *rust_tools, cargo_check, cargo_test, trace_tool, symbols, ckg_tool, explain, verifier_probe]

    def attach_budget_notice(tool_obj):
        if tool_obj.name == "read_file":
            return tool_obj
        original_func = tool_obj.func

        @wraps(original_func)
        def wrapped(*args, **kwargs):
            if tool_obj.name in ("read_lines", "search_code", "rust_symbols", "inspect_rust_neighborhood", "cargo_workspace"):
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
