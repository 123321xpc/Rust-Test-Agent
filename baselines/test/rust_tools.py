"""Rust-oriented, read-only tools operating on the agent's isolated checkout."""

import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool


def _format_context_matches(rel_posix, file_lines, matched_numbers, context_lines):
    windows = []
    total = len(file_lines)
    for num in matched_numbers:
        start = max(1, num - context_lines)
        end = min(total, num + context_lines)
        if windows and start <= windows[-1][1] + 1:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    matched_set = set(matched_numbers)
    blocks = []
    for start, end in windows:
        block = []
        for lineno in range(start, end + 1):
            sep = ":" if lineno in matched_set else "-"
            block.append(f"{rel_posix}:{lineno}{sep} {file_lines[lineno - 1]}")
        blocks.append("\n".join(block))
    return "\n--\n".join(blocks)


def _format_file_match_summary(
    total_matches: int,
    file_match_counts: dict[str, int],
    *,
    overflow: bool = False,
    max_results: int = 50,
    max_files: int = 50,
) -> str:
    if not file_match_counts:
        return "(no matches)"
    if overflow:
        header = (
            f"Search returned too many results ({total_matches} matches across {len(file_match_counts)} file(s), "
            f"exceeding max_results={max_results}; results truncated). "
            "Refine your query, narrow `path`, or set `exclude_tests=True`:"
        )
    else:
        header = f"Found {total_matches} matches across {len(file_match_counts)} files:"
    lines = [header]
    items = list(file_match_counts.items())
    for fpath, cnt in items[:max_files]:
        lines.append(f"  {fpath}: {cnt} match(es)")
    if len(items) > max_files:
        lines.append(f"  ... ({len(items) - max_files} more files omitted; narrow `path` or `query`)")
    return "\n".join(lines)


def extract_common_test_imports(manifest_file: Path, import_name: str = "") -> list[str]:
    """Extract common `use <crate>::...;` imports from existing `tests/*.rs` files in a crate."""
    tests_dir = manifest_file.parent / "tests"
    if not tests_dir.is_dir():
        return []
    seen: list[str] = []
    prefix = f"use {import_name}::" if import_name else "use "
    for tf in sorted(tests_dir.glob("*.rs"))[:4]:
        if tf.name.startswith(("test_regression_", "normal_regression_")):
            continue
        try:
            for line in tf.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
                s = line.strip()
                if s.startswith(prefix) and s.endswith(";") and len(s) <= 140:
                    if s not in seen:
                        seen.append(s)
                        if len(seen) >= 5:
                            return seen
        except OSError:
            continue
    return seen


def make_rust_tools(root, resolve_path, env=None):
    @tool
    def read_lines(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a UTF-8 file with 1-based inclusive line numbers (defaults to a 100-line window; at most 500 lines)."""
        try:
            if start_line < 1:
                raise ValueError("start_line must be at least 1")
            if end_line is None:
                end_line = start_line + 99
            if end_line < start_line:
                raise ValueError("end_line must be greater than or equal to start_line")
            clamped_notice = ""
            if end_line - start_line >= 500:
                end_line = start_line + 499
                clamped_notice = f"\n[Notice: requested range exceeded 500 lines; clamped to {start_line}-{end_line}]"
            target = resolve_path(path)
            if target.is_dir():
                return "\n".join(p.name + ("/" if p.is_dir() else "") for p in sorted(target.iterdir()) if p.name != ".git") or "(empty directory)"
            rel_posix = target.relative_to(root).as_posix()
            lines = []
            total_lines = 0
            with target.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    total_lines = number
                    if start_line <= number <= end_line:
                        lines.append(f"{number}: {line.rstrip(chr(10)).rstrip(chr(13))}")
            parts = [f"[File: {rel_posix} ({total_lines} lines total)]"]
            if total_lines == 0:
                parts.append("(empty file)")
            elif not lines:
                parts.append("(no lines in range)")
            else:
                if start_line > 1:
                    parts.append(f"({min(start_line - 1, total_lines)} lines above)")
                parts.extend(lines)
                actual_end = start_line + len(lines) - 1
                if actual_end < total_lines:
                    parts.append(f"({total_lines - actual_end} lines below)")
            return "\n".join(parts) + clamped_notice
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error reading lines: {exc}"

    @tool
    def search_code(query: str = "", path: str = ".", file_glob: str = "*.rs", regex: bool = False,
                    max_results: int = 50, context_lines: int = 0, multiline: bool = False,
                    exclude_tests: bool = False, summarize: bool = False) -> str:
        """Search UTF-8 code by literal text or regex, reporting repository-relative file and line; defaults to Rust files.

        Defaults to max_results=50; when matches exceed max_results, returns a per-file match summary instead of truncated code lines to avoid misleading partial results.
        Set context_lines (0 to 20) to include surrounding lines for each match.
        Set multiline=True to match cross-line signatures or statements where whitespace may span multiple lines.
        Set exclude_tests=True to skip tests/ and benches/ directories.
        Set summarize=True to return match counts grouped by file instead of all lines.
        Leave query empty with file_glob (e.g. '*.rs' or 'Cargo.toml') to list matching files across the repository.
        """
        try:
            if not 1 <= max_results <= 500:
                raise ValueError("max_results must be between 1 and 500")
            if not isinstance(context_lines, int) or isinstance(context_lines, bool) or not 0 <= context_lines <= 20:
                raise ValueError("context_lines must be an integer between 0 and 20")
            target = resolve_path(path)
            if not target.exists():
                raise ValueError("Search path does not exist")
            if target.is_file():
                files = [target]
            else:
                files = []
                skip_dirs = {".git", "target"} | ({"tests", "benches"} if exclude_tests else set())
                for directory, dirs, names in os.walk(target, followlinks=False):
                    dirs[:] = sorted(d for d in dirs if d not in skip_dirs and not (Path(directory) / d).is_symlink())
                    files.extend(Path(directory) / name for name in sorted(names) if fnmatch.fnmatch(name, file_glob))

            def _sort_key(p: Path):
                try:
                    rel = p.relative_to(root).as_posix()
                except ValueError:
                    rel = p.as_posix()
                if rel.startswith("src/") or "/src/" in rel:
                    return (0, rel)
                if "/" not in rel:
                    return (1, rel)
                if rel.startswith("examples/") or "/examples/" in rel:
                    return (2, rel)
                return (3, rel)

            files.sort(key=_sort_key)

            if not query:
                if file_glob:
                    matching_names = [f.relative_to(root).as_posix() for f in files]
                    if not matching_names:
                        return f"(no files matching '{file_glob}')"
                    out = "\n".join(matching_names[:max_results])
                    if len(matching_names) > max_results:
                        out += f"\n... ({len(matching_names) - max_results} more files; results truncated)"
                    return out
                raise ValueError("Provide a query or file_glob")

            if multiline:
                if regex:
                    pattern = re.compile(query, re.MULTILINE)
                else:
                    tokens = query.split()
                    raw_pat = r"\s+".join(re.escape(t) for t in tokens)
                    raw_pat = raw_pat.replace(r"\(", r"\(\s*").replace(r"\)", r",?\s*\)").replace(",", r",\s*")
                    pattern = re.compile(raw_pat, re.MULTILINE)
            else:
                pattern = re.compile(query if regex else re.escape(query))

            matches = []
            file_match_counts = {}
            total_matches = 0
            overflow = False

            for file in files:
                resolved = file.resolve()
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    continue
                try:
                    if multiline:
                        text = resolved.read_text(encoding="utf-8")
                        file_lines = text.splitlines()
                        matched_numbers = []
                        for m in pattern.finditer(text):
                            matched_numbers.append(text.count("\n", 0, m.start()) + 1)
                    else:
                        file_lines = [line.rstrip() for line in resolved.read_text(encoding="utf-8").splitlines()]
                        matched_numbers = [number for number, line in enumerate(file_lines, 1) if pattern.search(line)]
                except (OSError, UnicodeError):
                    continue

                if not matched_numbers:
                    continue

                rel_posix = file.relative_to(root).as_posix()
                file_match_counts[rel_posix] = len(matched_numbers)
                total_matches += len(matched_numbers)

                if not summarize and not overflow:
                    if total_matches > max_results:
                        overflow = True
                        matches.clear()
                    elif context_lines == 0:
                        for ln in matched_numbers:
                            line_content = file_lines[ln - 1] if ln <= len(file_lines) else ""
                            matches.append(f"{rel_posix}:{ln}: {line_content}")
                    else:
                        matches.append(_format_context_matches(rel_posix, file_lines, matched_numbers, context_lines))

            if summarize or overflow:
                return _format_file_match_summary(
                    total_matches,
                    file_match_counts,
                    overflow=overflow,
                    max_results=max_results,
                )

            if not matches:
                return "(no matches)"
            return "\n".join(matches) if context_lines == 0 else "\n--\n".join(matches)
        except (OSError, ValueError, re.error) as exc:
            return f"Error searching code: {exc}"

    @tool
    def cargo_workspace(manifest_path: str = "Cargo.toml") -> str:
        """Query Cargo workspace packages and targets using offline cargo metadata; manifest path is repository-relative."""
        try:
            manifest = resolve_path(manifest_path)
            if not manifest.is_file():
                raise ValueError("Manifest is not a file")
            result = subprocess.run(
                ["cargo", "metadata", "--format-version", "1", "--no-deps", "--offline", "--manifest-path", str(manifest)],
                cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=env,
            )
            if result.returncode != 0 and "--offline" in result.stderr:
                result = subprocess.run(
                    ["cargo", "metadata", "--format-version", "1", "--no-deps", "--manifest-path", str(manifest)],
                    cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=env,
                )
            if result.returncode:
                return f"Cargo metadata failed: {result.stderr.strip()}"
            metadata = json.loads(result.stdout)
            members = set(metadata["workspace_members"])
            def display_path(path):
                path = Path(path)
                return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
            packages = []
            for p in metadata["packages"]:
                if p["id"] not in members:
                    continue
                imp_name = p["name"].replace("-", "_")
                m_path = Path(p["manifest_path"])
                pkg_entry = {
                    "name": p["name"], "import_name": imp_name,
                    "version": p["version"], "id": p["id"],
                    "manifest_path": display_path(p["manifest_path"]),
                    "has_lib": any("lib" in t.get("kind", []) for t in p.get("targets", [])),
                    "features": list(p.get("features", {}).keys())[:25],
                    "default_features": p.get("features", {}).get("default", []),
                    "targets": [{"name": t["name"], "kind": t["kind"], "src_path": display_path(t["src_path"])} for t in p["targets"]],
                }
                common_imports = extract_common_test_imports(m_path, imp_name)
                if common_imports:
                    pkg_entry["common_test_imports"] = common_imports
                build_rs = m_path.parent / "build.rs"
                if build_rs.is_file():
                    try:
                        build_txt = build_rs.read_text(encoding="utf-8", errors="replace")[:65536]
                        cfgs = list(dict.fromkeys(re.findall(r"cargo:rustc-cfg=([A-Za-z0-9_]+)", build_txt)))
                        if cfgs:
                            pkg_entry["build_script_cfgs"] = cfgs[:20]
                    except OSError:
                        pass
                packages.append(pkg_entry)
            return json.dumps({"workspace_root": display_path(metadata["workspace_root"]), "workspace_members": packages}, ensure_ascii=False, indent=2)
        except (OSError, ValueError, subprocess.TimeoutExpired, KeyError) as exc:
            return f"Error querying Cargo workspace: {exc}"

    return [read_lines, search_code, cargo_workspace]
