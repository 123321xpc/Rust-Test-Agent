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


def make_rust_tools(root, resolve_path, env=None):
    @tool
    def read_lines(path: str, start_line: int = 1, end_line: int = 200) -> str:
        """Read a UTF-8 file with 1-based inclusive line numbers (at most 500 lines)."""
        try:
            if start_line < 1:
                raise ValueError("start_line must be at least 1")
            if end_line < start_line:
                raise ValueError("end_line must be greater than or equal to start_line")
            clamped_notice = ""
            if end_line - start_line >= 500:
                end_line = start_line + 499
                clamped_notice = f"\n[Notice: requested range exceeded 500 lines; clamped to {start_line}-{end_line}]"
            target = resolve_path(path)
            lines = []
            with target.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if number > end_line:
                        break
                    if number >= start_line:
                        lines.append(f"{number}: {line.rstrip(chr(10)).rstrip(chr(13))}")
            result = "\n".join(lines) if lines else "(no lines in range)"
            return result + clamped_notice
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error reading lines: {exc}"

    @tool
    def search_code(query: str, path: str = ".", file_glob: str = "*.rs", regex: bool = False,
                    max_results: int = 100, context_lines: int = 0, multiline: bool = False) -> str:
        """Search UTF-8 code by literal text or regex, reporting repository-relative file and line; defaults to Rust files.

        Set context_lines (0 to 20) to include surrounding lines for each match.
        Set multiline=True to match cross-line signatures or statements where whitespace may span multiple lines.
        """
        try:
            if not query or not 1 <= max_results <= 500:
                raise ValueError("Provide a query and max_results between 1 and 500")
            if not isinstance(context_lines, int) or isinstance(context_lines, bool) or not 0 <= context_lines <= 20:
                raise ValueError("context_lines must be an integer between 0 and 20")
            target = resolve_path(path)
            if not target.exists():
                raise ValueError("Search path does not exist")
            if target.is_file():
                files = [target]
            else:
                files = []
                for directory, dirs, names in os.walk(target, followlinks=False):
                    dirs[:] = sorted(d for d in dirs if d not in (".git", "target") and not (Path(directory) / d).is_symlink())
                    files.extend(Path(directory) / name for name in sorted(names) if fnmatch.fnmatch(name, file_glob))

            if multiline:
                if regex:
                    pattern = re.compile(query, re.MULTILINE)
                else:
                    tokens = query.split()
                    raw_pat = r"\s+".join(re.escape(t) for t in tokens)
                    raw_pat = raw_pat.replace(r"\(", r"\(\s*").replace(r"\)", r",?\s*\)").replace(",", r",\s*")
                    pattern = re.compile(raw_pat, re.MULTILINE)

                matches = []
                total_matches = 0
                for file in files:
                    resolved = file.resolve()
                    if not resolved.is_relative_to(root) or not resolved.is_file():
                        continue
                    try:
                        text = resolved.read_text(encoding="utf-8")
                    except (OSError, UnicodeError):
                        continue
                    file_lines = text.splitlines()
                    matched_numbers = []
                    for m in pattern.finditer(text):
                        line_no = text.count("\n", 0, m.start()) + 1
                        matched_numbers.append(line_no)
                        total_matches += 1
                        if total_matches >= max_results:
                            break
                    if matched_numbers:
                        rel_posix = file.relative_to(root).as_posix()
                        if context_lines == 0:
                            for ln in matched_numbers:
                                line_content = file_lines[ln - 1] if ln <= len(file_lines) else ""
                                matches.append(f"{rel_posix}:{ln}: {line_content}")
                        else:
                            matches.append(_format_context_matches(rel_posix, file_lines, matched_numbers, context_lines))
                    if total_matches >= max_results:
                        break
                if not matches:
                    return "(no matches)"
                output = "\n".join(matches) if context_lines == 0 else "\n--\n".join(matches)
                return output + ("\n(results truncated)" if total_matches >= max_results else "")

            pattern = re.compile(query if regex else re.escape(query))
            if context_lines == 0:
                matches = []
                for file in files:
                    resolved = file.resolve()
                    if not resolved.is_relative_to(root) or not resolved.is_file():
                        continue
                    try:
                        with resolved.open(encoding="utf-8") as stream:
                            for number, line in enumerate(stream, 1):
                                if pattern.search(line):
                                    matches.append(f"{file.relative_to(root).as_posix()}:{number}: {line.rstrip()}")
                                    if len(matches) > max_results:
                                        return "\n".join(matches[:max_results]) + "\n(results truncated)"
                    except (OSError, UnicodeError):
                        continue
                return "\n".join(matches) or "(no matches)"

            total_matches = 0
            truncated = False
            file_blocks = []
            for file in files:
                resolved = file.resolve()
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    continue
                try:
                    file_lines = [line.rstrip() for line in resolved.read_text(encoding="utf-8").splitlines()]
                except (OSError, UnicodeError):
                    continue
                matched_numbers = []
                for number, line in enumerate(file_lines, 1):
                    if pattern.search(line):
                        if total_matches >= max_results:
                            truncated = True
                            break
                        matched_numbers.append(number)
                        total_matches += 1
                if matched_numbers:
                    rel_posix = file.relative_to(root).as_posix()
                    file_blocks.append(_format_context_matches(rel_posix, file_lines, matched_numbers, context_lines))
                if truncated:
                    break
            if not file_blocks:
                return "(no matches)"
            output = "\n--\n".join(file_blocks)
            return output + ("\n(results truncated)" if truncated else "")
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
            packages = [{"name": p["name"], "version": p["version"], "id": p["id"],
                         "manifest_path": display_path(p["manifest_path"]),
                         "targets": [{"name": t["name"], "kind": t["kind"], "src_path": display_path(t["src_path"])} for t in p["targets"]]}
                        for p in metadata["packages"] if p["id"] in members]
            return json.dumps({"workspace_root": display_path(metadata["workspace_root"]), "workspace_members": packages}, ensure_ascii=False, indent=2)
        except (OSError, ValueError, subprocess.TimeoutExpired, KeyError) as exc:
            return f"Error querying Cargo workspace: {exc}"

    return [read_lines, search_code, cargo_workspace]
