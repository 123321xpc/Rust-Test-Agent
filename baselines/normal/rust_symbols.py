"""Approximate source-level Rust definition, caller, and reference lookup."""

import json
import os
import re
from pathlib import Path

from langchain_core.tools import tool


def _mask_noncode(source):
    """Mask comments and string/character literals without changing offsets."""
    chars = list(source)
    n, i = len(source), 0

    def mask(start, end):
        for k in range(start, end):
            if source[k] not in "\r\n":
                chars[k] = " "

    while i < n:
        start = i
        if source.startswith("//", i):
            end = source.find("\n", i)
            i = n if end < 0 else end
        elif source.startswith("/*", i):
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
        else:
            raw = re.match(r'(?:br|cr|r)(#*)"', source[i:i + 260]) if source[i] in "bcr" and (i == 0 or not (source[i - 1].isalnum() or source[i - 1] == "_")) else None
            if raw:
                closing = '"' + raw.group(1)
                end = source.find(closing, i + raw.end())
                i = n if end < 0 else end + len(closing)
            elif source[i] == '"':
                i += 1
                while i < n:
                    if source[i] == "\\":
                        i += 2
                    elif source[i] == '"':
                        i += 1
                        break
                    else:
                        i += 1
                i = min(i, n)
            elif source[i] == "'":
                # Distinguish character literals from lifetime names such as 'a.
                char = re.match(r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\r\n])'", source[i:i + 32])
                if char:
                    i += char.end()
                else:
                    i += 1
                    continue
            else:
                i += 1
                continue
        mask(start, i)
    return "".join(chars)


_IDENT = r"(?:r#)?(?:[^\W\d]|_)\w*"
_ITEM = re.compile(
    r"(?<![\w#])(?:(?P<kind>fn|struct|enum|trait|mod|union|type|const)\s+|(?P<static_kind>static)\s+(?:mut\s+)?)(?P<name>" + _IDENT + r")",
    re.UNICODE,
)
_MACRO = re.compile(r"(?<![\w#])macro_rules!\s*(?P<name>" + _IDENT + r")", re.UNICODE)
_IMPL = re.compile(r"(?<![\w#])impl\b(?P<header>[^;{}]{1,1000})(?=\{)", re.DOTALL)


def _matches_name(query: str, item_kind: str, item_name: str, exact: bool) -> bool:
    if not query:
        return True
    if not exact:
        return query in item_name
    clean_query = query.removeprefix("r#")
    if item_kind == "impl":
        return bool(re.search(r"(?<![\w#])(?:r#)?" + re.escape(clean_query) + r"(?!\w)", item_name))
    return item_name.removeprefix("r#") == clean_query


def _collect_rs_files(root: Path, target: Path, exclude_tests: bool) -> list[Path]:
    skip_dirs = {".git", "target"} | ({"tests", "benches"} if exclude_tests else set())
    files = []
    if target.is_file():
        if target.suffix == ".rs":
            files.append(target)
    elif target.is_dir():
        for directory, dirs, names in os.walk(target, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in skip_dirs and not (Path(directory) / d).is_symlink())
            files.extend(Path(directory) / f for f in sorted(names) if f.endswith(".rs"))
    return files


def _read_and_mask(file: Path, root: Path, mask_cache: dict | None = None):
    resolved = file.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        return None, None
    try:
        st = resolved.stat()
        key = (resolved, st.st_mtime_ns, st.st_size)
        if mask_cache is not None and key in mask_cache:
            return mask_cache[key]
        source = resolved.read_text(encoding="utf-8")
        code = _mask_noncode(source)
        if mask_cache is not None:
            if len(mask_cache) > 2000:
                mask_cache.clear()
            mask_cache[key] = (source, code)
        return source, code
    except (OSError, UnicodeError):
        return None, None


def find_rust_references(
    root: Path,
    name: str,
    target: Path | None = None,
    max_results: int = 100,
    exclude_tests: bool = False,
    callers_only: bool = False,
    mask_cache: dict | None = None,
) -> tuple[list[dict], bool]:
    """Find call sites or identifier references outside comments and strings."""
    root = Path(root).resolve()
    search_target = (target or root).resolve()
    clean_name = name.strip().removeprefix("r#")
    if not clean_name:
        return [], False

    escaped = re.escape(clean_name)
    if callers_only:
        pattern = re.compile(
            r"(?<![\w#])(?:r#)?" + escaped + r"(?:\s*(?:::<[^;{}()]{0,120}>)?\s*\(|!\s*[\(\[\{])",
            re.UNICODE,
        )
        decl_prefix = re.compile(
            r"(?<![\w#])(?:fn|macro_rules!)\s+$",
            re.UNICODE,
        )
    else:
        pattern = re.compile(r"(?<![\w#])(?:r#)?" + escaped + r"(?!\w)", re.UNICODE)
        decl_prefix = None

    files = _collect_rs_files(root, search_target, exclude_tests=exclude_tests)
    matches = []
    for file in files:
        source, code = _read_and_mask(file, root, mask_cache=mask_cache)
        if source is None:
            continue
        rel_path = file.relative_to(root).as_posix()
        seen_lines = set()
        for m in pattern.finditer(code):
            offset = m.start()
            if decl_prefix is not None:
                prefix_window = code[max(0, offset - 32):offset]
                if decl_prefix.search(prefix_window):
                    continue
            line = source.count("\n", 0, offset) + 1
            if line in seen_lines:
                continue
            seen_lines.add(line)
            start = source.rfind("\n", 0, offset) + 1
            end = source.find("\n", offset)
            context_line = source[start:end if end >= 0 else len(source)].strip()
            matches.append({"path": rel_path, "line": line, "context": context_line})
            if len(matches) > max_results:
                return matches[:max_results], True
    return matches, False


def make_rust_symbols_tool(root, resolve_path):
    mask_cache = {}

    @tool
    def rust_symbols(
        name: str = "",
        kind: str = "",
        path: str = ".",
        max_results: int = 100,
        exact: bool = False,
        exclude_tests: bool = False,
        outline: bool = False,
        callers: bool = False,
        references: bool = False,
    ) -> str:
        """Find Rust definitions, callers, or identifier references in repository-relative path.

        Default mode searches definitions (kinds: fn, struct, enum, trait, impl, mod, union, type, const, static, macro).
        Set exact=True for exact identifier matches; exclude_tests=True to skip tests/ and benches/.
        Set outline=True for an ordered structural summary of all definitions in target path/file.
        Set callers=True with a symbol name to find call sites (excluding the fn definition itself).
        Set references=True with a symbol name to find all non-comment/non-string identifier occurrences.
        Results are approximate text-based matches, not semantic resolution.
        """
        try:
            if callers and references:
                raise ValueError("Select at most one of callers and references")
            if (callers or references) and not name.strip():
                raise ValueError("Provide a symbol name when searching callers or references")
            if not 1 <= max_results <= 500:
                raise ValueError("max_results must be between 1 and 500")
            target = resolve_path(path)
            if not target.exists():
                raise ValueError("Path does not exist")

            if callers or references:
                results, truncated = find_rust_references(
                    root,
                    name=name,
                    target=target,
                    max_results=max_results,
                    exclude_tests=exclude_tests,
                    callers_only=callers,
                    mask_cache=mask_cache,
                )
                return json.dumps({
                    "method": "text_approximation",
                    "mode": "callers" if callers else "references",
                    "results": results,
                    "truncated": truncated,
                }, ensure_ascii=False)

            kinds = {"fn", "struct", "enum", "trait", "impl", "mod", "union", "type", "const", "static", "macro"}
            if kind and kind not in kinds:
                raise ValueError("Unknown kind; use fn, struct, enum, trait, impl, mod, union, type, const, static, or macro")

            files = _collect_rs_files(root, target, exclude_tests=exclude_tests)
            matches = []
            for file in files:
                source, code = _read_and_mask(file, root, mask_cache=mask_cache)
                if source is None:
                    continue
                found = [
                    (m.start(), m.group("kind") or m.group("static_kind"), m.group("name"))
                    for m in _ITEM.finditer(code)
                ]
                found.extend((m.start(), "macro", m.group("name")) for m in _MACRO.finditer(code))
                found.extend((m.start(), "impl", " ".join(source[m.start():m.end()].split())) for m in _IMPL.finditer(code))
                for offset, item_kind, item_name in sorted(found):
                    if (kind and kind != item_kind) or not _matches_name(name, item_kind, item_name, exact):
                        continue
                    line = source.count("\n", 0, offset) + 1
                    start = source.rfind("\n", 0, offset) + 1
                    end = source.find("\n", offset)
                    context_line = source[start:end if end >= 0 else len(source)].strip()
                    if "{" not in context_line and ";" not in context_line:
                        brace = source.find("{", offset)
                        semi = source.find(";", offset)
                        delims = [p for p in (brace, semi) if p >= 0]
                        if delims and min(delims) - offset < 250:
                            context = " ".join(source[start:min(delims)].split())
                        else:
                            context = context_line
                    else:
                        context = context_line
                    matches.append({"path": file.relative_to(root).as_posix(), "line": line, "kind": item_kind, "name": item_name, "context": context})
                    if len(matches) > max_results:
                        res = matches[:max_results]
                        if outline:
                            return json.dumps({"outline": res, "truncated": True}, ensure_ascii=False)
                        return json.dumps({"method": "text_approximation", "results": res, "truncated": True}, ensure_ascii=False)
            if outline:
                return json.dumps({"outline": matches, "truncated": False}, ensure_ascii=False)
            return json.dumps({"method": "text_approximation", "results": matches, "truncated": False}, ensure_ascii=False)
        except (OSError, ValueError) as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    return rust_symbols
