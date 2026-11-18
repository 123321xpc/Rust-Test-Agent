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
_PUB_USE = re.compile(r"(?<![\w#])pub(?:\s*\([^\)]+\))?\s+use\s+(?P<path>[^;{}]+);", re.UNICODE)


def _matches_name(query: str, item_kind: str, item_name: str, exact: bool) -> bool:
    if not query:
        return True
    if not exact:
        return query in item_name
    clean_query = query.removeprefix("r#")
    if item_kind in ("impl", "use"):
        return bool(re.search(r"(?<![\w#])(?:r#)?" + re.escape(clean_query) + r"(?!\w)", item_name))
    return item_name.removeprefix("r#") == clean_query


_MAX_WIDE_SCAN_BYTES = 400_000


def _collect_rs_files(
    root: Path,
    target: Path,
    exclude_tests: bool,
    priority_dir: Path | None = None,
    max_file_bytes: int | None = None,
) -> list[Path]:
    skip_dirs = {".git", "target", ".cargo"} | ({"tests", "benches"} if exclude_tests else set())
    files = []
    if target.is_file():
        if target.suffix == ".rs":
            files.append(target)
    elif target.is_dir():
        for directory, dirs, names in os.walk(target, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in skip_dirs and not (Path(directory) / d).is_symlink())
            for f in sorted(names):
                if not f.endswith(".rs"):
                    continue
                fpath = Path(directory) / f
                if max_file_bytes is not None:
                    try:
                        if fpath.stat().st_size > max_file_bytes:
                            continue
                    except OSError:
                        continue
                files.append(fpath)
    if priority_dir is not None and len(files) > 1:
        try:
            prio = Path(priority_dir).resolve()
            files.sort(key=lambda p: (0 if p.resolve().is_relative_to(prio) else 1, p.as_posix()))
        except Exception:
            pass
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


def _build_enclosing_blocks(source: str, code: str) -> list[tuple[int, int, str]]:
    open_braces = {}
    for m in re.finditer(r"(?<![\w#])(?:pub(?:\s*\([^\)]+\))?\s+)?impl\b(?P<header>[^;{}]{1,500})\{", code):
        open_braces[m.end() - 1] = " ".join(source[m.start():m.end() - 1].split())
    for m in re.finditer(r"(?<![\w#])(?:pub(?:\s*\([^\)]+\))?\s+)?trait\s+(?P<name>(?:r#)?[A-Za-z_]\w*)(?P<header>[^;{}]{0,500})\{", code):
        open_braces[m.end() - 1] = " ".join(source[m.start():m.end() - 1].split())

    blocks = []
    stack = []
    for idx, ch in enumerate(code):
        if ch == "{":
            hdr = open_braces.get(idx)
            stack.append((idx, hdr))
        elif ch == "}":
            if stack:
                start_idx, hdr = stack.pop()
                if hdr:
                    blocks.append((start_idx, idx, hdr))
    return blocks


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


def _extract_file_symbols(
    source: str,
    code: str,
    root: Path,
    file: Path,
    kind: str = "",
    query: str = "",
    exact: bool = False,
    multiline_context: bool = True,
) -> list[dict]:
    """Extract matching Rust symbol definitions from one masked source file."""
    enclosing_blocks = _build_enclosing_blocks(source, code)
    found = [
        (m.start(), m.group("kind") or m.group("static_kind"), m.group("name"))
        for m in _ITEM.finditer(code)
    ]
    found.extend((m.start(), "macro", m.group("name")) for m in _MACRO.finditer(code))
    found.extend((m.start(), "impl", " ".join(source[m.start():m.end()].split())) for m in _IMPL.finditer(code))
    found.extend((m.start(), "use", m.group("path").strip()) for m in _PUB_USE.finditer(code))

    entries = []
    rel_path = file.relative_to(root).as_posix()
    for offset, item_kind, item_name in sorted(found):
        if (kind and kind != item_kind) or not _matches_name(query, item_kind, item_name, exact):
            continue
        line = source.count("\n", 0, offset) + 1
        start = source.rfind("\n", 0, offset) + 1
        end = source.find("\n", offset)
        context_line = source[start:end if end >= 0 else len(source)].strip()
        if multiline_context and "{" not in context_line and ";" not in context_line:
            brace = source.find("{", offset)
            semi = source.find(";", offset)
            delims = [p for p in (brace, semi) if p >= 0]
            if delims and min(delims) - offset < 250:
                context = " ".join(source[start:min(delims)].split())
            else:
                context = context_line
        else:
            context = context_line

        prefix = code[max(0, offset - 50):offset]
        m_vis = re.search(r"\bpub(?:\s*\([^\)]+\))?(?=\s*(?:async\s+|unsafe\s+|const\s+|extern\s+(?:\"[^\"]+\"\s+)?)*$)", prefix)
        vis = " ".join(m_vis.group(0).split()) if m_vis else "private"

        enclosing = None
        for b_start, b_end, b_hdr in enclosing_blocks:
            if b_start <= offset <= b_end:
                if enclosing is None or b_start > enclosing[0]:
                    enclosing = (b_start, b_hdr)
        enclosing_text = enclosing[1] if enclosing else None

        attr_window = source[max(0, offset - 400):offset]
        attrs = [
            attr_match.group(1).strip()
            for attr_match in re.finditer(r"#\[\s*(derive\([^]]+\)|cfg\([^]]+\))\s*\]", attr_window)
        ]

        entry = {
            "path": rel_path,
            "line": line,
            "kind": item_kind,
            "name": item_name,
            "context": context,
            "vis": vis,
        }
        if enclosing_text and item_kind != "impl":
            entry["enclosing"] = enclosing_text
        if attrs:
            entry["attrs"] = attrs
        entries.append(entry)
    return entries


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

        Default mode searches definitions (kinds: fn, struct, enum, trait, impl, mod, union, type, const, static, macro, use).
        Returns item kind, name, visibility (pub/pub(crate)/private), enclosing impl/trait signature, and derive/cfg attributes when applicable.
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

            kinds = {"fn", "struct", "enum", "trait", "impl", "mod", "union", "type", "const", "static", "macro", "use"}
            if kind and kind not in kinds:
                raise ValueError("Unknown kind; use fn, struct, enum, trait, impl, mod, union, type, const, static, macro, or use")

            files = _collect_rs_files(root, target, exclude_tests=exclude_tests)
            matches = []
            for file in files:
                source, code = _read_and_mask(file, root, mask_cache=mask_cache)
                if source is None:
                    continue
                for entry in _extract_file_symbols(source, code, root, file, kind=kind, query=name, exact=exact):
                    matches.append(entry)
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


def lookup_symbol_details(
    root: Path,
    name: str,
    kind: str = "",
    exact: bool = True,
    max_results: int = 5,
    exclude_tests: bool = True,
    mask_cache: dict | None = None,
    priority_dir: Path | None = None,
    deadline: float | None = None,
) -> list[dict]:
    """Search for exact or fuzzy symbol definitions across the repository."""
    import time as _time

    root = Path(root).resolve()
    clean_name = name.strip().removeprefix("r#")
    if not clean_name:
        return []
    files = _collect_rs_files(root, root, exclude_tests=exclude_tests, max_file_bytes=_MAX_WIDE_SCAN_BYTES)
    prio_resolved = Path(priority_dir).resolve() if priority_dir is not None else None

    def _sort_key(p: Path):
        try:
            resolved = p.resolve()
            in_prio = 0 if (prio_resolved is not None and resolved.is_relative_to(prio_resolved)) else 1
            rel = resolved.relative_to(root).as_posix()
        except ValueError:
            in_prio = 1
            rel = p.as_posix()
        if rel.startswith("src/") or "/src/" in rel:
            return (in_prio, 0, rel)
        if "/" not in rel:
            return (in_prio, 1, rel)
        return (in_prio, 2, rel)

    files.sort(key=_sort_key)
    matches = []
    for file in files:
        if deadline is not None and _time.monotonic() >= deadline:
            break
        try:
            raw_source = file.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if clean_name not in raw_source:
            continue
        source, code = _read_and_mask(file, root, mask_cache=mask_cache)
        if source is None:
            continue
        matches.extend(_extract_file_symbols(
            source, code, root, file, kind=kind, query=clean_name, exact=exact, multiline_context=False,
        ))
        if len(matches) >= max_results * 3:
            break
        if prio_resolved is not None and len(matches) >= max_results:
            try:
                if not file.resolve().is_relative_to(prio_resolved):
                    break
            except Exception:
                pass

    def _match_priority(m):
        kind_prio = 0 if m["kind"] in ("struct", "enum", "trait", "fn", "type", "const", "mod", "union") else 1
        path_prio = 0 if m["path"].startswith("src/") else 1
        return (kind_prio, path_prio, m["path"], m["line"])

    matches.sort(key=_match_priority)
    return matches[:max_results]


def find_trait_implementations(
    root: Path,
    trait_name: str,
    max_results: int = 3,
    mask_cache: dict | None = None,
    priority_dir: Path | None = None,
    deadline: float | None = None,
) -> list[dict]:
    """Find existing 'impl ... Trait for ...' blocks in the repository."""
    import time as _time

    root = Path(root).resolve()
    clean_trait = trait_name.strip().removeprefix("r#")
    if not clean_trait:
        return []
    pattern = re.compile(
        r"(?<![\w#])impl\b(?P<header>[^;{}]{1,500}\b" + re.escape(clean_trait) + r"\b[^;{}]{0,200}\bfor\b[^;{}]{1,200})\{",
        re.DOTALL,
    )
    files = _collect_rs_files(root, root, exclude_tests=True, priority_dir=priority_dir, max_file_bytes=_MAX_WIDE_SCAN_BYTES)
    matches = []
    for file in files:
        if deadline is not None and _time.monotonic() >= deadline:
            break
        try:
            raw_source = file.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if clean_trait not in raw_source or "impl" not in raw_source:
            continue
        source, code = _read_and_mask(file, root, mask_cache=mask_cache)
        if source is None:
            continue
        for m in pattern.finditer(code):
            line = source.count("\n", 0, m.start()) + 1
            hdr = " ".join(m.group("header").split())
            matches.append({
                "path": file.relative_to(root).as_posix(),
                "line": line,
                "header": f"impl {hdr}",
            })
            if len(matches) >= max_results:
                return matches
    return matches


def find_reexports(
    root: Path,
    symbol_name: str,
    max_results: int = 3,
    mask_cache: dict | None = None,
    priority_dir: Path | None = None,
    deadline: float | None = None,
) -> list[dict]:
    """Find 'pub use ...::<symbol>' re-exports in the repository."""
    import time as _time

    root = Path(root).resolve()
    clean_name = symbol_name.strip().removeprefix("r#")
    if not clean_name:
        return []
    files = _collect_rs_files(root, root, exclude_tests=True, priority_dir=priority_dir, max_file_bytes=_MAX_WIDE_SCAN_BYTES)
    matches = []
    for file in files:
        if deadline is not None and _time.monotonic() >= deadline:
            break
        try:
            raw_source = file.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if clean_name not in raw_source or "pub" not in raw_source:
            continue
        source, code = _read_and_mask(file, root, mask_cache=mask_cache)
        if source is None:
            continue
        for m in _PUB_USE.finditer(code):
            use_path = m.group("path").strip()
            if use_path.endswith(f"::{clean_name}") or use_path == clean_name or f"::{clean_name}," in use_path or f"::{clean_name}}}" in use_path:
                line = source.count("\n", 0, m.start()) + 1
                matches.append({
                    "path": file.relative_to(root).as_posix(),
                    "line": line,
                    "reexport": f"pub use {use_path};",
                })
                if len(matches) >= max_results:
                    return matches
    return matches
