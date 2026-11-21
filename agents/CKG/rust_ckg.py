"""Lightweight, zero-dependency Rust Code Knowledge Graph (CKG) and token-aware structural parser.

Supports:
- Rust lexical tokenization (comments, nested block comments, raw/byte/c strings, lifetimes vs chars)
- Structural extraction (mod, pub use re-exports, struct/enum/trait/type/impl/fn/macro)
- Accurate function body spans & #[test]/#[cfg(test)] separation
- Incremental per-file cache keyed by (mtime_ns, size)
- Neighborhood queries (definitions, public re-export paths, trait impls, callers, callees)
- Multi-chain candidate fault propagation paths along call/trait/re-export edges
"""

import bisect
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import weakref
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


class RustToken:
    __slots__ = ("kind", "text", "start", "end", "line")

    def __init__(self, kind: str, text: str, start: int, end: int, line: int):
        self.kind = kind
        self.text = text
        self.start = start
        self.end = end
        self.line = line


def _compute_line_starts(source: str) -> list[int]:
    starts = [0]
    idx = 0
    while True:
        pos = source.find("\n", idx)
        if pos < 0:
            break
        starts.append(pos + 1)
        idx = pos + 1
    return starts


def _offset_to_line(line_starts: list[int], offset: int) -> int:
    return bisect.bisect_right(line_starts, offset)


def tokenize_rust(source: str) -> list[RustToken]:
    """Tokenize Rust source code, stripping comments/whitespace while preserving byte offsets and lines."""
    tokens: list[RustToken] = []
    n = len(source)
    i = 0
    line_starts = _compute_line_starts(source)

    while i < n:
        ch = source[i]

        if ch.isspace():
            i += 1
            continue

        # Line comment
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            end = source.find("\n", i + 2)
            i = n if end < 0 else end + 1
            continue

        # Nested block comment
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            depth = 1
            while i < n and depth > 0:
                if source[i] == "/" and i + 1 < n and source[i + 1] == "*":
                    depth += 1
                    i += 2
                elif source[i] == "*" and i + 1 < n and source[i + 1] == "/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            continue

        # Raw / byte-raw / c-raw string literals: r"...", r#"..."#, br#"..."#, cr#"..."#
        if ch in "bcr":
            raw_m = re.match(r'(?:br|cr|r)(#*)"', source[i:i + 64])
            if raw_m and (i == 0 or not (source[i - 1].isalnum() or source[i - 1] == "_")):
                hashes = raw_m.group(1)
                close_seq = '"' + hashes
                start = i
                content_start = i + raw_m.end()
                close_idx = source.find(close_seq, content_start)
                end = n if close_idx < 0 else close_idx + len(close_seq)
                tokens.append(RustToken("STRING", source[start:end], start, end, _offset_to_line(line_starts, start)))
                i = end
                continue
            if ch in ("b", "c") and i + 1 < n and source[i + 1] == '"':
                start = i
                i += 2
                while i < n:
                    if source[i] == "\\":
                        i += 2
                    elif source[i] == '"':
                        i += 1
                        break
                    else:
                        i += 1
                end = min(i, n)
                tokens.append(RustToken("STRING", source[start:end], start, end, _offset_to_line(line_starts, start)))
                continue

        # Normal string literal
        if ch == '"':
            start = i
            i += 1
            while i < n:
                if source[i] == "\\":
                    i += 2
                elif source[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            end = min(i, n)
            tokens.append(RustToken("STRING", source[start:end], start, end, _offset_to_line(line_starts, start)))
            continue

        # Byte char literal
        if ch == "b" and i + 1 < n and source[i + 1] == "'":
            char_m = re.match(r"b'(?:\\(?:x[0-9a-fA-F]{2}|.)|[^'\\\r\n])'", source[i:i + 16])
            if char_m:
                start = i
                end = i + char_m.end()
                tokens.append(RustToken("CHAR", source[start:end], start, end, _offset_to_line(line_starts, start)))
                i = end
                continue

        # Char literal vs lifetime
        if ch == "'":
            char_m = re.match(r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\r\n])'", source[i:i + 32])
            if char_m:
                start = i
                end = i + char_m.end()
                tokens.append(RustToken("CHAR", source[start:end], start, end, _offset_to_line(line_starts, start)))
                i = end
                continue
            life_m = re.match(r"'[A-Za-z_]\w*", source[i:i + 128])
            if life_m:
                start = i
                end = i + life_m.end()
                tokens.append(RustToken("LIFETIME", source[start:end], start, end, _offset_to_line(line_starts, start)))
                i = end
                continue
            tokens.append(RustToken("PUNCT", "'", i, i + 1, _offset_to_line(line_starts, i)))
            i += 1
            continue

        # Identifiers (including r#ident)
        if ch == "r" and i + 2 < n and source[i + 1] == "#" and (source[i + 2].isalpha() or source[i + 2] == "_"):
            ident_m = re.match(r"r#[A-Za-z_]\w*", source[i:i + 256])
            if ident_m:
                start = i
                end = i + ident_m.end()
                tokens.append(RustToken("IDENT", source[start:end], start, end, _offset_to_line(line_starts, start)))
                i = end
                continue

        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            tokens.append(RustToken("IDENT", source[start:i], start, i, _offset_to_line(line_starts, start)))
            continue

        # Numbers
        if ch.isdigit():
            start = i
            i += 1
            while i < n and (source[i].isalnum() or source[i] in "_."):
                if source[i] == "." and i + 1 < n and not source[i + 1].isdigit():
                    break
                i += 1
            tokens.append(RustToken("NUMBER", source[start:i], start, i, _offset_to_line(line_starts, start)))
            continue

        # Two-char operators
        two = source[i:i + 2]
        if two in ("::", "->", "=>", "..", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/="):
            tokens.append(RustToken("OP", two, i, i + 2, _offset_to_line(line_starts, i)))
            i += 2
            continue

        if ch == "#":
            tokens.append(RustToken("ATTR_HASH", "#", i, i + 1, _offset_to_line(line_starts, i)))
            i += 1
            continue

        if ch in "{}()[];,:<>.*&!?+-=/%|@^~":
            kind = "PUNCT" if ch in "{}()[];,:.<>" else "OP"
            tokens.append(RustToken(kind, ch, i, i + 1, _offset_to_line(line_starts, i)))
            i += 1
            continue

        i += 1

    return tokens


def _find_matching(tokens: list[RustToken], open_idx: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    n = len(tokens)
    i = open_idx
    while i < n:
        t = tokens[i].text
        if t == open_ch:
            depth += 1
        elif t == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def _find_matching_angles(tokens: list[RustToken], open_idx: int) -> int:
    depth = 0
    n = len(tokens)
    i = open_idx
    while i < n:
        t = tokens[i].text
        if t == "<":
            depth += 1
        elif t == ">":
            depth -= 1
            if depth == 0:
                return i
        elif t in ("{", ";"):
            return i - 1
        i += 1
    return open_idx


def _norm_snippet(source: str, start_off: int, end_off: int, limit: int = 240) -> str:
    txt = " ".join(source[start_off:end_off].split())
    return txt[:limit] if len(txt) > limit else txt


def _expand_use_tree(tokens: list[RustToken]) -> list[tuple[str, str]]:
    """Expand a Rust use token slice (after use, before semicolon) into [(full_path, exported_name)]."""
    def parse_group(idx: int, prefix: str) -> tuple[list[tuple[str, str]], int]:
        items: list[tuple[str, str]] = []
        n = len(tokens)
        while idx < n:
            if tokens[idx].text == "}":
                return items, idx + 1
            if tokens[idx].text == ",":
                idx += 1
                continue
            seg_parts: list[str] = []
            alias: str | None = None
            while idx < n and tokens[idx].text not in (",", "}", "{"):
                if tokens[idx].text == "as" and idx + 1 < n:
                    alias = tokens[idx + 1].text
                    idx += 2
                    break
                if tokens[idx].text != "::":
                    seg_parts.append(tokens[idx].text)
                idx += 1
            joined_seg = "::".join(seg_parts)
            full = f"{prefix}::{joined_seg}" if (prefix and joined_seg) else (prefix or joined_seg)
            if idx < n and tokens[idx].text == "{":
                sub_items, idx = parse_group(idx + 1, full)
                items.extend(sub_items)
            else:
                if joined_seg == "self":
                    base_name = prefix.split("::")[-1] if prefix else ""
                    items.append((prefix, alias or base_name))
                elif joined_seg == "*":
                    items.append((full, "*"))
                elif full:
                    base_name = full.split("::")[-1]
                    items.append((full, alias or base_name))
        return items, idx

    if not tokens:
        return []
    res, _ = parse_group(0, "")
    return res


def _derive_module_path(rel_posix: str) -> str:
    """Map repository-relative .rs path to an approximate crate-relative module path."""
    parts = list(Path(rel_posix).parts)
    if "src" in parts:
        idx = len(parts) - 1 - parts[::-1].index("src")
        sub = parts[idx + 1:]
    else:
        sub = parts
    if not sub:
        return "crate"
    stem_parts = list(sub)
    last = stem_parts[-1]
    if last in ("lib.rs", "main.rs"):
        stem_parts = stem_parts[:-1]
    elif last == "mod.rs":
        stem_parts = stem_parts[:-1]
    elif last.endswith(".rs"):
        stem_parts[-1] = last[:-3]
    clean = [p for p in stem_parts if p and p != "."]
    return "crate" + ("::" + "::".join(clean) if clean else "")


def _parse_visibility(tokens: list[RustToken], idx: int) -> tuple[str, int]:
    """Parse optional visibility starting at idx; returns (vis_str, next_idx)."""
    n = len(tokens)
    if idx < n and tokens[idx].text == "pub":
        if idx + 1 < n and tokens[idx + 1].text == "(":
            close_i = _find_matching(tokens, idx + 1, "(", ")")
            inner = "".join(t.text for t in tokens[idx + 2:close_i])
            return f"pub({inner})", close_i + 1
        return "pub", idx + 1
    return "private", idx


_SKIP_CALL_NAMES = frozenset({
    "if", "else", "for", "while", "loop", "match", "return", "let", "mut", "ref",
    "Some", "None", "Ok", "Err", "Box", "Vec", "String", "assert", "assert_eq",
    "assert_ne", "println", "eprintln", "format", "vec", "panic", "unreachable",
    "todo", "unimplemented", "cfg", "derive", "sizeof", "align_of", "drop",
})


def _extract_calls_and_types(tokens: list[RustToken], start_i: int, end_i: int) -> tuple[list[str], list[str]]:
    """Extract called function/method names and referenced CamelCase types inside a function body."""
    callees: list[str] = []
    seen_callees: set[str] = set()
    type_refs: list[str] = []
    seen_types: set[str] = set()

    def add_callee(name: str):
        clean = name.removeprefix("r#")
        if clean and clean not in _SKIP_CALL_NAMES and clean not in seen_callees:
            seen_callees.add(clean)
            callees.append(clean)

    i = start_i
    while i <= end_i and i < len(tokens):
        tok = tokens[i]
        if tok.kind == "IDENT":
            nxt = tokens[i + 1].text if i + 1 <= end_i else ""
            if nxt == "(":
                add_callee(tok.text)
                if i >= 2 and tokens[i - 1].text == "::" and tokens[i - 2].kind == "IDENT":
                    add_callee(f"{tokens[i - 2].text}::{tok.text}")
            elif nxt == "::" and i + 2 <= end_i and tokens[i + 2].text == "<":
                end_ang = _find_matching_angles(tokens, i + 2)
                if end_ang + 1 <= end_i and tokens[end_ang + 1].text == "(":
                    add_callee(tok.text)
            elif nxt == "!" and i + 2 <= end_i and tokens[i + 2].text in ("(", "[", "{"):
                add_callee(tok.text)

            raw = tok.text.removeprefix("r#")
            if raw and raw[0].isupper() and any(c.islower() for c in raw) and raw not in _SKIP_CALL_NAMES:
                if raw not in seen_types:
                    seen_types.add(raw)
                    type_refs.append(raw)
        i += 1
    return callees, type_refs


class ParsedRustFile:
    def __init__(self, rel_path: str, module_path: str):
        self.rel_path = rel_path
        self.module_path = module_path
        self.items: list[dict[str, Any]] = []
        self.reexports: list[dict[str, Any]] = []
        self.impl_blocks: list[dict[str, Any]] = []
        self.mod_decls: list[dict[str, Any]] = []
        self.call_edges: list[dict[str, Any]] = []


def parse_rust_file(source: str, rel_path: str) -> ParsedRustFile:
    """Parse a Rust file using token stream analysis into items, impls, re-exports, and call edges."""
    tokens = tokenize_rust(source)
    base_mod = _derive_module_path(rel_path)
    parsed = ParsedRustFile(rel_path=rel_path, module_path=base_mod)
    total_lines = source.count("\n") + 1

    def parse_block(tok_start: int, tok_end: int, mod_stack: list[str], enclosing_hdr: str | None, in_test_scope: bool):
        i = tok_start
        pending_attrs: list[str] = []
        pending_is_test = in_test_scope

        while i < tok_end:
            tok = tokens[i]

            if tok.kind == "ATTR_HASH":
                j = i + 1
                if j < tok_end and tokens[j].text == "!":
                    j += 1
                if j < tok_end and tokens[j].text == "[":
                    close_b = _find_matching(tokens, j, "[", "]")
                    attr_txt = _norm_snippet(
                        source,
                        tokens[j + 1].start if j + 1 < close_b else tokens[j].start,
                        tokens[close_b].start,
                        180,
                    )
                    if i + 1 < tok_end and tokens[i + 1].text != "!":
                        if attr_txt.startswith("derive(") or attr_txt.startswith("cfg(") or attr_txt == "test" or attr_txt.endswith("::test") or attr_txt == "macro_export":
                            pending_attrs.append(attr_txt)
                        if attr_txt == "test" or attr_txt.endswith("::test") or attr_txt == "cfg(test)":
                            pending_is_test = True
                    i = close_b + 1
                    continue
                i += 1
                continue

            if tok.text == ";":
                pending_attrs.clear()
                pending_is_test = in_test_scope
                i += 1
                continue

            vis, k = _parse_visibility(tokens, i)
            if k >= tok_end:
                break

            # pub use ...; or use ...;
            if tokens[k].text == "use":
                semi = k + 1
                while semi < tok_end and tokens[semi].text != ";":
                    semi += 1
                if vis != "private":
                    expanded = _expand_use_tree(tokens[k + 1:semi])
                    raw_use = _norm_snippet(source, tokens[i].start, tokens[min(semi, tok_end - 1)].end, 220)
                    for target_path, exported_name in expanded:
                        parsed.reexports.append({
                            "path": rel_path,
                            "line": tok.line,
                            "vis": vis,
                            "module_path": "::".join(mod_stack),
                            "target": target_path,
                            "exported_name": exported_name,
                            "reexport": raw_use,
                        })
                pending_attrs.clear()
                pending_is_test = in_test_scope
                i = semi + 1
                continue

            # mod foo; or mod foo { ... }
            if tokens[k].text == "mod" and k + 1 < tok_end and tokens[k + 1].kind == "IDENT":
                mod_name = tokens[k + 1].text
                cur_mod = "::".join(mod_stack)
                child_mod = f"{cur_mod}::{mod_name}"
                if k + 2 < tok_end and tokens[k + 2].text == ";":
                    parsed.mod_decls.append({
                        "path": rel_path,
                        "line": tok.line,
                        "name": mod_name,
                        "vis": vis,
                        "module_path": cur_mod,
                        "child_module": child_mod,
                        "inline": False,
                        "is_test": pending_is_test,
                    })
                    parsed.items.append({
                        "path": rel_path,
                        "line": tok.line,
                        "end_line": tok.line,
                        "kind": "mod",
                        "name": mod_name,
                        "vis": vis,
                        "module_path": cur_mod,
                        "context": _norm_snippet(source, tok.start, tokens[k + 2].end),
                        "is_test": pending_is_test,
                    })
                    pending_attrs.clear()
                    pending_is_test = in_test_scope
                    i = k + 3
                    continue
                if k + 2 < tok_end and tokens[k + 2].text == "{":
                    close_b = _find_matching(tokens, k + 2, "{", "}")
                    end_line = tokens[close_b].line
                    parsed.mod_decls.append({
                        "path": rel_path,
                        "line": tok.line,
                        "name": mod_name,
                        "vis": vis,
                        "module_path": cur_mod,
                        "child_module": child_mod,
                        "inline": True,
                        "is_test": pending_is_test,
                    })
                    parsed.items.append({
                        "path": rel_path,
                        "line": tok.line,
                        "end_line": end_line,
                        "kind": "mod",
                        "name": mod_name,
                        "vis": vis,
                        "module_path": cur_mod,
                        "context": _norm_snippet(source, tok.start, tokens[k + 1].end),
                        "is_test": pending_is_test,
                    })
                    child_test = pending_is_test or mod_name == "tests" or mod_name.startswith("__test_regression_")
                    parse_block(k + 3, close_b, mod_stack + [mod_name], None, child_test)
                    pending_attrs.clear()
                    pending_is_test = in_test_scope
                    i = close_b + 1
                    continue

            # macro_rules! foo { ... }
            if tokens[k].text == "macro_rules" and k + 2 < tok_end and tokens[k + 1].text == "!" and tokens[k + 2].kind == "IDENT":
                m_name = tokens[k + 2].text
                j = k + 3
                if j < tok_end and tokens[j].text in ("{", "(", "["):
                    close_ch = {"{": "}", "(": ")", "[": "]"}[tokens[j].text]
                    close_b = _find_matching(tokens, j, tokens[j].text, close_ch)
                    parsed.items.append({
                        "path": rel_path,
                        "line": tok.line,
                        "end_line": tokens[close_b].line,
                        "kind": "macro",
                        "name": m_name,
                        "vis": "pub" if any("macro_export" in a for a in pending_attrs) else vis,
                        "module_path": "::".join(mod_stack),
                        "context": f"macro_rules! {m_name}",
                        "is_test": pending_is_test,
                    })
                    pending_attrs.clear()
                    pending_is_test = in_test_scope
                    i = close_b + 1
                    continue

            is_unsafe = False
            if tokens[k].text == "unsafe" and k + 1 < tok_end and tokens[k + 1].text in ("trait", "impl"):
                is_unsafe = True
                k += 1

            # impl ... { ... }
            if tokens[k].text == "impl":
                j = k + 1
                ang = par = brk = 0
                brace_i = -1
                while j < tok_end:
                    t = tokens[j].text
                    if t == "<":
                        ang += 1
                    elif t == ">":
                        if j > 0 and tokens[j - 1].text != "-":
                            ang = max(0, ang - 1)
                    elif t == "(":
                        par += 1
                    elif t == ")":
                        par = max(0, par - 1)
                    elif t == "[":
                        brk += 1
                    elif t == "]":
                        brk = max(0, brk - 1)
                    elif t == "{" and ang == 0 and par == 0 and brk == 0:
                        brace_i = j
                        break
                    elif t == ";" and ang == 0 and par == 0 and brk == 0:
                        break
                    j += 1
                if brace_i > 0:
                    close_b = _find_matching(tokens, brace_i, "{", "}")
                    hdr_tokens = tokens[k:brace_i]
                    hdr_text = _norm_snippet(source, tokens[i].start, tokens[brace_i].start, 240)
                    for_pos = -1
                    ang_d = par_d = brk_d = 0
                    for hi, ht in enumerate(hdr_tokens):
                        if ht.text == "<": ang_d += 1
                        elif ht.text == ">" and (hi == 0 or hdr_tokens[hi - 1].text != "-"): ang_d = max(0, ang_d - 1)
                        elif ht.text == "(": par_d += 1
                        elif ht.text == ")": par_d = max(0, par_d - 1)
                        elif ht.text == "[": brk_d += 1
                        elif ht.text == "]": brk_d = max(0, brk_d - 1)
                        elif ht.text == "for" and ang_d == 0 and par_d == 0 and brk_d == 0:
                            for_pos = hi
                            break
                    trait_name = None
                    for_type = None
                    if for_pos > 0:
                        t_slice = hdr_tokens[1:for_pos]
                        if t_slice and t_slice[0].text == "<":
                            d = 0
                            cut = 0
                            for si, st in enumerate(t_slice):
                                if st.text == "<": d += 1
                                elif st.text == ">":
                                    d -= 1
                                    if d == 0:
                                        cut = si + 1
                                        break
                            t_slice = t_slice[cut:]
                        t_idents = [x.text for x in t_slice if x.kind == "IDENT"]
                        for si, st in enumerate(t_slice):
                            if st.text == "<":
                                t_idents = [x.text for x in t_slice[:si] if x.kind == "IDENT"]
                                break
                        if t_idents:
                            trait_name = t_idents[-1]
                        ty_slice = hdr_tokens[for_pos + 1:]
                        for si, st in enumerate(ty_slice):
                            if st.text == "where":
                                ty_slice = ty_slice[:si]
                                break
                        ty_idents = []
                        ang2 = 0
                        for st in ty_slice:
                            if st.text == "<": ang2 += 1
                            elif st.text == ">": ang2 = max(0, ang2 - 1)
                            elif ang2 == 0 and st.kind == "IDENT":
                                ty_idents.append(st.text)
                        if ty_idents:
                            for_type = ty_idents[-1]
                    else:
                        ty_slice = hdr_tokens[1:]
                        if ty_slice and ty_slice[0].text == "<":
                            d = 0
                            cut = 0
                            for si, st in enumerate(ty_slice):
                                if st.text == "<": d += 1
                                elif st.text == ">":
                                    d -= 1
                                    if d == 0:
                                        cut = si + 1
                                        break
                            ty_slice = ty_slice[cut:]
                        for si, st in enumerate(ty_slice):
                            if st.text == "where":
                                ty_slice = ty_slice[:si]
                                break
                        ty_idents = []
                        ang2 = 0
                        for st in ty_slice:
                            if st.text == "<": ang2 += 1
                            elif st.text == ">": ang2 = max(0, ang2 - 1)
                            elif ang2 == 0 and st.kind == "IDENT":
                                ty_idents.append(st.text)
                        if ty_idents:
                            for_type = ty_idents[-1]

                    impl_entry = {
                        "path": rel_path,
                        "line": tok.line,
                        "end_line": tokens[close_b].line,
                        "header": hdr_text,
                        "trait": trait_name,
                        "for_type": for_type,
                        "module_path": "::".join(mod_stack),
                        "is_test": pending_is_test,
                    }
                    parsed.impl_blocks.append(impl_entry)
                    parsed.items.append({
                        "path": rel_path,
                        "line": tok.line,
                        "end_line": tokens[close_b].line,
                        "kind": "impl",
                        "name": hdr_text,
                        "vis": vis,
                        "module_path": "::".join(mod_stack),
                        "context": hdr_text,
                        "is_test": pending_is_test,
                    })
                    parse_block(brace_i + 1, close_b, mod_stack, hdr_text, pending_is_test)
                    pending_attrs.clear()
                    pending_is_test = in_test_scope
                    i = close_b + 1
                    continue

            # trait Foo ... { ... }
            if tokens[k].text == "trait" and k + 1 < tok_end and tokens[k + 1].kind == "IDENT":
                tr_name = tokens[k + 1].text
                j = k + 2
                brace_i = -1
                ang = par = brk = 0
                while j < tok_end:
                    t = tokens[j].text
                    if t == "<": ang += 1
                    elif t == ">" and (j == 0 or tokens[j - 1].text != "-"): ang = max(0, ang - 1)
                    elif t == "(": par += 1
                    elif t == ")": par = max(0, par - 1)
                    elif t == "[": brk += 1
                    elif t == "]": brk = max(0, brk - 1)
                    elif t == "{" and ang == 0 and par == 0 and brk == 0:
                        brace_i = j
                        break
                    elif t == ";" and ang == 0 and par == 0 and brk == 0:
                        break
                    j += 1
                if brace_i > 0:
                    close_b = _find_matching(tokens, brace_i, "{", "}")
                    hdr_text = _norm_snippet(source, tokens[i].start, tokens[brace_i].start, 240)
                    parsed.items.append({
                        "path": rel_path,
                        "line": tok.line,
                        "end_line": tokens[close_b].line,
                        "kind": "trait",
                        "name": tr_name,
                        "vis": vis,
                        "module_path": "::".join(mod_stack),
                        "context": hdr_text,
                        "attrs": list(pending_attrs),
                        "is_test": pending_is_test,
                    })
                    parse_block(brace_i + 1, close_b, mod_stack, hdr_text, pending_is_test)
                    pending_attrs.clear()
                    pending_is_test = in_test_scope
                    i = close_b + 1
                    continue

            # fn (with optional qualifiers)
            fk = k
            fn_qualifiers: list[str] = []
            if is_unsafe:
                fn_qualifiers.append("unsafe")
            while fk < tok_end:
                t = tokens[fk].text
                if t in ("default", "async", "unsafe", "const"):
                    fn_qualifiers.append(t)
                    fk += 1
                elif t == "extern":
                    fn_qualifiers.append(t)
                    fk += 1
                    if fk < tok_end and tokens[fk].kind == "STRING":
                        fk += 1
                else:
                    break

            if fk < tok_end and tokens[fk].text == "fn" and fk + 1 < tok_end and tokens[fk + 1].kind == "IDENT":
                fn_name = tokens[fk + 1].text
                j = fk + 2
                par = brk = ang = 0
                saw_params = False
                brace_i = -1
                semi_i = -1
                while j < tok_end:
                    t = tokens[j].text
                    if t == "(":
                        if par == 0 and brk == 0 and ang == 0:
                            saw_params = True
                        par += 1
                    elif t == ")":
                        par = max(0, par - 1)
                    elif t == "[":
                        brk += 1
                    elif t == "]":
                        brk = max(0, brk - 1)
                    elif t == "<":
                        ang += 1
                    elif t == ">":
                        if j > 0 and tokens[j - 1].text != "-":
                            ang = max(0, ang - 1)
                    elif t == "{" and saw_params and par == 0 and brk == 0:
                        brace_i = j
                        break
                    elif t == ";" and par == 0 and brk == 0:
                        semi_i = j
                        break
                    j += 1

                if brace_i > 0:
                    close_b = _find_matching(tokens, brace_i, "{", "}")
                    sig_text = _norm_snippet(source, tokens[i].start, tokens[brace_i].start, 240)
                    callees, type_refs = _extract_calls_and_types(tokens, brace_i + 1, close_b - 1)
                    is_fn_test = pending_is_test or (fn_name.startswith("test_") and in_test_scope)
                    fn_entry = {
                        "path": rel_path,
                        "line": tok.line,
                        "end_line": tokens[close_b].line,
                        "kind": "fn",
                        "name": fn_name,
                        "vis": vis,
                        "module_path": "::".join(mod_stack),
                        "context": sig_text,
                        "has_body": True,
                        "is_const": "const" in fn_qualifiers,
                        "body_open_offset": tokens[brace_i].start,
                        "is_test": is_fn_test,
                        "callees": callees,
                        "type_refs": type_refs,
                    }
                    if enclosing_hdr:
                        fn_entry["enclosing"] = enclosing_hdr
                    if pending_attrs:
                        fn_entry["attrs"] = list(pending_attrs)
                    parsed.items.append(fn_entry)
                    caller_id = f"{rel_path}:{tok.line}:{fn_name}"
                    for ce in callees:
                        parsed.call_edges.append({
                            "caller_id": caller_id,
                            "caller": fn_name,
                            "caller_path": rel_path,
                            "caller_line": tok.line,
                            "caller_enclosing": enclosing_hdr,
                            "callee": ce,
                            "is_test": is_fn_test,
                        })
                    pending_attrs.clear()
                    pending_is_test = in_test_scope
                    i = close_b + 1
                    continue
                elif semi_i > 0:
                    sig_text = _norm_snippet(source, tokens[i].start, tokens[semi_i].start, 240)
                    fn_entry = {
                        "path": rel_path,
                        "line": tok.line,
                        "end_line": tokens[semi_i].line,
                        "kind": "fn",
                        "name": fn_name,
                        "vis": vis,
                        "module_path": "::".join(mod_stack),
                        "context": sig_text,
                        "has_body": False,
                        "is_const": "const" in fn_qualifiers,
                        "is_test": pending_is_test,
                    }
                    if enclosing_hdr:
                        fn_entry["enclosing"] = enclosing_hdr
                    if pending_attrs:
                        fn_entry["attrs"] = list(pending_attrs)
                    parsed.items.append(fn_entry)
                    pending_attrs.clear()
                    pending_is_test = in_test_scope
                    i = semi_i + 1
                    continue

            # struct / enum / union / type / const / static
            if tokens[k].text in ("struct", "enum", "union", "type", "const", "static"):
                item_kind = tokens[k].text
                name_idx = k + 1
                if item_kind == "static" and name_idx < tok_end and tokens[name_idx].text == "mut":
                    name_idx += 1
                if name_idx < tok_end and tokens[name_idx].kind == "IDENT":
                    item_name = tokens[name_idx].text
                    j = name_idx + 1
                    par = brk = ang = br_d = 0
                    end_idx = j
                    is_brace = False
                    while j < tok_end:
                        t = tokens[j].text
                        if t == "(": par += 1
                        elif t == ")": par = max(0, par - 1)
                        elif t == "[": brk += 1
                        elif t == "]": brk = max(0, brk - 1)
                        elif t == "<": ang += 1
                        elif t == ">" and (j == 0 or tokens[j - 1].text != "-"):
                            ang = max(0, ang - 1)
                        elif t == "{" and par == 0 and brk == 0 and br_d == 0 and item_kind in ("struct", "enum", "union"):
                            end_idx = _find_matching(tokens, j, "{", "}")
                            is_brace = True
                            break
                        elif t == "{":
                            br_d += 1
                        elif t == "}":
                            br_d = max(0, br_d - 1)
                        elif t == ";" and par == 0 and brk == 0 and br_d == 0:
                            end_idx = j
                            break
                        j += 1
                    end_idx = min(end_idx, tok_end - 1)
                    sig_end = tokens[j].start if (is_brace and j < tok_end) else tokens[end_idx].end
                    item_entry = {
                        "path": rel_path,
                        "line": tok.line,
                        "end_line": min(total_lines, tokens[end_idx].line),
                        "kind": item_kind,
                        "name": item_name,
                        "vis": vis,
                        "module_path": "::".join(mod_stack),
                        "context": _norm_snippet(source, tokens[i].start, sig_end, 220),
                        "is_test": pending_is_test,
                    }
                    if enclosing_hdr:
                        item_entry["enclosing"] = enclosing_hdr
                    if pending_attrs:
                        item_entry["attrs"] = list(pending_attrs)
                    parsed.items.append(item_entry)

                    if item_kind in ("struct", "enum"):
                        for attr in pending_attrs:
                            if attr.startswith("derive(") and attr.endswith(")"):
                                for d_tr in attr[7:-1].split(","):
                                    d_clean = d_tr.strip().split("::")[-1]
                                    if d_clean:
                                        parsed.impl_blocks.append({
                                            "path": rel_path,
                                            "line": tok.line,
                                            "end_line": tok.line,
                                            "header": f"#[derive({d_clean})] for {item_name}",
                                            "trait": d_clean,
                                            "for_type": item_name,
                                            "module_path": "::".join(mod_stack),
                                            "is_test": pending_is_test,
                                        })

                    pending_attrs.clear()
                    pending_is_test = in_test_scope
                    i = end_idx + 1
                    continue

            if tok.text == "{":
                close_b = _find_matching(tokens, i, "{", "}")
                pending_attrs.clear()
                pending_is_test = in_test_scope
                i = close_b + 1
                continue

            pending_attrs.clear()
            pending_is_test = in_test_scope
            i += 1

    parse_block(0, len(tokens), base_mod.split("::"), None, False)
    return parsed


_SEMANTICS_MOD: Any = None


def _load_semantics_module():
    global _SEMANTICS_MOD
    if _SEMANTICS_MOD is None:
        sem_path = Path(__file__).with_name("rust_semantics.py")
        if not sem_path.is_file():
            return None
        spec = importlib.util.spec_from_file_location("ckg_rust_semantics", sem_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _SEMANTICS_MOD = mod
    return _SEMANTICS_MOD


def _cleanup_tempdir(td: tempfile.TemporaryDirectory) -> None:
    try:
        td.cleanup()
    except OSError:
        pass


class RustCodeGraph:
    """Incremental repository-level Rust Code Knowledge Graph (CKG)."""

    def __init__(self, root: Path, env: dict[str, str] | None = None):
        self.root = Path(root).resolve()
        self.env = env
        self._file_cache: dict[Path, tuple[int, int, ParsedRustFile]] = {}
        self._pkg_roots: list[tuple[Path, str]] | None = None
        self.semantic: Any = None
        self._semantic_fp: str | None = None
        self._semantic_lock = threading.Lock()
        self._scratch_target_dir: tempfile.TemporaryDirectory | None = None
        self._last_deps_mtime_ns: int = 0
        self._deps_retry_fp: str | None = None
        self._last_semantic_error: str = ""

    def _semantic_env(self) -> dict[str, str]:
        base = dict(self.env if self.env is not None else os.environ)
        if not base.get("CARGO_TARGET_DIR"):
            if self._scratch_target_dir is None:
                self._scratch_target_dir = tempfile.TemporaryDirectory(prefix="ckg_sem_")
                weakref.finalize(self, _cleanup_tempdir, self._scratch_target_dir)
            base["CARGO_TARGET_DIR"] = self._scratch_target_dir.name
        return base

    @staticmethod
    def _target_deps_mtime_ns(sem_env: dict[str, str]) -> int:
        td = sem_env.get("CARGO_TARGET_DIR")
        if not td:
            return 0
        for sub in ("debug/deps", "release/deps"):
            p = Path(td) / sub
            try:
                if p.is_dir():
                    return p.stat().st_mtime_ns
            except OSError:
                continue
        return 0

    def _package_name_for_path(self, rel_path: str) -> str | None:
        fpath = (self.root / rel_path).resolve()
        while fpath != self.root.parent and fpath != fpath.parent:
            manifest = fpath / "Cargo.toml"
            if manifest.is_file():
                try:
                    content = manifest.read_text(encoding="utf-8", errors="replace")
                    m_sec = re.search(r"(?ms)^\[package\](.*?)(?:^\[|\Z)", content)
                    if m_sec:
                        m_name = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', m_sec.group(1))
                        if m_name:
                            return m_name.group(1).strip()
                except OSError:
                    pass
                break
            fpath = fpath.parent
        return None

    def ensure_semantic(
        self,
        *,
        target_symbols: list[str] | None = None,
        target_files: list[str] | None = None,
        need_mir: bool = False,
        force_mir_packages: list[str] | None = None,
        allow_subprocess: bool = True,
    ) -> Any:
        """Lazily build/refresh Stage 1 compiler definitions and optional Stage 2/3 MIR."""
        if not allow_subprocess:
            return self.semantic
        if self.semantic is not None and self._semantic_fp is None:
            return self.semantic
        has_manifest = (self.root / "Cargo.toml").is_file() or any(self.root.glob("*/Cargo.toml"))
        if not has_manifest:
            return None
        sem_mod = _load_semantics_module()
        if sem_mod is None:
            return None
        try:
            with self._semantic_lock:
                fp = sem_mod.compute_source_fingerprint(self.root)
                sem_env = self._semantic_env()
                deps_mtime = self._target_deps_mtime_ns(sem_env)
                focus_pkgs: list[str] = []
                for tf in target_files or ():
                    pkg = self._package_name_for_path(tf)
                    if pkg and pkg not in focus_pkgs:
                        focus_pkgs.append(pkg)
                should_retry_warm = (
                    self.semantic is not None
                    and not self.semantic.precise
                    and self._semantic_fp == fp
                    and deps_mtime > self._last_deps_mtime_ns
                    and self._deps_retry_fp != fp
                )
                if self.semantic is None or self._semantic_fp != fp or should_retry_warm:
                    if should_retry_warm:
                        self._deps_retry_fp = fp
                    self._last_deps_mtime_ns = deps_mtime
                    self.semantic = sem_mod.build_workspace_graph(
                        self.root, sem_env, fingerprint=fp, focus_packages=focus_pkgs or None,
                    )
                    self._semantic_fp = fp
                elif focus_pkgs and len(getattr(self.semantic, "packages", {})) > 6:
                    self.semantic.ensure_packages(focus_pkgs[:4], env=sem_env)
                if self.semantic is not None and (self.semantic.precise or force_mir_packages):
                    if force_mir_packages:
                        self.semantic.ensure_mir_for_packages(force_mir_packages, env=sem_env)
                    elif need_mir and target_symbols:
                        self.semantic.ensure_mir_for_symbols(target_symbols, env=sem_env)
                self._last_semantic_error = ""
        except Exception as exc:
            self._last_semantic_error = f"{type(exc).__name__}: {exc}"
            return self.semantic
        return self.semantic

    def _refresh_package_roots(self):
        roots: list[tuple[Path, str]] = []
        for dirpath, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in (".git", "target", ".cargo"))
            if "Cargo.toml" in files:
                mpath = Path(dirpath) / "Cargo.toml"
                try:
                    content = mpath.read_text(encoding="utf-8", errors="replace")
                    m_sec = re.search(r"(?ms)^\[package\](.*?)(?:^\[|\Z)", content)
                    if m_sec:
                        m_name = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', m_sec.group(1))
                        if m_name:
                            roots.append((Path(dirpath).resolve(), m_name.group(1).replace("-", "_")))
                except OSError:
                    pass
        roots.sort(key=lambda x: len(x[0].parts), reverse=True)
        self._pkg_roots = roots

    def crate_for_path(self, rel_path: str) -> str:
        if self._pkg_roots is None:
            self._refresh_package_roots()
        fpath = (self.root / rel_path).resolve()
        for pkg_dir, import_name in self._pkg_roots or ():
            if fpath.is_relative_to(pkg_dir):
                return import_name
        return "crate"

    def _collect_rs_files(self, exclude_tests: bool = True, max_file_bytes: int = 400_000) -> list[Path]:
        skip_dirs = {".git", "target", ".cargo"} | ({"tests", "benches"} if exclude_tests else set())
        out: list[Path] = []
        for dirpath, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in skip_dirs and not (Path(dirpath) / d).is_symlink())
            for f in sorted(files):
                if not f.endswith(".rs"):
                    continue
                fpath = Path(dirpath) / f
                if max_file_bytes is not None:
                    try:
                        if fpath.stat().st_size > max_file_bytes:
                            continue
                    except OSError:
                        continue
                out.append(fpath.resolve())

        def _prio(p: Path):
            rel = p.relative_to(self.root).as_posix()
            if rel.startswith("src/") or "/src/" in rel:
                return (0, rel)
            if "/" not in rel:
                return (1, rel)
            return (2, rel)

        out.sort(key=_prio)
        return out

    def get_parsed_file(self, fpath: Path) -> ParsedRustFile | None:
        resolved = fpath.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(self.root):
            return None
        try:
            st = resolved.stat()
            cached = self._file_cache.get(resolved)
            if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
                return cached[2]
            source = resolved.read_text(encoding="utf-8")
            rel = resolved.relative_to(self.root).as_posix()
            parsed = parse_rust_file(source, rel)
            self._file_cache[resolved] = (st.st_mtime_ns, st.st_size, parsed)
            return parsed
        except (OSError, UnicodeError):
            return None

    def _all_parsed(self, exclude_tests: bool = True) -> list[ParsedRustFile]:
        files = self._collect_rs_files(exclude_tests=exclude_tests)
        res = []
        for f in files:
            pf = self.get_parsed_file(f)
            if pf is not None:
                res.append(pf)
        return res

    def _public_paths_for_item(self, item: dict[str, Any], all_files: list[ParsedRustFile]) -> list[str]:
        crate_name = self.crate_for_path(item["path"])
        name = item["name"].removeprefix("r#")
        mod_path = item.get("module_path", "crate")
        paths: list[str] = []
        if item.get("vis") == "pub":
            direct = mod_path.replace("crate", crate_name, 1) if crate_name else mod_path
            paths.append(f"{direct}::{name}")
        for pf in all_files:
            pf_crate = self.crate_for_path(pf.rel_path)
            for rx in pf.reexports:
                if rx.get("vis") != "pub":
                    continue
                target = rx.get("target", "")
                exp_name = rx.get("exported_name", "")
                if exp_name == name or target.endswith(f"::{name}") or target == name:
                    rx_mod = rx.get("module_path", "crate").replace("crate", pf_crate, 1)
                    candidate = f"{rx_mod}::{exp_name}" if exp_name != "*" else target
                    if candidate not in paths:
                        paths.append(candidate)
        return paths[:6]

    @staticmethod
    def _item_tokens(it: dict[str, Any]) -> set[str]:
        refs = set(it.get("type_refs") or ())
        for field in ("enclosing", "context"):
            txt = it.get(field) or ""
            if txt:
                refs.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", txt))
        return refs

    def _resolve_callee_item(
        self,
        sem: Any,
        cname: str,
        caller_item: dict[str, Any],
        fn_defs: dict[str, list[dict[str, Any]]],
        by_loc: dict[tuple[str, int, str], dict[str, Any]],
        pf_by_path: dict[str, ParsedRustFile],
    ) -> dict[str, Any] | None:
        """Resolve a callee token to a concrete fn definition via MIR -> Stage 1 -> lexical ranking."""
        cb = cname.split("::")[-1].strip().removeprefix("r#")
        if not cb:
            return None

        if sem is not None and sem.precise:
            caller_sem = next(
                (it for it in sem.items_by_file_line.get((caller_item["path"], int(caller_item["line"])), ())
                 if it.name == caller_item["name"]),
                None,
            )
            if caller_sem is not None:
                for edge in sem.callees_by_caller.get(caller_sem.key, ()):
                    if edge.callee_name == cb:
                        hit = by_loc.get((edge.callee_file, edge.callee_line, edge.callee_name))
                        if hit is not None:
                            return hit

        cands = [x for x in fn_defs.get(cb, []) if x["kind"] == "fn"]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]

        if "::" in cname:
            owner = cname.split("::")[-2]
            owned = [
                x for x in cands
                if owner in (x.get("enclosing") or "") or x.get("module_path", "").endswith(owner)
            ]
            if len(owned) == 1:
                return owned[0]
            if owned:
                cands = owned

        if sem is not None and sem.precise:
            caller_types = self._item_tokens(caller_item)
            caller_pf = pf_by_path.get(caller_item["path"])
            caller_mod = caller_item.get("module_path", "")
            caller_sem = next(
                (it for it in sem.items_by_file_line.get((caller_item["path"], int(caller_item["line"])), ())
                 if it.name == caller_item["name"]),
                None,
            )
            caller_sem_mod = caller_sem.module_path if caller_sem else ""
            use_targets = {rx.get("target", "") for rx in (caller_pf.reexports if caller_pf else ())}

            def _sem_score(cand: dict[str, Any]) -> tuple[int, int, int, int, str]:
                sem_it = next(
                    (it for it in sem.items_by_file_line.get((cand["path"], int(cand["line"])), ())
                     if it.name == cand["name"]),
                    None,
                )
                type_hit = 1
                mod_hit = 1
                if sem_it is not None:
                    ot = (sem_it.owner_type or "").split("::")[-1]
                    tr = (sem_it.owner_trait or "").split("::")[-1]
                    if ot and ot in caller_types:
                        type_hit = 0
                    elif tr and tr in caller_types:
                        type_hit = 0
                    if caller_sem_mod and sem_it.module_path == caller_sem_mod:
                        mod_hit = 0
                    elif any(ut and sem_it.full_path.endswith(ut) for ut in use_targets):
                        mod_hit = 0
                elif caller_mod and cand.get("module_path") == caller_mod:
                    mod_hit = 0
                return (
                    type_hit,
                    mod_hit,
                    0 if cand.get("has_body") else 1,
                    0 if cand["path"] == caller_item["path"] else 1,
                    cand["path"],
                )

            cands.sort(key=_sem_score)
            return cands[0]

        cands.sort(key=lambda x: (0 if x.get("has_body") else 1, 0 if x["path"] == caller_item["path"] else 1, x["path"]))
        return cands[0]

    def query_neighborhood(
        self,
        target: str,
        kind: str = "",
        hops: int = 1,
        exclude_tests: bool = True,
        max_neighbors: int = 8,
        _focus_item: tuple[str, int, str] | None = None,
        _allow_semantic_subprocess: bool = True,
    ) -> dict[str, Any]:
        """Inspect a symbol (foo, Type::method) or file (src/lib.rs, src/lib.rs:42, src/lib.rs:symbol)."""
        clean = target.strip()
        if not clean:
            return {"error": "Provide a symbol name, Type::method, or file[:line]"}
        hops = max(1, min(3, hops))
        all_files = self._all_parsed(exclude_tests=exclude_tests)
        pf_by_path = {pf.rel_path: pf for pf in all_files}

        file_part = clean
        line_num: int | None = None
        symbol_name: str | None = None
        if ":" in clean and not clean.startswith("::"):
            fp, lp = clean.rsplit(":", 1)
            if lp.isdigit():
                file_part = fp.strip()
                line_num = int(lp)
            elif fp.strip().endswith(".rs") and lp.strip():
                file_part = fp.strip()
                symbol_name = lp.strip()

        if file_part.endswith(".rs"):
            fpath = (self.root / file_part).resolve()
            pf = self.get_parsed_file(fpath)
            if pf is None:
                return {"error": f"Rust file not found: {file_part}"}
            items = [it for it in pf.items if not (exclude_tests and it.get("is_test"))]
            if line_num is not None:
                enclosing_items = [it for it in items if it["line"] <= line_num <= it.get("end_line", it["line"])]
                enclosing_items.sort(key=lambda x: (x.get("end_line", x["line"]) - x["line"], -x["line"]))
                focus_items = enclosing_items[:3]
            elif symbol_name is not None:
                wanted = symbol_name.removeprefix("r#")
                wanted_base = wanted.split("::")[-1]
                wanted_owner = wanted.split("::")[-2] if "::" in wanted else ""
                focus_items = [
                    it for it in items
                    if it.get("name", "").removeprefix("r#") in (wanted, wanted_base)
                    and (not wanted_owner or wanted_owner in (it.get("enclosing") or ""))
                ]
                if not focus_items:
                    other_hits = [
                        f"{opf.rel_path}:{it['line']}"
                        for opf in all_files
                        if opf.rel_path != pf.rel_path
                        for it in opf.items
                        if not (exclude_tests and it.get("is_test"))
                        and it.get("name", "").removeprefix("r#") == wanted_base
                    ][:5]
                    err_res: dict[str, Any] = {
                        "error": f"Symbol not found in {file_part}: {symbol_name}",
                        "target": clean,
                        "available_symbols": [it.get("name", "") for it in items[:40]],
                    }
                    if other_hits:
                        err_res["matching_locations"] = other_hits
                    return err_res
            else:
                focus_items = items[:25]
            focus_names = [x.get("name", "") for x in focus_items if x.get("name")]
            sem = self.ensure_semantic(
                target_symbols=focus_names,
                target_files=[pf.rel_path],
                need_mir=(line_num is not None or symbol_name is not None),
                allow_subprocess=_allow_semantic_subprocess,
            )
            result: dict[str, Any] = {
                "target": clean,
                "mode": "file_line" if line_num is not None else ("file_symbol" if symbol_name else "file"),
                "precision": sem.mode if (sem is not None and sem.precise) else "lexical",
                "module_path": pf.module_path,
                "crate": self.crate_for_path(pf.rel_path),
                "items": focus_items,
                "reexports": pf.reexports[:15],
                "impl_blocks": [ib for ib in pf.impl_blocks if not (exclude_tests and ib.get("is_test"))][:15],
            }
            if (line_num is not None or symbol_name is not None) and focus_items:
                fn_item = next((x for x in focus_items if x["kind"] == "fn"), None)
                if fn_item:
                    nb = self.query_neighborhood(
                        fn_item["name"],
                        kind="fn",
                        hops=hops,
                        exclude_tests=exclude_tests,
                        max_neighbors=max_neighbors,
                        _focus_item=(fn_item["path"], int(fn_item["line"]), fn_item["name"]),
                        _allow_semantic_subprocess=_allow_semantic_subprocess,
                    )
                    result["callers"] = nb.get("callers", [])
                    result["callees"] = nb.get("callees", [])
                    result["precision"] = nb.get("precision", result["precision"])
            return result

        owner_filter = None
        sym_name = clean
        if "::" in clean:
            parts = [p.strip() for p in clean.split("::") if p.strip()]
            if len(parts) >= 2:
                owner_filter = parts[-2]
                sym_name = parts[-1]

        clean_sym = sym_name.removeprefix("r#")
        sym_files = list(dict.fromkeys(
            it["path"] for pf in all_files for it in pf.items
            if not (exclude_tests and it.get("is_test"))
            and it["name"].removeprefix("r#") == clean_sym
        ))[:4]
        sem = self.ensure_semantic(
            target_symbols=[clean, clean_sym],
            target_files=[_focus_item[0]] if _focus_item else sym_files,
            need_mir=(kind in ("", "fn")),
            allow_subprocess=_allow_semantic_subprocess,
        )
        sem_kind = "function" if kind == "fn" else kind
        sem_matches = sem.items_for_symbol(clean, kind=sem_kind) if (sem is not None and sem.precise) else []
        if _focus_item is not None and sem_matches:
            f_path, f_line, f_name = _focus_item
            exact_sem = [it for it in sem_matches if it.file == f_path and it.line == f_line and it.name == f_name]
            if exact_sem:
                sem_matches = exact_sem

        fn_defs: dict[str, list[dict[str, Any]]] = {}
        by_loc: dict[tuple[str, int, str], dict[str, Any]] = {}
        matched_defs: list[dict[str, Any]] = []
        sem_match_locs = {(it.file, it.line, it.name): it for it in sem_matches}

        for pf in all_files:
            for it in pf.items:
                if exclude_tests and it.get("is_test"):
                    continue
                if it["kind"] == "fn":
                    fn_defs.setdefault(it["name"], []).append(it)
                    by_loc[(it["path"], int(it["line"]), it["name"])] = it
                if kind and it["kind"] != kind:
                    continue
                if _focus_item is not None:
                    if (it["path"], int(it["line"]), it["name"]) != _focus_item:
                        continue
                    matched_defs.append(dict(it))
                    continue
                if it["name"].removeprefix("r#") == clean_sym:
                    if owner_filter and owner_filter not in (it.get("enclosing") or "") and owner_filter not in it.get("module_path", ""):
                        if (it["path"], int(it["line"]), it["name"]) not in sem_match_locs:
                            continue
                    matched_defs.append(dict(it))

        if _focus_item is None and "::" in clean and sem_match_locs:
            filtered = [d for d in matched_defs if (d["path"], int(d["line"]), d["name"]) in sem_match_locs]
            if filtered:
                matched_defs = filtered

        matched_defs.sort(key=lambda x: (0 if x.get("vis") == "pub" else 1, 0 if x.get("has_body") else 1, x["path"], x["line"]))
        for enriched in matched_defs[:5]:
            enriched["public_paths"] = self._public_paths_for_item(enriched, all_files)
            if sem is not None and sem.precise:
                sem_it = next(
                    (it for it in sem.items_by_file_line.get((enriched["path"], int(enriched["line"])), ())
                     if it.name == enriched["name"]),
                    None,
                )
                if sem_it is not None:
                    enriched["full_path"] = sem_it.full_path
                    if sem_it.owner_trait:
                        enriched["owner_trait"] = sem_it.owner_trait
                    if sem_it.owner_type:
                        enriched["owner_type"] = sem_it.owner_type

        trait_impls: list[dict[str, Any]] = []
        used_semantic_impls = False
        is_ambiguous_bare = False
        if sem is not None and sem.precise and sem_matches:
            sem_traits: list[str] = []
            sem_types: list[str] = []
            for it in sem_matches:
                if it.kind == "trait" and it.full_path not in sem_traits:
                    sem_traits.append(it.full_path)
                if it.kind in ("struct", "enum", "type", "union") and it.full_path not in sem_types:
                    sem_types.append(it.full_path)
                if it.owner_trait and it.owner_trait not in sem_traits and "::" in clean:
                    sem_traits.append(it.owner_trait)
                if it.owner_type and it.owner_type not in sem_types and "::" in clean:
                    sem_types.append(it.owner_type)
            if sem_traits or sem_types:
                used_semantic_impls = True
                is_ambiguous_bare = len(sem_traits) > 1 and "::" not in clean and _focus_item is None
                seen_imp: set[str] = set()
                seen_loc_tr: set[tuple[str, int, str]] = set()
                for imp in sem.dependents_of_trait_any(sem_traits) + sem.impls_for_type_any(sem_types):
                    if imp.key in seen_imp:
                        continue
                    seen_imp.add(imp.key)
                    t_short = imp.trait_name or ""
                    seen_loc_tr.add((imp.file, imp.line, t_short))
                    trait_impls.append({
                        "path": imp.file,
                        "line": imp.line,
                        "end_line": imp.end_line,
                        "header": imp.context,
                        "trait": t_short or None,
                        "trait_path": imp.trait_full_path,
                        "for_type": imp.type_name or None,
                        "for_type_path": imp.type_full_path,
                        "context": imp.context,
                        "precision": "compiler_ambiguous" if is_ambiguous_bare else "compiler",
                    })
                type_locs = {(it.file, it.line, it.name) for it in sem_matches if it.kind in ("struct", "enum")}
                for pf in all_files:
                    for ib in pf.impl_blocks:
                        if exclude_tests and ib.get("is_test"):
                            continue
                        if str(ib.get("header") or "").startswith("#[derive("):
                            if (ib.get("path"), int(ib.get("line") or 0), ib.get("for_type")) in type_locs:
                                key_lt = (str(ib.get("path")), int(ib.get("line") or 0), str(ib.get("trait") or ""))
                                if key_lt not in seen_loc_tr:
                                    seen_loc_tr.add(key_lt)
                                    trait_impls.append(ib)

        if not used_semantic_impls:
            for pf in all_files:
                for ib in pf.impl_blocks:
                    if exclude_tests and ib.get("is_test"):
                        continue
                    if ib.get("for_type") == sym_name or ib.get("trait") == sym_name or (owner_filter and ib.get("for_type") == owner_filter):
                        trait_impls.append(ib)

        callers: list[dict[str, Any]] = []
        seen_callers: set[str] = set()
        mir_pkgs = getattr(sem, "mir_packages_ok", set()) if (sem is not None and sem.precise) else set()
        if sem is not None and sem.precise and sem_matches and mir_pkgs:
            mir_frontier = [it.key for it in sem_matches]
            for hop in range(1, hops + 1):
                next_mir: list[str] = []
                for ckey in mir_frontier:
                    for edge in sem.callers_by_callee.get(ckey, ()):
                        cid = f"{edge.caller_file}:{edge.caller_line}:{edge.caller_name}"
                        if cid not in seen_callers and len(callers) < max_neighbors:
                            seen_callers.add(cid)
                            lex_caller = by_loc.get((edge.caller_file, edge.caller_line, edge.caller_name))
                            callers.append({
                                "hop": hop,
                                "caller": edge.caller_name,
                                "path": edge.caller_file,
                                "line": edge.caller_line,
                                "enclosing": lex_caller.get("enclosing") if lex_caller else None,
                                "full_path": edge.caller_full_path,
                                "precision": "compiler_mir",
                            })
                            next_mir.append(edge.caller_key)
                mir_frontier = next_mir
                if not mir_frontier:
                    break

        all_pkgs_mir = bool(sem is not None and sem.precise and sem.packages and not (set(sem.packages) - mir_pkgs))
        if not all_pkgs_mir:
            frontier = {sym_name}
            allowed_owners = {
                (it.owner_type or "").split("::")[-1]
                for it in sem_matches if it.owner_type
            }
            if owner_filter:
                allowed_owners.add(owner_filter)
            for hop in range(1, hops + 1):
                next_frontier: set[str] = set()
                for pf in all_files:
                    pf_pkg = sem.crate_to_pkg.get(self.crate_for_path(pf.rel_path)) if sem is not None else None
                    if pf_pkg and pf_pkg in mir_pkgs and sem_matches:
                        continue
                    for ce in pf.call_edges:
                        if exclude_tests and ce.get("is_test"):
                            continue
                        callee_str = ce["callee"]
                        callee_base = callee_str.split("::")[-1]
                        if callee_str in frontier or callee_base in frontier:
                            if "::" in callee_str and allowed_owners:
                                ce_owner = callee_str.split("::")[-2]
                                if ce_owner not in allowed_owners and ce_owner not in {it.module_path.split("::")[-1] for it in sem_matches}:
                                    continue
                            cid = ce["caller_id"]
                            if cid not in seen_callers and len(callers) < max_neighbors:
                                seen_callers.add(cid)
                                callers.append({
                                    "hop": hop,
                                    "caller": ce["caller"],
                                    "path": ce["caller_path"],
                                    "line": ce["caller_line"],
                                    "enclosing": ce.get("caller_enclosing"),
                                })
                                next_frontier.add(ce["caller"])
                frontier = next_frontier
                if not frontier:
                    break

        callees: list[dict[str, Any]] = []
        seen_callee_names: set[str] = set()
        for d in sorted(matched_defs, key=lambda x: (0 if x.get("has_body") else 1, x["path"]))[:5]:
            if sem is not None and sem.precise and mir_pkgs:
                caller_sem = next(
                    (it for it in sem.items_by_file_line.get((d["path"], int(d["line"])), ())
                     if it.name == d["name"]),
                    None,
                )
                if caller_sem is not None:
                    for edge in sem.callees_by_caller.get(caller_sem.key, ())[:max_neighbors]:
                        if edge.callee_full_path not in seen_callee_names:
                            seen_callee_names.add(edge.callee_full_path)
                            seen_callee_names.add(edge.callee_name)
                            callees.append({
                                "name": edge.callee_expr or edge.callee_full_path,
                                "defined_at": [f"{edge.callee_file}:{edge.callee_line}"],
                                "full_path": edge.callee_full_path,
                                "precision": "compiler_mir",
                            })
            for cname in d.get("callees", [])[:max_neighbors]:
                if cname not in seen_callee_names and cname.split("::")[-1] not in seen_callee_names:
                    seen_callee_names.add(cname)
                    resolved = self._resolve_callee_item(sem, cname, d, fn_defs, by_loc, pf_by_path)
                    c_defs = [f"{resolved['path']}:{resolved['line']}"] if resolved is not None else []
                    callees.append({"name": cname, "defined_at": c_defs})

        if sem is None or not sem.precise:
            precision = "lexical"
        elif is_ambiguous_bare or (len({it.full_path for it in sem_matches}) > 1 and "::" not in clean and _focus_item is None):
            precision = "compiler_ambiguous"
        else:
            precision = sem.mode

        return {
            "target": clean,
            "mode": "symbol",
            "precision": precision,
            "definitions": matched_defs[:5],
            "trait_impls": trait_impls[:6],
            "callers": callers,
            "callees": callees[:max_neighbors],
        }

    def build_fault_chains(
        self,
        seed_symbols: list[str],
        seed_files: list[str] | None = None,
        max_chains: int = 3,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Build 1..max_chains distinct candidate fault propagation paths across call/trait/re-export edges."""
        all_files = self._all_parsed(exclude_tests=True)
        pf_by_path = {pf.rel_path: pf for pf in all_files}
        fn_defs: dict[str, list[dict[str, Any]]] = {}
        by_loc: dict[tuple[str, int, str], dict[str, Any]] = {}
        for pf in all_files:
            for it in pf.items:
                if not it.get("is_test"):
                    fn_defs.setdefault(it["name"], []).append(it)
                    if it.get("kind") == "fn":
                        by_loc[(it["path"], int(it["line"]), it["name"])] = it

        chain_files = list(dict.fromkeys(
            list(seed_files or ()) + [
                it["path"]
                for sym in seed_symbols
                for it in fn_defs.get(sym.split("::")[-1].strip().removeprefix("r#"), ())[:2]
            ]
        ))[:4]
        sem = self.ensure_semantic(target_symbols=seed_symbols, target_files=chain_files, need_mir=True)
        chains: list[dict[str, Any]] = []
        seen_roots: set[tuple[str, int, str]] = set()

        for sym in seed_symbols:
            clean = sym.split("::")[-1].strip().removeprefix("r#")
            root_cands = sorted(fn_defs.get(clean, []), key=lambda x: (0 if x.get("has_body") else 1, x["path"], x["line"]))
            if sem is not None and sem.precise and "::" in sym:
                sem_roots = sem.items_for_symbol(sym)
                sem_locs = {(it.file, it.line, it.name) for it in sem_roots}
                narrowed = [x for x in root_cands if (x["path"], int(x["line"]), x["name"]) in sem_locs]
                if narrowed:
                    root_cands = narrowed
            for root_item in root_cands[:2]:
                rkey = (root_item["path"], int(root_item["line"]), root_item["name"])
                if rkey in seen_roots:
                    continue
                seen_roots.add(rkey)
                path_nodes = [{
                    "path": root_item["path"],
                    "line": root_item["line"],
                    "end_line": root_item.get("end_line", root_item["line"]),
                    "symbol": root_item["name"],
                    "kind": root_item["kind"],
                    "vis": root_item["vis"],
                    "enclosing": root_item.get("enclosing"),
                    "role": "entry_or_seed",
                }]
                cur_item = root_item
                visited_locs: set[tuple[str, int, str]] = {rkey}
                for depth in range(max_depth):
                    next_node = None
                    if sem is not None and sem.precise and sem.callees_by_caller:
                        cur_sem = next(
                            (it for it in sem.items_by_file_line.get((cur_item["path"], int(cur_item["line"])), ())
                             if it.name == cur_item["name"]),
                            None,
                        )
                        if cur_sem is not None:
                            for edge in sem.callees_by_caller.get(cur_sem.key, ()):
                                nkey = (edge.callee_file, edge.callee_line, edge.callee_name)
                                if nkey in visited_locs:
                                    continue
                                hit = by_loc.get(nkey)
                                if hit is not None:
                                    visited_locs.add(nkey)
                                    next_node = hit
                                    break
                    if next_node is None:
                        for cname in cur_item.get("callees", ()):
                            chosen = self._resolve_callee_item(sem, cname, cur_item, fn_defs, by_loc, pf_by_path)
                            if chosen is None:
                                continue
                            ckey = (chosen["path"], int(chosen["line"]), chosen["name"])
                            if ckey in visited_locs:
                                continue
                            visited_locs.add(ckey)
                            next_node = chosen
                            break
                    if not next_node:
                        break
                    path_nodes.append({
                        "path": next_node["path"],
                        "line": next_node["line"],
                        "end_line": next_node.get("end_line", next_node["line"]),
                        "symbol": next_node["name"],
                        "kind": next_node["kind"],
                        "vis": next_node["vis"],
                        "enclosing": next_node.get("enclosing"),
                        "role": "callee" if depth < max_depth - 1 else "inner_candidate",
                    })
                    cur_item = next_node

                chains.append({
                    "chain_id": f"C{len(chains) + 1}",
                    "seed": sym,
                    "nodes": path_nodes,
                })
                if len(chains) >= max_chains:
                    return chains

        for sf in seed_files or []:
            if len(chains) >= max_chains:
                break
            pf = self.get_parsed_file((self.root / sf).resolve())
            if pf is None:
                continue
            top_fns = [it for it in pf.items if it["kind"] == "fn" and not it.get("is_test")][:3]
            if top_fns:
                chains.append({
                    "chain_id": f"C{len(chains) + 1}",
                    "seed": sf,
                    "nodes": [{
                        "path": it["path"],
                        "line": it["line"],
                        "end_line": it.get("end_line", it["line"]),
                        "symbol": it["name"],
                        "kind": it["kind"],
                        "vis": it["vis"],
                        "enclosing": it.get("enclosing"),
                        "role": "file_candidate",
                    } for it in top_fns],
                })
        return chains[:max_chains]

    def _lexical_fn_slice(
        self,
        fn_item: dict[str, Any] | None,
        line_num: int | None,
        var: str,
        direction: str,
        reason: str = "",
        max_lines: int = 25,
    ) -> dict[str, Any]:
        """Lightweight lexical def-use fallback inside an enclosing function when MIR is unavailable."""
        if fn_item is None:
            return {"direction": direction, "var": var or None, "precision": "lexical", "slice_lines": [], "reason": reason}
        rel = fn_item["path"]
        s_line = int(fn_item.get("line") or 1)
        e_line = int(fn_item.get("end_line") or s_line)
        try:
            lines = (self.root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        seed_ln = line_num if (line_num and s_line <= line_num <= e_line) else s_line
        kw_skip = _SKIP_CALL_NAMES | {"let", "mut", "if", "else", "match", "return", "fn", "pub"}
        seed_vars = {var.split(".")[0], var.split(".")[-1]} - {""} if var else set()
        if not seed_vars and 1 <= seed_ln <= len(lines):
            seed_vars.update(
                tok for tok in re.findall(r"\b[a-z_][A-Za-z0-9_]*\b", lines[seed_ln - 1])
                if tok not in kw_skip
            )
        if not seed_vars and e_line >= s_line and lines:
            for ln in range(min(e_line, len(lines)), s_line - 1, -1):
                toks = [tok for tok in re.findall(r"\b[a-z_][A-Za-z0-9_]*\b", lines[ln - 1]) if tok not in kw_skip]
                if toks:
                    seed_vars.update(toks[:3])
                    break
        out_lines: list[dict[str, Any]] = []
        rng = range(s_line, min(e_line, len(lines)) + 1)
        for ln in rng:
            txt = lines[ln - 1].strip()
            toks = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", txt))
            if ln == seed_ln or (seed_vars & toks):
                out_lines.append({
                    "path": rel,
                    "line": ln,
                    "function": fn_item["name"],
                    "vars": sorted(seed_vars & toks)[:6],
                    "hop": 0,
                    "code": txt[:120],
                })
                if len(out_lines) >= max_lines:
                    break
        res = {
            "direction": direction,
            "var": var or None,
            "precision": "lexical",
            "root_function": fn_item["name"],
            "slice_lines": out_lines,
            "cross_function_hops": [],
        }
        if reason:
            res["reason"] = reason
        return res

    def slice_dataflow(
        self,
        target: str,
        direction: str = "backward",
        var: str = "",
        hops: int = 1,
        exclude_tests: bool = True,
        max_lines: int = 25,
        allow_semantic_subprocess: bool = True,
    ) -> dict[str, Any]:
        """Stage 3: Compute statement-level MIR def-use data-flow slice for a symbol or file[:line]."""
        clean = (target or "").strip()
        dir_norm = (direction or "backward").strip().lower()
        if dir_norm not in ("backward", "forward", "both"):
            dir_norm = "backward"
        if not clean:
            return {"error": "Provide a symbol name or file[:line] for dataflow slicing"}

        file_part = clean
        line_num: int | None = None
        sym_in_file: str | None = None
        if ":" in clean and not clean.startswith("::"):
            fp, lp = clean.rsplit(":", 1)
            if lp.isdigit():
                file_part = fp.strip()
                line_num = int(lp)
            elif fp.strip().endswith(".rs") and lp.strip():
                file_part = fp.strip()
                sym_in_file = lp.strip()

        all_files = self._all_parsed(exclude_tests=exclude_tests)
        fn_item: dict[str, Any] | None = None
        target_files: list[str] = []
        target_syms: list[str] = []

        if file_part.endswith(".rs"):
            pf = self.get_parsed_file((self.root / file_part).resolve())
            if pf is None:
                return {"error": f"Rust file not found: {file_part}"}
            target_files.append(pf.rel_path)
            items = [it for it in pf.items if not (exclude_tests and it.get("is_test")) and it.get("kind") == "fn"]
            if line_num is not None:
                cands = [it for it in items if it["line"] <= line_num <= it.get("end_line", it["line"])]
                cands.sort(key=lambda x: (x.get("end_line", x["line"]) - x["line"], -x["line"]))
                fn_item = cands[0] if cands else None
            elif sym_in_file:
                bare = sym_in_file.split("::")[-1].removeprefix("r#")
                owner = sym_in_file.split("::")[-2] if "::" in sym_in_file else ""
                cands = [
                    it for it in items
                    if it["name"].removeprefix("r#") == bare and (not owner or owner in (it.get("enclosing") or ""))
                ]
                fn_item = cands[0] if cands else None
            if fn_item:
                target_syms.append(fn_item["name"])
        else:
            bare = clean.split("::")[-1].removeprefix("r#")
            owner = clean.split("::")[-2] if "::" in clean else ""
            target_syms.append(clean)
            cands = [
                it for pf in all_files for it in pf.items
                if not (exclude_tests and it.get("is_test")) and it.get("kind") == "fn"
                and it["name"].removeprefix("r#") == bare
                and (not owner or owner in (it.get("enclosing") or "") or owner in it.get("module_path", ""))
            ]
            cands.sort(key=lambda x: (0 if x.get("vis") == "pub" else 1, 0 if x.get("has_body") else 1, x["path"], x["line"]))
            fn_item = cands[0] if cands else None
            target_files = list(dict.fromkeys(c["path"] for c in cands[:4]))

        force_pkgs: list[str] = []
        for tf in target_files:
            pkg = self._package_name_for_path(tf)
            if pkg and pkg not in force_pkgs:
                force_pkgs.append(pkg)
        sem = self.ensure_semantic(
            target_symbols=target_syms,
            target_files=target_files,
            need_mir=True,
            force_mir_packages=force_pkgs or None,
            allow_subprocess=allow_semantic_subprocess,
        )
        sem_mod = _load_semantics_module()
        if sem is not None and sem_mod is not None and sem.mir_functions:
            sem_items: list[Any] = []
            if fn_item is not None:
                sem_items = [
                    it for it in sem.items_by_file_line.get((fn_item["path"], int(fn_item["line"])), ())
                    if it.name == fn_item["name"]
                ]
            if not sem_items and target_syms:
                sem_items = sem.items_for_symbol(target_syms[0], kind="function")
            if not sem_items and file_part.endswith(".rs") and line_num is not None:
                sem_items = [
                    it for it in sem.items.values()
                    if it.file == file_part and it.line <= line_num <= it.end_line
                ]
            if not sem_items and fn_item is not None:
                sem_items = [
                    sem_mod.SemanticItem(
                        key=f"mir:{fn_item['path']}:{fn_item['line']}:{fn_item['name']}",
                        doc_id="",
                        name=fn_item["name"],
                        kind="function",
                        full_path=fn_item["name"],
                        module_path="",
                        crate="",
                        file=fn_item["path"],
                        line=int(fn_item["line"]),
                        end_line=int(fn_item.get("end_line") or fn_item["line"]),
                        col=0,
                    )
                ]
            if sem_items:
                mir_res = sem_mod.slice_mir_dataflow(
                    sem,
                    sem_items,
                    direction=dir_norm,
                    var=var,
                    target_line=line_num,
                    hops=hops,
                    max_lines=max_lines,
                )
                if mir_res.get("precision") == "compiler_mir_slice":
                    return mir_res
                return self._lexical_fn_slice(fn_item, line_num, var, dir_norm, reason=mir_res.get("reason", ""), max_lines=max_lines)

        reason = f"MIR unavailable ({self._last_semantic_error})" if self._last_semantic_error else "MIR unavailable"
        return self._lexical_fn_slice(fn_item, line_num, var, dir_norm, reason=reason, max_lines=max_lines)


def _auto_slice_direction(res: dict[str, Any], target: str, kind: str) -> str | None:
    """Decide whether an omitted/`auto` slice_mode should automatically run Stage 3 data-flow slicing."""
    if not res or res.get("error"):
        return None
    mode = res.get("mode")
    if mode == "file_line":
        if any(it.get("kind") == "fn" and it.get("has_body", True) for it in res.get("items", ())):
            return "both"
        return None
    if mode == "file_symbol":
        if any(it.get("kind") == "fn" and it.get("has_body", True) for it in res.get("items", ())):
            return "backward"
        return None
    if mode == "symbol":
        fn_defs = [d for d in res.get("definitions", ()) if d.get("kind") == "fn" and d.get("has_body", True)]
        if fn_defs and (len(fn_defs) == 1 or "::" in target or kind == "fn"):
            d0 = fn_defs[0]
            if int(d0.get("end_line", 0)) > int(d0.get("line", 0)):
                return "backward"
    return None


def make_rust_ckg_tool(root: Path, graph: RustCodeGraph | None = None, env: dict[str, str] | None = None):
    ckg = graph or RustCodeGraph(root, env=env)

    @tool
    def inspect_rust_neighborhood(
        target: str,
        kind: str = "",
        hops: int = 1,
        exclude_tests: bool = True,
        slice_mode: str = "auto",
        var: str = "",
    ) -> str:
        """Inspect a Rust symbol, Type::method, or file[:line] via the Rust Code Knowledge Graph (CKG).

        Target forms: "foo", "Type::method", "src/lib.rs", "src/lib.rs:42" (line),
        "src/lib.rs:method" or "src/lib.rs:Type::method" (symbol inside that file).

        Stage 3 data-flow slicing: `slice_mode="auto"` (default) automatically extracts statement-level
        def-use lines via MIR (or lexical fallback) when targeting a function or `file:line`.
        Pass `slice_mode="backward" | "forward" | "both"` (and optional `var="name"`) to specify direction.

        Returns definitions (with visibility and exact line spans), public pub use import paths,
        related impl Trait / #[derive] blocks, callers, callees, and dataflow_slice when applicable.
        Use this instead of chaining multiple search_code/rust_symbols calls before planning edits or tests.
        """
        try:
            res = ckg.query_neighborhood(target=target, kind=kind, hops=hops, exclude_tests=exclude_tests)
            sm = (slice_mode or "auto").strip().lower()
            explicit = sm in ("backward", "forward", "both")
            if not res.get("error"):
                slice_dir = sm if explicit else _auto_slice_direction(res, target, kind)
                if slice_dir:
                    sl = ckg.slice_dataflow(
                        target=target,
                        direction=slice_dir,
                        var=var.strip(),
                        hops=hops,
                        exclude_tests=exclude_tests,
                        max_lines=25 if explicit else 18,
                    )
                    if sl.get("slice_lines") or explicit:
                        res["dataflow_slice"] = sl
                    if sl.get("precision") == "compiler_mir_slice":
                        res["precision"] = "partial" if ckg.semantic.mode == "lexical" else ckg.semantic.mode
            return json.dumps(res, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    return inspect_rust_neighborhood, ckg
