"""Lightweight, zero-dependency Rust Code Knowledge Graph (CKG) and token-aware structural parser.

Supports:
- Rust lexical tokenization (comments, nested block comments, raw/byte/c strings, lifetimes vs chars)
- Structural extraction (mod, pub use re-exports, struct/enum/trait/type/impl/fn/macro)
- Accurate function body spans & #[test]/#[cfg(test)] separation (also used by rust_trace)
- Incremental per-file cache keyed by (mtime_ns, size)
- Neighborhood queries (definitions, public re-export paths, trait impls, callers, callees)
- Multi-chain candidate fault propagation paths along call/trait/re-export edges
"""

import bisect
import json
import os
import re
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


class RustCodeGraph:
    """Incremental repository-level Rust Code Knowledge Graph (CKG)."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._file_cache: dict[Path, tuple[int, int, ParsedRustFile]] = {}
        self._pkg_roots: list[tuple[Path, str]] | None = None

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

    def query_neighborhood(
        self,
        target: str,
        kind: str = "",
        hops: int = 1,
        exclude_tests: bool = True,
        max_neighbors: int = 12,
    ) -> dict[str, Any]:
        """Inspect a symbol (foo, Type::method) or file (src/lib.rs, src/lib.rs:42, src/lib.rs:symbol)."""
        clean = target.strip()
        if not clean:
            return {"error": "Provide a symbol name, Type::method, or file[:line]"}
        hops = max(1, min(3, hops))
        all_files = self._all_parsed(exclude_tests=exclude_tests)

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
                    return {
                        "error": f"Symbol not found in {file_part}: {symbol_name}",
                        "target": clean,
                        "available_symbols": [it.get("name", "") for it in items[:40]],
                    }
            else:
                focus_items = items[:25]
            result: dict[str, Any] = {
                "target": clean,
                "mode": "file_line" if line_num is not None else ("file_symbol" if symbol_name else "file"),
                "module_path": pf.module_path,
                "crate": self.crate_for_path(pf.rel_path),
                "items": focus_items,
                "reexports": pf.reexports[:15],
                "impl_blocks": [ib for ib in pf.impl_blocks if not (exclude_tests and ib.get("is_test"))][:15],
            }
            if (line_num is not None or symbol_name is not None) and focus_items:
                fn_item = next((x for x in focus_items if x["kind"] == "fn"), None)
                if fn_item:
                    nb = self.query_neighborhood(fn_item["name"], kind="fn", hops=hops, exclude_tests=exclude_tests, max_neighbors=max_neighbors)
                    result["callers"] = nb.get("callers", [])
                    result["callees"] = nb.get("callees", [])
            return result

        owner_filter = None
        sym_name = clean
        if "::" in clean:
            parts = [p.strip() for p in clean.split("::") if p.strip()]
            if len(parts) >= 2:
                owner_filter = parts[-2]
                sym_name = parts[-1]

        matched_defs: list[dict[str, Any]] = []
        clean_sym = sym_name.removeprefix("r#")
        for pf in all_files:
            for it in pf.items:
                if exclude_tests and it.get("is_test"):
                    continue
                if kind and it["kind"] != kind:
                    continue
                if it["name"].removeprefix("r#") == clean_sym:
                    if owner_filter and owner_filter not in (it.get("enclosing") or "") and owner_filter not in it.get("module_path", ""):
                        continue
                    matched_defs.append(dict(it))
        matched_defs.sort(key=lambda x: (0 if x.get("vis") == "pub" else 1, 0 if x.get("has_body") else 1, x["path"], x["line"]))
        for enriched in matched_defs[:8]:
            enriched["public_paths"] = self._public_paths_for_item(enriched, all_files)

        trait_impls: list[dict[str, Any]] = []
        for pf in all_files:
            for ib in pf.impl_blocks:
                if exclude_tests and ib.get("is_test"):
                    continue
                if ib.get("for_type") == sym_name or ib.get("trait") == sym_name or (owner_filter and ib.get("for_type") == owner_filter):
                    trait_impls.append(ib)

        callers: list[dict[str, Any]] = []
        seen_callers: set[str] = set()
        frontier = {sym_name}
        for hop in range(1, hops + 1):
            next_frontier: set[str] = set()
            for pf in all_files:
                for ce in pf.call_edges:
                    if exclude_tests and ce.get("is_test"):
                        continue
                    callee_base = ce["callee"].split("::")[-1]
                    if ce["callee"] in frontier or callee_base in frontier:
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
            for cname in d.get("callees", [])[:max_neighbors]:
                if cname not in seen_callee_names:
                    seen_callee_names.add(cname)
                    c_base = cname.split("::")[-1]
                    c_defs = []
                    for pf in all_files:
                        for it in pf.items:
                            if not (exclude_tests and it.get("is_test")) and it["kind"] == "fn" and it["name"] == c_base:
                                c_defs.append(f"{it['path']}:{it['line']}")
                                if len(c_defs) >= 2:
                                    break
                        if len(c_defs) >= 2:
                            break
                    callees.append({"name": cname, "defined_at": c_defs})

        return {
            "target": clean,
            "mode": "symbol",
            "definitions": matched_defs[:8],
            "trait_impls": trait_impls[:10],
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
        fn_defs: dict[str, list[dict[str, Any]]] = {}
        for pf in all_files:
            for it in pf.items:
                if not it.get("is_test"):
                    fn_defs.setdefault(it["name"], []).append(it)

        chains: list[dict[str, Any]] = []
        seen_roots: set[tuple[str, str]] = set()

        for sym in seed_symbols:
            clean = sym.split("::")[-1].strip().removeprefix("r#")
            root_cands = sorted(fn_defs.get(clean, []), key=lambda x: (0 if x.get("has_body") else 1, x["path"]))
            for root_item in root_cands[:2]:
                rkey = (root_item["path"], root_item["name"])
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
                cur_callees = list(root_item.get("callees", []))
                visited = {root_item["name"]}
                for depth in range(max_depth):
                    next_node = None
                    for cname in cur_callees:
                        cb = cname.split("::")[-1]
                        if cb in visited:
                            continue
                        cands = [x for x in fn_defs.get(cb, []) if x["kind"] == "fn"]
                        if cands:
                            cands.sort(key=lambda x: (0 if x.get("has_body") else 1, 0 if x["path"] == root_item["path"] else 1, x["path"]))
                            chosen = cands[0]
                            visited.add(cb)
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
                    cur_callees = list(next_node.get("callees", []))

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


def make_rust_ckg_tool(root: Path, graph: RustCodeGraph | None = None):
    ckg = graph or RustCodeGraph(root)

    @tool
    def inspect_rust_neighborhood(
        target: str,
        kind: str = "",
        hops: int = 1,
        exclude_tests: bool = True,
    ) -> str:
        """Inspect a Rust symbol, Type::method, or file[:line] via the Rust Code Knowledge Graph (CKG).

        Target forms: "foo", "Type::method", "src/lib.rs", "src/lib.rs:42" (line),
        "src/lib.rs:method" or "src/lib.rs:Type::method" (symbol inside that file).

        Returns definitions (with visibility and exact line spans), public pub use import paths,
        related impl Trait / #[derive] blocks, callers, and callees in one structured call.
        Use this instead of chaining multiple search_code/rust_symbols calls before planning edits or tests.
        """
        try:
            res = ckg.query_neighborhood(target=target, kind=kind, hops=hops, exclude_tests=exclude_tests)
            return json.dumps(res, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    return inspect_rust_neighborhood, ckg
