"""Zero-side-effect, type-safe hybrid dynamic tracer for Rust functions during cargo test."""

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool

_PRIMITIVE_DEBUG_TYPES = frozenset({
    "i8", "i16", "i32", "i64", "i128", "isize",
    "u8", "u16", "u32", "u64", "u128", "usize",
    "f32", "f64", "bool", "char", "str", "String",
    "Path", "PathBuf", "OsStr", "OsString",
})
_SAFE_WATCH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\(\)]*$")
_GENERIC_WRAP_RE = re.compile(r"^(?:Option|Vec|Box|Result)\s*<\s*([^<>]+?)\s*>$")


def _load_ckg_module():
    ckg_path = Path(__file__).with_name("rust_ckg.py")
    if not ckg_path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("rust_ckg_for_trace", ckg_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _load_symbols_module():
    sym_path = Path(__file__).with_name("rust_symbols.py")
    if not sym_path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("rust_symbols_for_trace", sym_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def parse_instrument_targets(root: Path, instrument_specs: list[str], symbols_module=None) -> dict[Path, set[str]]:
    """Parse instrument specifications into {file_path: set(fn_names)}."""
    root = Path(root).resolve()
    targets: dict[Path, set[str]] = {}
    for spec in instrument_specs:
        spec = spec.strip()
        if not spec:
            continue
        if ":" in spec:
            file_part, fn_part = spec.split(":", 1)
            fpath = (root / file_part.strip()).resolve()
            if not fpath.is_file():
                continue
            fns = {f.strip() for f in fn_part.split(",") if f.strip()}
            targets.setdefault(fpath, set()).update(fns)
        elif spec.endswith(".rs"):
            fpath = (root / spec).resolve()
            if fpath.is_file():
                targets.setdefault(fpath, set())
        else:
            if symbols_module is not None:
                details = symbols_module.lookup_symbol_details(root, name=spec, kind="fn", exact=True, max_results=2)
                for d in details:
                    fpath = (root / d["path"]).resolve()
                    if fpath.is_file():
                        targets.setdefault(fpath, set()).add(spec)
    return targets


def _extract_fn_params(source: str, sig_start_hint: int, brace_pos: int) -> list[tuple[str, str]]:
    """Extract simple `(ident, type_str)` pairs from a Rust function signature ending at `brace_pos`."""
    header = source[max(0, sig_start_hint):brace_pos]
    fn_idx = header.rfind("fn ")
    if fn_idx < 0:
        return []
    after_fn = header[fn_idx + 3:]
    # Skip function name + optional generic list <...>
    i = 0
    n = len(after_fn)
    while i < n and (after_fn[i].isalnum() or after_fn[i] in "_: "):
        if after_fn[i] == "(":
            break
        i += 1
    if i < n and after_fn[i] == "<":
        ang = 0
        while i < n:
            if after_fn[i] == "<":
                ang += 1
            elif after_fn[i] == ">" and (i == 0 or after_fn[i - 1] != "-"):
                ang -= 1
                if ang == 0:
                    i += 1
                    break
            i += 1
    open_p = after_fn.find("(", i)
    if open_p < 0:
        return []
    par = 0
    close_p = -1
    for idx in range(open_p, n):
        ch = after_fn[idx]
        if ch == "(":
            par += 1
        elif ch == ")":
            par -= 1
            if par == 0:
                close_p = idx
                break
    if close_p < 0:
        return []
    param_str = after_fn[open_p + 1:close_p]
    # Split top-level commas
    parts: list[str] = []
    cur: list[str] = []
    d_par = d_ang = d_brk = d_br = 0
    for idx, ch in enumerate(param_str):
        if ch == "(": d_par += 1
        elif ch == ")": d_par = max(0, d_par - 1)
        elif ch == "[": d_brk += 1
        elif ch == "]": d_brk = max(0, d_brk - 1)
        elif ch == "{": d_br += 1
        elif ch == "}": d_br = max(0, d_br - 1)
        elif ch == "<": d_ang += 1
        elif ch == ">" and (idx == 0 or param_str[idx - 1] != "-"): d_ang = max(0, d_ang - 1)
        if ch == "," and d_par == 0 and d_ang == 0 and d_brk == 0 and d_br == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())

    params: list[tuple[str, str]] = []
    for p in parts:
        if not p or "self" in p.split(":")[0].split():
            continue
        if ":" not in p:
            continue
        lhs, rhs = p.split(":", 1)
        lhs = lhs.strip()
        if lhs.startswith("mut "):
            lhs = lhs[4:].strip()
        if re.match(r"^[A-Za-z_]\w*$", lhs) and not lhs.startswith("_"):
            params.append((lhs, " ".join(rhs.split())))
    return params


def _is_type_debug_safe(type_str: str, debug_types: set[str]) -> bool:
    """Conservatively decide if `&arg` can be formatted with `{:?}` without risking E0277 or move errors."""
    t = " ".join(type_str.split()).strip()
    if not t or "mut " in t or "dyn " in t or "impl " in t or "fn(" in t or "*" in t:
        return False
    # Strip leading immutable reference & / &'a
    if t.startswith("&"):
        t = re.sub(r"^&\s*(?:'[A-Za-z_]\w*\s+)?", "", t).strip()
    if t.startswith("[") and t.endswith("]"):
        elem = t[1:-1].split(";", 1)[0].strip()
        return _is_type_debug_safe(elem, debug_types)
    m_gen = _GENERIC_WRAP_RE.match(t)
    if m_gen:
        inner = m_gen.group(1)
        return all(_is_type_debug_safe(part.strip(), debug_types) for part in inner.split(","))
    base = t.split("::")[-1].strip()
    if base in _PRIMITIVE_DEBUG_TYPES:
        return True
    if base in debug_types and "<" not in t:
        return True
    return False


def _find_fn_open_braces(source: str, target_fns: set[str], rel_display: str) -> tuple[list[tuple[str, int, int]], set[str]]:
    """Locate function body `{` offsets and Debug-derived types via token-aware CKG parser."""
    ckg_mod = _load_ckg_module()
    debug_types: set[str] = set()
    if ckg_mod is not None:
        try:
            parsed = ckg_mod.parse_rust_file(source, rel_display)
            for ib in parsed.impl_blocks:
                if ib.get("trait") == "Debug" and ib.get("for_type") and not ib.get("is_test"):
                    debug_types.add(str(ib["for_type"]))
            matches: list[tuple[str, int, int]] = []
            for item in parsed.items:
                if item.get("kind") != "fn" or not item.get("has_body"):
                    continue
                if item.get("is_test") or item.get("is_const"):
                    continue
                fn_name = item.get("name", "")
                if target_fns and fn_name not in target_fns:
                    continue
                b_off = item.get("body_open_offset")
                if isinstance(b_off, int) and 0 <= b_off < len(source) and source[b_off] == "{":
                    matches.append((fn_name, b_off, max(0, b_off - 400)))
            if matches or not target_fns:
                matches.sort(key=lambda x: x[1])
                return matches, debug_types
        except Exception:
            pass

    fn_pat = re.compile(
        r"(?<![\w#])(?:pub(?:\s*\([^)]+\))?\s+)?(?:async\s+|unsafe\s+|extern\s+(?:\"[^\"]+\"\s+)?)*fn\s+(?P<name>[A-Za-z_]\w*)\b",
        re.UNICODE,
    )
    matches = []
    for m in fn_pat.finditer(source):
        fn_name = m.group("name")
        if target_fns and fn_name not in target_fns:
            continue
        prefix = source[max(0, m.start() - 120):m.start()]
        if re.search(r"#\[\s*(?:[\w:]+::)?test\s*\]", prefix):
            continue
        if re.search(r"\bconst\s+$", source[max(0, m.start() - 16):m.start()]):
            continue
        brace_pos = source.find("{", m.end())
        if brace_pos == -1 or brace_pos - m.end() > 2000:
            continue
        matches.append((fn_name, brace_pos, m.start()))
    return matches, debug_types


def instrument_source_for_trace(
    source: str,
    target_fns: set[str],
    rel_display: str,
    watch_vars: list[str] | None = None,
    auto_watch_args: bool = False,
    known_debug_types: set[str] | None = None,
) -> tuple[str, int]:
    """Instrument target functions with non-destructive RAII trace probes and type-safe argument sampling."""
    matches, file_debug_types = _find_fn_open_braces(source, target_fns, rel_display)
    if not matches:
        return source, 0

    debug_types = set(file_debug_types) | set(known_debug_types or ())
    safe_vars = [v.strip() for v in (watch_vars or []) if v and _SAFE_WATCH_RE.match(v.strip())][:5]

    count = 0
    result = source
    for fn_name, brace_pos, sig_hint in reversed(matches):
        insert_offset = brace_pos + 1
        sub = result[insert_offset:]
        m_inner = re.match(r"(?:\s*(?://![^\n]*|#!\[[^\]]+\]))*\s*", sub)
        if m_inner:
            insert_offset += m_inner.end()

        arg_lines = ""
        if auto_watch_args:
            params = _extract_fn_params(source, sig_hint, brace_pos)
            for pname, ptype in params[:5]:
                if _is_type_debug_safe(ptype, debug_types):
                    arg_lines += (
                        f'    {{ let __t_arg = &({pname}); '
                        f'eprintln!("[RUST_TRACE_ARG] {fn_name}::{pname}={{:?}}", __t_arg); }}\n'
                    )
                else:
                    clean_ty = ptype.replace('"', "'")[:60]
                    arg_lines += (
                        f'    eprintln!("[RUST_TRACE_ARG] {fn_name}::{pname}=<unprintable: {clean_ty}>");\n'
                    )

        var_lines = ""
        for v in safe_vars:
            var_lines += f'    eprintln!("[RUST_TRACE_VAR] {fn_name}::{v}={{:?}}", &({v}));\n'

        probe = (
            "\n    struct __TraceGuard(&'static str, &'static str, u32, std::time::Instant);\n"
            "    impl Drop for __TraceGuard {\n"
            "        fn drop(&mut self) {\n"
            "            let __panic_tag = if std::thread::panicking() { \" [PANICKING]\" } else { \"\" };\n"
            "            eprintln!(\"[RUST_TRACE_EXIT] {}:{} {} elapsed={:?}{}\", self.0, self.2, self.1, self.3.elapsed(), __panic_tag);\n"
            "        }\n"
            "    }\n"
            f"    eprintln!(\"[RUST_TRACE_ENTER] {{}}:{{}} {{}}\", file!(), line!(), stringify!({fn_name}));\n"
            f"{arg_lines}"
            f"{var_lines}"
            f"    let __trace_guard = __TraceGuard(file!(), stringify!({fn_name}), line!(), std::time::Instant::now());\n"
        )
        result = result[:insert_offset] + probe + result[insert_offset:]
        count += 1

    return result, count


def format_trace_call_tree(raw_output: str, max_depth: int = 10) -> tuple[str, int]:
    """Parse [RUST_TRACE_ENTER], [RUST_TRACE_ARG], [RUST_TRACE_VAR], and [RUST_TRACE_EXIT] into a call tree."""
    events: list[tuple[str, str, str, str]] = []
    for line in raw_output.splitlines():
        line = line.strip()
        if line.startswith("[RUST_TRACE_ENTER]"):
            parts = line.split(maxsplit=2)
            if len(parts) >= 3:
                events.append(("enter", parts[1], parts[2], ""))
        elif line.startswith("[RUST_TRACE_ARG]"):
            val_text = line[len("[RUST_TRACE_ARG]"):].strip()
            events.append(("arg", "", val_text, ""))
        elif line.startswith("[RUST_TRACE_VAR]"):
            val_text = line[len("[RUST_TRACE_VAR]"):].strip()
            events.append(("var", "", val_text, ""))
        elif line.startswith("[RUST_TRACE_EXIT]"):
            parts = line.split(maxsplit=3)
            if len(parts) >= 3:
                elapsed = parts[3] if len(parts) >= 4 else ""
                events.append(("exit", parts[1], parts[2], elapsed))

    if not events:
        return "(no instrumented functions were invoked during this test run)", 0

    stack: list[tuple[str, str]] = []
    tree_lines: list[str] = []
    call_count = 0
    panic_frames: list[str] = []

    for action, loc, fn_name, elapsed_info in events:
        if action == "enter":
            call_count += 1
            depth = len(stack)
            indent = "  " * depth + ("├─ " if depth > 0 else "")
            stack.append((fn_name, loc))
            if depth < max_depth:
                tree_lines.append(f"{indent}{fn_name} ({loc})")
        elif action in ("arg", "var"):
            depth = len(stack)
            indent = "  " * depth + "│ "
            prefix = "arg" if action == "arg" else "var"
            if depth <= max_depth:
                tree_lines.append(f"{indent}{prefix} {fn_name[:180]}")
        elif action == "exit":
            if "[PANICKING]" in (elapsed_info or ""):
                frame = f"{fn_name} ({loc})"
                if frame not in panic_frames:
                    panic_frames.append(frame)
            if stack and stack[-1][0] == fn_name:
                stack.pop()
            elif stack:
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == fn_name:
                        stack = stack[:i]
                        break

    formatted = "\n".join(tree_lines[:200])
    if len(tree_lines) > 200:
        formatted += f"\n... [{len(tree_lines) - 200} trace entries truncated] ..."
    if panic_frames:
        formatted += "\n[Panic unwind detected (innermost -> outer): " + " -> ".join(panic_frames[:8]) + "]"
    return formatted, call_count


def _collect_repo_debug_types(ckg) -> set[str]:
    if ckg is None:
        return set()
    out: set[str] = set()
    try:
        for pf in ckg._all_parsed(exclude_tests=True):
            for ib in pf.impl_blocks:
                if ib.get("trait") == "Debug" and ib.get("for_type") and not ib.get("is_test"):
                    out.add(str(ib["for_type"]))
    except Exception:
        pass
    return out


def make_rust_trace_tool(root: Path, resolve_path, env=None, ckg=None):
    sym_mod = _load_symbols_module()
    root = Path(root).resolve()

    @tool
    def rust_trace(
        instrument: list[str],
        manifest_path: str = "Cargo.toml",
        package: str = "",
        test_target: str = "",
        test_filter: str = "",
        lib: bool = False,
        features: list[str] | None = None,
        all_features: bool = False,
        no_default_features: bool = False,
        target: str = "",
        watch_vars: list[str] | None = None,
        auto_watch_args: bool = True,
        timeout: int = 120,
    ) -> str:
        """Non-destructively trace Rust function execution and type-safe argument values during `cargo test`.

        Instruments target functions in memory/on disk with zero-dependency RAII probes, captures
        the call tree and Debug-safe parameter values (`auto_watch_args=True`), runs `cargo test -- --nocapture`,
        and atomically restores all files with their original mtime_ns.
        """
        if not instrument:
            return "Error: Provide at least one target to instrument (e.g. ['src/foo.rs:bar'] or ['bar'])."

        targets = parse_instrument_targets(root, instrument, symbols_module=sym_mod)
        if not targets:
            return f"Error: None of the specified instrumentation targets could be resolved: {instrument}"

        backup: dict[Path, tuple[bytes, int, int]] = {}
        for fpath in targets:
            st = fpath.stat()
            backup[fpath] = (fpath.read_bytes(), st.st_mode, st.st_mtime_ns)

        known_debug = _collect_repo_debug_types(ckg) if auto_watch_args else set()

        def _restore():
            for fpath, (data, mode, mtime_ns) in backup.items():
                fpath.write_bytes(data)
                fpath.chmod(mode)
                try:
                    os.utime(fpath, ns=(mtime_ns, mtime_ns))
                except Exception:
                    pass

        def _execute(vars_to_watch, sample_args: bool):
            instrumented_count = 0
            for fpath, fns in targets.items():
                content = backup[fpath][0].decode("utf-8", errors="replace")
                rel_display = fpath.relative_to(root).as_posix()
                new_content, n_mod = instrument_source_for_trace(
                    content, fns, rel_display,
                    watch_vars=vars_to_watch,
                    auto_watch_args=sample_args,
                    known_debug_types=known_debug,
                )
                if n_mod > 0:
                    fpath.write_bytes(new_content.encode("utf-8"))
                    instrumented_count += n_mod

            if instrumented_count == 0:
                return None, 0, "Error: Could not locate function definitions matching the specified instrument list."

            manifest = resolve_path(manifest_path)
            cmd = ["cargo", "test", "--manifest-path", str(manifest)]
            if (root / "Cargo.lock").is_file() or (manifest.parent / "Cargo.lock").is_file():
                cmd.append("--locked")
            if package:
                cmd.append(f"--package={package}")
            if test_target:
                cmd.append(f"--test={test_target}")
                cmd.append("--no-fail-fast")
            if lib:
                cmd.append("--lib")
            if features:
                cmd.append("--features=" + ",".join(features))
            if all_features:
                cmd.append("--all-features")
            if no_default_features:
                cmd.append("--no-default-features")
            if target:
                cmd.append(f"--target={target}")
            if test_filter:
                cmd.append(test_filter)
            cmd.extend(["--", "--nocapture"])

            test_env = dict(env) if env is not None else os.environ.copy()
            test_env.setdefault("RUST_BACKTRACE", "1")
            res = subprocess.run(
                cmd, cwd=root, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout, env=test_env,
            )
            return res, instrumented_count, None

        try:
            try:
                res, instrumented_count, err = _execute(watch_vars, auto_watch_args)
            except subprocess.TimeoutExpired:
                return f"Error: cargo test timed out after {timeout} seconds during trace execution."
            if err:
                return err

            fallback_note = ""
            if (watch_vars or auto_watch_args) and res.returncode != 0:
                combined_err = (res.stderr or "") + "\n" + (res.stdout or "")
                if any(tok in combined_err for tok in ("E0277", "E0382", "E0597", "doesn't implement", "cannot find value")):
                    _restore()
                    try:
                        res, instrumented_count, err = _execute(None, False)
                        fallback_note = "\n[Note: watch_vars failed Debug/scope check and automatically fell back to call-tree-only tracing.]"
                    except subprocess.TimeoutExpired:
                        return f"Error: cargo test timed out after {timeout} seconds during trace execution."
                    if err:
                        return err

            tree_text, total_calls = format_trace_call_tree((res.stderr or "") + "\n" + (res.stdout or ""))
            test_status = "passed" if res.returncode == 0 else f"failed (exit_code={res.returncode})"
            guidance = (
                "\n\n[Minimal Modification Guidance]:\n"
                "- Prioritize fixing logic inside the innermost function (callee) where incorrect values originate.\n"
                "- Avoid refactoring or changing line spans in caller/forwarding functions, as line shifts can break downstream doc-tests."
            )
            return (
                f"[Rust Dynamic Trace Report]\n"
                f"Test status: {test_status}\n"
                f"Instrumented functions: {instrumented_count} across {len(targets)} file(s)\n"
                f"Total traced function calls: {total_calls}{fallback_note}\n\n"
                f"Call Hierarchy:\n{tree_text}{guidance}"
            )
        finally:
            _restore()

    return rust_trace
