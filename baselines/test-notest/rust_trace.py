"""Zero-side-effect dynamic tracing tool for Rust functions during cargo test."""

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool


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


def parse_instrument_targets(root: Path, instrument_specs: list[str], symbols_module=None) -> dict[Path, set[str]]:
    """Parse instrument specifications into {file_path: set(fn_names)}."""
    root = Path(root).resolve()
    targets = {}
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


_SAFE_WATCH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\(\)]*$")


def _find_fn_open_braces(source: str, target_fns: set[str], rel_display: str) -> list[tuple[str, int]]:
    """Locate function body '{' offsets using token-aware CKG parser, skipping tests/const-fn/macro decls."""
    ckg_mod = _load_ckg_module()
    if ckg_mod is not None:
        try:
            parsed = ckg_mod.parse_rust_file(source, rel_display)
            matches: list[tuple[str, int]] = []
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
                    matches.append((fn_name, b_off))
            if matches or not target_fns:
                matches.sort(key=lambda x: x[1])
                return matches
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
        matches.append((fn_name, brace_pos))
    return matches


def instrument_source_for_trace(
    source: str,
    target_fns: set[str],
    rel_display: str,
    watch_vars: list[str] | None = None,
) -> tuple[str, int]:
    """Instrument target functions with non-destructive RAII trace probes."""
    matches = _find_fn_open_braces(source, target_fns, rel_display)
    if not matches:
        return source, 0

    safe_vars = [v.strip() for v in (watch_vars or []) if v and _SAFE_WATCH_RE.match(v.strip())][:5]

    count = 0
    result = source
    for fn_name, brace_pos in reversed(matches):
        insert_offset = brace_pos + 1
        sub = result[insert_offset:]
        m_inner = re.match(r"(?:\s*(?://![^\n]*|#!\[[^\]]+\]))*\s*", sub)
        if m_inner:
            insert_offset += m_inner.end()

        var_lines = ""
        for v in safe_vars:
            var_lines += f'    eprintln!("[RUST_TRACE_VAR] {fn_name}::{v}={{:?}}", &({v}));\n'

        probe = (
            "\n    struct __TraceGuard(&'static str, &'static str, u32, std::time::Instant);\n"
            "    impl Drop for __TraceGuard {\n"
            "        fn drop(&mut self) {\n"
            "            eprintln!(\"[RUST_TRACE_EXIT] {}:{} {} elapsed={:?}\", self.0, self.2, self.1, self.3.elapsed());\n"
            "        }\n"
            "    }\n"
            f"    eprintln!(\"[RUST_TRACE_ENTER] {{}}:{{}} {{}}\", file!(), line!(), stringify!({fn_name}));\n"
            f"{var_lines}"
            f"    let __trace_guard = __TraceGuard(file!(), stringify!({fn_name}), line!(), std::time::Instant::now());\n"
        )
        result = result[:insert_offset] + probe + result[insert_offset:]
        count += 1

    return result, count


def format_trace_call_tree(raw_output: str, max_depth: int = 10) -> tuple[str, int]:
    """Parse [RUST_TRACE_ENTER], [RUST_TRACE_VAR], and [RUST_TRACE_EXIT] into a tree representation."""
    events = []
    for line in raw_output.splitlines():
        line = line.strip()
        if line.startswith("[RUST_TRACE_ENTER]"):
            parts = line.split(maxsplit=2)
            if len(parts) >= 3:
                loc, fn_name = parts[1], parts[2]
                events.append(("enter", loc, fn_name, ""))
        elif line.startswith("[RUST_TRACE_VAR]"):
            val_text = line[len("[RUST_TRACE_VAR]"):].strip()
            events.append(("var", "", val_text, ""))
        elif line.startswith("[RUST_TRACE_EXIT]"):
            parts = line.split(maxsplit=3)
            if len(parts) >= 3:
                loc, fn_name = parts[1], parts[2]
                elapsed = parts[3] if len(parts) >= 4 else ""
                events.append(("exit", loc, fn_name, elapsed))

    if not events:
        return "(no instrumented functions were invoked during this test run)", 0

    stack = []
    tree_lines = []
    call_count = 0

    for action, loc, fn_name, elapsed in events:
        if action == "enter":
            call_count += 1
            depth = len(stack)
            indent = "  " * depth + ("├─ " if depth > 0 else "")
            stack.append((fn_name, loc))
            if depth < max_depth:
                tree_lines.append(f"{indent}{fn_name} ({loc})")
        elif action == "var":
            depth = len(stack)
            indent = "  " * depth + "│ "
            if depth <= max_depth:
                tree_lines.append(f"{indent}var {fn_name[:180]}")
        elif action == "exit":
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
    return formatted, call_count


def make_rust_trace_tool(root: Path, resolve_path, env=None):
    """Build the rust_trace tool."""
    root = Path(root).resolve()
    spec = importlib.util.spec_from_file_location("symbols_trace", Path(__file__).with_name("rust_symbols.py"))
    sym_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sym_mod)

    @tool
    def rust_trace(
        instrument: list[str],
        test_target: str = "",
        test_filter: str = "",
        manifest_path: str = "Cargo.toml",
        package: str = "",
        features: list[str] | None = None,
        all_features: bool = False,
        no_default_features: bool = False,
        target: str = "",
        timeout: int = 180,
        lib: bool = False,
        watch_vars: list[str] | None = None,
    ) -> str:
        """Dynamically trace execution of Rust functions during cargo test without permanent code changes.

        - instrument: List of targets in format 'path/to/file.rs:fn_name', 'path/to/file.rs', or bare 'fn_name'.
        - test_target: Integration test target name (e.g. 'test_regression_mybug').
        - test_filter: Optional test name substring filter.
        - lib: Set True to test library unit tests (cargo test --lib).
        - watch_vars: Optional list of local variable/argument expressions (must implement Debug);
          automatically retries without watch_vars if Debug is not satisfied.

        Instruments target functions with zero-dependency RAII trace probes, executes cargo test,
        and cleanly restores all files. Does not invalidate working copy verification status.
        """
        if not instrument:
            return "Error: Provide at least one target to instrument (e.g. ['src/foo.rs:bar'] or ['bar']). "

        targets = parse_instrument_targets(root, instrument, symbols_module=sym_mod)
        if not targets:
            return f"Error: None of the specified instrumentation targets could be resolved: {instrument}"

        backup = {}
        for fpath in targets:
            st = fpath.stat()
            backup[fpath] = (fpath.read_bytes(), st.st_mode, st.st_mtime_ns)

        def _restore():
            for fpath, (data, mode, mtime_ns) in backup.items():
                fpath.write_bytes(data)
                fpath.chmod(mode)
                try:
                    os.utime(fpath, ns=(mtime_ns, mtime_ns))
                except Exception:
                    pass

        def _execute(vars_to_watch):
            instrumented_count = 0
            for fpath, fns in targets.items():
                content = backup[fpath][0].decode("utf-8")
                rel_display = fpath.relative_to(root).as_posix()
                new_content, n_mod = instrument_source_for_trace(
                    content, fns, rel_display, watch_vars=vars_to_watch
                )
                if n_mod > 0:
                    fpath.write_bytes(new_content.encode("utf-8"))
                    instrumented_count += n_mod

            if instrumented_count == 0:
                return None, 0, "Error: Could not locate function definitions matching the specified instrument list."

            manifest = resolve_path(manifest_path)
            cmd = ["cargo", "test", "--manifest-path", str(manifest)]
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
                res, instrumented_count, err = _execute(watch_vars)
            except subprocess.TimeoutExpired:
                return f"Error: cargo test timed out after {timeout} seconds during trace execution."
            if err:
                return err

            fallback_note = ""
            if watch_vars and res.returncode != 0:
                combined_err = (res.stderr or "") + "\n" + (res.stdout or "")
                if "E0277" in combined_err or "doesn't implement" in combined_err or "cannot find value" in combined_err:
                    _restore()
                    try:
                        res, instrumented_count, err = _execute(None)
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
