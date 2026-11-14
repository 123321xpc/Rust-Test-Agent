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


def _strip_in_module_tests(text: str) -> str:
    """Strip temporary in-module regression test blocks from Rust source code."""
    pattern = r"(?ms)\r?\n// __TEST_REGRESSION_START: [^\n]+\r?\n.*?// __TEST_REGRESSION_END: [^\n]+(?:\r?\n)?"
    return re.sub(pattern, "", text)

def _check_rust_syntax(path_str: str, content: str) -> None:
    """Validate delimiter balance, unclosed comments/strings, and E0753 before writing .rs files."""
    if not path_str.endswith(".rs") or not content.strip():
        return
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
                raise ValueError(
                    f"Syntax guardrail error in {path_str}:{line}: unexpected closing delimiter '{ch}' "
                    "with no matching opening delimiter. Edit rejected to prevent cascading errors."
                )
            top, top_line, _ = stack.pop()
            if top != pairs[ch]:
                raise ValueError(
                    f"Syntax guardrail error in {path_str}:{line}: mismatched delimiter '{ch}', "
                    f"expected closing for '{top}' from line {top_line}. Edit rejected to prevent cascading errors."
                )
    if stack:
        top, top_line, _ = stack[-1]
        raise ValueError(
            f"Syntax guardrail error in {path_str}:{top_line}: unclosed opening delimiter '{top}' "
            "(reaches end of file without matching closing bracket). Edit rejected to prevent cascading errors."
        )

    has_item = False
    first_item_line = None
    for lineno, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            if (s.startswith("//!") or s.startswith("#![")) and has_item:
                raise ValueError(
                    f"Syntax guardrail error in {path_str}:{lineno}: inner attribute or doc comment ('{s[:25]}') "
                    f"found after code declarations at line {first_item_line} (Rust compiler error E0753). "
                    "Inner attributes/docs must be placed at the very top of the file/module before any item or use declarations. Edit rejected."
                )
            continue
        if s.startswith("#!["):
            if has_item:
                raise ValueError(
                    f"Syntax guardrail error in {path_str}:{lineno}: inner attribute ('{s[:25]}') "
                    f"found after code declarations at line {first_item_line} (Rust compiler error E0753). "
                    "Inner attributes/docs must be placed at the very top of the file/module before any item or use declarations. Edit rejected."
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
    old_first_doc_line = old_text[:old_doc_starts[0]].count("\n") + 1
    new_first_doc_line = new_text[:new_doc_starts[0]].count("\n") + 1
    impacted = any(old_text[:pos].count("\n") + 1 >= first_diff for pos in old_doc_starts)
    if impacted:
        sign = f"+{line_delta}" if line_delta > 0 else f"{line_delta}"
        return (
            f"[Doctest Line-Shift Notice in {path_str}]: edit changed line count by {sign} (from line {first_diff}+), "
            f"shifting downstream Rust doc-tests (e.g. line {old_first_doc_line} -> {new_first_doc_line}). "
            "If existing tests target exact doctest line numbers, keep edits line-neutral or preserve vertical spacing."
        )
    return ""


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
        self._checkpoints = {}
        self._best_verified_snapshot = None
        self._workspace_deps_cache = None
        self.require_reproduction = require_reproduction
        self.post_strip_check = post_strip_check
        self.rust_env = None
        self.rust_info = None
        self._budget_client = None
        self._budget_warning_threshold = 10
        self._package_manifests_cache = None

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
                        if t_info.get("is_in_module"):
                            cmd_str = f"cargo_test(manifest_path='{manifest.as_posix()}', lib=True, test_filter='{name}')"
                        else:
                            cmd_str = f"cargo_test(manifest_path='{manifest.as_posix()}', test_target='{name}')"
                        note = self._reproduction_notes.get(key)
                        missing_repro.append(f"{cmd_str} ({note})" if note else cmd_str)
                if missing_repro:
                    parts.append("confirm failure (Red phase): " + ", ".join(missing_repro))

            missing_reg = []
            for manifest, name in targets:
                if (manifest, name) not in self._passed_regressions:
                    t_info = all_targets_info.get((manifest, name), {})
                    if t_info.get("is_in_module"):
                        missing_reg.append(f"cargo_test(manifest_path='{manifest.as_posix()}', lib=True, test_filter='{name}')")
                    else:
                        missing_reg.append(f"cargo_test(manifest_path='{manifest.as_posix()}', test_target='{name}')")
            if missing_reg:
                parts.append("pass regression tests: " + ", ".join(missing_reg))
            return "; ".join(parts)

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

    def record_check_result(self, exit_code: int, diagnostics: list[dict]) -> str | None:
        with self._lock:
            errors = [d for d in diagnostics if d.get("level") == "error"]
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
                "verified": self.verified(),
            }

    def _restore_snapshot(self, snap: dict):
        with self._lock:
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

            self.originals = dict(snap["originals"])
            self.regression_tests = set(snap["regression_tests"])
            self.in_module_regressions = dict(snap.get("in_module_regressions", {}))
            self._reproduced_regressions = {k: dict(v) for k, v in snap["reproduced_regressions"].items()}
            self._passed_regressions = set(snap["passed_regressions"])
            self._passed_regression_configs = {k: dict(v) for k, v in snap["passed_regression_configs"].items()}
            self._regression_test_names = {k: set(v) for k, v in snap["regression_test_names"].items()}
            self._reproduction_notes = dict(snap["reproduction_notes"])
            self._passed_suites.clear()
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

        prod_changed = [p for p in self.changed_paths() if p not in self.regression_tests]
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
                    self._reproduced_regressions[(rel_manifest, name)] = {
                        "clean_red": True,
                        "runtime_failure": executed > 0,
                        "compile_reproduction": executed == 0,
                        "executed": executed,
                        "failures": failures,
                        "returncode": res.returncode,
                    }
                    self._reproduction_notes.pop((rel_manifest, name), None)
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

    def run_post_strip_check(self, selected_paths: list[Path]):
        if not shutil.which("cargo"):
            return
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
            return

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
                    if first_cfg.get("features"):
                        cmd.append("--features=" + ",".join(first_cfg["features"]))
                    if first_cfg.get("all_features"):
                        cmd.append("--all-features")
                    if first_cfg.get("no_default_features"):
                        cmd.append("--no-default-features")
                    if first_cfg.get("target"):
                        cmd.append(f"--target={first_cfg['target']}")

                res = subprocess.run(
                    cmd,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=120,
                    env=self.rust_env,
                )
                if res.returncode != 0:
                    rel_manifest = manifest.relative_to(root).as_posix() if manifest.is_relative_to(root) else str(manifest)
                    err_snippet = (res.stderr.strip() or res.stdout.strip())[-3000:]
                    raise ValueError(
                        f"Post-strip verification failed for {rel_manifest} under 'cargo check --tests'. "
                        "When temporary regression tests were excluded, your changes broke compilation "
                        f"of production code or existing test targets:\n{err_snippet}"
                    )
            downstream_pkgs = self._find_downstream_workspace_packages(valid_manifests)
            ws_manifest = root / "Cargo.toml"
            for dep_pkg in downstream_pkgs:
                dep_res = subprocess.run(
                    ["cargo", "check", "--manifest-path", str(ws_manifest), f"--package={dep_pkg}"],
                    cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=120, env=self.rust_env,
                )
                if dep_res.returncode != 0:
                    err_snippet = (dep_res.stderr.strip() or dep_res.stdout.strip())[-3000:]
                    is_env_error = any(kw in err_snippet for kw in (
                        "requires rustc", "failed to select a version", "failed to download",
                        "could not find", "Blocking waiting for file lock", "network is unreachable"
                    ))
                    if is_env_error:
                        continue
                    raise ValueError(
                        f"Post-strip downstream workspace check failed for dependent package '{dep_pkg}'. "
                        f"Your changes broke downstream crate compilation:\n{err_snippet}"
                    )
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
            return ""


    def invalidate_write(self):
        with self._lock:
            self._revision += 1
            self._verified_revision = None
            self._passed_regressions.clear()
            self._passed_regression_configs.clear()
            self._regression_test_names.clear()
            self._passed_suites.clear()
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
                all_targets_info = self.get_regression_targets()
                for (relative_manifest, name), tinfo in all_targets_info.items():
                    target_manifest = (self.root / relative_manifest).resolve()
                    same_manifest = (target_manifest == effective_manifest or (not package and (self.root / manifest_path).resolve() == target_manifest))
                    if not same_manifest:
                        continue
                    matched = False
                    if not tinfo["is_in_module"]:
                        matched = (test_target == name and not test_filter)
                    else:
                        matched = (lib and (test_filter == name or test_filter in name) and not test_target)
                    if matched:
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
                is_suite = (not test_target and not test_filter and not lib and not bin_target)
                if is_suite:
                    self._passed_suites.add(effective_manifest)
                targets = set(self.get_regression_targets().keys())
                reproduction_ok = (not self.require_reproduction) or all(self._is_target_reproduced(t) for t in targets)
                if targets and targets <= self._passed_regressions and reproduction_ok:
                    self._verified_revision = self._revision
                    if not self.require_reproduction or self.reproduced_cleanly():
                        self._best_verified_snapshot = self._capture_snapshot()

    def fail_test(self, token, manifest_path="Cargo.toml", package="", test_target="",
                  test_filter="", lib=False, bin_target="", tests=False,
                  executed=0, failures=None, returncode=None, failed_target="",
                  compile_errors=None):
        with self._lock:
            if token == (self._revision, self._test_generation):
                self._passed_suites.clear()
                effective_manifest = self._resolve_manifest_target(manifest_path, package)
                all_targets_info = self.get_regression_targets()
                for (relative_manifest, name), tinfo in all_targets_info.items():
                    target_manifest = (self.root / relative_manifest).resolve()
                    key = (relative_manifest, name)
                    same_package = target_manifest == effective_manifest
                    matched = False
                    if not tinfo["is_in_module"]:
                        matched = (same_package and test_target == name and not test_filter)
                    else:
                        matched = (same_package and lib and (test_filter == name or test_filter in name) and not test_target)
                    if matched or (not test_target and not test_filter and (failed_target == name or set(failures or []) & self._regression_test_names.get(key, set()))):
                        self._passed_regressions.discard(key)
                        self._passed_regression_configs.pop(key, None)
                        self._regression_test_names.pop(key, None)
                        self._verified_revision = None
                    if (target_manifest == effective_manifest or not package) and matched:
                        clean_red = not any(p not in self.regression_tests for p in self.changed_paths())
                        if executed == 0 and returncode is None:
                            self._reproduced_regressions.pop(key, None)
                            self._reproduction_notes[key] = "test failed to run or timed out"
                            continue
                        runtime_failure = executed > 0 and (returncode != 0 or returncode is None or bool(failures))
                        compile_repro = False
                        if executed == 0 and returncode is not None and returncode != 0:
                            err_codes = {d.get("code") for d in (compile_errors or []) if d.get("code")}
                            if clean_red and bool(err_codes & self._API_COMPILE_ERROR_CODES):
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
    def read_file(path: str) -> str:
        """List a directory or read small UTF-8 files (files over 250 lines are previewed; use read_lines for source files); paths are repository-relative."""
        try:
            target = resolve_path(path)
            if target.is_dir():
                return "\n".join(p.name + ("/" if p.is_dir() else "") for p in sorted(target.iterdir()) if p.name != ".git") or "(empty directory)"
            content = target.read_bytes().decode("utf-8")
            lines = content.splitlines()
            if len(lines) > 250 or len(content) > 15000:
                head = "\n".join(lines[:120])
                tail = "\n".join(lines[-30:])
                return (
                    f"{head}\n"
                    f"... [File has {len(lines)} lines ({len(content)} chars); displaying first 120 and last 30 lines. "
                    f"Use read_lines('{path}', start_line, end_line) to inspect specific ranges, "
                    f"or search_code to locate definitions] ...\n"
                    f"{tail}"
                )
            return content
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error reading {path}: {exc}"

    @tool
    @synchronized
    def write_file(path: str, content: str | None) -> str:
        """Write UTF-8 contents in the working copy; null deletes a file.

        Any write invalidates the submitted patch. Call generate_patch after
        all edits, selecting only files that belong in the final fix.
        """
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
            if relative not in state.originals:
                st = target.stat() if target.is_file() else None
                state.originals[relative] = ((target.read_bytes(), st.st_mode, st.st_mtime_ns) if st else None)
            if content is not None:
                _check_rust_syntax(relative.as_posix(), content)
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

                orig_text = target.read_text(encoding="utf-8")
                clean_base_text = _strip_in_module_tests(orig_text)
                _check_rust_syntax(relative.as_posix(), content)

                test_block = (
                    f"\n// __TEST_REGRESSION_START: {clean_name}__\n"
                    f"#[cfg(test)]\n"
                    f"mod {clean_name} {{\n"
                    f"    use super::*;\n"
                    f"{content}\n"
                    f"}}\n"
                    f"// __TEST_REGRESSION_END: {clean_name}__\n"
                )
                updated_text = clean_base_text + test_block

                st = target.stat()
                state.originals.setdefault(relative, (orig_text.encode("utf-8"), st.st_mode, st.st_mtime_ns))
                state.invalidate_write()
                state.regression_tests.add(relative)
                state.in_module_regressions[relative] = {
                    "name": clean_name,
                    "manifest": manifest_rel,
                    "content": content,
                }
                reg_key = (manifest_rel, clean_name)
                state._reproduced_regressions.pop(reg_key, None)
                state._passed_regression_configs.pop(reg_key, None)
                state._reproduction_notes.pop(reg_key, None)

                target.write_bytes(updated_text.encode("utf-8"))
                msg = f"Mounted temporary in-module regression test '{clean_name}' at end of {relative.as_posix()}."
                if run_now:
                    run_res = cargo_test.func(manifest_path=manifest_rel.as_posix(), lib=True, test_filter=clean_name)
                    msg += f"\n\n[Auto-run result for {clean_name}]:\n{run_res}"
                return msg

            target = resolve_path(path)
            relative = target.relative_to(root)
            if (target.parent.name != "tests" or not (target.name.startswith("test_regression_") or target.name.startswith("normal_regression_"))
                    or target.suffix != ".rs" or not isinstance(content, str) or not content.strip()):
                raise ValueError("Use a nonempty test at tests/test_regression_<name>.rs, or use target_src='src/lib.rs' for in-module test")
            if relative not in state.regression_tests and target.exists():
                raise ValueError("Regression test must be a new file")
            state.invalidate_write()
            state.originals.setdefault(relative, None)
            state.regression_tests.add(relative)
            reg_key = (relative.parent.parent / "Cargo.toml", relative.stem)
            state._reproduced_regressions.pop(reg_key, None)
            state._passed_regression_configs.pop(reg_key, None)
            state._reproduction_notes.pop(reg_key, None)
            _check_rust_syntax(relative.as_posix(), content)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
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
        try:
            target = resolve_path(path)
            relative = target.relative_to(root)
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
            write_result = write_file.func(path, updated)
            if not write_result.startswith("Updated "):
                return write_result
            snippet = _format_edit_snippet(updated, line_ranges, context=6)
            out_text = f"{write_result}\n{snippet}" if snippet else write_result
            if target.suffix == ".rs" and relative not in state.regression_tests:
                orig_raw = state.originals.get(relative)
                if orig_raw and orig_raw[0] is not None:
                    orig_code = orig_raw[0].decode("utf-8", errors="replace")
                    shift_warn = _detect_doctest_line_shift(orig_code, updated, relative.as_posix())
                    if shift_warn:
                        out_text += f"\n{shift_warn}" 
            if target.suffix == ".rs" and relative not in state.regression_tests:
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
            return out_text
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error editing {path}: {exc}"

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
            if relative in state.in_module_regressions:
                info = state.in_module_regressions.pop(relative, None)
                state.regression_tests.discard(relative)
                if info:
                    reg_key = (info["manifest"], info["name"])
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
                state.invalidate_write()
                return f"Reverted {relative.as_posix()} to its original state."
            if relative not in state.originals and relative not in state.regression_tests:
                if not target.exists():
                    raise ValueError(f"File does not exist: {path}")
                if not target.is_file():
                    raise ValueError("Path is not a regular file")
                return f"No changes to revert for {relative.as_posix()}."
            original = state.originals.pop(relative, None)
            state.regression_tests.discard(relative)
            reg_key = (relative.parent.parent / "Cargo.toml", relative.stem)
            state._reproduced_regressions.pop(reg_key, None)
            state._passed_regression_configs.pop(reg_key, None)
            state._reproduction_notes.pop(reg_key, None)
            state.invalidate_write()
            if original is None:
                if target.exists() or target.is_symlink():
                    target.unlink()
                return f"Reverted {relative.as_posix()} (removed newly created file)."
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original[0])
            target.chmod(original[1])
            if len(original) >= 3 and original[2] is not None:
                try:
                    os.utime(target, ns=(original[2], original[2]))
                except Exception:
                    pass
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
            targets = set(state.get_regression_targets().keys())
            if state.require_reproduction and targets and targets <= state._passed_regressions and not state.reproduced_cleanly():
                state.verify_pre_fix_reproduction(targets)
            if changed and not state.verified():
                missing = state.missing_verification_steps()
                detail = f" Missing: {missing}." if missing else ""
                return ("After the last write, pass each temporary regression test with cargo_test "
                        f"using test_target and no filter, with at least one test executed.{detail}")
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
                return (f"Unsubmitted preview. Changed files:\n{names}\n{repro_notice}{test_compile_warn}{doctest_notice}"
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
                state.run_post_strip_check(selected)
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
        enrich_diagnostics=diagnostics.attach_diagnostic_snippets,
        on_test_feedback=state.record_test_result,
    )
    explain = load_local("rust_error_explain").make_rust_error_explain_tool(root, rust_env)
    symbols_mod = load_local("rust_symbols")
    impact_mask_cache = {}
    trace_tool = load_local("rust_trace").make_rust_trace_tool(root, resolve_path, rust_env)
    symbols = symbols_mod.make_rust_symbols_tool(root, resolve_path)
    all_tools = [read_file, write_file, write_regression_test, edit_file, revert_file, checkpoint, generate_patch,
                 *rust_tools, cargo_check, cargo_test, trace_tool, symbols, explain]

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
