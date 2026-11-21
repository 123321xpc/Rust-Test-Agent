"""Compiler-precise Rust definition graph (Stage 1), MIR call edges (Stage 2), and data-flow slicing (Stage 3).

Uses `cargo doc --output-format=json` (with `RUSTC_BOOTSTRAP=1`) to obtain
compiler-assigned item identities, full module paths, and trait -> impl -> method
membership, eliminating same-name collisions across modules. Stage 2 lazily runs
single-package `cargo rustc -- -Zunpretty=mir` and joins resolved call targets
back to Stage 1 definitions. Stage 3 extracts statement-level def-use and
control-dependence slices across intra- and 1-hop inter-procedural MIR paths.
Any compiler failure degrades cleanly to the lexical CKG.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BOOTSTRAP_ENV = "RUSTC_BOOTSTRAP"
RUSTDOC_FLAGS = "--output-format=json -Zunstable-options --document-private-items"
DOC_TIMEOUT_SEC = 180
MIR_TIMEOUT_SEC = 90
MAX_ERR_CHARS = 1600

_MIR_FN_RE = re.compile(r"^fn\s+(.+?)\((.*)\)\s*(?:->\s*.*?)?\{\s*$")
_MIR_IMPL_SPAN_RE = re.compile(r"<impl at ([^:]+):(\d+):(\d+):[^>]+>::([A-Za-z_][A-Za-z0-9_]*)$")
_MIR_CALL_RE = re.compile(r"^\s*(?:_[0-9]+\s*=\s*)?(<.+?>::[A-Za-z_][A-Za-z0-9_:<>]*|[A-Za-z_][A-Za-z0-9_:<>]*)\(.*\)\s*->\s*\[")
_MIR_SPAN_RE = re.compile(r"//\s*scope\s+\d+\s+at\s+(.+?):(\d+):(\d+):\s*(\d+):(\d+)\s*$")
_MIR_DEBUG_RE = re.compile(r"^\s*debug\s+([A-Za-z_][A-Za-z0-9_]*)\s*=>\s*(_\d+)\s*;")
_MIR_BB_RE = re.compile(r"^\s*(bb\d+):\s*\{\s*$")
_MIR_FIELD_PROJ_RE = re.compile(r"\(\s*(?:\*\s*)?(_\d+)(?:\s+as\s+[A-Za-z0-9_:<>]+)?\s*\)\.(\d+)")
_MIR_PLAIN_FIELD_RE = re.compile(r"\b(_\d+)\.(\d+)\b")
_MIR_LOCAL_RE = re.compile(r"\b(_\d+)\b")
_MIR_BB_TARGET_RE = re.compile(r"\b(bb\d+)\b")


@dataclass
class SemanticItem:
    """One compiler-resolved Rust definition."""

    key: str
    doc_id: str
    name: str
    kind: str
    full_path: str
    module_path: str
    crate: str
    file: str
    line: int
    end_line: int
    col: int
    is_trait_method: bool = False
    owner_trait: str | None = None
    owner_type: str | None = None
    impl_key: str | None = None
    is_public: bool = False


@dataclass
class SemanticImpl:
    """One `impl` block (`impl Trait for Type` or inherent `impl Type`)."""

    key: str
    doc_id: str
    crate: str
    file: str
    line: int
    end_line: int
    col: int
    trait_full_path: str | None
    trait_name: str | None
    type_full_path: str | None
    type_name: str | None
    context: str
    is_negative: bool = False
    blanket: bool = False
    method_keys: list[str] = field(default_factory=list)


@dataclass
class SemanticCallEdge:
    """One MIR-resolved call site joined with Stage 1 definitions."""

    caller_key: str
    caller_full_path: str
    caller_name: str
    caller_file: str
    caller_line: int
    callee_key: str
    callee_full_path: str
    callee_name: str
    callee_file: str
    callee_line: int
    callee_expr: str


@dataclass
class MirStmt:
    """One statement or terminator inside a MIR basic block."""

    bb: str
    idx: int
    file: str | None
    line: int
    end_line: int
    defs: set[str]
    weak_defs: set[str]
    uses: set[str]
    is_switch: bool = False
    is_return: bool = False
    call_callee_key: str | None = None
    call_callee_expr: str | None = None
    call_arg_places: list[set[str]] = field(default_factory=list)
    call_dest: str | None = None


@dataclass
class MirFunction:
    """Parsed MIR body with CFG, reaching-definition def-use, and control dependence."""

    item_key: str
    full_path: str
    name: str
    file: str
    line: int
    end_line: int
    param_count: int
    debug_vars: dict[str, str] = field(default_factory=dict)  # _1 -> var_name
    var_to_locals: dict[str, set[str]] = field(default_factory=dict)
    stmts: dict[tuple[str, int], MirStmt] = field(default_factory=dict)
    block_order: list[str] = field(default_factory=list)
    block_stmts: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    succs: dict[str, list[str]] = field(default_factory=dict)
    backward_deps: dict[tuple[str, int], set[tuple[str, int]]] = field(default_factory=dict)
    forward_deps: dict[tuple[str, int], set[tuple[str, int]]] = field(default_factory=dict)
    built_deps: bool = False


@dataclass
class SemanticGraph:
    """Stage 1 compiler definition graph for one target crate."""

    crate_root: Path
    package: str
    crate: str
    items: dict[str, SemanticItem] = field(default_factory=dict)
    impls: dict[str, SemanticImpl] = field(default_factory=dict)
    by_full_path: dict[str, list[str]] = field(default_factory=dict)
    precise: bool = True
    degrade_reason: str | None = None


@dataclass
class SemanticWorkspace:
    """Merged workspace-level Stage 1 definition graph + lazy Stage 2 MIR call edges."""

    root: Path
    fingerprint: str
    packages: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # pkg -> [(kind, crate_name)]
    crate_to_pkg: dict[str, str] = field(default_factory=dict)
    graphs: dict[str, SemanticGraph] = field(default_factory=dict)
    items: dict[str, SemanticItem] = field(default_factory=dict)
    impls: dict[str, SemanticImpl] = field(default_factory=dict)
    by_full_path: dict[str, list[SemanticItem]] = field(default_factory=dict)
    items_by_name: dict[str, list[SemanticItem]] = field(default_factory=dict)
    items_by_file_line: dict[tuple[str, int], list[SemanticItem]] = field(default_factory=dict)
    impls_by_file_line: dict[tuple[str, int], SemanticImpl] = field(default_factory=dict)
    impls_by_trait: dict[str, list[SemanticImpl]] = field(default_factory=dict)
    impls_by_type: dict[str, list[SemanticImpl]] = field(default_factory=dict)
    impls_by_trait_name: dict[str, list[SemanticImpl]] = field(default_factory=dict)
    impls_by_type_name: dict[str, list[SemanticImpl]] = field(default_factory=dict)
    callers_by_callee: dict[str, list[SemanticCallEdge]] = field(default_factory=dict)
    callees_by_caller: dict[str, list[SemanticCallEdge]] = field(default_factory=dict)
    mir_functions: dict[str, MirFunction] = field(default_factory=dict)
    mir_packages_ok: set[str] = field(default_factory=set)
    mir_packages_attempted: set[str] = field(default_factory=set)
    members_ok: list[str] = field(default_factory=list)
    members_failed: dict[str, str] = field(default_factory=dict)
    precise: bool = False
    mode: str = "lexical"

    def ensure_packages(self, pkgs: list[str], env: dict[str, str] | None = None) -> bool:
        """Lazily document and ingest additional packages (for large >6-crate workspaces)."""
        base_env = dict(env or os.environ)
        added = False
        for pname in pkgs:
            if pname not in self.packages or pname in self.members_ok or pname in self.members_failed:
                continue
            targets = self.packages[pname]
            p_target, p_err, tmp_p = _run_cargo_doc(self.root, base_env, package=pname)
            try:
                if p_err is not None:
                    self.members_failed[pname] = p_err
                    continue
                loaded_any = False
                for _k, cname in targets:
                    jpath = p_target / "doc" / f"{cname}.json"
                    if jpath.is_file():
                        try:
                            doc = json.loads(jpath.read_text(encoding="utf-8"))
                            self._ingest_graph(cname, pname, doc)
                            loaded_any = True
                            added = True
                        except (OSError, json.JSONDecodeError) as exc:
                            self.members_failed[pname] = str(exc)[:MAX_ERR_CHARS]
                if not loaded_any and pname not in self.members_failed:
                    self.members_failed[pname] = "rustdoc JSON output not found"
            finally:
                if tmp_p is not None:
                    tmp_p.cleanup()
        self._refresh_mode()
        return added

    def _ingest_graph(self, cname: str, pname: str, doc: dict[str, Any]) -> None:
        g = parse_rustdoc_json(doc, self.root, pname, cname)
        self.graphs[cname] = g
        if pname not in self.members_ok:
            self.members_ok.append(pname)
        self.members_failed.pop(pname, None)
        for key, item in g.items.items():
            self.items[key] = item
            self.by_full_path.setdefault(item.full_path, []).append(item)
            self.items_by_name.setdefault(item.name, []).append(item)
            self.items_by_file_line.setdefault((item.file, item.line), []).append(item)
        for key, imp in g.impls.items():
            self.impls[key] = imp
            self.impls_by_file_line[(imp.file, imp.line)] = imp
            if imp.trait_full_path:
                self.impls_by_trait.setdefault(imp.trait_full_path, []).append(imp)
            if imp.trait_name:
                self.impls_by_trait_name.setdefault(imp.trait_name, []).append(imp)
            if imp.type_full_path:
                self.impls_by_type.setdefault(imp.type_full_path, []).append(imp)
            if imp.type_name:
                self.impls_by_type_name.setdefault(imp.type_name, []).append(imp)

    def _refresh_mode(self) -> None:
        self.precise = bool(self.members_ok)
        if not self.precise:
            self.mode = "lexical"
        elif self.mir_packages_ok and not self.members_failed:
            self.mode = "compiler_mir"
        elif self.members_failed:
            self.mode = "partial"
        else:
            self.mode = "compiler"

    def resolve_full_paths(self, symbols: list[str]) -> list[str]:
        """Map bare or qualified symbol names to matching full_paths."""
        out: list[str] = []
        for raw in symbols:
            sym = (raw or "").strip().removeprefix("r#")
            if not sym:
                continue
            if "::" in sym:
                for full, items in self.by_full_path.items():
                    if full == sym or full.endswith(f"::{sym}"):
                        if full not in out:
                            out.append(full)
                if out:
                    continue
            bare = sym.split("::")[-1]
            for it in self.items_by_name.get(bare, ()):
                if it.full_path not in out:
                    out.append(it.full_path)
        return out

    def items_for_symbol(self, target: str, *, kind: str = "") -> list[SemanticItem]:
        """Resolve a bare or qualified name to matching SemanticItems."""
        clean = (target or "").strip().removeprefix("r#")
        if not clean:
            return []
        if "::" in clean:
            matches: list[SemanticItem] = []
            for full, items in self.by_full_path.items():
                if full == clean or full.endswith(f"::{clean}"):
                    for it in items:
                        if not kind or it.kind == kind:
                            matches.append(it)
            if matches:
                return matches
            parts = [p for p in clean.split("::") if p]
            if len(parts) >= 2:
                owner_path = "::".join(parts[:-1])
                owner, bare = parts[-2], parts[-1]
                for it in self.items_by_name.get(bare, ()):
                    if kind and it.kind != kind:
                        continue
                    if (
                        (it.owner_type and (it.owner_type == owner_path or it.owner_type.endswith(f"::{owner_path}") or (len(parts) == 2 and it.owner_type.endswith(f"::{owner}"))))
                        or (it.owner_trait and (it.owner_trait == owner_path or it.owner_trait.endswith(f"::{owner_path}") or (len(parts) == 2 and it.owner_trait.endswith(f"::{owner}"))))
                        or (it.module_path == owner_path or it.module_path.endswith(f"::{owner_path}"))
                    ):
                        matches.append(it)
                if matches:
                    return matches
        bare = clean.split("::")[-1]
        return [it for it in self.items_by_name.get(bare, ()) if not kind or it.kind == kind]

    def has_ambiguous_name(self, name: str) -> bool:
        clean = (name or "").strip().split("::")[-1].removeprefix("r#")
        return len(self.items_by_name.get(clean, ())) >= 2

    def dependents_of_trait_any(self, full_paths: list[str]) -> list[SemanticImpl]:
        """Return unique SemanticImpls implementing any of the given trait full_paths."""
        out: list[SemanticImpl] = []
        seen: set[str] = set()
        for fp in full_paths:
            for imp in self.impls_by_trait.get(fp, ()):
                if imp.key not in seen:
                    seen.add(imp.key)
                    out.append(imp)
        return out

    def impls_for_type_any(self, full_paths: list[str]) -> list[SemanticImpl]:
        """Return unique SemanticImpls targeting any of the given type full_paths."""
        out: list[SemanticImpl] = []
        seen: set[str] = set()
        for fp in full_paths:
            for imp in self.impls_by_type.get(fp, ()):
                if imp.key not in seen:
                    seen.add(imp.key)
                    out.append(imp)
        return out

    def ensure_mir_for_symbols(
        self,
        symbols: list[str],
        env: dict[str, str] | None = None,
        *,
        force_packages: list[str] | None = None,
    ) -> bool:
        """Lazily run Stage 2 single-package MIR when queried/changed symbols have name collisions."""
        pkgs: list[str] = list(force_packages or ())
        if not pkgs:
            if not self.precise:
                return False
            ambiguous = any(self.has_ambiguous_name(s) for s in symbols if s)
            if not ambiguous:
                return False
            seen_pkgs: set[str] = set()
            for sym in symbols:
                for it in self.items_for_symbol(sym):
                    pkg = self.crate_to_pkg.get(it.crate)
                    if pkg and pkg not in seen_pkgs:
                        seen_pkgs.add(pkg)
                        pkgs.append(pkg)
            # Include remaining workspace packages only if workspace is small
            if len(self.packages) <= 4:
                for pkg in self.packages:
                    if pkg not in seen_pkgs and len(pkgs) < 4:
                        seen_pkgs.add(pkg)
                        pkgs.append(pkg)
        updated = False
        for pkg in pkgs:
            if pkg not in self.packages or pkg in self.mir_packages_attempted:
                continue
            self.mir_packages_attempted.add(pkg)
            if self._collect_mir_for_package(pkg, env=env):
                self.mir_packages_ok.add(pkg)
                updated = True
        self._refresh_mode()
        return updated

    def ensure_mir_for_packages(
        self,
        pkgs: list[str],
        env: dict[str, str] | None = None,
    ) -> bool:
        """Run Stage 2+3 MIR collection for explicit package(s) regardless of symbol ambiguity."""
        updated = False
        for pkg in pkgs:
            if pkg not in self.packages or pkg in self.mir_packages_attempted:
                continue
            self.mir_packages_attempted.add(pkg)
            if self._collect_mir_for_package(pkg, env=env):
                self.mir_packages_ok.add(pkg)
                updated = True
        self._refresh_mode()
        return updated

    def _collect_mir_for_package(self, pkg: str, env: dict[str, str] | None = None) -> bool:
        targets = self.packages.get(pkg) or []
        selected_flag: list[str] = []
        crate_name = pkg.replace("-", "_")
        lib_t = next((t for t in targets if t[0] in ("lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro")), None)
        bin_t = next((t for t in targets if t[0] == "bin"), None)
        if lib_t:
            selected_flag = ["--lib"]
            crate_name = lib_t[1].replace("-", "_")
        elif bin_t:
            selected_flag = ["--bin", bin_t[1]]
            crate_name = bin_t[1].replace("-", "_")
        else:
            return False

        run_env = dict(env or os.environ)
        run_env[BOOTSTRAP_ENV] = "1"
        run_env.pop("RUSTC_WRAPPER", None)
        tmp_dir_obj = None
        if not run_env.get("CARGO_TARGET_DIR"):
            tmp_dir_obj = tempfile.TemporaryDirectory(prefix="ckg_mir_")
            run_env["CARGO_TARGET_DIR"] = tmp_dir_obj.name
        try:
            cmd = ["cargo", "rustc", "--offline", "-p", pkg, *selected_flag, "--", "-Zunpretty=mir", "-Zmir-include-spans"]
            proc = subprocess.run(
                cmd,
                cwd=self.root,
                env=run_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=MIR_TIMEOUT_SEC,
            )
            if proc.returncode != 0 and "mir-include-spans" in (proc.stderr or ""):
                cmd = ["cargo", "rustc", "--offline", "-p", pkg, *selected_flag, "--", "-Zunpretty=mir"]
                proc = subprocess.run(
                    cmd,
                    cwd=self.root,
                    env=run_env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=MIR_TIMEOUT_SEC,
                )
            if proc.returncode != 0 or not proc.stdout:
                return False
            edges, mir_fns = parse_mir_edges_and_functions(proc.stdout, crate_name, self)
            for edge in edges:
                self._add_mir_edge(edge)
            for mf in mir_fns:
                self.mir_functions[mf.item_key] = mf
            return bool(edges or mir_fns)
        except (OSError, subprocess.SubprocessError):
            return False
        finally:
            if tmp_dir_obj is not None:
                tmp_dir_obj.cleanup()

    def _add_mir_edge(self, edge: SemanticCallEdge) -> None:
        existing_callees = self.callees_by_caller.setdefault(edge.caller_key, [])
        if not any(e.callee_key == edge.callee_key for e in existing_callees):
            existing_callees.append(edge)
        existing_callers = self.callers_by_callee.setdefault(edge.callee_key, [])
        if not any(e.caller_key == edge.caller_key for e in existing_callers):
            existing_callers.append(edge)


def compute_source_fingerprint(root: Path) -> str:
    """Hash Cargo.toml and non-target .rs file mtimes/sizes for cache invalidation."""
    entries: list[str] = []
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "target", ".cargo"))
        for fname in sorted(files):
            if fname == "Cargo.toml" or fname.endswith(".rs"):
                fpath = Path(dirpath) / fname
                try:
                    st = fpath.stat()
                    rel = fpath.relative_to(root).as_posix()
                    entries.append(f"{rel}:{st.st_mtime_ns}:{st.st_size}")
                except OSError:
                    continue
    return hashlib.sha1("\n".join(entries).encode("utf-8")).hexdigest()


def _discover_workspace_packages(root: Path, env: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """Return {package_name: [(target_kind, crate_name_underscored), ...]}."""
    try:
        proc = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
            members = set(data.get("workspace_members") or ())
            out: dict[str, list[tuple[str, str]]] = {}
            for pkg in data.get("packages") or ():
                if members and pkg.get("id") not in members:
                    continue
                pname = pkg.get("name")
                if not pname:
                    continue
                tlist: list[tuple[str, str]] = []
                for tgt in pkg.get("targets") or ():
                    kinds = tgt.get("kind") or []
                    tname = tgt.get("name")
                    if not tname:
                        continue
                    for k in kinds:
                        if k in ("lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro", "bin"):
                            tlist.append((k, tname.replace("-", "_")))
                            break
                if tlist:
                    out[pname] = tlist
            if out:
                return out
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    # Minimal fallback if metadata cannot run
    out = {}
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "target", ".cargo"))
        if "Cargo.toml" in files:
            try:
                content = (Path(dirpath) / "Cargo.toml").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            m_sec = re.search(r"(?ms)^\[package\](.*?)(?:^\[|\Z)", content)
            if m_sec:
                m_name = re.search(r'(?m)^name\s*=\s*["\']([^"\']+)["\']', m_sec.group(1))
                if m_name:
                    pname = m_name.group(1).strip()
                    out[pname] = [("lib", pname.replace("-", "_"))]
    return out


def _norm_repo_file(root: Path, filename: str) -> str | None:
    if not filename:
        return None
    p = Path(filename)
    try:
        resolved = p.resolve() if p.is_absolute() else (root / p).resolve()
        if resolved.is_relative_to(root):
            return resolved.relative_to(root).as_posix()
    except OSError:
        return None
    return None


def _unwrap_inner(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return (item_kind, inner_payload) across old (format_version <= 25) and modern rustdoc JSON."""
    inner = entry.get("inner")
    outer_kind = entry.get("kind")
    if outer_kind:
        return str(outer_kind), inner if isinstance(inner, dict) else {}
    if isinstance(inner, dict) and inner:
        kind = str(next(iter(inner.keys())))
        payload = inner[kind]
        return kind, payload if isinstance(payload, dict) else {}
    return "", {}


def _entry_kind(entry: dict[str, Any], path_entry: dict[str, Any] | None) -> str:
    kind = ""
    if path_entry and path_entry.get("kind"):
        kind = str(path_entry["kind"])
    else:
        kind, _ = _unwrap_inner(entry)
    if kind == "method":
        return "function"
    if kind == "typedef":
        return "type_alias"
    return kind


def _resolve_doc_path(
    doc_id: Any,
    paths: dict[str, Any],
    items_by_doc_id: dict[str, SemanticItem],
    fallback_name: str | None = None,
) -> str | None:
    if doc_id is not None:
        key = str(doc_id)
        pentry = paths.get(key)
        if pentry and pentry.get("path"):
            return "::".join(pentry["path"])
        it = items_by_doc_id.get(key)
        if it is not None:
            return it.full_path
    return fallback_name


def _extract_for_type(imp: dict[str, Any], paths: dict[str, Any], items_by_doc_id: dict[str, SemanticItem]) -> tuple[str | None, str | None]:
    for_obj = imp.get("for") or {}
    if not isinstance(for_obj, dict):
        return None, None
    # format_version <= 25 uses {"kind": "resolved_path", "inner": {...}}
    if "kind" in for_obj and "inner" in for_obj:
        fk = str(for_obj.get("kind") or "")
        fi = for_obj.get("inner")
        if fk == "resolved_path" and isinstance(fi, dict):
            name_fb = fi.get("path") or fi.get("name") or None
            full = _resolve_doc_path(fi.get("id"), paths, items_by_doc_id, fallback_name=name_fb)
            short = (full or name_fb or "").split("::")[-1] or None
            return full, short
        if fk in ("primitive", "generic") and fi:
            val = str(fi)
            return val, val
        return None, None
    if "resolved_path" in for_obj:
        rp = for_obj["resolved_path"] or {}
        name_fb = rp.get("path") or rp.get("name") or None
        full = _resolve_doc_path(rp.get("id"), paths, items_by_doc_id, fallback_name=name_fb)
        short = (full or name_fb or "").split("::")[-1] or None
        return full, short
    if "primitive" in for_obj:
        prim = str(for_obj["primitive"])
        return prim, prim
    if "generic" in for_obj:
        gen = str(for_obj["generic"])
        return gen, gen
    return None, None


def parse_rustdoc_json(doc: dict[str, Any], root: Path, package: str, crate: str) -> SemanticGraph:
    """Parse one `rustdoc --output-format=json` payload into a SemanticGraph."""
    graph = SemanticGraph(crate_root=root, package=package, crate=crate)
    index: dict[str, Any] = doc.get("index") or {}
    paths: dict[str, Any] = doc.get("paths") or {}

    items_by_doc_id: dict[str, SemanticItem] = {}
    module_children: dict[str, str] = {}

    # Pass 1: discover module membership so private/unpathed items retain full module paths
    root_mod_id = doc.get("root")
    if root_mod_id is not None and str(root_mod_id) in index:
        stack: list[tuple[str, str]] = [(str(root_mod_id), crate)]
        seen_mods: set[str] = set()
        while stack:
            mid, mpath = stack.pop()
            if mid in seen_mods:
                continue
            seen_mods.add(mid)
            mentry = index.get(mid) or {}
            mkind, mod_inner = _unwrap_inner(mentry)
            if mkind != "module" or not isinstance(mod_inner, dict):
                continue
            for child_id in mod_inner.get("items") or ():
                cid = str(child_id)
                module_children[cid] = mpath
                centry = index.get(cid) or {}
                ckind, _ = _unwrap_inner(centry)
                if ckind == "module" and centry.get("name"):
                    stack.append((cid, f"{mpath}::{centry['name']}"))

    # Pass 2: create SemanticItems for every non-impl/non-field item with a repo span
    for doc_id, entry in index.items():
        if not isinstance(entry, dict):
            continue
        item_kind_raw, _ = _unwrap_inner(entry)
        if item_kind_raw in ("impl", "struct_field", "variant"):
            continue
        name = entry.get("name")
        if not name:
            continue
        span = entry.get("span") or {}
        rel_file = _norm_repo_file(root, span.get("filename") or "")
        begin = span.get("begin") or []
        end = span.get("end") or begin
        if not rel_file or len(begin) < 2:
            continue
        pentry = paths.get(str(doc_id))
        if pentry and pentry.get("path"):
            seg = pentry["path"]
            full_path = "::".join(seg)
            mod_path = "::".join(seg[:-1]) if len(seg) > 1 else crate
        else:
            mod_path = module_children.get(str(doc_id), crate)
            full_path = f"{mod_path}::{name}"
        key = f"{rel_file}:{begin[0]}:{begin[1]}"
        item = SemanticItem(
            key=key,
            doc_id=str(doc_id),
            name=str(name),
            kind=_entry_kind(entry, pentry),
            full_path=full_path,
            module_path=mod_path,
            crate=crate,
            file=rel_file,
            line=int(begin[0]),
            end_line=int(end[0]) if end else int(begin[0]),
            col=int(begin[1]),
            is_public=(entry.get("visibility") == "public"),
        )
        items_by_doc_id[str(doc_id)] = item
        graph.items[key] = item

    # Pass 3a: annotate trait methods with their owning trait
    for doc_id, entry in index.items():
        if not isinstance(entry, dict):
            continue
        ekind, epayload = _unwrap_inner(entry)
        if ekind != "trait":
            continue
        tr_item = items_by_doc_id.get(str(doc_id))
        if tr_item is None:
            continue
        for mid in epayload.get("items") or ():
            m_item = items_by_doc_id.get(str(mid))
            if m_item is None:
                continue
            m_item.is_trait_method = True
            m_item.owner_trait = tr_item.full_path
            m_item.full_path = f"{tr_item.full_path}::{m_item.name}"

    # Pass 3b: process user-written impl blocks and attach methods
    for doc_id, entry in index.items():
        if not isinstance(entry, dict):
            continue
        ekind, imp = _unwrap_inner(entry)
        if ekind != "impl":
            continue
        if imp.get("is_synthetic") or imp.get("synthetic"):
            continue
        span = entry.get("span") or {}
        rel_file = _norm_repo_file(root, span.get("filename") or "")
        begin = span.get("begin") or []
        end = span.get("end") or begin
        if not rel_file or len(begin) < 2:
            continue
        tr_ref = imp.get("trait") or {}
        trait_full: str | None = None
        trait_short: str | None = None
        if isinstance(tr_ref, dict) and (tr_ref.get("id") is not None or tr_ref.get("path") or tr_ref.get("name")):
            tr_fb = tr_ref.get("path") or tr_ref.get("name")
            trait_full = _resolve_doc_path(tr_ref.get("id"), paths, items_by_doc_id, fallback_name=tr_fb)
            trait_short = (trait_full or tr_fb or "").split("::")[-1] or None

        type_full, type_short = _extract_for_type(imp, paths, items_by_doc_id)
        impl_key = f"impl:{crate}:{doc_id}"
        method_keys: list[str] = []
        for mid in imp.get("items") or ():
            m_item = items_by_doc_id.get(str(mid))
            if m_item is None:
                continue
            m_item.is_trait_method = trait_full is not None
            m_item.owner_trait = trait_full
            m_item.owner_type = type_full
            m_item.impl_key = impl_key
            if trait_full and type_full:
                m_item.full_path = f"<{type_full} as {trait_full}>::{m_item.name}"
            elif type_full:
                m_item.full_path = f"{type_full}::{m_item.name}"
            method_keys.append(m_item.key)

        ctx = (
            f"impl {trait_full} for {type_full or '?'}"
            if trait_full
            else f"impl {type_full or '?'}"
        )
        graph.impls[impl_key] = SemanticImpl(
            key=impl_key,
            doc_id=str(doc_id),
            crate=crate,
            file=rel_file,
            line=int(begin[0]),
            end_line=int(end[0]) if end else int(begin[0]),
            col=int(begin[1]),
            trait_full_path=trait_full,
            trait_name=trait_short,
            type_full_path=type_full,
            type_name=type_short,
            context=ctx,
            is_negative=bool(imp.get("is_negative") or imp.get("negative")),
            blanket=bool(imp.get("blanket_impl")),
            method_keys=method_keys,
        )

    for item in graph.items.values():
        graph.by_full_path.setdefault(item.full_path, []).append(item.key)
    return graph


def _run_cargo_doc(root: Path, env: dict[str, str], package: str | None = None) -> tuple[Path, str | None, tempfile.TemporaryDirectory | None]:
    run_env = dict(env)
    run_env[BOOTSTRAP_ENV] = "1"
    run_env["RUSTDOCFLAGS"] = RUSTDOC_FLAGS
    tmp_dir_obj: tempfile.TemporaryDirectory | None = None
    if run_env.get("CARGO_TARGET_DIR"):
        target_dir = Path(run_env["CARGO_TARGET_DIR"])
    else:
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="ckg_doc_")
        target_dir = Path(tmp_dir_obj.name)
        run_env["CARGO_TARGET_DIR"] = str(target_dir)
    cmd = ["cargo", "doc", "--offline", "--no-deps"]
    if package:
        cmd.extend(["-p", package])
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=run_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=DOC_TIMEOUT_SEC,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[:MAX_ERR_CHARS]
            return target_dir, err or f"cargo doc exited {proc.returncode}", tmp_dir_obj
        return target_dir, None, tmp_dir_obj
    except (OSError, subprocess.SubprocessError) as exc:
        return target_dir, str(exc)[:MAX_ERR_CHARS], tmp_dir_obj


def build_workspace_graph(
    root: Path,
    env: dict[str, str] | None = None,
    *,
    fingerprint: str | None = None,
    focus_packages: list[str] | None = None,
) -> SemanticWorkspace:
    """Build a Stage 1 SemanticWorkspace for `root`, degrading cleanly per package on failure."""
    root = Path(root).resolve()
    fp = fingerprint or compute_source_fingerprint(root)
    ws = SemanticWorkspace(root=root, fingerprint=fp)

    has_manifest = (root / "Cargo.toml").is_file() or any(root.glob("*/Cargo.toml"))
    if not has_manifest or not shutil.which("cargo"):
        ws.mode = "lexical"
        return ws

    base_env = dict(env or os.environ)
    packages = _discover_workspace_packages(root, base_env)
    if not packages:
        ws.mode = "lexical"
        return ws
    ws.packages = packages
    for pname, targets in packages.items():
        for _k, cname in targets:
            ws.crate_to_pkg[cname] = pname

    # For large monorepos (>6 packages), only document focused packages to stay fast and avoid timeouts.
    selected_pkgs = [p for p in (focus_packages or ()) if p in packages][:4]
    if len(packages) > 6:
        if selected_pkgs:
            ws.ensure_packages(selected_pkgs, env=base_env)
        return ws

    # Small/medium workspace: try one workspace-wide `cargo doc` first, falling back per-package on error.
    target_dir, ws_err, tmp_ws = _run_cargo_doc(root, base_env, package=None)
    try:
        if ws_err is None:
            for pname, targets in packages.items():
                loaded_any = False
                for _k, cname in targets:
                    jpath = target_dir / "doc" / f"{cname}.json"
                    if jpath.is_file():
                        try:
                            ws._ingest_graph(cname, pname, json.loads(jpath.read_text(encoding="utf-8")))
                            loaded_any = True
                        except (OSError, json.JSONDecodeError) as exc:
                            ws.members_failed[pname] = str(exc)[:MAX_ERR_CHARS]
                if not loaded_any and pname not in ws.members_failed:
                    ws.members_failed[pname] = "rustdoc JSON output not found"
            ws._refresh_mode()
        elif len(packages) == 1:
            ws.members_failed[next(iter(packages))] = ws_err
            ws._refresh_mode()
        else:
            ws.ensure_packages(list(packages), env=base_env)
    finally:
        if tmp_ws is not None:
            tmp_ws.cleanup()
    return ws


def _strip_generics(text: str) -> str:
    """Strip generic/lifetime angle-bracketed args `<...>` outside UFCS `<T as Tr>`."""
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()


def _split_ufcs(expr: str) -> tuple[str, str | None, str] | None:
    """Parse `<Type as Trait>::method` or `<Type>::method` -> (type_str, trait_str|None, method)."""
    if not expr.startswith("<"):
        return None
    depth = 0
    close_idx = -1
    for i, ch in enumerate(expr):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx <= 0 or not expr[close_idx + 1:].startswith("::"):
        return None
    inside = expr[1:close_idx]
    method = expr[close_idx + 3:].split("::<", 1)[0].split("<", 1)[0].strip()
    # Find top-level ` as ` inside <...>
    depth = 0
    as_pos = -1
    i = 0
    while i <= len(inside) - 4:
        ch = inside[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0 and inside[i:i + 4] == " as ":
            as_pos = i
            break
        i += 1
    if as_pos >= 0:
        ty_raw = _strip_generics(inside[:as_pos]).lstrip("&").removeprefix("mut ").strip()
        tr_raw = _strip_generics(inside[as_pos + 4:]).strip()
        return ty_raw, tr_raw or None, method
    ty_raw = _strip_generics(inside).lstrip("&").removeprefix("mut ").strip()
    return ty_raw, None, method


def _clean_mir_path(text: str) -> str:
    """Strip generic args and collapse dangling `::` (e.g. `foo::<u8>` -> `foo`)."""
    return "::".join(p for p in _strip_generics(text).split("::") if p)


def _match_mir_caller(header_name: str, crate_name: str, ws: SemanticWorkspace) -> SemanticItem | None:
    m_imp = _MIR_IMPL_SPAN_RE.search(header_name)
    if m_imp:
        raw_file, line_s, _col_s, method_name = m_imp.groups()
        norm_file = _norm_repo_file(ws.root, raw_file)
        line_num = int(line_s)
        if norm_file:
            imp = ws.impls_by_file_line.get((norm_file, line_num))
            if imp is not None:
                for mk in imp.method_keys:
                    it = ws.items.get(mk)
                    if it is not None and it.name == method_name:
                        return it
            for it in ws.items_by_name.get(method_name, ()):
                if it.file == norm_file and it.line >= line_num:
                    return it
    clean = _clean_mir_path(header_name)
    if not clean:
        return None
    for cand in (f"{crate_name}::{clean}", clean):
        items = ws.by_full_path.get(cand)
        if items:
            return items[0]
    crate_cands = [
        it for it in ws.items_by_name.get(clean.split("::")[-1], ())
        if it.crate == crate_name and (it.full_path == clean or it.full_path.endswith(f"::{clean}"))
    ]
    return crate_cands[0] if len(crate_cands) == 1 else None


def _match_mir_callee(callee_expr: str, caller_crate: str, ws: SemanticWorkspace) -> SemanticItem | None:
    ufcs = _split_ufcs(callee_expr)
    if ufcs is not None:
        ty_path, tr_path, method = ufcs
        ty_short = ty_path.split("::")[-1]
        tr_short = tr_path.split("::")[-1] if tr_path else None
        is_self_ty = ty_short in ("Self", "")
        cands: list[SemanticItem] = []
        for it in ws.items_by_name.get(method, ()):
            if not it.owner_type:
                if not (is_self_ty and tr_path and it.is_trait_method and it.owner_trait):
                    continue
            elif not (it.owner_type == ty_path or it.owner_type.endswith(f"::{ty_path}") or it.owner_type.split("::")[-1] == ty_short):
                continue
            if tr_path:
                if not it.owner_trait:
                    continue
                if not (it.owner_trait == tr_path or it.owner_trait.endswith(f"::{tr_path}") or it.owner_trait.split("::")[-1] == tr_short):
                    continue
            cands.append(it)
        if len(cands) == 1:
            return cands[0]
        exact_tr = [c for c in cands if tr_path and c.owner_trait and (c.owner_trait == tr_path or c.owner_trait.endswith(f"::{tr_path}"))]
        if len(exact_tr) == 1:
            return exact_tr[0]
        same_crate = [c for c in (exact_tr or cands) if c.crate == caller_crate]
        return same_crate[0] if len(same_crate) == 1 else (exact_tr or cands or [None])[0]

    clean = _clean_mir_path(callee_expr)
    if not clean:
        return None
    for cand in (f"{caller_crate}::{clean}", clean):
        items = ws.by_full_path.get(cand)
        if len(items or ()) == 1:
            return items[0]
    parts = clean.split("::")
    if len(parts) >= 2:
        owner, method = parts[-2], parts[-1]
        # Inherent method or qualified path: Type::method or mod::fn
        inh = [
            it for it in ws.items_by_name.get(method, ())
            if not it.is_trait_method and (
                (it.owner_type and (it.owner_type == owner or it.owner_type.endswith(f"::{owner}")))
                or it.full_path.endswith(f"::{clean}")
            )
        ]
        if len(inh) == 1:
            return inh[0]
        same_crate = [it for it in inh if it.crate == caller_crate]
        if len(same_crate) == 1:
            return same_crate[0]
        any_match = [
            it for it in ws.items_by_name.get(method, ())
            if it.full_path.endswith(f"::{clean}")
            or (it.owner_type and (it.owner_type == owner or it.owner_type.endswith(f"::{owner}")))
        ]
        if any_match:
            return any_match[0]
    else:
        cands = [it for it in ws.items_by_name.get(clean, ()) if it.crate == caller_crate and not it.owner_type]
        if len(cands) == 1:
            return cands[0]
        all_free = [it for it in ws.items_by_name.get(clean, ()) if not it.owner_type]
        if len(all_free) == 1:
            return all_free[0]
    return None


def _split_top_level_csv(text: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    par = ang = brk = 0
    for ch in text:
        if ch == "(":
            par += 1
        elif ch == ")":
            par = max(0, par - 1)
        elif ch == "<":
            ang += 1
        elif ch == ">":
            ang = max(0, ang - 1)
        elif ch == "[":
            brk += 1
        elif ch == "]":
            brk = max(0, brk - 1)
        elif ch == "," and par == 0 and ang == 0 and brk == 0:
            seg = "".join(cur).strip()
            if seg:
                parts.append(seg)
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _find_top_level_assign(stmt: str) -> int:
    par = ang = brk = 0
    in_str = False
    escape = False
    for i, ch in enumerate(stmt):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "(":
            par += 1
        elif ch == ")":
            par = max(0, par - 1)
        elif ch == "<":
            ang += 1
        elif ch == ">":
            ang = max(0, ang - 1)
        elif ch == "[":
            brk += 1
        elif ch == "]":
            brk = max(0, brk - 1)
        elif ch == "=" and par == 0 and ang == 0 and brk == 0:
            prev_c = stmt[i - 1] if i > 0 else ""
            next_c = stmt[i + 1] if i + 1 < len(stmt) else ""
            if prev_c not in ("=", "!", "<", ">") and next_c not in ("=", ">"):
                return i
    return -1


def _extract_places(expr: str, aliases: dict[str, str] | None = None) -> set[str]:
    cleaned = re.sub(r'"(?:\\.|[^"\\])*"', '""', expr)
    places: set[str] = set()
    for loc, fld in _MIR_FIELD_PROJ_RE.findall(cleaned):
        places.add(f"{loc}.{fld}")
        places.add(loc)
        if aliases and loc in aliases:
            base = aliases[loc].split(".")[0]
            places.add(f"{base}.{fld}")
            places.add(base)
    for loc, fld in _MIR_PLAIN_FIELD_RE.findall(cleaned):
        places.add(f"{loc}.{fld}")
        places.add(loc)
        if aliases and loc in aliases:
            base = aliases[loc].split(".")[0]
            places.add(f"{base}.{fld}")
            places.add(base)
    for loc in _MIR_LOCAL_RE.findall(cleaned):
        places.add(loc)
        if aliases and loc in aliases:
            places.add(aliases[loc])
    return places


def _extract_lhs_places(lhs: str, aliases: dict[str, str] | None = None) -> tuple[set[str], set[str]]:
    """Return (strong_defs, weak_defs) for an assignment LHS."""
    for loc, fld in _MIR_FIELD_PROJ_RE.findall(lhs) or _MIR_PLAIN_FIELD_RE.findall(lhs):
        defs = {f"{loc}.{fld}"}
        weak = {loc}
        if aliases and loc in aliases:
            base = aliases[loc].split(".")[0]
            defs.add(f"{base}.{fld}")
            weak.add(base)
        return defs, weak
    locs = _MIR_LOCAL_RE.findall(lhs)
    if not locs:
        return set(), set()
    loc = locs[0]
    if lhs.strip().startswith("*") and aliases and loc in aliases:
        target = aliases[loc]
        return set(), {loc, target, target.split(".")[0]}
    return {loc}, set()


def _parse_normal_succs(targets_str: str) -> list[str]:
    # Strip unwind branch so panic cleanup paths don't pollute normal data-flow slices
    head = targets_str.split("unwind", 1)[0]
    out: list[str] = []
    for bb in _MIR_BB_TARGET_RE.findall(head):
        if bb not in out:
            out.append(bb)
    return out


def _parse_mir_stmt(
    stmt_text: str,
    file: str | None,
    l1: int,
    l2: int,
    bb: str,
    idx: int,
    aliases: dict[str, str],
    mut_locals: set[str],
    crate_name: str,
    ws: SemanticWorkspace,
) -> tuple[MirStmt | None, list[str]]:
    """Parse one MIR statement/terminator into MirStmt and its normal CFG successors."""
    s = stmt_text.strip().rstrip(";").strip()
    if not s or s.startswith(("StorageLive(", "StorageDead(", "FakeRead(", "AscribeUserType(", "Retag(")) or s in ("nop", "unreachable", "resume", "abort"):
        return None, []

    succs: list[str] = []
    arrow_idx = s.rfind("-> [")
    head = s
    if arrow_idx >= 0:
        head = s[:arrow_idx].strip()
        succs = _parse_normal_succs(s[arrow_idx + 4:])
    elif s.startswith("goto ->"):
        m_g = _MIR_BB_TARGET_RE.search(s)
        if m_g:
            succs = [m_g.group(1)]
        return None, succs

    if head == "return":
        return MirStmt(bb=bb, idx=idx, file=file, line=l1, end_line=l2, defs=set(), weak_defs=set(), uses={"_0"}, is_return=True), []

    if head.startswith("switchInt("):
        inner = head[len("switchInt("):-1] if head.endswith(")") else head[len("switchInt("):]
        return MirStmt(
            bb=bb, idx=idx, file=file, line=l1, end_line=l2,
            defs=set(), weak_defs=set(), uses=_extract_places(inner, aliases),
            is_switch=True,
        ), succs

    if head.startswith("assert("):
        cond_part = _split_top_level_csv(head[len("assert("):-1] if head.endswith(")") else head)[0:1]
        uses = _extract_places(cond_part[0], aliases) if cond_part else set()
        return MirStmt(
            bb=bb, idx=idx, file=file, line=l1, end_line=l2,
            defs=set(), weak_defs=set(), uses=uses,
        ), succs

    # Call terminator: `[dest = ] callee(args...)` with `-> [...]`
    if arrow_idx >= 0:
        assign_pos = _find_top_level_assign(head)
        lhs = head[:assign_pos].strip() if assign_pos >= 0 else ""
        call_rhs = head[assign_pos + 1:].strip() if assign_pos >= 0 else head
        open_p = call_rhs.rfind("(")
        if open_p > 0 and call_rhs.endswith(")"):
            callee_expr = call_rhs[:open_p].strip()
            args_str = call_rhs[open_p + 1:-1]
            arg_places = [_extract_places(a, aliases) for a in _split_top_level_csv(args_str)]
            uses: set[str] = set()
            weak_defs: set[str] = set()
            for ap in arg_places:
                uses.update(ap)
                for p in ap:
                    if p in mut_locals and p in aliases:
                        weak_defs.add(aliases[p])
                        weak_defs.add(aliases[p].split(".")[0])
            defs, lhs_weak = _extract_lhs_places(lhs, aliases) if lhs else (set(), set())
            weak_defs.update(lhs_weak)
            dest_local = next(iter(defs), None) if len(defs) == 1 else None
            callee_item = _match_mir_callee(callee_expr, crate_name, ws)
            return MirStmt(
                bb=bb, idx=idx, file=file, line=l1, end_line=l2,
                defs=defs, weak_defs=weak_defs, uses=uses,
                call_callee_key=callee_item.key if callee_item else None,
                call_callee_expr=callee_expr,
                call_arg_places=arg_places,
                call_dest=dest_local,
            ), succs

    assign_pos = _find_top_level_assign(head)
    if assign_pos >= 0:
        lhs = head[:assign_pos].strip()
        rhs = head[assign_pos + 1:].strip()
        defs, weak_defs = _extract_lhs_places(lhs, aliases)
        rhs_clean = rhs.strip()
        if len(defs) == 1 and (rhs_clean.startswith("&mut ") or rhs_clean.startswith("&")):
            is_mut = rhs_clean.startswith("&mut ")
            ref_places = _extract_places(rhs_clean, None)
            base = next((p for p in ref_places if "." in p), next(iter(ref_places), None))
            lhs_loc = next(iter(defs))
            if base:
                aliases[lhs_loc] = base
                if is_mut:
                    mut_locals.add(lhs_loc)
        uses = _extract_places(rhs, aliases)
        return MirStmt(
            bb=bb, idx=idx, file=file, line=l1, end_line=l2,
            defs=defs, weak_defs=weak_defs, uses=uses,
        ), succs

    return MirStmt(
        bb=bb, idx=idx, file=file, line=l1, end_line=l2,
        defs=set(), weak_defs=set(), uses=_extract_places(head, aliases),
    ), succs


def _build_mir_deps(mf: MirFunction) -> None:
    """Compute reaching definitions and control-dependence edges for one MirFunction."""
    if mf.built_deps or not mf.block_order:
        return
    mf.built_deps = True
    entry_bb = mf.block_order[0]
    preds: dict[str, list[str]] = {b: [] for b in mf.block_order}
    for b in mf.block_order:
        for s in mf.succs.get(b, ()):
            if s in preds and b not in preds[s]:
                preds[s].append(b)

    entry_defs: dict[str, set[Any]] = {
        f"_{p}": {("__PARAM__", p)} for p in range(1, mf.param_count + 1)
    }
    in_map: dict[str, dict[str, set[Any]]] = {b: {} for b in mf.block_order}
    out_map: dict[str, dict[str, set[Any]]] = {b: {} for b in mf.block_order}
    in_map[entry_bb] = {k: set(v) for k, v in entry_defs.items()}

    def _apply_stmt(cur: dict[str, set[Any]], stmt: MirStmt, sid: tuple[str, int]) -> None:
        for d in stmt.defs:
            if "." not in d:
                for k in list(cur.keys()):
                    if k == d or k.startswith(f"{d}."):
                        del cur[k]
            cur[d] = {sid}
        for wd in stmt.weak_defs:
            cur.setdefault(wd, set()).add(sid)

    work = list(mf.block_order)
    in_work = set(work)
    while work:
        b = work.pop(0)
        in_work.discard(b)
        merged: dict[str, set[Any]] = {k: set(v) for k, v in entry_defs.items()} if b == entry_bb else {}
        for p in preds.get(b, ()):
            for place, defs in out_map[p].items():
                merged.setdefault(place, set()).update(defs)
        in_map[b] = merged
        cur = {k: set(v) for k, v in merged.items()}
        for sid in mf.block_stmts.get(b, ()):
            _apply_stmt(cur, mf.stmts[sid], sid)
        if cur != out_map[b]:
            out_map[b] = cur
            for s in mf.succs.get(b, ()):
                if s in in_map and s not in in_work:
                    work.append(s)
                    in_work.add(s)

    # Link uses to reaching definitions
    for b in mf.block_order:
        cur = {k: set(v) for k, v in in_map[b].items()}
        for sid in mf.block_stmts.get(b, ()):
            stmt = mf.stmts[sid]
            for u in stmt.uses:
                rdefs = cur.get(u)
                if not rdefs and "." in u:
                    rdefs = cur.get(u.split(".")[0])
                for d in rdefs or ():
                    if isinstance(d, tuple) and len(d) == 2 and d[0] == "__PARAM__":
                        mf.backward_deps.setdefault(sid, set()).add(d)
                    elif d != sid:
                        mf.backward_deps.setdefault(sid, set()).add(d)
                        mf.forward_deps.setdefault(d, set()).add(sid)
            _apply_stmt(cur, stmt, sid)

    # Control-dependence via iterative post-dominators (links guarded branches to switchInt)
    all_bbs = set(mf.block_order)
    postdom: dict[str, set[str]] = {}
    for b in mf.block_order:
        normal_s = [s for s in mf.succs.get(b, ()) if s in all_bbs]
        postdom[b] = {b} if not normal_s else set(all_bbs)
    changed = True
    while changed:
        changed = False
        for b in reversed(mf.block_order):
            normal_s = [s for s in mf.succs.get(b, ()) if s in all_bbs]
            if not normal_s:
                continue
            inter = set.intersection(*(postdom[s] for s in normal_s))
            new_pd = {b} | inter
            if new_pd != postdom[b]:
                postdom[b] = new_pd
                changed = True

    for b in mf.block_order:
        sids = mf.block_stmts.get(b, ())
        if not sids:
            continue
        term_id = sids[-1]
        if not mf.stmts[term_id].is_switch:
            continue
        normal_s = [s for s in mf.succs.get(b, ()) if s in all_bbs]
        if len(set(normal_s)) < 2:
            continue
        pd_b = postdom[b]
        q = [s for s in normal_s if s not in pd_b]
        controlled: set[str] = set()
        while q:
            nb = q.pop(0)
            if nb in controlled or nb in pd_b:
                continue
            controlled.add(nb)
            for nxt in mf.succs.get(nb, ()):
                if nxt in all_bbs and nxt not in pd_b and nxt not in controlled:
                    q.append(nxt)
        for cb in controlled:
            for csid in mf.block_stmts.get(cb, ()):
                mf.backward_deps.setdefault(csid, set()).add(term_id)
                mf.forward_deps.setdefault(term_id, set()).add(csid)


def parse_mir_edges_and_functions(
    mir_text: str, crate_name: str, ws: SemanticWorkspace
) -> tuple[list[SemanticCallEdge], list[MirFunction]]:
    """Parse Stage 2 call edges and Stage 3 span-annotated MirFunctions from MIR text."""
    from collections import Counter

    edges: list[SemanticCallEdge] = []
    mir_fns: list[MirFunction] = []
    seen_pairs: set[tuple[str, str]] = set()
    cur_caller: SemanticItem | None = None
    cur_fn: MirFunction | None = None
    cur_bb: str | None = None
    cur_aliases: dict[str, str] = {}
    cur_muts: set[str] = set()
    in_ctfe = False
    brace_depth = 0
    stmt_buf: list[str] = []

    def _flush_stmt():
        nonlocal stmt_buf
        if not stmt_buf or (cur_caller is None and cur_fn is None) or cur_bb is None:
            stmt_buf = []
            return
        raw_stmt = " ".join(stmt_buf).strip()
        stmt_buf = []
        m_span = _MIR_SPAN_RE.search(raw_stmt)
        norm_file: str | None = None
        l1 = l2 = 0
        body = raw_stmt
        if m_span:
            body = raw_stmt[:m_span.start()].rstrip()
            norm_file = _norm_repo_file(ws.root, m_span.group(1).strip())
            l1, l2 = int(m_span.group(2)), int(m_span.group(4))
        idx = len(cur_fn.block_stmts.get(cur_bb, ())) if cur_fn is not None else 0
        parsed_stmt, succs = _parse_mir_stmt(
            body, norm_file, l1, l2, cur_bb, idx, cur_aliases, cur_muts, crate_name, ws
        )
        if cur_caller is not None and parsed_stmt is not None and parsed_stmt.call_callee_key:
            callee_item = ws.items.get(parsed_stmt.call_callee_key)
            if callee_item is not None and (cur_caller.key, callee_item.key) not in seen_pairs:
                seen_pairs.add((cur_caller.key, callee_item.key))
                edges.append(SemanticCallEdge(
                    caller_key=cur_caller.key,
                    caller_full_path=cur_caller.full_path,
                    caller_name=cur_caller.name,
                    caller_file=cur_caller.file,
                    caller_line=cur_caller.line,
                    callee_key=callee_item.key,
                    callee_full_path=callee_item.full_path,
                    callee_name=callee_item.name,
                    callee_file=callee_item.file,
                    callee_line=callee_item.line,
                    callee_expr=parsed_stmt.call_callee_expr or callee_item.full_path,
                ))
        if cur_fn is not None:
            if succs:
                cur_fn.succs.setdefault(cur_bb, []).extend(s for s in succs if s not in cur_fn.succs.get(cur_bb, ()))
            if parsed_stmt is not None:
                sid = (cur_bb, idx)
                cur_fn.stmts[sid] = parsed_stmt
                cur_fn.block_stmts.setdefault(cur_bb, []).append(sid)

    for line in mir_text.splitlines():
        stripped = line.strip()
        if stripped == "// MIR FOR CTFE":
            in_ctfe = True
            continue
        if brace_depth == 0:
            m_fn = _MIR_FN_RE.match(stripped)
            if m_fn:
                if in_ctfe:
                    brace_depth = 1
                    continue
                hdr_name = m_fn.group(1).strip()
                is_closure = bool(re.search(r"::\{(?:closure|async_block|coroutine)#\d+\}", hdr_name))
                lookup_hdr = re.sub(r"::\{(?:closure|async_block|coroutine)#\d+\}.*$", "", hdr_name) if is_closure else hdr_name
                m_imp = _MIR_IMPL_SPAN_RE.search(lookup_hdr)
                imp_norm_file = _norm_repo_file(ws.root, m_imp.group(1)) if m_imp else None
                if m_imp and imp_norm_file is None:
                    cur_caller = None
                    cur_fn = None
                    cur_aliases = {}
                    cur_muts = set()
                    cur_bb = None
                    stmt_buf = []
                    brace_depth = 1
                    continue
                cur_caller = _match_mir_caller(lookup_hdr, crate_name, ws)
                if cur_caller is not None and not is_closure:
                    cur_fn = MirFunction(
                        item_key=cur_caller.key,
                        full_path=cur_caller.full_path,
                        name=cur_caller.name,
                        file=cur_caller.file,
                        line=cur_caller.line,
                        end_line=cur_caller.end_line,
                        param_count=len(_split_top_level_csv(m_fn.group(2))),
                    )
                elif not is_closure:
                    clean_hdr = _clean_mir_path(lookup_hdr)
                    fb_name = (m_imp.group(4) if m_imp else (clean_hdr.split("::")[-1] if clean_hdr else ""))
                    if fb_name and not fb_name.startswith("{"):
                        cur_fn = MirFunction(
                            item_key="",
                            full_path=f"{crate_name}::{clean_hdr}" if clean_hdr else fb_name,
                            name=fb_name,
                            file=imp_norm_file or "",
                            line=0,
                            end_line=0,
                            param_count=len(_split_top_level_csv(m_fn.group(2))),
                        )
                    else:
                        cur_fn = None
                else:
                    cur_fn = None
                cur_aliases = {}
                cur_muts = set()
                cur_bb = None
                stmt_buf = []
                brace_depth = 1
                continue
            in_ctfe = False
            continue

        if in_ctfe:
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                brace_depth = 0
                in_ctfe = False
            continue

        if cur_caller is not None or cur_fn is not None:
            if cur_bb is None:
                if cur_fn is not None:
                    m_dbg = _MIR_DEBUG_RE.match(stripped)
                    if m_dbg:
                        vname, loc = m_dbg.groups()
                        cur_fn.debug_vars.setdefault(loc, vname)
                        cur_fn.var_to_locals.setdefault(vname, set()).add(loc)
                m_bb = _MIR_BB_RE.match(stripped)
                if m_bb:
                    cur_bb = m_bb.group(1)
                    if cur_fn is not None:
                        if cur_bb not in cur_fn.block_order:
                            cur_fn.block_order.append(cur_bb)
                        cur_fn.block_stmts.setdefault(cur_bb, [])
                        cur_fn.succs.setdefault(cur_bb, [])
            else:
                if stripped.startswith("//"):
                    pass  # skip multi-line ConstOperand annotations
                elif stripped == "}":
                    _flush_stmt()
                    cur_bb = None
                else:
                    stmt_buf.append(stripped)
                    if ";" in stripped.split("//", 1)[0]:
                        _flush_stmt()

        brace_depth += stripped.count("{") - stripped.count("}")
        if brace_depth <= 0:
            _flush_stmt()
            if cur_fn is not None and cur_fn.stmts:
                if not cur_fn.item_key:
                    repo_stmts = [st for st in cur_fn.stmts.values() if st.file and st.line > 0]
                    if repo_stmts:
                        if not cur_fn.file:
                            cur_fn.file = Counter(st.file for st in repo_stmts).most_common(1)[0][0]
                        same_f = [st for st in repo_stmts if st.file == cur_fn.file] or repo_stmts
                        if not same_f[0].file:
                            cur_fn = None
                        else:
                            if not cur_fn.file:
                                cur_fn.file = same_f[0].file
                            cur_fn.line = min(st.line for st in same_f)
                            cur_fn.end_line = max(max(st.line, st.end_line) for st in same_f)
                            cur_fn.item_key = f"mir:{cur_fn.file}:{cur_fn.line}:{cur_fn.name}"
                            mir_fns.append(cur_fn)
                else:
                    mir_fns.append(cur_fn)
            brace_depth = 0
            cur_caller = None
            cur_fn = None
            cur_bb = None

    return edges, mir_fns


def parse_mir_edges(mir_text: str, crate_name: str, ws: SemanticWorkspace) -> list[SemanticCallEdge]:
    """Extract caller -> callee edges from `rustc -Zunpretty=mir` and join with Stage 1 items."""
    edges, _ = parse_mir_edges_and_functions(mir_text, crate_name, ws)
    return edges


def slice_mir_dataflow(
    ws: SemanticWorkspace,
    target_items: list[SemanticItem],
    *,
    direction: str = "backward",
    var: str = "",
    target_line: int | None = None,
    hops: int = 1,
    max_lines: int = 35,
) -> dict[str, Any]:
    """Compute intra- and 1-hop inter-procedural MIR def-use slice for target item(s)."""
    dir_norm = direction if direction in ("backward", "forward", "both") else "backward"
    hops = max(0, min(2, hops))
    root_mf: MirFunction | None = None
    for it in target_items:
        if it.key in ws.mir_functions:
            root_mf = ws.mir_functions[it.key]
            break
    if root_mf is None and target_items:
        tfile = target_items[0].file
        tname = target_items[0].name
        r_start = target_items[0].line
        r_end = max(r_start, target_items[0].end_line)
        if target_line is not None:
            cands = [mf for mf in ws.mir_functions.values() if mf.file == tfile and mf.line <= target_line <= mf.end_line]
            if cands:
                cands.sort(key=lambda m: (0 if m.name == tname else 1, m.end_line - m.line))
                root_mf = cands[0]
        if root_mf is None:
            cands = [
                mf for mf in ws.mir_functions.values()
                if mf.file == tfile and mf.line <= r_end and r_start <= mf.end_line
            ]
            if cands:
                cands.sort(key=lambda m: (0 if m.name == tname else 1, abs(m.line - r_start), m.end_line - m.line))
                root_mf = cands[0]
    if root_mf is None:
        return {"precision": "lexical", "reason": "MIR body unavailable for target"}

    _build_mir_deps(root_mf)
    if not any(s.file for s in root_mf.stmts.values()):
        return {"precision": "lexical", "reason": "MIR lacked source spans"}

    file_cache: dict[str, list[str]] = {}

    def _get_file_lines(fpath: str) -> list[str]:
        if fpath not in file_cache:
            try:
                file_cache[fpath] = (ws.root / fpath).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                file_cache[fpath] = []
        return file_cache[fpath]

    var_locals: set[str] = set()
    var_tail = ""
    if var:
        var_base = var.split(".")[0].strip()
        var_tail = var.split(".")[-1].strip()
        var_locals.update(root_mf.var_to_locals.get(var_base, ()))
        if var_tail and var_tail != var_base:
            var_locals.update(root_mf.var_to_locals.get(var_tail, ()))

    def _seed_stmts(mf: MirFunction) -> list[tuple[str, int]]:
        _build_mir_deps(mf)
        seeds: list[tuple[str, int]] = []
        field_hits: list[tuple[str, int]] = []
        if var_tail and not var_locals:
            v_re = re.compile(rf"\b{re.escape(var_tail)}\b")
            for sid, st in mf.stmts.items():
                if st.file and st.line > 0:
                    flines = _get_file_lines(st.file)
                    if 1 <= st.line <= len(flines) and v_re.search(flines[st.line - 1]):
                        field_hits.append(sid)
        if target_line is not None:
            line_hits = [
                sid for sid, st in mf.stmts.items()
                if st.file == mf.file and st.line <= target_line <= max(st.line, st.end_line)
            ]
            if var_locals:
                v_hits = [sid for sid in line_hits if (mf.stmts[sid].defs | mf.stmts[sid].uses) & var_locals]
                if v_hits:
                    return v_hits
            if field_hits:
                f_set = set(field_hits)
                lf_hits = [sid for sid in line_hits if sid in f_set]
                if lf_hits:
                    return lf_hits
            if line_hits:
                return line_hits
        if var_locals:
            v_hits = [sid for sid, st in mf.stmts.items() if (st.defs | st.uses) & var_locals]
            if v_hits:
                return v_hits
        if field_hits:
            return field_hits
        if dir_norm in ("backward", "both"):
            seeds.extend(
                sid for sid, st in mf.stmts.items()
                if st.is_return or "_0" in st.defs or bool(st.weak_defs)
            )
        if dir_norm in ("forward", "both"):
            for sid, deps in mf.backward_deps.items():
                if any(isinstance(d, tuple) and d[0] == "__PARAM__" for d in deps):
                    seeds.append(sid)
        return seeds or list(mf.stmts.keys())[:2]

    seed_sids = _seed_stmts(root_mf)
    visited: set[tuple[str, tuple[str, int]]] = set()
    collected: dict[tuple[str, int], tuple[str, set[str], int, int]] = {}  # (file, line) -> (fn_name, vars, hop, min_dist)
    cross_hops: list[dict[str, Any]] = []
    seen_cross: set[tuple[str, str, int, str]] = set()

    def _record_stmt(mf: MirFunction, sid: tuple[str, int], hop: int, dist: int) -> None:
        st = mf.stmts.get(sid)
        if st is None or not st.file or st.line <= 0:
            return
        vnames: set[str] = set()
        for p in st.defs | st.uses:
            base = p.split(".")[0]
            if base in mf.debug_vars:
                vnames.add(mf.debug_vars[base])
        key = (st.file, st.line)
        prev = collected.get(key)
        if prev is None:
            collected[key] = (mf.name, vnames, hop, dist)
        else:
            prev[1].update(vnames)
            if hop < prev[2] or dist < prev[3]:
                collected[key] = (prev[0], prev[1], min(prev[2], hop), min(prev[3], dist))

    def _run_dir(one_dir: str) -> None:
        q: list[tuple[MirFunction, tuple[str, int], int, set[str] | None, int]] = [
            (root_mf, sid, 0, None, 0) for sid in seed_sids
        ]
        while q:
            mf, sid, hop, trig, dist = q.pop(0)
            vkey = (mf.item_key, sid)
            if vkey in visited:
                continue
            visited.add(vkey)
            _build_mir_deps(mf)
            _record_stmt(mf, sid, hop, dist)
            st = mf.stmts.get(sid)
            if st is None:
                continue

            if one_dir == "backward":
                for dep in mf.backward_deps.get(sid, ()):
                    if isinstance(dep, tuple) and len(dep) == 2 and dep[0] == "__PARAM__":
                        if hop < hops:
                            p_idx = int(dep[1])
                            for edge in ws.callers_by_callee.get(mf.item_key, ())[:3]:
                                caller_mf = ws.mir_functions.get(edge.caller_key)
                                if caller_mf is None:
                                    continue
                                _build_mir_deps(caller_mf)
                                for csid, cst in caller_mf.stmts.items():
                                    if cst.call_callee_key == mf.item_key:
                                        ck = (mf.full_path, caller_mf.full_path, cst.line, "caller_arg")
                                        if ck not in seen_cross:
                                            seen_cross.add(ck)
                                            cross_hops.append({"from": mf.full_path, "to": caller_mf.full_path, "line": cst.line, "relation": "caller_arg"})
                                        _record_stmt(caller_mf, csid, hop + 1, dist + 1)
                                        arg_p = cst.call_arg_places[p_idx - 1] if 1 <= p_idx <= len(cst.call_arg_places) else set()
                                        arg_defs = [
                                            d for d in caller_mf.backward_deps.get(csid, ())
                                            if not (isinstance(d, tuple) and d[0] == "__PARAM__")
                                            and arg_p and ((caller_mf.stmts[d].defs | caller_mf.stmts[d].weak_defs) & arg_p)
                                        ]
                                        if arg_defs:
                                            for d in arg_defs:
                                                q.append((caller_mf, d, hop + 1, None, dist + 1))
                                        else:
                                            q.append((caller_mf, csid, hop + 1, None, dist + 1))
                    else:
                        q.append((mf, dep, hop, None, dist + 1))
                if hop < hops and st.call_callee_key and st.call_callee_key in ws.mir_functions:
                    callee_mf = ws.mir_functions[st.call_callee_key]
                    _build_mir_deps(callee_mf)
                    ck = (mf.full_path, callee_mf.full_path, st.line, "callee_return")
                    if ck not in seen_cross:
                        seen_cross.add(ck)
                        cross_hops.append({"from": mf.full_path, "to": callee_mf.full_path, "line": st.line, "relation": "callee_return"})
                    for csid, cst in callee_mf.stmts.items():
                        if cst.is_return or "_0" in cst.defs:
                            q.append((callee_mf, csid, hop + 1, None, dist + 1))
            else:
                active_defs = st.defs | st.weak_defs | (st.uses if trig is None else set())
                for nxt in mf.forward_deps.get(sid, ()):
                    nst = mf.stmts.get(nxt)
                    ntrig = (active_defs & nst.uses) if nst is not None else active_defs
                    q.append((mf, nxt, hop, ntrig or active_defs, dist + 1))
                if hop < hops and st.call_callee_key and st.call_callee_key in ws.mir_functions:
                    callee_mf = ws.mir_functions[st.call_callee_key]
                    _build_mir_deps(callee_mf)
                    used_places = trig if trig is not None else (st.defs | st.uses)
                    matched_params = {
                        i + 1 for i, ap in enumerate(st.call_arg_places)
                        if ap & used_places
                    }
                    if not matched_params and st.call_arg_places:
                        matched_params = {1}
                    ck = (mf.full_path, callee_mf.full_path, st.line, "callee_param")
                    if ck not in seen_cross:
                        seen_cross.add(ck)
                        cross_hops.append({"from": mf.full_path, "to": callee_mf.full_path, "line": st.line, "relation": "callee_param"})
                    for csid, deps in callee_mf.backward_deps.items():
                        if any(isinstance(d, tuple) and d[0] == "__PARAM__" and d[1] in matched_params for d in deps):
                            q.append((callee_mf, csid, hop + 1, {f"_{p}" for p in matched_params}, dist + 1))
                if hop < hops and (st.is_return or "_0" in st.defs or any(d.split(".")[0] in {f"_{p}" for p in range(1, mf.param_count + 1)} for d in st.defs)):
                    for edge in ws.callers_by_callee.get(mf.item_key, ())[:3]:
                        caller_mf = ws.mir_functions.get(edge.caller_key)
                        if caller_mf is None:
                            continue
                        _build_mir_deps(caller_mf)
                        for csid, cst in caller_mf.stmts.items():
                            if cst.call_callee_key == mf.item_key:
                                ck = (mf.full_path, caller_mf.full_path, cst.line, "caller_return")
                                if ck not in seen_cross:
                                    seen_cross.add(ck)
                                    cross_hops.append({"from": mf.full_path, "to": caller_mf.full_path, "line": cst.line, "relation": "caller_return"})
                                _record_stmt(caller_mf, csid, hop + 1, dist + 1)
                                ctrig = ({cst.call_dest} if cst.call_dest else set()) | cst.weak_defs
                                for nxt in caller_mf.forward_deps.get(csid, ()):
                                    q.append((caller_mf, nxt, hop + 1, ctrig or None, dist + 1))

    direction_note = ""
    if dir_norm == "backward":
        _run_dir("backward")
        param_locals = {f"_{p}" for p in range(1, root_mf.param_count + 1)}
        param_only_var = (
            bool(var)
            and bool(var_locals)
            and var_locals <= param_locals
            and not any((st.defs | st.weak_defs) & var_locals for st in root_mf.stmts.values())
        )
        if param_only_var and sum(1 for v in collected.values() if v[2] == 0) <= 3:
            visited.clear()
            _run_dir("forward")
            direction_note = "input parameter has no intra-function defs; expanded forward"
    else:
        if dir_norm == "both":
            _run_dir("backward")
            visited.clear()
        _run_dir("forward")

    all_keys = list(collected.keys())
    if len(all_keys) <= max_lines:
        selected_keys = sorted(all_keys, key=lambda x: (collected[x][2], x[0], x[1]))
    else:
        hop_ge1 = sorted(
            [k for k in all_keys if collected[k][2] >= 1],
            key=lambda x: (collected[x][2], collected[x][3], x[0], x[1]),
        )
        chosen: list[tuple[str, int]] = []
        chosen_set: set[tuple[str, int]] = set()
        if hop_ge1:
            cross_quota = min(len(hop_ge1), max(2, max_lines // 4))
            seen_cross_fn: set[tuple[str, str]] = set()
            for k in hop_ge1:
                fn_tag = (k[0], collected[k][0])
                if fn_tag not in seen_cross_fn:
                    seen_cross_fn.add(fn_tag)
                    chosen.append(k)
                    chosen_set.add(k)
                    if len(chosen) >= cross_quota:
                        break
            for k in hop_ge1:
                if len(chosen) >= cross_quota:
                    break
                if k not in chosen_set:
                    chosen.append(k)
                    chosen_set.add(k)

        def _intra_rank(k: tuple[str, int]) -> tuple[int, int, int, int, str, int]:
            fn_name, _v, hop_val, dist_val = collected[k]
            same_file = k[0] == root_mf.file
            near_target = 0 if (same_file and target_line is not None and abs(k[1] - target_line) <= 5) else 1
            line_dist = abs(k[1] - target_line) if (same_file and target_line is not None) else 0
            return (dist_val, near_target, hop_val, line_dist, k[0], k[1])

        for k in sorted(all_keys, key=_intra_rank):
            if len(chosen) >= max_lines:
                break
            if k not in chosen_set:
                chosen.append(k)
                chosen_set.add(k)
        selected_keys = sorted(chosen, key=lambda x: (collected[x][2], x[0], x[1]))

    slice_lines: list[dict[str, Any]] = []
    for (fpath, lnum) in selected_keys:
        flines = _get_file_lines(fpath)
        code_txt = flines[lnum - 1].strip()[:120] if 1 <= lnum <= len(flines) else ""
        fn_name, vnames, hop, _dist = collected[(fpath, lnum)]
        slice_lines.append({
            "path": fpath,
            "line": lnum,
            "function": fn_name,
            "vars": sorted(vnames)[:6],
            "hop": hop,
            "code": code_txt,
        })

    out: dict[str, Any] = {
        "direction": dir_norm,
        "var": var or None,
        "precision": "compiler_mir_slice",
        "root_function": root_mf.full_path,
        "slice_lines": slice_lines,
        "cross_function_hops": cross_hops[:8],
    }
    if direction_note:
        out["direction_note"] = direction_note
    return out
