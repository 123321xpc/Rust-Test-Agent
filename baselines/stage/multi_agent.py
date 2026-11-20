"""Adaptive multi-agent orchestration for Rust repository-level issue resolution.

Architecture:
  Layer 0 (Deterministic): Rust CKG, rustc diagnostics + auto-lookup
  Agent 1 (Navigator & Reproducer): Issue contract extraction, 1-3 fault chains, Red test
  Adaptive Gate (AdaCoder-inspired): Fast Path (simple single-file) vs Planned Path (complex/multi-file/trait)
  Agent 2 (Rust Architect / Suggester): Clean-context root-cause arbitration & Rust contract RepairPlan
  Agent 3 (Rust Fixer): Minimal edit_file loop + compiler/test verification + bounded failure escalation
  Agent 4 (Patch QA & Verifier): Deterministic Red-Green/post-strip/hygiene gate + independent LLM audit
"""

import functools
import hashlib
import os
import json
import re
import time
import difflib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# File-based loading also supports agent.py and isolated trajectory replay.
import importlib.util
_units_spec = importlib.util.spec_from_file_location("test_repair_units", Path(__file__).with_name("repair_units.py"))
_repair_units = importlib.util.module_from_spec(_units_spec)
_units_spec.loader.exec_module(_repair_units)
RepairUnitSpec = _repair_units.RepairUnitSpec
_candidates_spec = importlib.util.spec_from_file_location("test_repair_candidates", Path(__file__).with_name("repair_candidates.py"))
_repair_candidates = importlib.util.module_from_spec(_candidates_spec)
_candidates_spec.loader.exec_module(_repair_candidates)
RepairCandidate = _repair_candidates.RepairCandidate
_impact_spec = importlib.util.spec_from_file_location("test_rust_impact", Path(__file__).with_name("rust_impact.py"))
_rust_impact = importlib.util.module_from_spec(_impact_spec)
_impact_spec.loader.exec_module(_rust_impact)
_tools_spec = importlib.util.spec_from_file_location("test_tools_helpers", Path(__file__).with_name("tools.py"))
_tools_helpers = importlib.util.module_from_spec(_tools_spec)
_tools_spec.loader.exec_module(_tools_helpers)
_inline_test_line_ranges = _tools_helpers.inline_test_line_ranges
_touched_line_hits_inline_tests = _tools_helpers.touched_line_hits_inline_tests


class _StageBudgetReached(Exception):
    def __init__(self, role: str, used: int, cap: int):
        super().__init__(f"Stage '{role}' reached its budget cap ({used}/{cap})")
        self.role = role
        self.used = used
        self.cap = cap


class _StageReserveReached(Exception):
    def __init__(self, role: str, remaining: int, reserve: int):
        super().__init__(f"Global API reserve reached during '{role}' (remaining={remaining}, reserve={reserve})")
        self.role = role
        self.remaining = remaining
        self.reserve = reserve


class _AutoEscalate(Exception):
    def __init__(self, reason: str, detail: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


_TRIVIAL_BACKTICKS = frozenset({
    "true", "false", "none", "some", "ok", "err", "self", "mut", "pub", "fn",
    "struct", "enum", "trait", "impl", "use", "mod", "crate", "super", "let",
    "cargo", "rustc", "rust", "test", "tests", "lib", "main", "src", "str",
    "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize",
    "f32", "f64", "bool", "char", "vec", "string", "option", "result",
})


def _strip_fenced_code_blocks(text: str) -> str:
    return re.sub(r"```[\s\S]*?```", "", text or "")


def _strip_inline_commands(text: str) -> str:
    """Remove fenced blocks, URLs, and multi-token inline shell commands before scanning CLI flags."""
    cleaned = _strip_fenced_code_blocks(text or "")
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    return re.sub(
        r"\x60([^\x60\n]+)\x60",
        lambda m: (
            " "
            if (re.search(r"[\s|;&]", m.group(1).strip()) and not m.group(1).strip().startswith("--"))
            else m.group(0)
        ),
        cleaned,
    )


def _extract_cli_flags(text: str) -> list[str]:
    """Extract unique CLI long flags (`--flag`) outside fenced blocks, URLs, and shell commands."""
    cleaned = _strip_inline_commands(text)
    return list(dict.fromkeys(
        re.findall(r"(?<![A-Za-z0-9_-])(--[a-z][a-z0-9_-]*)(?!\.\d|[A-Za-z0-9_-])", cleaned)
    ))


def _extract_requirement_checklist(
    issue: str,
    new_api_candidates: list[str],
    companion_api_hints: list[str],
    expected_result_excerpt: str,
    cli_flags: list[str] | None = None,
) -> list[dict[str, str]]:
    """Decompose multi-part issue expectations into up to 5 orthogonal requirement items."""
    clean_issue = _strip_fenced_code_blocks(issue or "")
    items: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    def _add_req(kind: str, target: str, summary: str) -> None:
        clean_sum = " ".join((summary or "").split())[:160]
        key = (target or clean_sum[:60]).strip().lower()
        if not key or key in seen_keys or len(items) >= 5:
            return
        seen_keys.add(key)
        items.append({
            "req_id": f"req_{len(items) + 1}",
            "kind": kind,
            "target": target,
            "summary": clean_sum,
        })

    for sym in new_api_candidates:
        _add_req("api", sym, f"Implement and verify requested API `{sym}`")

    if any("is_empty" in h for h in companion_api_hints):
        _add_req("companion_api", "is_empty", "Check/add companion `is_empty(&self) -> bool` alongside `len`")

    for m in re.finditer(
        r"(?i)\bboth\s+\x60([A-Za-z_][A-Za-z0-9_:]*)\x60\s*(?:and|&|,)\s*\x60([A-Za-z_][A-Za-z0-9_:]*)\x60"
        r"|\b(?:add|implement|introduce|expose|provide|support)\s+(?:[A-Za-z0-9_.-]+\s+){0,4}\x60([A-Za-z_][A-Za-z0-9_:]*)\x60\s*(?:and|,)\s*\x60([A-Za-z_][A-Za-z0-9_:]*)\x60",
        clean_issue,
    ):
        for g in (m.group(1), m.group(2), m.group(3), m.group(4)):
            if g and g.lower() not in _TRIVIAL_BACKTICKS:
                _add_req("sibling_variant", g, f"Cover paired variant/symbol `{g}`")

    flags = _extract_cli_flags(issue or "") if cli_flags is None else list(cli_flags)
    if len(flags) >= 2 or (len(flags) == 1 and items):
        for flag in flags[:3]:
            _add_req("cli_flag", flag, f"Verify `{flag}` CLI behavior and combinations")

    bullet_src = _strip_fenced_code_blocks(expected_result_excerpt) or clean_issue
    for raw_line in bullet_src.splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)]|-\s*\[[ xX]\])\s+(.+)$", raw_line)
        if not m:
            continue
        line = " ".join(m.group(1).split())
        if not (14 <= len(line) <= 180):
            continue
        if not (re.search(r"\b(?:should|must|needs?\s+to|add|support|return|handle|emit|panic|error|reject|allow|preserve|both)\b", line, re.I) or "`" in line or "--" in line):
            continue
        target_m = re.search(
            r"\x60([A-Za-z_][A-Za-z0-9_:]*)\x60|(?<![A-Za-z0-9_-])(--[a-z][a-z0-9_-]*)(?!\.\d|[A-Za-z0-9_-])",
            _strip_inline_commands(line),
        )
        target = (target_m.group(1) or target_m.group(2)) if target_m else ""
        if target.lower() in _TRIVIAL_BACKTICKS:
            target = ""
        _add_req("requirement", target, line)

    return items


def _audit_target_proxy_coverage(issue_contract: dict[str, Any], repro_contents: str) -> dict[str, Any]:
    """Check whether temporary regression tests cover decomposed requirement checklist items."""
    checklist = issue_contract.get("requirement_checklist") or []
    text = repro_contents or ""
    test_fns = re.findall(r"#\[(?:[A-Za-z0-9_]+::)*test\]\s*(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    missing_targets: list[str] = []
    if len(checklist) >= 2 and text.strip():
        low_text = text.lower()
        for item in checklist:
            tgt = str(item.get("target") or "").strip()
            if not tgt:
                continue
            if tgt.startswith("--"):
                bare = tgt.lstrip("-")
                variants = (tgt.lower(), bare.lower(), bare.replace("-", "_").lower())
            else:
                leaf = tgt.split("::")[-1]
                variants = (tgt.lower(), leaf.lower()) if leaf else (tgt.lower(),)
            if not any(v in low_text for v in variants) and tgt not in missing_targets:
                missing_targets.append(tgt)
    return {
        "checklist_count": len(checklist),
        "test_fn_count": len(test_fns),
        "test_fns": test_fns[:8],
        "missing_targets": missing_targets,
        "monolithic_warning": bool(len(checklist) >= 2 and text.strip() and len(test_fns) <= 1),
    }


def extract_issue_contract(issue: str) -> dict[str, Any]:
    """Deterministically extract requested symbols, API candidates, and Rust complexity signals."""
    backticked = re.findall(r"\x60([A-Za-z_][A-Za-z0-9_:]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)\x60", issue or "")
    requested_symbols: list[str] = []
    seen: set[str] = set()
    for item in backticked:
        clean = item.strip(":")
        if not clean or clean.lower() in _TRIVIAL_BACKTICKS:
            continue
        if clean not in seen:
            seen.add(clean)
            requested_symbols.append(clean)

    # Identify explicitly requested new API names (e.g., "Add timestamp_micros to Utc")
    new_api_candidates: list[str] = []
    add_patterns = (
        r"(?i)\b(?:add|implement|introduce|expose|provide|support)\s+(?:a\s+|an\s+|the\s+)?(?:new\s+)?(?:public\s+)?(?:method|fn|function|trait|struct|enum|type|field|variant|api)?\s*\x60([A-Za-z_][A-Za-z0-9_:]*)\x60",
        r"(?i)\x60([A-Za-z_][A-Za-z0-9_]*)\x60\s+(?:method|function|fn|trait|struct|enum)\s+(?:should|must|needs?\s+to)\s+(?:be\s+)?(?:added|implemented|exposed)",
    )
    for pat in add_patterns:
        for m in re.finditer(pat, issue or ""):
            sym = m.group(1).split("::")[-1].strip()
            if sym and sym.lower() not in _TRIVIAL_BACKTICKS and sym not in new_api_candidates:
                new_api_candidates.append(sym)

    low = (issue or "").lower()
    mentions_trait_or_lifetime = bool(
        re.search(r"\b(?:trait|lifetime|where\s+[a-z]:|dyn\s+[a-z_]|send\s*\+\s*sync|asref|borrow<)\b", low)
        or re.search(r"<\s*[A-Z][A-Za-z0-9_]*\s*:\s*[A-Z]", issue or "")
        or re.search(r"(?:<|&\s*(?:mut\s+)?|\blifetime\s+)'(?!static\b|_+\b)[a-z_][a-z0-9_]*\b(?!')", issue or "")
        or re.search(r"(?<![A-Za-z0-9_])'(?!static\b|_+\b)[a-z_][a-z0-9_]*\s*(?::|\+)", issue or "")
    )
    mentions_macro = bool(re.search(r"\b(?:macro|proc_macro|derive\(|macro_rules)\b", low))
    mentions_new_api = bool(
        new_api_candidates
        or re.search(r"\b(?:add\s+(?:method|function|support|api|a\s+new)|feature\s+request|new\s+method|missing\s+method)\b", low)
    )
    mentions_feature_flag = bool(re.search(r"\b(?:feature\s+flag|cfg\(|--features|--all-features|no_default_features|no_std)\b", low))
    cli_flags = _extract_cli_flags(issue or "")
    mentions_cli_or_additive_flag = bool(
        cli_flags
        or re.search(r"(?:\banother\s+(?:argument|flag|option)\b|\bnew\s+(?:flag|option|argument|field)\b)", low)
    )
    mentions_test_change = bool(
        re.search(r"\b(?:test|tests|snapshot|snapshots|fixture|fixtures|assert(?:ion)?|golden)\b", low)
        and re.search(r"\b(?:add|update|fix|change|modify|refresh|regression|expected|failing|remove)\b", low)
    )
    requests_documentation_change = bool(re.search(
        r"\b(?:add|update|fix|correct|revise|clarify|rewrite|improve)\s+"
        r"(?:(?:the|a|an|existing|project|public|relevant)\s+){0,4}"
        r"(?:documentation|docs?|readme|rustdoc|doc(?:umentation)?\s+comments?)\b"
        r"|\b(?:documentation|docs?|readme|rustdoc|doc\s+comments?)\s+"
        r"(?:should|must|needs?)\s+(?:to\s+)?(?:be\s+)?"
        r"(?:added|updated|fixed|corrected|revised|clarified|rewritten|improved)\b",
        low,
    ))
    mentions_documentation = requests_documentation_change or bool(
        re.search(r"\b(?:documentation|docs?|readme|changelog|release\s+notes?|rustdoc|doc\s+comment)\b", low)
        and re.search(r"\b(?:add|update|fix|change|modify|refresh|document|correct|write)\b", low)
    )
    expected_m = re.search(
        r"(?ims)^#{1,4}\s*(?:expected(?:\s+result|\s+behavior)?|proposed(?:\s+fix|\s+solution)?|requirements?)\b.*?(?=^#{1,4}\s|\Z)",
        issue or "",
    )
    expected_result_excerpt = expected_m.group(0).strip()[:600] if expected_m else ""

    referenced_files: list[str] = []
    seen_files: set[str] = set()
    for m in re.finditer(r"(?:blob/[0-9a-fA-F]{7,40}/|(?<![A-Za-z0-9_./-]))([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.rs)\b", issue or ""):
        fpath = m.group(1).lstrip("./")
        if fpath and fpath not in seen_files and not _is_non_prod_path(fpath):
            seen_files.add(fpath)
            referenced_files.append(fpath)

    companion_api_hints: list[str] = []
    if (
        "len" in new_api_candidates
        or (mentions_new_api and any(s.split("::")[-1] == "len" for s in requested_symbols))
    ):
        companion_api_hints.append(
            "When adding `pub fn len(&self)`, check whether `pub fn is_empty(&self) -> bool` is also expected by Rust/Clippy conventions."
        )
    requirement_checklist = _extract_requirement_checklist(
        issue or "",
        new_api_candidates,
        companion_api_hints,
        expected_result_excerpt,
        cli_flags=cli_flags,
    )

    return {
        "requested_symbols": requested_symbols[:20],
        "new_api_candidates": new_api_candidates[:10],
        "companion_api_hints": companion_api_hints,
        "requirement_checklist": requirement_checklist,
        "referenced_files": referenced_files[:10],
        "expected_result_excerpt": expected_result_excerpt,
        "signals": {
            "mentions_trait_or_lifetime": mentions_trait_or_lifetime,
            "mentions_macro": mentions_macro,
            "mentions_new_api": mentions_new_api,
            "mentions_feature_flag": mentions_feature_flag,
            "mentions_cli_or_additive_flag": mentions_cli_or_additive_flag,
            "mentions_test_change": mentions_test_change,
            "mentions_documentation": mentions_documentation,
            "requests_documentation_change": requests_documentation_change,
        },
    }


def allocate_stage_budgets(max_api_calls: int | None) -> dict[str, int | None]:
    """Allocate per-stage caps and final QA reserve from a single global API_LIMIT."""
    if max_api_calls is None:
        return {
            "total": None,
            "navigator_cap": 30,
            "suggester_cap": 8,
            "fixer_attempt_cap": 60,
            "verifier_cap": 5,
            "verifier_reserve": 5,
        }
    if max_api_calls <= 6:
        return {
            "total": max_api_calls,
            "navigator_cap": max(1, max_api_calls // 2),
            "suggester_cap": 1,
            "fixer_attempt_cap": max(1, max_api_calls - 1),
            "verifier_cap": 1,
            "verifier_reserve": 0,
        }
    if max_api_calls <= 20:
        reserve = 2
        return {
            "total": max_api_calls,
            "navigator_cap": max(4, int(max_api_calls * 0.30)),
            "suggester_cap": 3,
            "fixer_attempt_cap": max(4, int(max_api_calls * 0.45)),
            "verifier_cap": 2,
            "verifier_reserve": reserve,
        }
    reserve = max(3, min(5, max_api_calls // 25))
    return {
        "total": max_api_calls,
        "navigator_cap": max(8, min(32, int(max_api_calls * 0.22))),
        "suggester_cap": max(4, min(8, int(max_api_calls * 0.08))),
        "fixer_attempt_cap": max(14, int(max_api_calls * 0.38)),
        "verifier_cap": max(3, min(8, reserve + 3)),
        "verifier_reserve": reserve,
    }


def _client_calls_made(client: Any) -> int:
    val = getattr(client, "stage_calls_made", None)
    if not isinstance(val, int) or isinstance(val, bool):
        val = getattr(client, "calls_made", None)
    return val if isinstance(val, int) and not isinstance(val, bool) else 0


def _client_remaining_calls(client: Any) -> int | None:
    val = getattr(client, "remaining_calls", None)
    return val if isinstance(val, int) and not isinstance(val, bool) else None


def _is_non_prod_path(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/").lstrip("./")
    return (
        norm.startswith(("tests/", "benches/", "examples/"))
        or "/tests/" in norm
        or "/benches/" in norm
        or "/examples/" in norm
    )


def route_after_navigation(blackboard: dict[str, Any], state: Any) -> tuple[str, list[str]]:
    """AdaCoder-inspired adaptive gate: choose 'fast' vs 'planned' path."""
    bundle = blackboard.get("localization") or {}
    contract = blackboard.get("issue_contract") or {}
    signals = contract.get("signals") or {}
    chains = bundle.get("candidate_chains") or []
    reasons: list[str] = []

    distinct_files: set[str] = set()
    root_files: set[str] = set()
    for ch in chains:
        for node in ch.get("nodes", []):
            p = node.get("path")
            if not p or _is_non_prod_path(p):
                continue
            distinct_files.add(p)
            nrole = (node.get("role") or "").strip().lower()
            if nrole in ("root_candidate", "modify"):
                root_files.add(p)

    effective_files = root_files if root_files else distinct_files

    if bundle.get("needs_architecture_planning"):
        reasons.append("navigator requested architecture planning")
    if len(chains) > 1 and len(distinct_files) > 1:
        reasons.append(f"multiple candidate fault chains ({len(chains)})")
    if bundle.get("unresolved_chain_ids") and bundle.get("ambiguity_reason"):
        reasons.append("navigator could not distinguish root-cause chains")
    if len(effective_files) > 2:
        reasons.append(f"repair spans {len(effective_files)} candidate files")
    for sig_key, sig_label in (
        ("mentions_new_api", "issue requests a new public API"),
        ("mentions_cli_or_additive_flag", "issue adds or modifies CLI/additive options"),
        ("mentions_trait_or_lifetime", "issue involves trait/lifetime/generic bounds"),
        ("mentions_macro", "issue involves macros"),
    ):
        if signals.get(sig_key):
            reasons.append(sig_label)

    ckg = getattr(state, "ckg", None)
    if ckg is not None and effective_files and not reasons:
        try:
            surface = _rust_impact.build_impact_surface(ckg, sorted(effective_files))
            extra_consumers = [
                p for p in (surface.get("consumer_paths") or [])
                if p not in effective_files and not _is_non_prod_path(p)
            ]
            if len(extra_consumers) >= 2:
                reasons.append(f"CKG impact surface spans {len(extra_consumers)} external consumer file(s)")
            elif len(surface.get("trait_impls") or []) >= 2:
                reasons.append("CKG impact surface spans multiple trait/impl sites")
        except Exception:
            pass

    # Record downstream workspace dependents on blackboard for Fixer awareness without forcing Planned Path.
    if effective_files and getattr(state, "_find_downstream_workspace_packages", None):
        try:
            root = state.root.resolve()
            manifests = set()
            for rel_f in effective_files:
                cur = (root / rel_f).parent
                while cur != root.parent:
                    if (cur / "Cargo.toml").is_file():
                        manifests.add((cur / "Cargo.toml").resolve())
                        break
                    cur = cur.parent
            if manifests:
                deps = state._find_downstream_workspace_packages(list(manifests))
                if deps:
                    blackboard["downstream_dependents"] = deps
        except Exception:
            pass

    if reasons:
        return "planned", reasons
    return "fast", ["localized repair with at most two candidate files"]


def _build_snippet_context(root: Path, chains: list[dict[str, Any]], max_chars: int = 7500) -> str:
    """Pre-load compact line-numbered snippets around candidate chain nodes for clean Suggester handoff."""
    root = Path(root).resolve()
    blocks: list[str] = []
    seen_ranges: set[tuple[str, int, int]] = set()
    total = 0

    for ch in chains:
        for node in ch.get("nodes", []):
            rel = node.get("path")
            if not rel:
                continue
            fpath = (root / rel).resolve()
            if not fpath.is_file() or not fpath.is_relative_to(root):
                continue
            l_start = max(1, int(node.get("line_start") or node.get("line") or 1) - 8)
            l_end = int(node.get("line_end") or node.get("line") or l_start) + 25
            if l_end - l_start > 70:
                l_end = l_start + 70
            key = (rel, l_start, l_end)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            l_end = min(len(lines), l_end)
            excerpt = "\n".join(f"{ln}: {lines[ln - 1]}" for ln in range(l_start, l_end + 1))
            block = f"--- {rel}:{l_start}-{l_end} ({node.get('symbol', '')}) ---\n{excerpt}"
            if total + len(block) > max_chars:
                return "\n\n".join(blocks)
            blocks.append(block)
            total += len(block)
    return "\n\n".join(blocks)


def _priority_paths_from_blackboard(blackboard: dict[str, Any] | None) -> set[str]:
    """Identify core issue-referenced and root_candidate paths to retain in shared memory."""
    if not isinstance(blackboard, dict):
        return set()
    priorities: set[str] = set()
    contract = blackboard.get("issue_contract") or {}
    for rf in contract.get("referenced_files") or []:
        if rf:
            priorities.add(str(rf).replace("\\", "/"))
    loc = blackboard.get("localization") or {}
    for ch in (loc.get("candidate_chains") or loc.get("ckg_suggested_chains") or []):
        for node in ch.get("nodes") or []:
            p = node.get("path")
            if p and (str(node.get("role") or "").strip().lower() in ("root_candidate", "modify") or not priorities):
                priorities.add(str(p).replace("\\", "/"))
    return priorities


def _build_shared_context(root: Path, blackboard: dict[str, Any], chains: list[dict[str, Any]], max_chars: int = 12500) -> str:
    """Combine chain node excerpts with cached read excerpts & CKG notes to eliminate duplicate file reads."""
    parts: list[str] = []
    used = 0

    chain_ctx = _build_snippet_context(root, chains, max_chars=min(6500, max_chars // 2))
    if chain_ctx:
        parts.append(chain_ctx)
        used += len(chain_ctx)

    shared_reads = blackboard.get("shared_read_excerpts") or {}
    priority_paths = _priority_paths_from_blackboard(blackboard)
    ordered_reads = sorted(
        shared_reads.items(),
        key=lambda kv: (0 if kv[0].split(":", 1)[0].replace("\\", "/") in priority_paths else 1),
    )
    for key, text_val in ordered_reads:
        if not text_val:
            continue
        block = f"--- Cached Read ({key}) ---\n{text_val}"
        if used + len(block) > max_chars - 1500:
            break
        if text_val[:80] in chain_ctx:
            continue
        parts.append(block)
        used += len(block)

    ckg_notes = blackboard.get("shared_ckg_notes") or {}
    for tgt, note_val in ckg_notes.items():
        if not note_val:
            continue
        block = f"--- CKG Neighborhood ({tgt}) ---\n{note_val}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


def _compact_handoff_text(value: Any, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def build_handoff_card(
    issue_contract: dict[str, Any],
    blackboard: dict[str, Any],
    *,
    det_qa: dict[str, Any] | None = None,
    qa_feedback: str = "",
    failure_route: str = "",
    active_unit: dict[str, Any] | None = None,
) -> str:
    """Build a compact five-line Fixer handoff from structured state.

    Scopes edits/constraints to the active repair unit when present so large plans
    never lose downstream contract obligations via fixed-index truncation.
    """
    loc = blackboard.get("localization") or {}
    plan = blackboard.get("repair_plan") or {}
    chains = loc.get("candidate_chains") or loc.get("ckg_suggested_chains") or []
    root_paths = []
    for chain in chains[:3]:
        for node in (chain.get("nodes") or [])[:3]:
            path = node.get("path")
            if path and path not in root_paths:
                root_paths.append(path)

    unit_ledger = blackboard.get("repair_units") or {}
    verified_units = [
        u.get("unit_id") for u in (unit_ledger.get("units") or [])
        if u.get("status") == "verified" and u.get("unit_id")
    ]
    all_edits = plan.get("ordered_edits") or []
    unit_work = (active_unit or {}).get("work") or []
    if unit_work:
        selected_edits = unit_work[:10]
        extra_edits = max(0, len(all_edits) - len(selected_edits))
    else:
        priority_set = _priority_paths_from_blackboard(blackboard) | set(root_paths)
        prioritized = sorted(
            all_edits,
            key=lambda e: (0 if str(e.get("file") or "").replace("\\", "/") in priority_set else 1),
        )
        selected_edits = prioritized[:8]
        extra_edits = max(0, len(all_edits) - len(selected_edits))

    edits = [
        _compact_handoff_text({
            "file": edit.get("file"),
            "symbol": edit.get("symbol"),
            "action": edit.get("action"),
            "description": edit.get("description"),
        }, 260)
        for edit in selected_edits
    ]
    if extra_edits:
        edits.append(f"(+{extra_edits} more plan edit(s) in other units)")

    api_items: list[str] = []
    for item in (
        list(issue_contract.get("requested_symbols") or [])
        + list(issue_contract.get("new_api_candidates") or [])
        + list(plan.get("api_checklist") or [])
    ):
        s = str(item).strip()
        if s and s not in api_items:
            api_items.append(s)
    contract = ", ".join(api_items) or "no named API"
    expected = _compact_handoff_text(issue_contract.get("expected_result_excerpt"), 260) or "follow the issue's stated behavior"
    checklist = issue_contract.get("requirement_checklist") or []
    req_suffix = ""
    if len(checklist) >= 2:
        req_parts = [f"{r.get('req_id')}:{r.get('target') or _compact_handoff_text(r.get('summary'), 48)}" for r in checklist[:5]]
        req_suffix = f"; reqs=[{', '.join(req_parts)}]"[:220]

    unit_paths = set((active_unit or {}).get("paths") or [])
    raw_invariants = list(plan.get("rust_invariants") or [])
    for hint in issue_contract.get("companion_api_hints") or []:
        if hint not in raw_invariants:
            raw_invariants.append(hint)
    raw_forbidden = list(plan.get("forbidden_edits") or [])
    if unit_paths:
        raw_invariants.sort(key=lambda x: (0 if any(p in str(x) for p in unit_paths) else 1))
        raw_forbidden.sort(key=lambda x: (0 if any(p in str(x) for p in unit_paths) else 1))
    invariants = "; ".join(_compact_handoff_text(x, 180) for x in raw_invariants[:6]) or "preserve unrelated behavior"
    forbidden = "; ".join(_compact_handoff_text(x, 160) for x in raw_forbidden[:6]) or "none recorded"

    qa = _compact_handoff_text(qa_feedback, 420) or "none"
    if plan.get("_qa_conflicts"):
        qa += f"; plan_conflicts_superseded_by_qa={_compact_handoff_text(plan['_qa_conflicts'], 220)}"
    if det_qa:
        qa = _compact_handoff_text(
            f"deterministic_ok={det_qa.get('ok')}; missing={det_qa.get('missing_contract_apis') or []}; "
            f"doctest_shifts={(det_qa.get('hygiene') or {}).get('doctest_shifts') or []}; feedback={qa}",
            520,
        )
    unit_tag = ""
    if active_unit and active_unit.get("unit_id"):
        unit_tag = f" unit={active_unit['unit_id']}"
        if active_unit.get("expected_outcome"):
            unit_tag += f" (expect={_compact_handoff_text(active_unit['expected_outcome'], 140)})"
    if verified_units:
        unit_tag += f" verified_units={verified_units}"

    return "\n".join([
        f"1. Contract: APIs={contract}; expected={expected}{req_suffix}",
        f"2. Root: paths={', '.join(root_paths[:6]) or 'relocalize from issue'}; summary={_compact_handoff_text(loc.get('summary'), 260)}",
        f"3. Plan: chain={plan.get('selected_chain_id') or 'direct'}{unit_tag}; edits={' | '.join(edits) or 'minimal root-cause edit'}",
        f"4. Constraints: invariants={_compact_handoff_text(invariants, 520)}; forbidden={_compact_handoff_text(forbidden, 420)}",
        f"5. QA/route: {qa}; failure_route={failure_route or 'normal'}",
    ])


def _classify_failure_route(reason: str, stuck_status: dict[str, Any] | None = None) -> str:
    """Classify a failed repair for targeted Navigator or Architect routing."""
    text = (reason or "") + " " + json.dumps(stuck_status or {}, ensure_ascii=False)
    low = text.lower()
    localization_signals = (
        "wrong file", "wrong module", "wrong symbol", "unrelated file", "out of scope",
        "localization", "mislocalized", "candidate chain", "no relevant source",
    )
    trait_signals = (
        "trait", "lifetime", "borrow", "type mismatch", "bound", "associated type",
        "method takes", "expected .* argument", "compiler error",
    )
    if any(signal in low for signal in localization_signals):
        return "navigator"
    if any(signal in low for signal in trait_signals):
        return "architect"
    return "architect"


def _store_excerpt_with_priority(
    store: dict[str, str], key: str, value: str, *, limit: int, priority_paths: set[str],
) -> None:
    if len(store) >= limit and key not in store:
        evict = next(
            (rk for rk in store if rk.split(":", 1)[0].replace("\\", "/") not in priority_paths),
            next(iter(store)),
        )
        store.pop(evict, None)
    store[key] = value


def _merge_activity_into_blackboard(blackboard: dict[str, Any], activity: dict[str, Any]) -> None:
    reads = blackboard.setdefault("shared_read_excerpts", {})
    priority_paths = _priority_paths_from_blackboard(blackboard)
    for k, v in (activity.get("read_excerpts") or {}).items():
        _store_excerpt_with_priority(reads, k, v, limit=14, priority_paths=priority_paths)
    ckg = blackboard.setdefault("shared_ckg_notes", {})
    for k, v in (activity.get("ckg_notes") or {}).items():
        if len(ckg) >= 8 and k not in ckg:
            ckg.pop(next(iter(ckg)), None)
        ckg[k] = v


def _invalidate_cached_path(blackboard: dict[str, Any], activity: dict[str, Any], rel_path: str) -> None:
    for store in (blackboard.get("shared_read_excerpts"), activity.get("read_excerpts")):
        if not isinstance(store, dict):
            continue
        for k in [k for k in store if k == rel_path or k.startswith(rel_path + ":")]:
            store.pop(k, None)


_MILESTONE_KEYS = (
    "m1_reproduced",
    "m2_compiled",
    "m3_regression_green",
    "m4_smoke_passed",
    "m5_qa_approved",
)


def _compute_milestone_state(
    state: Any,
    *,
    det_qa: dict[str, Any] | None = None,
    ver_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute deterministic milestone flags and monotonic progress score."""
    selected = _extract_selected_source_paths(state) if hasattr(state, "changed_paths") else []
    stuck = state.stuck_status() if hasattr(state, "stuck_status") else {}
    check_errs = stuck.get("last_check_errors") or []
    failed_tests = (stuck.get("last_test_failure") or {}).get("failed_tests") or []
    repro_clean = bool(state.reproduced_cleanly()) if hasattr(state, "reproduced_cleanly") else False
    repro_any = bool(state.reproduced()) if hasattr(state, "reproduced") else False
    m1 = repro_clean or (repro_any and not getattr(state, "require_reproduction", True))
    if det_qa is not None:
        m2 = bool(selected) and det_qa.get("post_strip_status") != "failed" and not check_errs
    else:
        m2 = bool(selected) and not check_errs
    m3 = bool(state.verified()) if hasattr(state, "verified") else False
    smoke_passed = sum(
        1 for item in ((det_qa or {}).get("existing_test_smoke") or [])
        if isinstance(item, dict) and item.get("status") == "passed"
    )
    m4 = bool(det_qa and det_qa.get("ok") and smoke_passed > 0)
    m5 = bool(ver_payload and str(ver_payload.get("decision") or "").upper() == "APPROVE")
    flags = {
        "m1_reproduced": m1,
        "m2_compiled": m2,
        "m3_regression_green": m3,
        "m4_smoke_passed": m4,
        "m5_qa_approved": m5,
    }
    achieved = sum(1 for k in _MILESTONE_KEYS if flags[k])
    score = (
        int(m1),
        int(m2),
        int(m3),
        int(m4),
        int(m5),
        smoke_passed,
        -len(check_errs),
        -len(failed_tests),
    )
    return {
        "flags": flags,
        "achieved": achieved,
        "total": len(_MILESTONE_KEYS),
        "progress_score": list(score),
    }


def _update_milestones_and_kpi(
    ctx: SimpleNamespace,
    role: str,
    *,
    det_qa: dict[str, Any] | None = None,
    ver_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update milestone states and attribute newly achieved milestones to `role`."""
    cur = _compute_milestone_state(ctx.state, det_qa=det_qa, ver_payload=ver_payload)
    prev = ctx.blackboard.get("milestones") or {k: False for k in _MILESTONE_KEYS}
    credits = ctx.blackboard.setdefault("milestone_credits", {})
    newly_achieved = [k for k in _MILESTONE_KEYS if cur["flags"][k] and not prev.get(k)]
    if newly_achieved:
        credits[role] = int(credits.get(role, 0)) + len(newly_achieved)
        ctx.blackboard.setdefault("milestone_history", []).append({
            "role": role,
            "newly_achieved": newly_achieved,
            "achieved_total": cur["achieved"],
        })
    merged = {k: bool(prev.get(k) or cur["flags"][k]) if k == "m1_reproduced" else cur["flags"][k] for k in _MILESTONE_KEYS}
    ctx.blackboard["milestones"] = merged
    cur["flags"] = merged
    cur["achieved"] = sum(1 for k in _MILESTONE_KEYS if merged[k])
    return cur


def _should_prune_suggester_replan(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Prune redundant -> suggester loops when prior replans produced zero milestone progress."""
    replan_count = int(ctx.blackboard.get("replan_count") or 0)
    if replan_count < 1:
        return False, ""
    if ctx.candidate_policy and ctx.candidate_policy.data.get("pending"):
        return False, ""
    lock_rep = ctx.state.locked_regression_report() if hasattr(ctx.state, "locked_regression_report") else {}
    if lock_rep.get("violations") and lock_rep.get("locked"):
        return False, ""
    last_score = ctx.blackboard.get("last_replan_score")
    if not last_score:
        return False, ""
    cur_score = _compute_milestone_state(ctx.state)["progress_score"]
    if tuple(cur_score) <= tuple(last_score):
        zero_runs = int(ctx.blackboard.get("zero_progress_replans") or 0) + 1
        ctx.blackboard["zero_progress_replans"] = zero_runs
        if zero_runs >= 1:
            return True, f"Zero milestone progress after {replan_count} replan(s); pruning redundant architect loop"
    else:
        ctx.blackboard["zero_progress_replans"] = 0
    return False, ""


def _append_cognitive_entry(
    ctx: SimpleNamespace,
    *,
    source: str,
    actual: str,
    expected: str = "",
    falsified: str = "",
) -> None:
    """Append an Expected vs Actual reflection record for Cognitive Evolving Planning."""
    ledger = ctx.blackboard.setdefault("cognitive_ledger", [])
    changed = [p.as_posix() for p in _extract_selected_source_paths(ctx.state)] if hasattr(ctx.state, "changed_paths") else []
    exp_text = expected or str(ctx.blackboard.get("last_plan_expectation") or "resolve root cause and pass regression/existing tests")
    entry = {
        "step": len(ledger) + 1,
        "source": source,
        "changed_files": changed[:6],
        "expected": _compact_handoff_text(exp_text, 160),
        "actual": _compact_handoff_text(actual, 220),
        "falsified": _compact_handoff_text(falsified, 160),
    }
    ledger.append(entry)
    if len(ledger) > 6:
        del ledger[:-6]


def _format_cognitive_ledger(blackboard: dict[str, Any], max_entries: int = 5) -> str:
    ledger = blackboard.get("cognitive_ledger") or []
    if not ledger:
        return ""
    lines = ["Cognitive Evolving Ledger (Expected -> Actual -> Falsified; do NOT repeat falsified attempts):"]
    for item in ledger[-max_entries:]:
        files = ",".join(item.get("changed_files") or []) or "none"
        fal = f" | falsified={item['falsified']}" if item.get("falsified") else ""
        lines.append(
            f"- #{item.get('step')} [{item.get('source')}] files=[{files}] "
            f"expected={item.get('expected')} | actual={item.get('actual')}{fal}"
        )
    return "\n".join(lines)


def _reconcile_plan_with_qa_rejection(blackboard: dict[str, Any], ver_payload: dict[str, Any] | None) -> None:
    """Annotate active RepairPlan with QA rejections so handoff card does not conflict with QA feedback."""
    plan = blackboard.get("repair_plan")
    if not isinstance(plan, dict) or not ver_payload:
        return
    if str(ver_payload.get("plan_consistency") or "").lower() == "conflict":
        plan["_qa_conflicts"] = list(ver_payload.get("plan_conflicts") or [])[:5]
    reqs = list(ver_payload.get("missing_requirements") or []) + list(ver_payload.get("objections") or [])
    if reqs:
        plan["_qa_required_fixes"] = [str(r)[:180] for r in reqs[:6]]


def _refresh_sibling_contracts(ctx: SimpleNamespace) -> str:
    """Probe sibling trait impls and existing test expectations for requested/localized symbols."""
    ckg = getattr(ctx.state, "ckg", None)
    if ckg is None:
        return ""
    loc = ctx.blackboard.get("localization") or {}
    chains = loc.get("candidate_chains") or loc.get("ckg_suggested_chains") or []
    symbols: list[str] = list(ctx.issue_contract.get("requested_symbols") or []) + list(ctx.issue_contract.get("new_api_candidates") or [])
    for ch in chains[:3]:
        for node in (ch.get("nodes") or [])[:4]:
            sym = node.get("symbol")
            if sym and sym not in symbols:
                symbols.append(sym)
    try:
        contracts = _rust_impact.collect_sibling_contracts(ckg, symbols[:10])
        text = _rust_impact.format_sibling_contracts(contracts, max_chars=1200)
    except Exception:
        contracts, text = {"sibling_impls": [], "test_contracts": []}, ""
    ctx.blackboard["sibling_contracts"] = contracts
    ctx.blackboard["sibling_contracts_text"] = text
    return text


def _emit_coordination_metrics(ctx: SimpleNamespace) -> None:
    """Emit coordination KPI, duplicate work, and efficiency metrics."""
    if not ctx.trace:
        return
    milestones = _compute_milestone_state(
        ctx.state,
        ver_payload=((ctx.blackboard.get("qa_history") or [])[-1] if ctx.blackboard.get("qa_history") else None),
    )
    merged = ctx.blackboard.get("milestones") or milestones["flags"]
    achieved = sum(1 for k in _MILESTONE_KEYS if merged.get(k))
    total_m = len(_MILESTONE_KEYS)
    credits = dict(ctx.blackboard.get("milestone_credits") or {})
    roles = ("navigator", "suggester", "fixer", "verifier")
    role_kpi = {r: round(int(credits.get(r, 0)) / total_m, 4) for r in roles}
    overall_kpi = round(sum(role_kpi.values()) / len(roles), 4)
    coord_stats = ctx.blackboard.get("coord_stats") or {}
    downstream_reads = int(coord_stats.get("downstream_reads") or 0)
    duplicate_reads = int(coord_stats.get("duplicate_reads") or 0)
    dup_rate = round(duplicate_reads / downstream_reads, 4) if downstream_reads > 0 else 0.0
    total_trans = int(coord_stats.get("total_transitions") or 0)
    zero_delta_handoffs = int(coord_stats.get("zero_delta_handoffs") or 0)
    t_metrics = ctx.trace.metrics() if hasattr(ctx.trace, "metrics") else {}
    t_metrics = t_metrics if isinstance(t_metrics, dict) else {}
    raw_api = t_metrics.get("api_calls")
    api_calls = int(raw_api) if isinstance(raw_api, (int, float)) else _client_calls_made(ctx.client)
    raw_tok = t_metrics.get("total_tokens")
    total_tokens = int(raw_tok) if isinstance(raw_tok, (int, float)) else 0
    eff_per_call = round(achieved / max(1, api_calls), 4)
    eff_per_100k = round((achieved * 100_000) / total_tokens, 4) if total_tokens > 0 else None
    ctx.trace.record(
        "coordination_metrics",
        milestones=merged,
        milestones_achieved=achieved,
        milestones_total=total_m,
        role_kpi=role_kpi,
        overall_kpi=overall_kpi,
        downstream_reads=downstream_reads,
        duplicate_reads=duplicate_reads,
        duplicate_work_rate=dup_rate,
        total_transitions=total_trans,
        zero_delta_handoffs=zero_delta_handoffs,
        replan_count=int(ctx.blackboard.get("replan_count") or 0),
        pruned_replans=int(ctx.blackboard.get("pruned_replans") or 0),
        coordination_efficiency_per_call=eff_per_call,
        coordination_efficiency_per_100k_tokens=eff_per_100k,
    )


def _prune_unreproduced_regression_targets(state: Any, trace: Any = None) -> list[str]:
    """Remove orphaned/broken standalone regression test files if another clean Red target exists."""
    targets = state.get_regression_targets()
    if len(targets) <= 1:
        return []
    repro_keys = [
        k for k in targets
        if state._is_target_reproduced(k) and (state._reproduced_regressions.get(k) or {}).get("clean_red")
    ]
    dead_keys = [k for k in targets if k not in repro_keys]
    if not repro_keys or not dead_keys:
        return []

    pruned: list[str] = []
    root = state.root.resolve()
    with state._lock:
        for key in dead_keys:
            tinfo = targets.get(key, {})
            if tinfo.get("is_in_module"):
                continue
            rel_path = tinfo.get("path")
            if rel_path is None:
                continue
            fpath = root / rel_path
            try:
                if fpath.is_file() and state.originals.get(rel_path) is None:
                    fpath.unlink()
            except OSError:
                continue
            state.originals.pop(rel_path, None)
            state.regression_tests.discard(rel_path)
            state._reproduced_regressions.pop(key, None)
            state._passed_regressions.discard(key)
            state._passed_regression_configs.pop(key, None)
            state._regression_test_names.pop(key, None)
            state._reproduction_notes.pop(key, None)
            pruned.append(rel_path.as_posix())

    if pruned and trace:
        trace.record("pruned_dead_regression_targets", targets=pruned)
    return pruned


def _extract_selected_source_paths(state: Any) -> list[Path]:
    return [
        p for p in state.changed_paths()
        if (p not in state.regression_tests or p in state.in_module_regressions)
    ]


def _relaxed_source_patch(state: Any, selected: list[Path] | None = None) -> str:
    """Return the current source diff even when optional verification gates failed."""
    selected = selected or _extract_selected_source_paths(state)
    if not selected:
        return ""
    try:
        patch = state.diff_hygiene_report(selected).get("patch", "")
    except Exception:
        patch = ""
    if not patch:
        try:
            patch = state.build_patch(selected)
        except Exception:
            return ""
    if patch:
        state.patch = patch
    return patch


def _check_missing_contract_apis(issue_contract: dict[str, Any], patch_text: str, state: Any) -> list[str]:
    """Check if explicitly requested new API names are missing from added patch lines and source."""
    candidates = issue_contract.get("new_api_candidates") or []
    if not candidates or not patch_text:
        return []
    added_lines = "\n".join(
        line[1:] for line in patch_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    missing: list[str] = []
    for sym in candidates:
        if re.search(rf"\b{re.escape(sym)}\b", added_lines):
            continue
        if getattr(state, "ckg", None) is not None:
            try:
                nb = state.ckg.query_neighborhood(sym, exclude_tests=True)
                if nb.get("definitions"):
                    continue
            except Exception:
                pass
        missing.append(sym)
    return missing


def _detect_added_inline_tests(patch_text: str) -> list[str]:
    """Flag #[cfg(test)], #[test], or new test functions added directly into non-test src/ files."""
    if not patch_text:
        return []
    findings: list[str] = []
    cur_file = ""
    in_mod_tests_hunk = False
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:].strip()
            in_mod_tests_hunk = False
            continue
        if line.startswith("@@ "):
            in_mod_tests_hunk = bool(re.search(r"\bmod\s+(?:tests?|[a-z0-9_]*_tests)\b", line))
            continue
        if not cur_file.endswith(".rs") or "/tests/" in cur_file or cur_file.startswith("tests/"):
            continue
        if cur_file.endswith("_test.rs") or cur_file.endswith("tests.rs"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            body = line[1:].strip()
            if (
                body.startswith("#[cfg(test)]")
                or re.match(r"^#\[(?:[\w:]+::)?test(?:\s*\(.*\))?\]$", body)
                or (in_mod_tests_hunk and re.match(r"^(?:async\s+)?fn\s+test_[A-Za-z0-9_]+\s*\(", body))
            ):
                tag = f"{cur_file}: {body}"
                if tag not in findings:
                    findings.append(tag)
    return findings[:6]


def _detect_modified_inline_tests(patch_text: str, state: Any = None) -> list[str]:
    """Flag hunks in production src/ files that edit pre-existing inline #[cfg(test)] / mod tests functions."""
    if not patch_text:
        return []
    findings: list[str] = []
    cur_file = ""
    in_mod_tests_hunk = False
    hunk_touched: list[tuple[str, int]] = []
    hunk_old_line = 0
    hunk_has_delete = False
    hunk_mentions_test_fn = False
    ranges_cache: dict[str, list[tuple[int, int]]] = {}

    def _in_orig_test_range(fpath: str, touched: list[tuple[str, int]]) -> bool:
        if state is None or not touched:
            return False
        if fpath not in ranges_cache:
            orig_raw = getattr(state, "originals", {}).get(Path(fpath))
            if orig_raw and orig_raw[0] is not None:
                ranges_cache[fpath] = _inline_test_line_ranges(orig_raw[0].decode("utf-8", errors="replace"))
            else:
                ranges_cache[fpath] = []
        return _touched_line_hits_inline_tests(tuple(touched), ranges_cache[fpath])

    def _flush():
        if cur_file and cur_file.endswith(".rs") and not _is_non_prod_path(cur_file):
            if not (cur_file.endswith("_test.rs") or cur_file.endswith("tests.rs")):
                if hunk_has_delete and (in_mod_tests_hunk or hunk_mentions_test_fn
                                       or _in_orig_test_range(cur_file, hunk_touched)):
                    tag = f"{cur_file}: modified existing inline test hunk"
                    if tag not in findings:
                        findings.append(tag)

    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            _flush()
            cur_file = line[6:].strip()
            in_mod_tests_hunk = False
            hunk_touched = []
            hunk_old_line = 0
            hunk_has_delete = False
            hunk_mentions_test_fn = False
            continue
        if line.startswith("@@ "):
            _flush()
            in_mod_tests_hunk = bool(re.search(r"\bmod\s+(?:tests?|[a-z0-9_]*_tests)\b", line))
            m = re.match(r"^@@ -(\d+)(?:,(\d+))?", line)
            hunk_old_line = int(m.group(1)) if m else 0
            hunk_touched = []
            hunk_has_delete = False
            hunk_mentions_test_fn = False
            continue
        if line.startswith("-") and not line.startswith("---"):
            hunk_has_delete = True
            hunk_touched.append(("line", hunk_old_line))
            hunk_old_line += 1
            body = line[1:].strip()
            if "#[test]" in body or re.search(r"\bfn\s+test_[A-Za-z0-9_]+\b", body):
                hunk_mentions_test_fn = True
        elif line.startswith("+") and not line.startswith("+++"):
            gap = hunk_old_line - 1
            if ("gap", gap) not in hunk_touched:
                hunk_touched.append(("gap", gap))
        elif line.startswith(" "):
            hunk_old_line += 1
    _flush()
    return findings[:6]


def _patch_file_changes(patch_text: str) -> dict[str, dict[str, Any]]:
    """Parse changed lines from a unified diff without interpreting code tokens."""
    changes: dict[str, dict[str, Any]] = {}
    old_path = ""
    current = ""
    for line in (patch_text or "").splitlines():
        if line.startswith("--- "):
            old_path = line[4:].split("\t", 1)[0].strip()
            continue
        if line.startswith("+++ "):
            new_path = line[4:].split("\t", 1)[0].strip()
            raw = new_path if new_path != "/dev/null" else old_path
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            current = raw
            changes.setdefault(current, {"added": [], "removed": [], "new": old_path == "/dev/null"})
            if current and new_path == "/dev/null":
                changes[current]["deleted"] = True
            continue
        if not current or line.startswith(("@@ ", "diff ", "index ", "new file ", "deleted file ", "old mode ", "new mode ", "Binary files")):
            if line.startswith("Binary files") and current:
                changes[current]["binary"] = True
            continue
        if line.startswith("+") and not line.startswith("+++"):
            changes[current]["added"].append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            changes[current]["removed"].append(line[1:])
    return changes


def _patch_hunks(patch_text: str):
    """Yield (path, old_start, old_count, has_added, has_removed, touched) for each hunk.

    `touched` holds the OLD-file positions the hunk actually changes, as a tuple of
    ("line", n) for removed lines and ("gap", n) for insertion gaps (n = the old line
    the new content is inserted after). Context lines are excluded.

    Inline-test overlap checks must use this: a hunk whose context merely abuts a
    `#[cfg(test)]` block is not an edit to that block.
    """
    current = ""
    old_path = ""
    hunk: tuple[str, int, int, bool, bool, tuple[tuple[str, int], ...]] | None = None
    old_line = 0

    def flush():
        nonlocal hunk
        if hunk is not None:
            yield hunk
        hunk = None

    for line in (patch_text or "").splitlines():
        if line.startswith("--- "):
            old_path = line[4:].split("\t", 1)[0].strip()
            continue
        if line.startswith("+++ "):
            for item in flush():
                yield item
            new_path = line[4:].split("\t", 1)[0].strip()
            raw = new_path if new_path != "/dev/null" else old_path
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            current = raw
            continue
        if line.startswith("@@ "):
            for item in flush():
                yield item
            match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+", line)
            if match:
                old_line = int(match.group(1))
                hunk = (current, old_line, int(match.group(2) or "1"), False, False, ())
            continue
        if hunk is None:
            continue
        path, start, count, has_added, has_removed, touched = hunk
        if line.startswith("+") and not line.startswith("+++"):
            has_added = True
            gap = old_line - 1
            if ("gap", gap) not in touched:
                touched = touched + (("gap", gap),)
        elif line.startswith("-") and not line.startswith("---"):
            has_removed = True
            touched = touched + (("line", old_line),)
            old_line += 1
        elif line.startswith(" "):
            old_line += 1
        hunk = (path, start, count, has_added, has_removed, touched)
    for item in flush():
        yield item


def _is_documentation_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name.startswith(("readme", "changelog", "history", "news", "contributing")) or name in {
        "license", "copying", "notice",
    } or Path(path).suffix.lower() in {".md", ".markdown", ".rst", ".txt"}


def _is_benchmark_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = Path(normalized).name
    return (
        normalized.startswith("tests/")
        or "/tests/" in normalized
        or name == "tests.rs"
        or name.endswith("_test.rs")
        or name.endswith((".snap", ".snap.new"))
        or "/snapshots/" in normalized
    )


def _rust_line_is_non_substantive(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return stripped.startswith(("//", "/*", "*", "*/", "#[doc", "#![doc", "#[cfg(test)]", "#[test]"))


def _issue_allows_test_or_doc_changes(issue_contract: dict[str, Any] | None) -> tuple[bool, bool]:
    signals = (issue_contract or {}).get("signals") or {}
    return bool(signals.get("mentions_test_change")), bool(signals.get("mentions_documentation"))


def _detect_non_substantive_patch(
    patch_text: str,
    selected: list[Path] | None = None,
    issue_contract: dict[str, Any] | None = None,
) -> list[str]:
    """Reject only docs-only or comment-only Rust diffs; preserve real non-Rust production edits."""
    changes = _patch_file_changes(patch_text)
    if not changes:
        return ["empty diff"]
    _, docs_allowed = _issue_allows_test_or_doc_changes(issue_contract)
    if docs_allowed:
        return []
    rust_paths: list[str] = []
    rust_substantive: list[str] = []
    other_substantive: list[str] = []
    documentation: list[str] = []
    for path, data in changes.items():
        if _is_documentation_path(path):
            documentation.append(path)
            continue
        if path.lower().endswith(".rs"):
            rust_paths.append(path)
            changed_lines = list(data.get("added") or []) + list(data.get("removed") or [])
            if data.get("binary") or any(not _rust_line_is_non_substantive(line) for line in changed_lines):
                rust_substantive.append(path)
        else:
            # Cargo manifests, templates, shell scripts, and other non-doc files are production edits.
            other_substantive.append(path)
    if rust_substantive or other_substantive:
        return []
    if rust_paths:
        return ["Rust changes contain only comments/blank lines: " + ", ".join(rust_paths[:6])]
    if documentation:
        return ["Documentation-only changes: " + ", ".join(documentation[:6])]
    return ["No substantive production-code changes detected"]


def _documentation_only_patch(state: Any, issue_contract: dict[str, Any]) -> bool:
    if not (issue_contract.get("signals") or {}).get("requests_documentation_change"):
        return False
    selected = _extract_selected_source_paths(state)
    if not selected:
        return False
    changes = _patch_file_changes(state.build_patch(selected))
    if not changes:
        return False
    for path, data in changes.items():
        if data.get("binary"):
            return False
        if _is_documentation_path(path):
            continue
        if not path.lower().endswith(".rs"):
            return False
        changed_lines = list(data.get("added") or []) + list(data.get("removed") or [])
        if not changed_lines or any(not _rust_line_is_non_substantive(line) for line in changed_lines):
            return False
    return True


def _unreproduced_qa_allowed(state: Any, reason: str, *, check_original: bool = False) -> bool:
    if len(reason.strip()) < 20:
        return False
    targets = set(state.get_regression_targets().keys())
    if not targets or not targets <= state._passed_regressions:
        return False
    missing_red = [
        key for key in targets
        if not (state._is_target_reproduced(key)
                and (state._reproduced_regressions.get(key) or {}).get("clean_red"))
    ]
    if not missing_red:
        return False
    for key in missing_red:
        if check_original:
            state._reproduction_notes.pop(key, None)
            try:
                state.verify_pre_fix_reproduction([key])
            except ValueError:
                pass
        if state._reproduction_notes.get(key) != "passed on unpatched original code":
            return False
    return True


def _detect_benchmark_test_conflicts(
    patch_text: str,
    state: Any,
    issue_contract: dict[str, Any] | None = None,
) -> list[str]:
    """Find pre-existing test/snapshot edits that can collide with SWE-bench test_patch."""
    allows_tests, _ = _issue_allows_test_or_doc_changes(issue_contract)
    if allows_tests or not patch_text:
        return []
    findings: list[str] = []
    changes = _patch_file_changes(patch_text)
    regression_tests = set(getattr(state, "regression_tests", set()))
    in_module = set(getattr(state, "in_module_regressions", {}))
    originals = getattr(state, "originals", {})
    for path, data in changes.items():
        rel = Path(path)
        if rel in regression_tests and rel not in in_module:
            continue
        if _is_benchmark_test_path(path):
            findings.append(f"{path}: existing test/snapshot file modified")
    for path, start, count, has_added, has_removed, touched in _patch_hunks(patch_text):
        if not path.lower().endswith(".rs") or _is_non_prod_path(path) or _is_benchmark_test_path(path):
            continue
        original = originals.get(Path(path))
        if not original or original[0] is None or not (has_added or has_removed):
            continue
        ranges = _inline_test_line_ranges(original[0].decode("utf-8", errors="replace"))
        if not touched:
            continue
        if _touched_line_hits_inline_tests(touched, ranges):
            findings.append(f"{path}: hunk overlaps a pre-existing inline #[cfg(test)]/mod tests region")
    return list(dict.fromkeys(findings))[:8]


def _classify_smoke_failure(item: dict[str, Any]) -> str:
    if item.get("_from_existing_failure_log"):
        return "unknown"
    declared = str(item.get("classification") or "").strip().lower()
    if declared in {"environment", "code"}:
        return declared
    details = json.dumps(item, ensure_ascii=False).lower()
    if any(token in details for token in (
        "network", "download", "proxy", "connection refused", "could not resolve",
        "failed to select a version", "requires rustc", "no such file or directory",
    )):
        return "environment"
    if any(token in details for token in (
        "error[e", "could not compile", "unresolved import", "mismatched types",
        "expected ", "found ", "no method named", "this function takes", "trait bound",
    )):
        return "code"
    if str(item.get("status", "")).lower() in {"timeout", "error", "no_tests"}:
        return "unknown"
    # A raw `_existing_test_failures` entry has only failure names and no paired
    # baseline; keep it inconclusive rather than attributing a pre-existing
    # failure to the current patch.  Structured smoke results include execution
    # counts (or compile diagnostics) and can be classified as code failures.
    if item.get("tests_executed", 0) or item.get("executed", 0) or item.get("compile_errors"):
        return "code"
    if any(token in details for token in ("test result: failed", "assertion failed", "panicked at", "test failed")):
        return "code"
    return "unknown"


def _smoke_gate_report(state: Any, smoke_results: list[dict[str, Any]] | None = None) -> dict[str, list[str]]:
    """Block confirmed compilation regressions and report other smoke failures."""
    items = [
        {**dict(item), "_from_existing_failure_log": True}
        for item in (getattr(state, "_existing_test_failures", []) or [])
    ]
    if smoke_results:
        items.extend(smoke_results)
    cached = getattr(state, "_existing_test_smoke_cache", None)
    if not smoke_results and cached and len(cached) > 1:
        items.extend(cached[1] or [])
    warnings: list[str] = []
    blocking: list[str] = []
    seen: set[str] = set()
    current_revision = getattr(state, "_revision", None)
    for item in items:
        status = str(item.get("status") or "failed").lower()
        if status not in {"failed", "timeout", "error", "environment_error"}:
            continue
        observed = item.get("observed_revision")
        if observed is not None and current_revision is not None and observed != current_revision:
            continue
        classification = _classify_smoke_failure(item)
        baseline = str(item.get("baseline_status") or "").lower()
        target = item.get("target") or item.get("test_target") or "existing-test-smoke"
        raw_detail = str(item.get("details") or item.get("failure_details") or "")
        detail_lines = [ln for ln in raw_detail.splitlines() if not ln.lstrip().startswith("stored new snapshot ")]
        detail = ("\n".join(detail_lines) if detail_lines else raw_detail).strip().replace("\n", " ")[:420]
        key = f"{target}:{status}:{detail}"
        if key in seen:
            continue
        seen.add(key)
        message = f"{target} status={status} classification={classification} observed_revision={observed}"
        if detail:
            message += f": {detail}"
        new_failures = item.get("new_failed_tests") or []
        if baseline == "failed" and new_failures:
            message += "; newly failing tests: " + ", ".join(new_failures[:10])
        elif classification == "code" and baseline in {"", "unknown", "none"}:
            message += " (no confirmed passing baseline)"
        baseline_label = str(item.get("baseline_label") or "")
        if baseline_label:
            message += f" [{baseline_label}]"
        compile_failure = bool(item.get("compile_errors")) or bool(re.search(
            r"error\[E\d+\]|could not compile", str(item.get("details") or ""), re.IGNORECASE
        ))
        if status == "failed" and classification == "code" and baseline == "passed" and compile_failure:
            blocking.append("New compilation failure: " + message)
        else:
            warnings.append("Smoke check: " + message)
    return {"blocking": blocking, "warnings": warnings}


def _submission_gate_report(
    state: Any,
    issue_contract: dict[str, Any] | None,
    patch_text: str,
    selected: list[Path],
    smoke_results: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    findings = _detect_non_substantive_patch(patch_text, selected, issue_contract)
    findings.extend(_detect_benchmark_test_conflicts(patch_text, state, issue_contract))
    if smoke_results is None and callable(getattr(state, "run_existing_test_smoke", None)):
        try:
            smoke_results = state.run_existing_test_smoke(selected)
        except Exception as exc:
            smoke_results = [{
                "target": "existing-test-smoke", "status": "error", "classification": "unknown",
                "observed_revision": getattr(state, "_revision", None), "details": str(exc),
            }]
    smoke = _smoke_gate_report(state, smoke_results)
    return {
        "blocking": list(dict.fromkeys(findings + smoke["blocking"])),
        "warnings": list(dict.fromkeys(smoke["warnings"])),
    }


def _plan_conflict_explanation_has_issue_basis(explanation: str, issue_contract: dict[str, Any] | None) -> bool:
    text = str(explanation or "").strip()
    if len(text) < 20:
        return False
    low = text.lower()
    contract = issue_contract or {}
    symbols = [str(item).lower() for item in (contract.get("requested_symbols") or []) if len(str(item)) >= 3]
    context_words = ("issue", "contract", "requirement", "expected", "behavior", "requested", "spec", "bug", "fix", "panic")
    evidence_words = ("test", "verification", "caller", "public api", "regression", "compile", "runtime", "cargo", "probe", "diff", "check", "assert", "pass", "fail", "symbol", "impl")
    has_context = any(token in low for token in context_words)
    has_evidence_or_symbol = any(token in low for token in evidence_words) or any(token in low for token in symbols)
    return has_context and has_evidence_or_symbol


def _prune_invalid_regression_targets(state: Any, trace: Any = None) -> list[str]:
    """Remove un-runnable standalone regression files (e.g. virtual-workspace root tests/) that never compile/run."""
    targets = state.get_regression_targets()
    if not targets:
        return []
    root = state.root.resolve()
    pruned: list[str] = []
    with state._lock:
        for key, tinfo in list(targets.items()):
            if tinfo.get("is_in_module"):
                continue
            rel_path = tinfo.get("path")
            if rel_path is None:
                continue
            manifest_abs = root / key[0]
            note = (state._reproduction_notes.get(key) or "").lower()
            is_virtual_ws = False
            if manifest_abs.is_file() and not (manifest_abs.parent / "src").is_dir():
                try:
                    mc = manifest_abs.read_text(encoding="utf-8", errors="replace")
                    if re.search(r"(?m)^\[workspace\]", mc) and not re.search(r"(?m)^\[package\]", mc):
                        is_virtual_ws = True
                except OSError:
                    pass
            if is_virtual_ws or "no test target named" in note:
                fpath = root / rel_path
                try:
                    if fpath.is_file() and state.originals.get(rel_path) is None:
                        fpath.unlink()
                except OSError:
                    continue
                state.originals.pop(rel_path, None)
                state.regression_tests.discard(rel_path)
                state._reproduced_regressions.pop(key, None)
                state._passed_regressions.discard(key)
                state._passed_regression_configs.pop(key, None)
                state._regression_test_names.pop(key, None)
                state._reproduction_notes.pop(key, None)
                pruned.append(rel_path.as_posix())
        remaining_targets = set(state.get_regression_targets().keys())
        if remaining_targets and remaining_targets <= state._passed_regressions:
            repro_ok = (not state.require_reproduction) or all(state._is_target_reproduced(t) for t in remaining_targets)
            if repro_ok:
                state._verified_revision = state._revision
    if pruned and trace:
        trace.record("pruned_invalid_regression_targets", targets=pruned)
    return pruned


def _synthesize_ordered_edits_from_chains(chains: list[dict[str, Any]], max_steps: int = 4) -> list[dict[str, str]]:
    edits: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ch in chains or []:
        hyp = (ch.get("hypothesis") or "").strip()[:180]
        for node in ch.get("nodes", []):
            p = (node.get("path") or "").strip()
            if not p or _is_non_prod_path(p):
                continue
            sym = (node.get("symbol") or "").strip()
            key = (p, sym)
            if key in seen:
                continue
            seen.add(key)
            l_start = node.get("line_start") or 1
            l_end = node.get("line_end") or l_start
            edits.append({
                "file": p,
                "symbol": sym,
                "line_range": f"{l_start}-{l_end}",
                "action": "modify",
                "description": f"Inspect and fix root cause in {sym or p}. {hyp}".strip(),
            })
            if len(edits) >= max_steps:
                return edits
    return edits


def _clean_and_audit_fallback_patch(state: Any, issue_contract: dict[str, Any], patch_text: str, trace: Any = None) -> str:
    """Strip leftover scratch helper scripts and enforce hygiene/inline-test/doctest checks on fallback patches."""
    if not patch_text:
        return ""
    selected = _extract_selected_source_paths(state)
    if not selected:
        return ""
    hygiene = state.diff_hygiene_report(selected)
    scratch = set(hygiene.get("scratch_files") or [])
    if scratch:
        root = state.root.resolve()
        with state._lock:
            for sf in scratch:
                sp = Path(sf)
                if state.originals.get(sp) is None:
                    fpath = root / sp
                    try:
                        if fpath.is_file() or fpath.is_symlink():
                            fpath.unlink()
                    except OSError:
                        pass
                    state.originals.pop(sp, None)
        selected = [p for p in selected if p.as_posix() not in scratch]
        if not selected:
            state.patch = ""
            return ""
        if state.post_strip_check:
            try:
                state.run_post_strip_check(selected)
            except Exception:
                state.patch = ""
                return ""
        hygiene = state.diff_hygiene_report(selected)
        patch_text = hygiene.get("patch", "")
        if trace:
            trace.record("stripped_scratch_from_fallback", scratch=sorted(scratch))
    if (
        not patch_text
        or hygiene.get("added_dbg_lines")
        or _detect_added_inline_tests(patch_text)
    ):
        state.patch = ""
        return ""
    gate = _submission_gate_report(state, issue_contract, patch_text, selected)
    if gate["blocking"]:
        state.patch = ""
        return ""
    if trace and (gate["blocking"] or gate["warnings"]):
        trace.record("submission_gate", path="verified_fallback", blocking=[], warnings=gate["blocking"] + gate["warnings"])
    state.patch = patch_text
    return patch_text


def _detect_comment_only_doctest_shifts(patch_text: str, state: Any, selected: list[Path]) -> list[str]:
    """Flag files where hunks above existing doctests only changed comments/docs/blank lines and shifted line counts."""
    if not patch_text or not selected:
        return []
    root = state.root.resolve()
    doc_line_by_file: dict[str, int] = {}
    for rel in selected:
        if rel.suffix != ".rs":
            continue
        orig_raw = state.originals.get(rel)
        if not orig_raw or orig_raw[0] is None:
            continue
        orig_c = orig_raw[0].decode("utf-8", errors="replace")
        starts = [orig_c[:m.start()].count("\n") + 1 for m in re.finditer(r"(?m)^\s*(?:#\[doc\s*=\s*|///\s*)```", orig_c)]
        if starts:
            doc_line_by_file[rel.as_posix()] = max(starts)

    if not doc_line_by_file:
        return []

    cur_file = ""
    hunks_above: dict[str, list[list[str]]] = {}
    cur_hunk: list[str] | None = None
    cur_hunk_start = 10**9
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            if cur_hunk is not None and cur_file in doc_line_by_file and cur_hunk_start <= doc_line_by_file[cur_file]:
                hunks_above.setdefault(cur_file, []).append(cur_hunk)
            cur_file = line[6:].strip()
            cur_hunk = None
            cur_hunk_start = 10**9
            continue
        if line.startswith("@@ "):
            if cur_hunk is not None and cur_file in doc_line_by_file and cur_hunk_start <= doc_line_by_file[cur_file]:
                hunks_above.setdefault(cur_file, []).append(cur_hunk)
            m = re.match(r"^@@ -(\d+)", line)
            cur_hunk_start = int(m.group(1)) if m else 10**9
            cur_hunk = []
            continue
        if cur_hunk is not None and (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---")):
            cur_hunk.append(line)
    if cur_hunk is not None and cur_file in doc_line_by_file and cur_hunk_start <= doc_line_by_file[cur_file]:
        hunks_above.setdefault(cur_file, []).append(cur_hunk)

    bad: list[str] = []
    for fpath, hlist in hunks_above.items():
        for h in hlist:
            h_delta = sum(1 if ln.startswith("+") else -1 for ln in h)
            if h_delta == 0:
                continue
            all_comment_or_blank = all(
                ln[1:].strip() == "" or ln[1:].lstrip().startswith(("//", "///", "//!", "/*", "*", "*/", "#[doc"))
                for ln in h
            )
            doc_or_blank_delta = sum(
                (1 if ln.startswith("+") else -1)
                for ln in h
                if ln[1:].strip() == "" or ln[1:].lstrip().startswith(("///", "//!", "#[doc"))
            )
            if all_comment_or_blank:
                bad.append(f"{fpath} (delta={h_delta:+d} from doc/comment/blank lines above doctests)")
                break
    return bad


def _is_env_dependency_blocker(text: str) -> bool:
    low = (text or "").lower()
    return (
        "modifying cargo.lock is prohibited" in low
        or "failed to select a version" in low
        or "perhaps a crate was updated and forgotten to be re-vendored" in low
    )


def _prune_secondary_unverified_regression_targets(state: Any, trace: Any = None) -> list[str]:
    """Drop extra unverified standalone regression tests (e.g. late *_tmp.rs files) when a primary test passed."""
    targets = state.get_regression_targets()
    if len(targets) <= 1 or not state._passed_regressions:
        return []
    keep_keys = [
        k for k in targets
        if k in state._passed_regressions and (not state.require_reproduction or state._is_target_reproduced(k))
    ]
    if not keep_keys:
        keep_keys = [k for k in targets if k in state._passed_regressions]
    if not keep_keys:
        return []

    drop_keys = [k for k in targets if k not in keep_keys]
    pruned: list[str] = []
    root = state.root.resolve()
    with state._lock:
        for key in drop_keys:
            tinfo = targets.get(key, {})
            rel_path = tinfo.get("path")
            if rel_path is None:
                continue
            fpath = root / rel_path
            if tinfo.get("is_in_module"):
                try:
                    if fpath.is_file():
                        cur_text = fpath.read_bytes().decode("utf-8", errors="replace")
                        stripped = re.sub(
                            r"(?ms)\r?\n// __TEST_REGRESSION_START: [^\n]+\r?\n.*?// __TEST_REGRESSION_END: [^\n]+(?:\r?\n)?",
                            "",
                            cur_text,
                        )
                        if stripped != cur_text:
                            stripped_bytes = stripped.encode("utf-8")
                            fpath.write_bytes(stripped_bytes)
                            orig_entry = state.originals.get(rel_path)
                            if orig_entry is not None and stripped_bytes == orig_entry[0]:
                                state.originals.pop(rel_path, None)
                except OSError:
                    continue
                state.in_module_regressions.pop(rel_path, None)
                state.regression_tests.discard(rel_path)
                state._reproduced_regressions.pop(key, None)
                state._passed_regressions.discard(key)
                state._passed_regression_configs.pop(key, None)
                state._regression_test_names.pop(key, None)
                state._reproduction_notes.pop(key, None)
                pruned.append(f"{rel_path.as_posix()}::{tinfo.get('name', '')}")
                continue
            try:
                if fpath.is_file() and state.originals.get(rel_path) is None:
                    fpath.unlink()
            except OSError:
                continue
            state.originals.pop(rel_path, None)
            state.regression_tests.discard(rel_path)
            state._reproduced_regressions.pop(key, None)
            state._passed_regressions.discard(key)
            state._passed_regression_configs.pop(key, None)
            state._regression_test_names.pop(key, None)
            state._reproduction_notes.pop(key, None)
            pruned.append(rel_path.as_posix())
        remaining_targets = set(state.get_regression_targets().keys())
        if remaining_targets and remaining_targets <= state._passed_regressions:
            repro_ok = (not state.require_reproduction) or all(state._is_target_reproduced(t) for t in remaining_targets)
            if repro_ok:
                state._verified_revision = state._revision
    if pruned and trace:
        trace.record("pruned_secondary_regression_targets", targets=pruned)
    return pruned


def _detect_static_manifest_tests(state: Any) -> list[str]:
    root = state.root.resolve()
    flagged: list[str] = []
    for _key, info in state.get_regression_targets().items():
        if info.get("is_in_module"):
            continue
        rel_path = info.get("path")
        if rel_path is None:
            continue
        fpath = root / rel_path
        if not fpath.is_file():
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "Cargo.toml" not in content or not ("include_str!" in content or "read_to_string" in content or "fs::read" in content):
            continue
        non_manifest = re.sub(r"Cargo\.toml", "", content)
        if re.search(r"\b(?:Command|cargo_bin|cargo)\b", non_manifest, re.IGNORECASE):
            continue
        cleaned_uses = re.sub(
            r"(?m)^\s*use\s+(?:std|core|alloc|tempfile|toml|serde|anyhow|regex|assert_cmd|predicates)\b[^;]*;",
            "",
            content,
        )
        if re.search(r"(?m)^\s*(?:use|extern\s+crate)\s+[A-Za-z_]\w*", cleaned_uses):
            continue
        flagged.append(rel_path.as_posix())
    return flagged


def _find_untested_integration_targets(state: Any, selected: list[Path]) -> list[str]:
    root = state.root.resolve()
    executed = getattr(state, "_executed_test_targets", []) or []
    ran_targets = {(x.get("manifest"), x.get("target")) for x in executed}
    untested: list[str] = []
    seen_manifests: set[Path] = set()
    for sel in selected:
        cur = (root / sel).parent
        while cur != root.parent:
            mf = cur / "Cargo.toml"
            if mf.is_file():
                seen_manifests.add(mf.resolve())
                break
            cur = cur.parent
    for mf in sorted(seen_manifests):
        rel_mf = mf.relative_to(root).as_posix() if mf.is_relative_to(root) else mf.as_posix()
        if (rel_mf, "tests") in ran_targets or (rel_mf, "default") in ran_targets:
            continue
        tests_dir = mf.parent / "tests"
        if not tests_dir.is_dir():
            continue
        for tf in sorted(tests_dir.glob("*.rs")):
            if tf.name.startswith(("test_regression_", "normal_regression_")):
                continue
            if (rel_mf, tf.stem) not in ran_targets:
                untested.append(f"{rel_mf}:{tf.stem}")
    return untested[:8]


def _try_degraded_patch_fallback(
    state: Any,
    trace: Any = None,
    allow_unverified_compiling: bool = False,
    issue_contract: dict[str, Any] | None = None,
    blackboard: dict[str, Any] | None = None,
) -> str:
    """Recover verified/compiling snapshots when possible, enforce hygiene, and record verification gaps."""
    if getattr(state, "_best_verified_snapshot", None) is not None and hasattr(state, "auto_submit_verified_patch"):
        try:
            verified = state.auto_submit_verified_patch(allow_snapshot_fallback=True)
            if verified:
                cleaned = _clean_and_audit_fallback_patch(state, issue_contract or {}, verified, trace=trace)
                if cleaned:
                    return cleaned
        except Exception:
            pass

    from_compiling_snapshot = False
    patch_text = _relaxed_source_patch(state)
    if (
        not patch_text
        and allow_unverified_compiling
        and getattr(state, "_last_compiling_prod_snapshot", None) is not None
        and hasattr(state, "_capture_snapshot")
        and hasattr(state, "_restore_snapshot")
    ):
        cur_snap = state._capture_snapshot()
        try:
            state._restore_snapshot(state._last_compiling_prod_snapshot)
            selected = _extract_selected_source_paths(state)
            if selected:
                if getattr(state, "post_strip_check", False):
                    state.run_post_strip_check(selected)
                patch_text = _relaxed_source_patch(state, selected)
                from_compiling_snapshot = bool(patch_text)
        except Exception:
            patch_text = ""
        if not patch_text:
            state._restore_snapshot(cur_snap)

    if not patch_text:
        return ""

    selected = _extract_selected_source_paths(state)
    hygiene = state.diff_hygiene_report(selected) if selected and hasattr(state, "diff_hygiene_report") else {}
    scratch = set(hygiene.get("scratch_files") or [])
    if scratch:
        root = state.root.resolve()
        with state._lock:
            for sf in scratch:
                sp = Path(sf)
                if state.originals.get(sp) is None:
                    fpath = root / sp
                    try:
                        if fpath.is_file() or fpath.is_symlink():
                            fpath.unlink()
                    except OSError:
                        pass
                    state.originals.pop(sp, None)
        selected = [p for p in selected if p.as_posix() not in scratch]
        patch_text = _relaxed_source_patch(state, selected) if selected else ""
        if not patch_text:
            state.patch = ""
            return ""
        hygiene = state.diff_hygiene_report(selected)

    if (
        hygiene.get("added_dbg_lines")
        or _detect_added_inline_tests(patch_text)
        or _detect_benchmark_test_conflicts(patch_text, state, issue_contract)
    ):
        state.patch = ""
        if trace:
            trace.record("degraded_fallback_rejected", reason="hygiene_or_test_isolation_conflict")
        return ""

    passed_regs = len(getattr(state, "_passed_regressions", ()))
    qa_hist = (blackboard or {}).get("qa_history") or []
    last_qa = qa_hist[-1] if qa_hist else {}
    last_qa_decision = str(last_qa.get("decision") or "")
    verification_label = (
        "compiling_snapshot" if from_compiling_snapshot
        else "unverified_after_qa_reject" if last_qa_decision == "REJECT"
        else "unverified_no_passed_regressions" if passed_regs == 0
        else "unverified_partial"
    )
    if trace:
        trace.record(
            "auto_submit_degraded",
            patch=patch_text,
            passed_regressions=passed_regs,
            reproduced_cleanly=bool(state.reproduced_cleanly()) if hasattr(state, "reproduced_cleanly") else False,
            unverified_compiling=from_compiling_snapshot,
            last_qa_decision=last_qa_decision or None,
            last_qa_objections=list(last_qa.get("objections") or [])[:4],
            verification=verification_label,
        )
    return patch_text


def run_deterministic_qa_checks(
    state: Any, issue_contract: dict[str, Any], run_doctest_on_shift: bool = True,
    unreproduced_reason: str = "",
) -> dict[str, Any]:
    """Collect verification findings without blocking a non-empty source patch."""
    _prune_invalid_regression_targets(state)
    changed = state.changed_paths()
    selected = _extract_selected_source_paths(state)
    if not changed or not selected:
        return {"ok": False, "reason": "No production source changes were made."}

    warnings = []
    documentation_only = _documentation_only_patch(state, issue_contract)
    targets = set(state.get_regression_targets().keys())
    if not targets and not documentation_only:
        warnings.append("No temporary regression test was created via write_regression_test.")

    if targets and not (targets <= state._passed_regressions):
        warnings.append(f"Pending regression verification: {state.missing_verification_steps()}")

    if targets and not documentation_only and state.require_reproduction and not state.reproduced_cleanly():
        try:
            state.verify_pre_fix_reproduction(targets)
        except Exception as exc:
            warnings.append(f"Paired reproduction check failed: {exc}")

    repro_notes = getattr(state, "_reproduction_notes", {}) or {}
    repro_map = getattr(state, "_reproduced_regressions", {}) or {}
    passed_on_original = bool(targets) and any(
        repro_notes.get(k) == "passed on unpatched original code"
        or (repro_map.get(k) or {}).get("soft_fallback")
        for k in targets
    )
    unreproduced_allowed = _unreproduced_qa_allowed(state, unreproduced_reason)
    if targets and not documentation_only and not unreproduced_allowed and state.require_reproduction and not state.reproduced_cleanly():
        if "Paired reproduction was not confirmed on the original code" not in warnings:
            warnings.append("Paired reproduction was not confirmed on the original code")

    if not documentation_only and not unreproduced_allowed and not state.verified():
        warnings.append(f"Verification incomplete: {state.missing_verification_steps()}")

    hygiene = state.diff_hygiene_report(selected)
    if hygiene.get("added_dbg_lines"):
        warnings.append(f"Debug macro dbg!(...) left in patch: {hygiene['added_dbg_lines']}")
    if hygiene.get("scratch_files"):
        warnings.append(f"Temporary helper/scratch files left in repository: {hygiene['scratch_files']}")

    post_strip_status = "not_run"
    if state.post_strip_check:
        try:
            post_strip = state.run_post_strip_check(selected) or {}
            post_strip_status = post_strip.get("status", "completed_without_reported_error")
            warnings.extend(post_strip.get("warnings", []))
        except Exception as exc:
            return {"ok": False, "reason": f"Post-strip check failed: {exc}"}

    smoke_results = []
    if hasattr(state, "run_existing_test_smoke"):
        try:
            smoke_results = state.run_existing_test_smoke(selected)
        except Exception as exc:
            smoke_results = [{"target": "existing-test-smoke", "status": "error", "details": str(exc)}]

    patch_text = hygiene.get("patch", "")
    if not patch_text:
        return {"ok": False, "reason": "No non-empty source patch was generated."}
    added_inline_tests = _detect_added_inline_tests(patch_text)
    if added_inline_tests:
        warnings.append(f"Inline tests added in production source files: {added_inline_tests}")
    modified_inline_tests = _detect_modified_inline_tests(patch_text, state=state)
    signals = issue_contract.get("signals") or {}
    if modified_inline_tests and (signals.get("mentions_new_api") or signals.get("mentions_cli_or_additive_flag")):
        warnings.append(f"Modified existing inline tests in production source files: {modified_inline_tests}")

    submission_gate = _submission_gate_report(state, issue_contract, patch_text, selected, smoke_results)
    if submission_gate["blocking"]:
        return {"ok": False, "reason": "Submission gate failed: " + "; ".join(submission_gate["blocking"])}
    warnings.extend(submission_gate["warnings"])

    comment_shifts = _detect_comment_only_doctest_shifts(patch_text, state, selected) if hygiene.get("doctest_shifts") else []

    missing_apis = _check_missing_contract_apis(issue_contract, patch_text, state)
    existing_failures = [dict(x) for x in getattr(state, "_existing_test_failures", [])]
    existing_failures.extend(item for item in smoke_results if item.get("status") == "failed")
    root_res = state.root.resolve()
    selected_set = {p.as_posix() for p in selected}
    untouched_ref_files = [
        rf for rf in (issue_contract.get("referenced_files") or [])
        if rf not in selected_set and (root_res / rf).is_file()
    ]
    static_manifest_tests = _detect_static_manifest_tests(state)
    untested_integration_targets = _find_untested_integration_targets(state, selected)
    target_proxy_coverage = _audit_target_proxy_coverage(issue_contract, _collect_regression_test_contents(state))
    if target_proxy_coverage["missing_targets"]:
        warnings.append(
            "Target proxy requirement coverage gap: temporary regression test does not reference "
            f"{target_proxy_coverage['missing_targets']}"
        )

    doctest_status = "not_run"
    if run_doctest_on_shift and hygiene.get("doctest_shifts"):
        import subprocess
        import shutil
        if shutil.which("cargo"):
            root = state.root.resolve()
            first_src = root / selected[0]
            cur = first_src.parent
            manifest = None
            while cur != root.parent:
                if (cur / "Cargo.toml").is_file():
                    manifest = cur / "Cargo.toml"
                    break
                cur = cur.parent
            if manifest is not None:
                try:
                    doc_cmd = ["cargo", "test", "--doc", "--manifest-path", str(manifest)]
                    if (root / "Cargo.lock").is_file():
                        doc_cmd.append("--locked")
                    doc_res = subprocess.run(
                        doc_cmd, cwd=root, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=90, env=state.rust_env,
                    )
                    doctest_status = "passed" if doc_res.returncode == 0 else f"failed (exit={doc_res.returncode}): {(doc_res.stdout + doc_res.stderr)[-800:]}"
                except subprocess.TimeoutExpired:
                    doctest_status = "timeout_skipped"
                except Exception as exc:
                    doctest_status = f"error: {exc}"

    return {
        "ok": True,
        "selected": selected,
        "hygiene": hygiene,
        "missing_contract_apis": missing_apis,
        "added_inline_tests": added_inline_tests,
        "modified_inline_tests": modified_inline_tests,
        "existing_test_failures": existing_failures,
        "existing_test_smoke": smoke_results,
        "existing_test_smoke_labels": [
            {
                "target": item.get("target"),
                "label": item.get("baseline_label"),
                "classification": item.get("classification"),
                "baseline_status": item.get("baseline_status"),
            }
            for item in (smoke_results or [])
            if item.get("baseline_label")
        ],
        "reproduction": ("not_applicable_documentation" if documentation_only else
                         "paired_red_green" if targets and state.reproduced_cleanly() and state.verified() else
                         "unreproduced_original_passed" if (unreproduced_allowed or passed_on_original) else "unverified"),
        "passed_on_original": passed_on_original,
        "unreproduced_reason": unreproduced_reason,
        "post_strip_status": post_strip_status,
        "submission_gate_warnings": warnings,
        "untouched_referenced_files": untouched_ref_files,
        "static_manifest_tests": static_manifest_tests,
        "untested_integration_targets": untested_integration_targets,
        "target_proxy_coverage": target_proxy_coverage,
        "executed_test_targets": list(getattr(state, "_executed_test_targets", []) or []),
        "doctest_status": doctest_status,
        "comment_doctest_shifts": comment_shifts,
    }


def _collect_regression_test_contents(state: Any, max_chars: int = 3000) -> str:
    root = state.root.resolve()
    parts: list[str] = []
    for rel, info in state.get_regression_targets().items():
        if info.get("is_in_module"):
            parts.append(f"[In-module test {info['name']} in {info['path'].as_posix()}]:\n{info.get('content', '')[:1200]}")
        else:
            fpath = root / info["path"]
            if fpath.is_file():
                try:
                    txt = fpath.read_text(encoding="utf-8", errors="replace")
                    parts.append(f"[{info['path'].as_posix()}]:\n{txt[:1500]}")
                except OSError:
                    pass
    out = "\n\n".join(parts)
    return out[:max_chars]


def _make_handoff_tools(
    role: str,
    state: Any,
    blackboard: dict[str, Any],
    activity: dict[str, Any],
    *,
    allow_escalate: bool = True,
):
    """Create role-specific structured handoff tools that populate the blackboard cleanly."""
    holder: dict[str, Any] = {"done": False, "payload": None}
    candidates = activity.get("candidate_policy")
    candidate_tools = []
    if candidates and role in ("suggester", "fixer"):
        @tool
        def request_candidate_comparison(
            trigger: str, reason: str, evidence_ids: list[str] | None = None,
            chain_ids: list[str] | None = None,
        ) -> str:
            """Request two brief mechanisms ONLY for ambiguous_chains, verification_counterevidence, or observed scope_growth."""
            if trigger == "ambiguous_chains":
                error = candidates.ambiguity(chain_ids or [], reason)
            elif trigger == "verification_counterevidence":
                error = candidates.counterevidence(reason, evidence_ids or [])
            elif trigger == "scope_growth":
                pending = candidates.data["pending"]
                error = None if pending and pending["kind"] == "scope_growth" else "No sustained scope growth observed by Coordinator."
            else:
                error = "Unknown trigger; ordinary complexity/compiler errors alone do not require a second mechanism."
            if error:
                return f"Error: {error}"
            if role == "fixer":
                holder.update(done=True, payload={"action": "escalate", "candidate_comparison": True,
                                                  "reason": "Brief candidate comparison: " + reason})
                return "Handing current patch/evidence to Architect for brief comparison; no rollback or second implementation yet."
            return "Comparison gate opened. Call compare_repair_candidates, then submit only the selected detailed plan."

        candidate_tools.append(request_candidate_comparison)
        if role == "suggester":
            @tool
            def compare_repair_candidates(candidates_brief: list[RepairCandidate], next_check: str = "") -> str:
                """Compare exactly two SHORT causal mechanisms before implementing a second one. Not two full plans."""
                try:
                    result = candidates.compare([c.model_dump() for c in candidates_brief], next_check=next_check)
                except ValueError as exc:
                    return f"Error in candidate comparison: {exc}"
                return json.dumps(result, ensure_ascii=False)

            candidate_tools.append(compare_repair_candidates)

    if role == "navigator":
        class ChainNode(BaseModel):
            path: str = Field(description="Repository-relative file path")
            symbol: str = Field(default="", description="Function, struct, trait, or impl symbol name")
            line_start: int = Field(default=1, description="Approximate start line")
            line_end: int = Field(default=1, description="Approximate end line")
            vis: str = Field(default="unknown", description="Visibility: pub, pub(crate), or private")
            role: str = Field(default="candidate", description="entry_or_symptom | intermediate | root_candidate")

        class FaultChain(BaseModel):
            chain_id: str = Field(default="C1", description="Chain identifier such as C1, C2")
            hypothesis: str = Field(description="Why this chain could explain the root cause")
            nodes: list[ChainNode] = Field(default_factory=list)

        @tool
        def submit_localization_bundle(
            summary: str,
            candidate_chains: list[FaultChain],
            reproduction_status: str = "not_reproduced",
            requested_apis: list[str] | None = None,
            needs_architecture_planning: bool = False,
            notes: str = "",
            unresolved_chain_ids: list[str] | None = None,
            ambiguity_reason: str = "",
        ) -> str:
            """Finish exploration/reproduction and hand off structured findings to the Architect or Fixer."""
            chains_dict = [c.model_dump() for c in candidate_chains]
            if unresolved_chain_ids:
                ambiguous = [c for c in chains_dict if c["chain_id"] in unresolved_chain_ids]
                if len({c["chain_id"] for c in ambiguous}) < 2 or not ambiguity_reason.strip():
                    return "Error: unresolved_chain_ids must name two known chains with a concrete ambiguity_reason."
            if not chains_dict and getattr(state, "ckg", None) is not None:
                chains_dict = state.ckg.build_fault_chains(
                    blackboard["issue_contract"].get("requested_symbols", []),
                    seed_files=activity.get("files_read", [])[:4],
                    max_chains=3,
                )
            holder["done"] = True
            holder["payload"] = {
                "summary": summary,
                "candidate_chains": chains_dict[:3],
                "reproduction_status": reproduction_status,
                "requested_apis": requested_apis or blackboard["issue_contract"].get("requested_symbols", []),
                "needs_architecture_planning": needs_architecture_planning,
                "notes": notes,
                "unresolved_chain_ids": unresolved_chain_ids or [],
                "ambiguity_reason": ambiguity_reason,
            }
            return "LocalizationBundle recorded. Handing off to next stage."

        return [submit_localization_bundle], holder

    if role == "suggester":
        class EditStep(BaseModel):
            file: str = Field(description="Repository-relative file path to modify")
            symbol: str = Field(default="", description="Target function/struct/impl/trait")
            line_range: str = Field(default="", description="Target line range e.g. '120-145'")
            action: str = Field(default="modify", description="modify | add | preserve")
            description: str = Field(description="Precise semantic change required")
            rationale: str = Field(default="", description="Why this change fixes root cause without regressions")

        @tool
        def submit_repair_plan(
            selected_chain_id: str,
            arbitration_rationale: str,
            ordered_edits: list[EditStep],
            rust_invariants: list[str] | None = None,
            api_checklist: list[str] | None = None,
            verification_targets: list[str] | None = None,
            forbidden_edits: list[str] | None = None,
            repair_units: list[RepairUnitSpec] | None = None,
            candidate_id: str = "",
        ) -> str:
            """Submit the structured Rust RepairPlan to guide the Fixer."""
            payload = {
                "action": "plan",
                "selected_chain_id": selected_chain_id,
                "arbitration_rationale": arbitration_rationale,
                "ordered_edits": [e.model_dump() for e in ordered_edits],
                "rust_invariants": rust_invariants or [],
                "api_checklist": api_checklist or [],
                "verification_targets": verification_targets or [],
                "forbidden_edits": forbidden_edits or [],
                "repair_units": [u.model_dump() for u in repair_units or []],
            }
            if not payload["ordered_edits"]:
                return "Error: ordered_edits must include at least one concrete EditStep (empty replans discard unit structure). Submit again."
            try:
                _repair_units.normalize_units(payload, [])
            except ValueError as exc:
                return f"Error in repair units: {exc}. Correct the plan and submit again."
            if candidates:
                error = candidates.plan_gate(payload, candidate_id)
                if error:
                    payload.setdefault("gate_warnings", []).append(error)
            holder["done"] = True
            holder["payload"] = payload
            return "RepairPlan recorded. Handing off to Fixer."

        @tool
        def request_relocalization(reason: str) -> str:
            """Request Navigator to re-explore if all candidate chains are irrelevant to the issue."""
            holder["done"] = True
            holder["payload"] = {"action": "relocalize", "reason": reason}
            return "Relocalization requested."

        @tool
        def release_locked_regression(name: str, reason: str) -> str:
            """Release the Red-phase lock on a regression test you confirmed is factually wrong.

            The lock protects a confirmed Red oracle from being weakened by the Fixer. When the
            locked test itself is wrong (wrong crate path, impossible assertion, wrong expected
            value), the repair cannot proceed until the lock is released. Each test can be
            released only once, and the release is recorded for the Verifier.
            """
            unlock = getattr(state, "unlock_regression", None)
            if not callable(unlock):
                return "Error: unlocking a locked regression test is not available in this session."
            result = unlock(name=name, reason=reason, approved_by="architect")
            if not result.get("ok"):
                return "Error: " + str(result.get("error", "unlock refused"))
            unlock_entry = {
                "name": result.get("unlocked", name),
                "path": result.get("path", ""),
                "reason": result.get("reason", reason),
            }
            activity.setdefault("unlocked_regressions", []).append(unlock_entry)
            holder["done"] = True
            holder["payload"] = {
                "action": "unlock",
                "unlocked": unlock_entry["name"],
                "path": unlock_entry["path"],
                "reason": unlock_entry["reason"],
            }
            return (
                "Released the Red-phase lock on '" + str(result.get("unlocked", name)) + "' ("
                + str(result.get("path", "")) + "). The Fixer may now correct that test; the release "
                "and its reason are recorded and will be shown to the Verifier."
            )

        return [submit_repair_plan, request_relocalization, release_locked_regression] + candidate_tools, holder

    if role == "fixer":
        coordinator = activity.get("unit_coordinator")

        @tool
        def complete_repair_unit(summary: str) -> str:
            """Close the current dependency-closed unit for Coordinator's targeted Cargo verification."""
            if not coordinator or not coordinator.active:
                return "No active repair unit. Finish final verification and call submit_for_qa."
            holder["done"] = True
            holder["payload"] = {"action": "unit_complete", "summary": summary}
            return "Unit handed to Coordinator for verification; do not continue editing."

        @tool
        def submit_for_qa(summary: str, unreproduced_reason: str = "") -> str:
            """Submit edits to QA; explain why a passing original test cannot reproduce the issue if needed."""
            changed = state.changed_paths()
            selected = _extract_selected_source_paths(state)
            if not changed or not selected:
                return "Cannot submit for QA: no source code changes detected in src/."
            holder["done"] = True
            holder["payload"] = {"action": "qa", "summary": summary,
                                 "unreproduced_reason": unreproduced_reason}
            return "Submitted to Patch QA & Verifier."

        @tool
        def escalate_for_replan(reason: str) -> str:
            """Escalate back to Architect/Suggester when stuck on trait/lifetime/design errors or wrong localization."""
            holder["done"] = True
            holder["payload"] = {"action": "escalate", "reason": reason}
            return "Escalation recorded. Handing off to Architect for replanning."

        @tool
        def append_repair_unit(
            unit_id: str,
            file: str,
            description: str,
            symbol: str = "",
            depends_on: list[str] | None = None,
            expected_outcome: str = "",
        ) -> str:
            """Dynamically append a missing companion/consumer repair unit without wiping verified units."""
            if not coordinator:
                return "Error: unit coordinator is unavailable."
            try:
                res = coordinator.append_unit(
                    unit_id=unit_id,
                    description=description,
                    paths=[file],
                    symbol=symbol,
                    depends_on=depends_on,
                    expected_outcome=expected_outcome,
                )
            except ValueError as exc:
                return f"Error appending repair unit: {exc}"
            return json.dumps(res, ensure_ascii=False)

        handoffs = [submit_for_qa]
        if coordinator:
            handoffs.extend([complete_repair_unit, append_repair_unit])
        if allow_escalate:
            handoffs.append(escalate_for_replan)
            handoffs.extend(candidate_tools)
        return handoffs, holder

    if role == "verifier":
        @tool
        def submit_qa_decision(
            decision: str,
            summary: str,
            missing_requirements: list[str] | None = None,
            objections: list[str] | None = None,
            plan_consistency: str = "not_checked",
            plan_conflicts: list[str] | None = None,
            explanation: str = "",
        ) -> str:
            """Finalize QA audit with decision='APPROVE' or 'REJECT'.

            When plan_consistency='conflict', provide non-empty plan_conflicts AND a concrete
            `explanation` citing the issue requirement/contract and test/compile/runtime/caller evidence.
            """
            norm = decision.strip().upper()
            if norm not in ("APPROVE", "REJECT"):
                return "Error: decision must be 'APPROVE' or 'REJECT'."
            consistency = plan_consistency.strip().lower()
            if consistency not in ("consistent", "conflict", "not_checked", "not_applicable"):
                return "Error: plan_consistency must be consistent, conflict, not_checked, or not_applicable."
            has_plan = bool(blackboard.get("repair_plan"))
            if not has_plan:
                # Without an Architect plan there is nothing to compare against, so
                # 'consistent' is meaningless. Normalize it instead of rejecting the
                # whole decision and burning a full verifier round. A 'conflict'
                # without a plan is a real reporting error and stays an error.
                if consistency == "consistent":
                    consistency = "not_applicable"
                    plan_conflicts = []
            if consistency == "conflict" and not plan_conflicts:
                return "Error: describe the observed plan_conflicts with supporting evidence."
            if plan_conflicts and consistency != "conflict":
                return "Error: set plan_consistency='conflict' when reporting plan_conflicts."
            if consistency == "conflict" and not _plan_conflict_explanation_has_issue_basis(
                explanation, blackboard.get("issue_contract")
            ):
                return (
                    "Error: explain how the conflict relates to the issue contract and concrete test, caller, "
                    "API, compile, or runtime evidence in the `explanation` parameter (not only `summary`)."
                )
            if consistency == "not_applicable" and has_plan:
                return "Error: an Architect plan exists; audit it or select not_checked."
            if consistency in ("consistent", "conflict") and not has_plan:
                return "Error: no Architect plan exists; select not_applicable or not_checked."
            if norm == "REJECT" and not (objections or missing_requirements):
                return "Error: provide concrete objections or missing_requirements when rejecting."
            holder["done"] = True
            holder["payload"] = {
                "decision": norm,
                "summary": summary,
                "missing_requirements": missing_requirements or [],
                "objections": objections or [],
                "plan_consistency": consistency,
                "plan_conflicts": plan_conflicts or [],
                "explanation": explanation,
            }
            return f"QA decision recorded: {norm}"

        return [submit_qa_decision], holder

    return [], holder


def _clone_tool_with_func(t_obj: Any, new_func: Any) -> Any:
    """Create a shallow copy of a LangChain StructuredTool with wrapped func without mutating shared tools."""
    if hasattr(t_obj, "model_copy"):
        return t_obj.model_copy(update={"func": new_func})
    import copy
    cloned = copy.copy(t_obj)
    cloned.func = new_func
    return cloned


_READ_ONLY_BROWSE_TOOLS = frozenset({
    "read_file", "read_lines", "search_code", "cargo_workspace",
    "rust_symbols", "inspect_rust_neighborhood", "rust_error_explain",
})
_PROGRESS_ACTION_TOOLS = frozenset({
    "edit_file", "write_file", "write_regression_test", "revert_file",
    "checkpoint", "cargo_check", "cargo_test",
})


def _wrap_tools_for_role(
    role: str,
    tools_by_name: dict[str, Any],
    allowed_names: list[str],
    handoff_tools: list[Any],
    holder: dict[str, Any],
    state: Any,
    client: Any,
    stage_start_calls: int,
    stage_cap: int | None,
    activity: dict[str, Any],
    auto_escalate_on_stuck: bool = False,
    blackboard: dict[str, Any] | None = None,
):
    activity.setdefault("read_excerpts", {})
    activity.setdefault("ckg_notes", {})
    selected = []
    for name in allowed_names:
        t_obj = tools_by_name.get(name)
        if t_obj is None:
            continue
        orig_func = t_obj.func

        @functools.wraps(orig_func)
        def _wrapped(*args, __fn=orig_func, __tname=name, **kwargs):
            coordinator = activity.get("unit_coordinator") if role == "fixer" else None
            revision_before = getattr(state, "_revision", 0)
            if coordinator:
                if holder.get("done"):
                    raise _repair_units.UnitBoundary("Structured unit handoff already submitted")
                # StructuredTool normally passes kwargs; binding covers direct callers too.
                import inspect
                arguments = inspect.signature(__fn).bind_partial(*args, **kwargs).arguments
                blocked = coordinator.before_tool(__tname, arguments)
                if blocked:
                    return blocked
            if stage_cap is not None:
                rem_stage = max(0, stage_cap - (_client_calls_made(client) - stage_start_calls))
                activity["calls_remaining_in_stage"] = rem_stage
            if __tname in ("read_lines", "read_file"):
                p = kwargs.get("path") or (args[0] if args else "")
                if p and p not in activity["files_read"]:
                    activity["files_read"].append(str(p))
                if p and blackboard is not None and role in ("suggester", "fixer", "verifier"):
                    s_l = kwargs.get("start_line")
                    e_l = kwargs.get("end_line")
                    r_key = f"{p}:{s_l}-{e_l}" if (s_l is not None or e_l is not None) else str(p)
                    cached_reads = blackboard.get("shared_read_excerpts") or {}
                    stats = blackboard.setdefault("coord_stats", {})
                    stats["downstream_reads"] = int(stats.get("downstream_reads", 0)) + 1
                    if r_key in cached_reads or str(p) in cached_reads:
                        stats["duplicate_reads"] = int(stats.get("duplicate_reads", 0)) + 1
            elif __tname == "inspect_rust_neighborhood":
                tgt = kwargs.get("target") or (args[0] if args else "")
                if tgt and tgt not in activity["symbols_queried"]:
                    activity["symbols_queried"].append(str(tgt))

            out = __fn(*args, **kwargs)
            if coordinator:
                evidence_id = coordinator.observe(__tname, arguments, out, revision_before)
                if evidence_id and isinstance(out, str):
                    if __tname in ("cargo_check", "cargo_test"):
                        result = json.loads(out)
                        result["candidate_evidence_id"] = evidence_id
                        out = json.dumps(result, ensure_ascii=False)
                    else:
                        out += f"\n[Coordinator execution evidence: {evidence_id}, revision={state._revision}]"

            if isinstance(out, str):
                if __tname in ("read_lines", "read_file"):
                    p = str(kwargs.get("path") or (args[0] if args else ""))
                    if p and not p.startswith("tests/") and "/tests/" not in p and not out.startswith("Error") and ("1: " in out[:40] or (":" in out[:12] and "\n" in out)):
                        s_l = kwargs.get("start_line")
                        e_l = kwargs.get("end_line")
                        key = f"{p}:{s_l}-{e_l}" if (s_l is not None or e_l is not None) else p
                        _store_excerpt_with_priority(
                            activity["read_excerpts"],
                            key,
                            out[:1600],
                            limit=10,
                            priority_paths=_priority_paths_from_blackboard(blackboard),
                        )
                elif __tname == "inspect_rust_neighborhood":
                    tgt = str(kwargs.get("target") or (args[0] if args else ""))
                    if tgt and '"error"' not in out[:60]:
                        activity["ckg_notes"][tgt] = out[:1200]
                elif __tname in ("edit_file", "write_file", "revert_file"):
                    p = str(kwargs.get("path") or (args[0] if args else ""))
                    if p and blackboard is not None:
                        _invalidate_cached_path(blackboard, activity, p)

            if isinstance(out, str) and __tname in ("cargo_check", "cargo_test", "edit_file", "write_file") and _is_env_dependency_blocker(out):
                hits = activity.get("env_blocker_hits", 0) + 1
                activity["env_blocker_hits"] = hits
                if blackboard is not None:
                    blackboard["environment_blocker"] = True
                if hits >= 2:
                    out += (
                        "\n[Environment Blocker Hint: Offline Cargo registry lacks this dependency version and Cargo.lock is read-only. "
                        "Do not keep toggling unavailable versions; revert unavailable Cargo.toml bumps and check Cargo.lock for an already-locked crate/version or code-level fix.]"
                    )
                if auto_escalate_on_stuck and hits >= 3:
                    raise _AutoEscalate(
                        "Repeated offline Cargo dependency/lockfile environment blocker (>=3 attempts)",
                        {"env_blocker_hits": hits, "last_output": out[-400:]},
                    )

            if isinstance(out, str):
                if role == "navigator":
                    used_stage = (_client_calls_made(client) - stage_start_calls) if stage_cap is not None else 0
                    if used_stage >= 12 and not state.get_regression_targets() and "[Navigator pacing notice:" not in out:
                        out += (
                            "\n[Navigator pacing notice: 12+ calls used without write_regression_test yet. "
                            "Synthesize findings, create ONE temporary regression test, and call submit_localization_bundle soon.]"
                        )
                    elif state.reproduced() and __tname in ("read_file", "read_lines", "search_code", "rust_symbols", "inspect_rust_neighborhood"):
                        activity["post_repro_reads"] = activity.get("post_repro_reads", 0) + 1
                        if activity["post_repro_reads"] >= 2 and "[Navigator pacing notice:" not in out:
                            out += (
                                "\n[Navigator pacing notice: Red reproduction is already confirmed. "
                                "Call submit_localization_bundle now to preserve budget for Fixer.]"
                            )
                elif role == "fixer":
                    if __tname in _READ_ONLY_BROWSE_TOOLS:
                        activity["read_only_streak"] = activity.get("read_only_streak", 0) + 1
                    elif __tname in _PROGRESS_ACTION_TOOLS:
                        activity["read_only_streak"] = 0
                    if not coordinator:
                        if __tname in ("edit_file", "write_file") and not out.startswith("Error"):
                            activity["edits_since_check"] = activity.get("edits_since_check", 0) + 1
                            if activity["edits_since_check"] >= 4:
                                out += (
                                    "\n[Pacing notice: 4+ source edits made without running cargo_check or cargo_test. "
                                    "Run cargo_check or cargo_test now to catch compiler/test errors early.]"
                                )
                        elif __tname in ("cargo_check", "cargo_test"):
                            activity["edits_since_check"] = 0
                        if __tname == "write_regression_test":
                            prev_passed = len(getattr(state, "_passed_regressions", set()))
                            activity["reg_write_streak"] = activity.get("reg_write_streak", 0) + 1
                            if prev_passed > activity.get("last_passed_reg_count", 0):
                                activity["reg_write_streak"] = 0
                            activity["last_passed_reg_count"] = prev_passed
                            if activity["reg_write_streak"] >= 4:
                                out += (
                                    "\n[Pacing notice: multiple write_regression_test edits without passing; "
                                    "inspect existing tests/ helpers or run existing crate tests instead of rewriting boilerplate.]"
                                )
                        elif __tname == "cargo_test" and len(getattr(state, "_passed_regressions", set())) > activity.get("last_passed_reg_count", 0):
                            activity["reg_write_streak"] = 0
                            activity["last_passed_reg_count"] = len(getattr(state, "_passed_regressions", set()))
                    streak = activity.get("read_only_streak", 0)
                    if streak >= 10 and not auto_escalate_on_stuck:
                        out += (
                            f"\n[Pacing warning: {streak} consecutive read-only calls without edits or Cargo checks. "
                            "Apply a concrete edit, run cargo_check/cargo_test, or submit now.]"
                        )

            if auto_escalate_on_stuck and role == "fixer":
                streak = activity.get("read_only_streak", 0)
                if streak >= 16:
                    raise _AutoEscalate(
                        f"Fixer read-only exploration stalled ({streak} consecutive browse calls without edits or Cargo checks); clarify localization or concrete next edit",
                        {"read_only_streak": streak},
                    )
            if auto_escalate_on_stuck and role == "fixer" and not coordinator:
                if activity.get("reg_write_streak", 0) >= 6:
                    raise _AutoEscalate(
                        "Repeated temporary regression test rewrites without passing (>=6 attempts)",
                        {"reg_write_streak": activity.get("reg_write_streak", 0)},
                    )
            if auto_escalate_on_stuck and __tname in ("cargo_check", "cargo_test"):
                stuck = state.stuck_status()
                if stuck["compile_repeats"] >= 3 or stuck["test_repeats"] >= 3:
                    raise _AutoEscalate(
                        f"Repeated identical {'compiler' if stuck['compile_repeats'] >= 3 else 'test'} failure (>=3 revisions)",
                        stuck,
                    )
            return out

        selected.append(_clone_tool_with_func(t_obj, _wrapped))

    selected.extend(handoff_tools)
    return selected


def _make_guarded_llm(
    chat_model_factory: Any,
    model: str,
    api_key: str,
    base_url: str | None,
    client: Any,
    compress_fn: Any,
    role: str,
    start_calls: int,
    stage_cap: int | None,
    reserve_floor: int,
    latest_msgs_ref: dict[str, Any] | None = None,
    handoff_holder: dict[str, Any] | None = None,
):
    llm = chat_model_factory(
        model=model, api_key=api_key, base_url=base_url,
        temperature=0, timeout=180, max_retries=3, http_client=client,
    )
    if hasattr(llm, "bind_tools") and callable(getattr(llm, "bind_tools")):
        orig_bind = llm.bind_tools

        def _guarded_bind(*args, **kwargs):
            bound = orig_bind(*args, **kwargs)

            def _guard_and_compress(msgs):
                if latest_msgs_ref is not None:
                    latest_msgs_ref["msgs"] = msgs
                if role == "fixer" and handoff_holder and handoff_holder.get("done"):
                    raise _repair_units.UnitBoundary("Structured handoff reached")
                used = _client_calls_made(client) - start_calls
                if stage_cap is not None and used >= stage_cap:
                    raise _StageBudgetReached(role, used, stage_cap)
                rem = _client_remaining_calls(client)
                if reserve_floor > 0 and rem is not None and rem <= reserve_floor:
                    raise _StageReserveReached(role, rem, reserve_floor)
                return compress_fn(msgs)

            return RunnableLambda(_guard_and_compress) | bound

        try:
            object.__setattr__(llm, "bind_tools", _guarded_bind)
        except Exception:
            pass
    return llm


def _run_role_agent(
    role: str,
    system_prompt: str,
    user_prompt: str,
    role_tools: list[Any],
    holder: dict[str, Any],
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    client: Any,
    trace: Any,
    compress_fn: Any,
    create_agent_fn: Any,
    chat_model_factory: Any,
    role_callback_cls: Any,
    stage_cap: int | None,
    reserve_floor: int = 0,
    attempt: int = 1,
    max_continuations: int = 2,
    continuation_fn: Any = None,
    handoff_only_tools: list[Any] | None = None,
    wrapup_prompt: str | None = None,
    history: list[Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Execute one bounded sub-agent role with clean context, forced handoff wrap-up, and shared TraceHTTPClient."""
    start_calls = _client_calls_made(client)
    started_at = time.perf_counter()
    if trace:
        trace.record("stage_start", agent_role=role, attempt=attempt, stage_cap=stage_cap, calls_made_start=start_calls)

    latest_msgs_ref: dict[str, Any] = {"msgs": None}
    explore_cap = (stage_cap - 1) if (handoff_only_tools and stage_cap is not None and stage_cap >= 2) else stage_cap

    llm = _make_guarded_llm(
        chat_model_factory, model, api_key, base_url, client, compress_fn,
        role, start_calls, explore_cap, reserve_floor, latest_msgs_ref=latest_msgs_ref,
        handoff_holder=holder,
    )
    agent = create_agent_fn(model=llm, tools=role_tools, system_prompt=system_prompt)
    import sys
    config: dict[str, Any] = {"recursion_limit": sys.maxsize}
    if trace:
        cb = role_callback_cls(trace, role=role, attempt=attempt) if role_callback_cls else trace
        config["callbacks"] = [cb]

    messages = (list(compress_fn(history)) if history else []) + [{"role": "user", "content": user_prompt}]
    outcome = "completed"
    try:
        try:
            for step in range(max_continuations + 1):
                result = agent.invoke({"messages": messages}, config=config)
                if isinstance(result, dict) and isinstance(result.get("messages"), list):
                    latest_msgs_ref["msgs"] = result["messages"]
                if holder.get("done"):
                    break
                if step >= max_continuations or not isinstance(result, dict) or not isinstance(result.get("messages"), list):
                    outcome = "no_handoff"
                    break
                cont_msg = continuation_fn(role, step) if continuation_fn else (
                    f"Complete your role ({role}) now by calling your structured handoff tool."
                )
                messages = compress_fn(result["messages"]) + [{"role": "user", "content": cont_msg}]
        except _StageBudgetReached as exc:
            outcome = "stage_cap_reached"
            if trace:
                trace.record("stage_cap", agent_role=role, used=exc.used, cap=exc.cap)
        except _StageReserveReached as exc:
            outcome = "reserve_reached"
            if trace:
                trace.record("stage_reserve", agent_role=role, remaining=exc.remaining, reserve=exc.reserve)
        except _AutoEscalate as exc:
            outcome = "auto_escalate"
            holder["done"] = True
            holder["payload"] = {"action": "escalate", "reason": exc.reason, "detail": exc.detail}
        except _repair_units.UnitBoundary as exc:
            outcome = "unit_boundary"
            if not holder.get("done"):
                holder["payload"] = {"action": "unit_boundary", "reason": str(exc)}

        if (
            not holder.get("done")
            and handoff_only_tools
            and outcome in ("stage_cap_reached", "no_handoff")
        ):
            base_msgs = latest_msgs_ref["msgs"] or messages
            base_wrap_msg = wrapup_prompt or f"Exploration budget reached. Call your structured handoff tool for {role} IMMEDIATELY."
            for wrap_try in range(2):
                rem = _client_remaining_calls(client)
                if rem is not None and (rem <= reserve_floor or (wrap_try > 0 and rem <= reserve_floor + 1)):
                    break
                if wrap_try > 0 and trace:
                    trace.record("wrapup_retry", agent_role=role, attempt=attempt)
                wrap_start = _client_calls_made(client)
                wrap_llm = _make_guarded_llm(
                    chat_model_factory, model, api_key, base_url, client, compress_fn,
                    role, wrap_start, 1, reserve_floor,
                )
                wrap_agent = create_agent_fn(model=wrap_llm, tools=handoff_only_tools, system_prompt=system_prompt)
                msg_now = (
                    base_wrap_msg
                    if wrap_try == 0
                    else (
                        f"Your previous handoff tool call for {role} failed JSON parsing or did not invoke the tool. "
                        "Call the handoff tool NOW with short, valid JSON (keep every description <= 200 chars, no long quotes)."
                    )
                )
                wrap_input = compress_fn(base_msgs) + [{"role": "user", "content": msg_now}]
                try:
                    wrap_agent.invoke({"messages": wrap_input}, config=config)
                except (_StageBudgetReached, _StageReserveReached):
                    pass
                except RuntimeError as exc:
                    if "Model API call limit reached" not in str(exc):
                        raise
                if holder.get("done"):
                    outcome = "completed"
                    break
    finally:
        holder["history"] = latest_msgs_ref["msgs"] or messages
        if trace:
            trace.record(
                "stage_end",
                agent_role=role,
                attempt=attempt,
                outcome=outcome,
                calls_used=_client_calls_made(client) - start_calls,
                elapsed_seconds=round(time.perf_counter() - started_at, 3),
            )

    return holder.get("payload"), outcome


def _run_scheduled_fixer(system_prompt, user_prompt, role_tools, holder, *, coordinator,
                         allow_escalate, **kwargs):
    """Continue the repair conversation across unit checks within one attempt budget."""
    client = kwargs["client"]
    start = _client_calls_made(client)
    total_cap = kwargs.pop("stage_cap")
    history = kwargs.pop("history", None)
    slices = idle_slices = 0
    feedback = ""
    injected_digests: dict[str, str] = {}
    last_diff_digest = ""
    while True:
        used = max(_client_calls_made(client) - start, slices)
        if total_cap is not None and used >= total_cap:
            return None, "stage_cap_reached"
        unit = coordinator.select()
        if not unit and not coordinator.all_done():
            return {"action": "escalate", "reason": "No dependency-closed repair unit is ready"}, "unit_blocked"
        holder.update(done=False, payload=None)
        remaining_cap = total_cap - used if total_cap is not None else None
        slice_cap = remaining_cap
        current_ctx = ""
        if slices:
            paths = (unit["paths"] if unit else [p.as_posix() for p in coordinator.state.changed_paths()])
            file_blocks: list[str] = []
            for path in paths[:6]:
                p = coordinator.state.root / path
                if p.is_file():
                    raw_txt = p.read_text(encoding="utf-8", errors="replace")
                    dig = hashlib.sha256(raw_txt.encode("utf-8", errors="replace")).hexdigest()[:16]
                    if injected_digests.get(path) != dig:
                        injected_digests[path] = dig
                        file_blocks.append(f"\n{path}:\n{raw_txt[:1800]}\n")
            diff_scope = set(paths) | (unit.get("touched", set()) if unit else set())
            diff_paths = [p for p in coordinator.state.changed_paths() if p.as_posix() in diff_scope]
            diff = coordinator.state.build_patch(diff_paths) if diff_paths else "(No changes in this unit yet)"
            diff_dig = hashlib.sha256(diff.encode("utf-8", errors="replace")).hexdigest()[:16]
            diff_block = ""
            if diff_dig != last_diff_digest:
                last_diff_digest = diff_dig
                diff_block = "\nCurrent unit diff:\n" + diff[:3500]
            if file_blocks or diff_block:
                current_ctx = (
                    ("\nCURRENT modified source excerpts (unchanged files omitted):\n" + "".join(file_blocks)[:5000] if file_blocks else "")
                    + diff_block
                )
            else:
                current_ctx = "\n(Unit files and diff unchanged since previous slice.)"
        policy = getattr(coordinator, "candidate_policy", None)
        base_prompt = (
            f"Continue repair on unit '{unit['unit_id']}'.\n" if unit else "Continue final repair and verification.\n"
        ) if (slices >= 1 and history) else user_prompt
        prompt = base_prompt + coordinator.prompt() + (policy.prompt() if policy else "") + current_ctx + "\nCoordinator feedback:\n" + feedback
        revision_before = coordinator.state._revision
        done_before = sum(u["status"] in {"verified", "completed_unverified"} for u in coordinator.units)
        payload, outcome = _run_role_agent(
            "fixer", system_prompt, prompt, role_tools, holder, stage_cap=slice_cap, history=history, **kwargs,
        )
        history = holder.get("history") or history
        slices += 1
        action = (payload or {}).get("action")
        if action == "qa":
            return payload, outcome
        if policy:
            policy.repeated_test_failure()
        if action == "escalate" or outcome == "reserve_reached":
            coordinator.state.reset_stuck_counters()
            return payload, outcome
        if coordinator.active:
            # Check at closed units and bounded exploration/edit boundaries.
            # Never mark an incomplete/coupled unit finished on compile alone.
            active = coordinator._unit(coordinator.active)
            changed = bool(coordinator.delta_paths(active))
            if changed or action in ("unit_complete", "qa"):
                ok, feedback = coordinator.verify(complete=action in ("unit_complete", "qa"),
                                                   summary=(payload or {}).get("summary", ""))
                if action == "qa" and ok and coordinator.all_done():
                    return payload, outcome
                if policy:
                    policy.repeated_test_failure()
                stuck = coordinator.state.stuck_status()
                if not ok and active.get("refuted_by") and allow_escalate:
                    return {
                        "action": "escalate",
                        "reason": f"Repair unit '{active['unit_id']}' refuted ({active['refuted_by']}): {feedback[-1200:]}",
                        "detail": stuck,
                    }, "unit_blocked"
                if not ok and action != "unit_complete" and allow_escalate and (stuck["compile_repeats"] >= 3 or stuck["test_repeats"] >= 3):
                    return {"action": "escalate", "reason": "Repeated unit verification failure: " + feedback,
                            "detail": stuck}, "unit_blocked"
            else:
                coordinator.steps = coordinator.unchecked_edits = 0
                feedback = "No edits since unit start. Implement current unit from supplied excerpts or escalate with a concrete blocker."
        else:
            feedback = "All units checked. Perform final regression/existing-test verification and call submit_for_qa."
        progressed = (coordinator.state._revision != revision_before
                      or sum(u["status"] in {"verified", "completed_unverified"} for u in coordinator.units) > done_before)
        idle_slices = 0 if progressed else idle_slices + 1
        lock_rep = (
            coordinator.state.locked_regression_report()
            if (not progressed and allow_escalate and hasattr(coordinator.state, "locked_regression_report"))
            else {}
        )
        if lock_rep.get("violations") and lock_rep.get("locked") and not coordinator.state.verified():
            reason = "Fixer hit Red-phase lock violation(s) without progress; inspect locked oracle and release/replan if wrong. " + feedback[-1200:]
            return {"action": "escalate", "reason": reason}, "unit_blocked"
        if idle_slices >= 2:
            reason = "Two bounded Fixer slices without implementation/verification progress. " + feedback[-1800:]
            if allow_escalate:
                return {"action": "escalate", "reason": reason}, "unit_blocked"
            return None, "stage_cap_reached"
        rem = _client_remaining_calls(client)
        if rem is not None and rem <= kwargs.get("reserve_floor", 0):
            return None, "reserve_reached"


def _feature_sha256(value: Any) -> tuple[bool, str | None, int]:
    if value is None:
        return False, None, 0
    if isinstance(value, str):
        present = bool(value.strip())
        raw = value
    elif isinstance(value, bool):
        present = True
        raw = "true" if value else "false"
    elif isinstance(value, (int, float)):
        present = True
        raw = str(value)
    else:
        present = bool(value)
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16] if present else None
    return present, digest, len(raw)


def _feature_meta(
    value: Any,
    *,
    consumers: list[str] | None = None,
    prompt_char_len: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    present, digest, char_len = _feature_sha256(value)
    meta: dict[str, Any] = {
        "present": present,
        "sha256": digest,
        "char_len": char_len,
    }
    if isinstance(value, (list, dict)):
        meta["count"] = len(value)
    if consumers is not None:
        meta["consumers"] = list(consumers)
    if prompt_char_len is not None:
        meta["prompt_char_len"] = int(prompt_char_len)
    if extra:
        meta.update(extra)
    return meta


def _record_handoff_snapshot(
    trace: Any,
    *,
    stage: str,
    from_role: str,
    to_role: str,
    attempt: int,
    features: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    if not trace:
        return
    payload: dict[str, Any] = {
        "stage": stage,
        "from_role": from_role,
        "to_role": to_role,
        "attempt": int(attempt),
        "features": features,
    }
    if extra:
        payload.update(extra)
    trace.record("handoff_feature_snapshot", **payload)


def _build_issue_contract_features(issue: str, issue_contract: dict[str, Any]) -> dict[str, Any]:
    req_syms = list(issue_contract.get("requested_symbols") or [])
    new_apis = list(issue_contract.get("new_api_candidates") or [])
    ref_files = list(issue_contract.get("referenced_files") or [])
    exp_excerpt = issue_contract.get("expected_result_excerpt") or ""
    signals = dict(issue_contract.get("signals") or {})
    return {
        "issue_text": _feature_meta(issue or "", consumers=["navigator", "suggester", "fixer", "verifier"], prompt_char_len=len(issue or "")),
        "requested_symbols": _feature_meta(req_syms, consumers=["navigator", "suggester", "fixer", "verifier"], extra={"items": req_syms[:20]}),
        "new_api_candidates": _feature_meta(new_apis, consumers=["navigator", "suggester", "fixer", "verifier"], extra={"items": new_apis[:10]}),
        "referenced_files": _feature_meta(ref_files, consumers=["navigator", "suggester", "fixer", "verifier"], extra={"items": ref_files[:10]}),
        "expected_result_excerpt": _feature_meta(exp_excerpt, consumers=["navigator", "suggester", "fixer", "verifier"]),
        "issue_signals": _feature_meta(signals, consumers=["router", "suggester", "verifier"], extra={"signals": signals}),
    }


def _build_localization_features(
    blackboard: dict[str, Any],
    state: Any,
    *,
    repro_contents: str = "",
) -> dict[str, Any]:
    loc = blackboard.get("localization") or {}
    chains = list(loc.get("candidate_chains") or [])
    distinct_files: list[str] = []
    root_candidates: list[dict[str, str]] = []
    node_count = 0
    for ch in chains:
        for node in ch.get("nodes") or []:
            node_count += 1
            p = str(node.get("path") or "")
            if p and p not in distinct_files:
                distinct_files.append(p)
            nrole = str(node.get("role") or "").strip().lower()
            if nrole in ("root_candidate", "modify"):
                root_candidates.append({"path": p, "symbol": str(node.get("symbol") or ""), "vis": str(node.get("vis") or "unknown")})
    repro_meta = loc.get("repro_meta") or {}
    repro_targets = list(repro_meta.get("targets") or [])
    shared_reads = blackboard.get("shared_read_excerpts") or {}
    shared_ckg = blackboard.get("shared_ckg_notes") or {}
    downstream = list(blackboard.get("downstream_dependents") or [])
    req_apis = list(loc.get("requested_apis") or [])
    repro_status = str(loc.get("reproduction_status") or "not_reproduced")
    return {
        "localization_summary": _feature_meta(loc.get("summary") or "", consumers=["suggester", "fixer"]),
        "candidate_chains": _feature_meta(
            chains,
            consumers=["router", "suggester", "fixer"],
            extra={
                "chain_count": len(chains),
                "node_count": node_count,
                "distinct_files": distinct_files[:15],
                "root_candidates": root_candidates[:10],
                "ckg_suggested_count": len(loc.get("ckg_suggested_chains") or []),
            },
        ),
        "requested_apis": _feature_meta(req_apis, consumers=["suggester", "fixer"], extra={"items": req_apis[:20]}),
        "needs_architecture_planning": _feature_meta(
            bool(loc.get("needs_architecture_planning")),
            consumers=["router"],
            extra={"value": bool(loc.get("needs_architecture_planning"))},
        ),
        "reproduction_status": _feature_meta(
            repro_status,
            consumers=["router", "suggester", "fixer"],
            extra={
                "status": repro_status,
                "target_count": len(repro_targets),
                "red_verified_count": sum(1 for t in repro_targets if t.get("red_verified")),
                "clean_red_count": sum(1 for t in repro_targets if t.get("clean_red")),
                "in_module_count": sum(1 for t in repro_targets if t.get("is_in_module")),
            },
        ),
        "repro_test_contents": _feature_meta(repro_contents, consumers=["suggester", "fixer", "verifier"]),
        "shared_read_excerpts": _feature_meta(
            shared_reads,
            consumers=["suggester", "fixer"],
            extra={"keys": list(shared_reads.keys())[:14]},
        ),
        "shared_ckg_notes": _feature_meta(
            shared_ckg,
            consumers=["suggester", "fixer"],
            extra={"targets": list(shared_ckg.keys())[:8]},
        ),
        "downstream_dependents": _feature_meta(downstream, consumers=["fixer", "verifier"], extra={"items": downstream[:15]}),
        "navigator_notes": _feature_meta(loc.get("notes") or "", consumers=["suggester", "fixer"]),
        "env_blocker": _feature_meta(
            bool(blackboard.get("environment_blocker", False)),
            consumers=["router", "suggester", "fixer"],
            extra={"value": bool(blackboard.get("environment_blocker", False))},
        ),
    }


def _build_repair_plan_features(plan: dict[str, Any] | None, escalations: list[dict[str, Any]]) -> dict[str, Any]:
    plan = plan or {}
    edits = list(plan.get("ordered_edits") or [])
    invariants = list(plan.get("rust_invariants") or [])
    api_checklist = list(plan.get("api_checklist") or [])
    ver_targets = list(plan.get("verification_targets") or [])
    forbidden = list(plan.get("forbidden_edits") or [])
    return {
        "selected_chain_id": _feature_meta(plan.get("selected_chain_id") or "", consumers=["fixer"], extra={"value": plan.get("selected_chain_id") or ""}),
        "arbitration_rationale": _feature_meta(plan.get("arbitration_rationale") or "", consumers=["fixer"]),
        "ordered_edits": _feature_meta(
            edits,
            consumers=["fixer"],
            extra={
                "files": [str(e.get("file") or "") for e in edits[:10]],
                "symbols": [str(e.get("symbol") or "") for e in edits[:10]],
                "actions": [str(e.get("action") or "") for e in edits[:10]],
            },
        ),
        "rust_invariants": _feature_meta(invariants, consumers=["fixer"]),
        "api_checklist": _feature_meta(api_checklist, consumers=["fixer"], extra={"items": api_checklist[:15]}),
        "verification_targets": _feature_meta(ver_targets, consumers=["fixer"], extra={"items": ver_targets[:15]}),
        "forbidden_edits": _feature_meta(forbidden, consumers=["fixer"]),
        "escalation_history": _feature_meta(escalations, consumers=["suggester", "fixer"]),
    }


def _build_fixer_handoff_features(
    state: Any,
    blackboard: dict[str, Any],
    *,
    det_qa: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        selected = _extract_selected_source_paths(state) if hasattr(state, "changed_paths") else []
    except Exception:
        selected = []
    selected_strs = [p.as_posix() for p in selected]
    stuck = state.stuck_status() if hasattr(state, "stuck_status") else {}
    verified_flag = bool(state.verified()) if hasattr(state, "verified") else False
    repro_clean_flag = bool(state.reproduced_cleanly()) if hasattr(state, "reproduced_cleanly") else False
    missing_steps = state.missing_verification_steps() if hasattr(state, "missing_verification_steps") else ""
    passed_regs = sorted(f"{k[0].as_posix()}::{k[1]}" for k in (getattr(state, "_passed_regressions", set()) or set()))
    exec_targets = list(getattr(state, "_executed_test_targets", []) or [])
    err_codes = [
        e.get("code")
        for e in (stuck.get("last_check_errors") or [])
        if isinstance(e, dict) and e.get("code")
    ]
    failed_tests = list((stuck.get("last_test_failure") or {}).get("failed_tests") or [])
    patch_bytes = 0
    if det_qa and isinstance(det_qa.get("hygiene"), dict):
        patch_bytes = len((det_qa["hygiene"].get("patch") or "").encode("utf-8"))

    features: dict[str, Any] = {
        "changed_files": _feature_meta(
            selected_strs,
            consumers=["qa", "verifier"],
            extra={"paths": selected_strs[:20], "patch_bytes": patch_bytes},
        ),
        "regression_verification": _feature_meta(
            {
                "verified": verified_flag,
                "reproduced_cleanly": repro_clean_flag,
                "passed_regressions": passed_regs,
                "executed_targets_count": len(exec_targets),
                "missing": missing_steps,
            },
            consumers=["qa", "verifier"],
            extra={
                "verified": verified_flag,
                "reproduced_cleanly": repro_clean_flag,
                "passed_count": len(passed_regs),
                "executed_targets_count": len(exec_targets),
            },
        ),
        "stuck_status": _feature_meta(
            stuck,
            consumers=["router", "suggester"],
            extra={
                "compile_repeats": int(stuck.get("compile_repeats", 0) or 0),
                "test_repeats": int(stuck.get("test_repeats", 0) or 0),
                "error_codes": err_codes[:5],
                "failed_tests": failed_tests[:5],
            },
        ),
        "env_blocker": _feature_meta(
            bool(blackboard.get("environment_blocker", False)),
            consumers=["router", "suggester"],
            extra={"value": bool(blackboard.get("environment_blocker", False))},
        ),
    }

    if det_qa is not None:
        hygiene = det_qa.get("hygiene") or {}
        det_summary = {
            "ok": bool(det_qa.get("ok")),
            "reason": str(det_qa.get("reason") or "")[:240],
            "missing_contract_apis": list(det_qa.get("missing_contract_apis") or [])[:10],
            "added_inline_tests_count": len(det_qa.get("added_inline_tests") or []),
            "modified_inline_tests_count": len(det_qa.get("modified_inline_tests") or []),
            "doctest_shifts": list(hygiene.get("doctest_shifts") or [])[:8],
            "doctest_status": str(det_qa.get("doctest_status") or "not_run")[:120],
            "existing_test_failures_count": len(det_qa.get("existing_test_failures") or []),
            "untouched_referenced_files": list(det_qa.get("untouched_referenced_files") or [])[:8],
            "static_manifest_tests": list(det_qa.get("static_manifest_tests") or [])[:6],
            "untested_integration_targets_count": len(det_qa.get("untested_integration_targets") or []),
            "added_dbg_lines_count": len(hygiene.get("added_dbg_lines") or []),
            "scratch_files_count": len(hygiene.get("scratch_files") or []),
        }
        features["deterministic_qa"] = _feature_meta(det_summary, consumers=["verifier", "fixer"], extra=det_summary)
    return features


def _build_verifier_handoff_features(ver_payload: dict[str, Any] | None) -> dict[str, Any]:
    ver_payload = ver_payload or {}
    missing = list(ver_payload.get("missing_requirements") or [])
    objections = list(ver_payload.get("objections") or [])
    conflicts = list(ver_payload.get("plan_conflicts") or [])
    return {
        "qa_summary": _feature_meta(ver_payload.get("summary") or "", consumers=["fixer", "submission"]),
        "missing_requirements": _feature_meta(missing, consumers=["fixer"], extra={"items": missing[:10]}),
        "objections": _feature_meta(objections, consumers=["fixer"], extra={"items": [str(x)[:160] for x in objections[:8]]}),
        "plan_consistency": _feature_meta(ver_payload.get("plan_consistency") or "not_checked", consumers=["fixer", "submission"],
                                          extra={"value": ver_payload.get("plan_consistency") or "not_checked"}),
        "plan_conflicts": _feature_meta(conflicts, consumers=["fixer"], extra={"items": conflicts[:10]}),
    }


def _record_plan_consistency_audit(trace: Any, blackboard: dict[str, Any], patch: str,
                                   payload: dict[str, Any] | None, *, attempt: int, outcome: str) -> None:
    if not trace:
        return
    payload = payload or {}
    trace.record(
        "plan_consistency_audit", attempt=attempt, outcome=outcome,
        status=payload.get("plan_consistency") or "not_checked",
        conflicts=list(payload.get("plan_conflicts") or []),
        plan=_feature_meta(blackboard.get("repair_plan")), patch=_feature_meta(patch),
    )


def _format_qa_repair_feedback(payload: dict[str, Any] | None, *, round_number: int) -> str:
    """Turn a Verifier rejection into an explicit, actionable Fixer checklist."""
    payload = payload or {}
    objections = [str(item) for item in (payload.get("objections") or [])]
    missing = [str(item) for item in (payload.get("missing_requirements") or [])]
    conflicts = [str(item) for item in (payload.get("plan_conflicts") or [])]
    lines = [
        f"Verifier rejected the candidate patch on QA round {round_number}.",
        "Do not resubmit the same diff. Inspect the current diff, repair every applicable item below, rerun the affected verification, and submit only after the repair is reflected in the diff.",
    ]
    if objections:
        lines.append("objections (actionable):")
        lines.extend(f"- {item}" for item in objections)
    if missing:
        lines.append("missing_requirements:")
        lines.extend(f"- {item}" for item in missing)
    if conflicts:
        lines.append("plan_conflicts (informational; reconcile only if they expose an issue-contract or verification defect):")
        lines.extend(f"- {item}" for item in conflicts)
    if not objections and not missing and not conflicts:
        lines.append("The Verifier returned no structured details; inspect the complete QA summary and re-audit the candidate diff before resubmitting.")
    lines.append("If a plan item is wrong, update the implementation decision in your summary and keep the patch aligned with the issue contract and visible tests.")
    return "\n".join(lines)


def _record_consistency_blocked(trace: Any, blackboard: dict[str, Any], patch: str,
                                payload: dict[str, Any] | None, *, reason: str, attempt: int,
                                decision: str | None = None) -> None:
    """Record that a patch was emitted without a completed Verifier repair cycle."""
    payload = payload or {}
    blackboard["consistency_blocked"] = True
    blackboard["consistency_block_reason"] = reason
    if trace:
        trace.record(
            "consistency_blocked",
            reason=reason,
            attempt=attempt,
            decision=decision or payload.get("decision", "REJECT"),
            plan_consistency=payload.get("plan_consistency", "not_checked"),
            conflicts=list(payload.get("plan_conflicts") or []),
            objections=list(payload.get("objections") or []),
            missing_requirements=list(payload.get("missing_requirements") or []),
            patch=_feature_meta(patch),
        )








def _format_repro_status_for_blackboard(state: Any) -> tuple[str, dict[str, Any]]:
    targets = state.get_regression_targets()
    if not targets:
        return "not_reproduced", {"targets": []}
    target_summaries = []
    status = "not_reproduced"
    for key, info in targets.items():
        rep = state._reproduced_regressions.get(key, {})
        if rep.get("runtime_failure"):
            status = "runtime_failure"
        elif rep.get("compile_reproduction") and status == "not_reproduced":
            status = "compile_reproduction"
        target_summaries.append({
            "manifest": key[0].as_posix(),
            "name": key[1],
            "is_in_module": info.get("is_in_module", False),
            "path": info["path"].as_posix(),
            "red_verified": bool(rep.get("runtime_failure") or rep.get("compile_reproduction")),
            "clean_red": bool(rep.get("clean_red", False)),
            "failures": rep.get("failures", [])[:5],
        })
    return status, {"targets": target_summaries}


def _fixer_budget_hint(st: Any, remaining: int) -> str:
    source_changes = _extract_selected_source_paths(st)
    if st.verified() and source_changes:
        return "Verification passed -> call submit_for_qa(summary=...) immediately."
    missing = st.missing_verification_steps() if hasattr(st, "missing_verification_steps") else ""
    if not source_changes:
        return "No production edits made yet. Apply the minimal fix and run required tests before budget runs out."
    return f"Verification still pending ({missing}). Run the pending cargo_test check and call submit_for_qa(summary=...)."


def _repair_progress_signature(state: Any) -> tuple:
    changes = []
    for path in sorted(_extract_selected_source_paths(state)):
        file_path = state.root / path
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest() if file_path.is_file() else None
        changes.append((path.as_posix(), digest))
    targets = []
    for key in sorted(state.get_regression_targets()):
        target_hash = state._target_sha256(key)
        reproduced = state._reproduced_regressions.get(key) or {}
        targets.append((str(key[0]), key[1], target_hash, key in state._passed_regressions,
                        bool(reproduced.get("clean_red"))))
    return (tuple(changes), tuple(targets), state.missing_verification_steps())


_VERIFIER_TOOLS = (
    "read_lines", "read_file", "search_code", "rust_symbols",
    "inspect_rust_neighborhood", "verifier_probe",
)


def _is_valid_external_unreproduced_reason(
    reason: str,
    issue_contract: dict[str, Any] | None = None,
    *,
    state: Any = None,
    blackboard: dict[str, Any] | None = None,
    issue_text: str = "",
) -> bool:
    """Allow skipping Red reproduction only when external constraints match actual runtime/issue evidence."""
    text = (reason or "").strip()
    if len(text) < 20:
        return False
    signals = (issue_contract or {}).get("signals") or {}
    if signals.get("requests_documentation_change"):
        return True
    low = text.lower()
    if state is None and blackboard is None and not issue_text:
        external_markers = (
            "documentation", "doc-only", "rustdoc", "readme", "changelog",
            "windows", "macos", "darwin", "freebsd", "aarch64", "cross-compil",
            "target triple", "platform-specific", "os-specific",
            "network", "offline", "github", "download", "internet", "proxy", "remote",
            "tty", "interactive", "terminal", "gui", "display",
        )
        return any(marker in low for marker in external_markers)

    issue_low = (issue_text or "").lower()
    diff_low = ""
    if state is not None and hasattr(state, "build_patch") and hasattr(state, "changed_paths"):
        try:
            sel = _extract_selected_source_paths(state)
            if sel:
                diff_low = (state.build_patch(sel) or "").lower()
        except Exception:
            diff_low = ""
    evidence_low = issue_low + "\n" + diff_low

    if any(m in low for m in ("documentation", "doc-only", "rustdoc", "readme", "changelog")):
        if signals.get("mentions_documentation") or (state is not None and _documentation_only_patch(state, issue_contract or {})):
            return True

    if any(m in low for m in ("network", "offline", "github", "download", "internet", "proxy", "remote")):
        if (blackboard or {}).get("environment_blocker"):
            return True
        if state is not None:
            env_texts = [
                json.dumps(getattr(state, "_existing_test_failures", []) or [], ensure_ascii=False),
                json.dumps((state.stuck_status() or {}).get("last_test_failure") or {}, ensure_ascii=False) if hasattr(state, "stuck_status") else "",
                json.dumps(list((getattr(state, "_reproduction_notes", {}) or {}).values()), ensure_ascii=False),
            ]
            combined = "\n".join(env_texts).lower()
            if _is_env_dependency_blocker(combined) or any(t in combined for t in ("network", "could not resolve", "connection refused", "timed out")):
                return True
        if any(t in issue_low for t in ("network", "offline", "http", "https", "github", "download", "registry")):
            return True

    platform_markers = ("windows", "macos", "darwin", "freebsd", "aarch64", "cross-compil", "target triple", "platform-specific", "os-specific")
    matched_plat = [m for m in platform_markers if m in low]
    if matched_plat and (any(m in evidence_low for m in matched_plat) or "target_os" in evidence_low or "target_arch" in evidence_low or "cfg(" in diff_low):
        return True

    interactive_markers = ("tty", "interactive", "terminal", "gui", "display")
    matched_ui = [m for m in interactive_markers if m in low]
    if matched_ui and any(t in evidence_low for t in (*interactive_markers, "isatty", "termios", "crossterm", "ratatui", "stdin")):
        return True

    return False


def _changed_lines_by_file(state: Any, rels: list[str]) -> dict[str, list[int]]:
    """Map each changed Rust file to the 1-based line numbers that differ from the original."""
    originals = getattr(state, "originals", None)
    root = getattr(state, "root", None)
    if not isinstance(originals, dict) or root is None:
        return {}
    out: dict[str, list[int]] = {}
    for rel in rels:
        if not rel.endswith(".rs"):
            continue
        entry = originals.get(rel)
        if not entry or entry[0] is None:
            continue
        try:
            old_text = entry[0].decode("utf-8", errors="replace")
            new_text = (Path(root) / rel).read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError):
            continue
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        changed: list[int] = []
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
        for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            changed.extend(range(j1 + 1, j2 + 1))
        if changed:
            out[rel] = changed
    return out


def _build_impact_prompt_block(state: Any, selected: list[Path] | None = None) -> str:
    """Build the CKG-derived producer->consumer impact surface for the current patch."""
    ckg = getattr(state, "ckg", None)
    if ckg is None:
        return ""
    paths = selected if selected is not None else _extract_selected_source_paths(state)
    rels = [p.as_posix() for p in paths] if paths else []
    if not rels:
        try:
            rels = [p.as_posix() for p in state.changed_paths()]
        except Exception:
            rels = []
    try:
        surface = _rust_impact.build_impact_surface(
            ckg, rels, changed_lines=_changed_lines_by_file(state, rels)
        )
    except Exception:
        return ""
    text = _rust_impact.format_impact_surface(surface)
    if not text:
        return ""
    return (
        "CKG Impact Surface (auto-derived; verify these consumers before approving):\n"
        f"{text}\n\n"
    )


def _build_red_lock_prompt_block(state: Any) -> str:
    """Report the Red-phase test lock status so the Verifier knows the oracle is immutable."""
    reporter = getattr(state, "locked_regression_report", None)
    if not callable(reporter):
        return ""
    try:
        report = reporter()
    except Exception:
        return ""
    if not report.get("locked"):
        return ""
    lines = [
        "Locked Red-phase Regression Test(s) (immutable contract oracle; the Fixer cannot rewrite them):",
    ]
    for entry in report.get("locked", []):
        lines.append(
            f"  * {entry.get('path')} :: {entry.get('name')} (sha256={str(entry.get('locked_sha256'))[:12]}, status={entry.get('status')})"
        )
    if report.get("tampered"):
        lines.append(
            "WARNING: a locked regression test was modified or removed after it was confirmed Red. "
            "Treat the current test as an unreliable oracle and REJECT with the concrete tampering evidence."
        )
    for violation in report.get("violations", []) or []:
        lines.append(
            f"  * lock violation: {violation.get('action')} on {violation.get('path')} "
            f"(x{violation.get('count', 1)}) {violation.get('detail', '')}".rstrip()
        )
    unlocked = getattr(state, "_unlocked_regressions", None) or {}
    for meta in unlocked.values():
        lines.append(
            f"  * lock RELEASED by the Architect on {meta.get('path')} :: {meta.get('name')} "
            f"- reason: {meta.get('reason', '')}"
        )
        lines.append(
            "    The Architect confirmed this oracle was factually wrong; judge the replacement test on its own merits."
        )
    return "\n".join(lines) + "\n\n"


def _build_verifier_prompt(
    issue: str,
    issue_contract: dict[str, Any],
    det_qa: dict[str, Any],
    state: Any,
    blackboard: dict[str, Any],
) -> str:
    """Build a unified, evidence-rich QA prompt shared by continuous and staged repair modes."""
    selected = det_qa["selected"]
    hygiene = det_qa["hygiene"]
    missing_apis = det_qa.get("missing_contract_apis") or []
    existing_failures = det_qa.get("existing_test_failures") or []
    latest_test_failure = (state.stuck_status() or {}).get("last_test_failure") or {}
    added_inline_tests = det_qa.get("added_inline_tests") or []
    repro_contents = _collect_regression_test_contents(state)
    verifier_plan = blackboard.get("repair_plan")
    qa_history = blackboard.get("qa_history") or []
    verifier_patch = hygiene.get("patch") or state.build_patch(selected)
    passed_orig_banner = (
        " [CRITICAL: temporary regression test PASSED on the unpatched original checkout]"
        if det_qa.get("passed_on_original") else ""
    )
    qa_hist_block = (
        f"Previous Verifier Decisions (most recent 2):\n{json.dumps(qa_history[-2:], ensure_ascii=False, indent=2)}\n\n"
        if qa_history else ""
    )
    impact_block = _build_impact_prompt_block(state, selected)
    red_lock_block = _build_red_lock_prompt_block(state)
    sibling_block = (blackboard.get("sibling_contracts_text") or "").strip()
    sibling_section = f"{sibling_block}\n\n" if sibling_block else ""
    proxy_cov = det_qa.get("target_proxy_coverage") or {}
    proxy_cov_line = (
        f"- Target Proxy Requirement Coverage: checklist={proxy_cov.get('checklist_count', 0)}, "
        f"test_fns={proxy_cov.get('test_fns') or []}, missing_targets={proxy_cov.get('missing_targets') or []}\n"
        if (proxy_cov.get("checklist_count", 0) >= 2 or proxy_cov.get("missing_targets"))
        else ""
    )
    return (
        f"Issue:\n{issue}\n\n"
        f"Issue Contract:\n{json.dumps(issue_contract, ensure_ascii=False, indent=2)}\n\n"
        f"Deterministic Checks:\n"
        f"- Reproduction evidence: {det_qa.get('reproduction')}{passed_orig_banner}; "
        f"passed_on_original={bool(det_qa.get('passed_on_original'))}; "
        f"reason: {det_qa.get('unreproduced_reason') or '(none)'}\n"
        f"- Post-strip cargo check (--tests + workspace dependents): {det_qa.get('post_strip_status', 'not_run')}\n"
        f"- Observed Existing Repository Test Failures (check observed_revision): {json.dumps(existing_failures, ensure_ascii=False) if existing_failures else 'None observed'}\n"
        f"- Existing Test Smoke Results (timeouts and no-tests are not passes): {json.dumps(det_qa.get('existing_test_smoke') or [], ensure_ascii=False)}\n"
        f"- Smoke Baseline Labels (BASELINE_UNCONFIRMED_NOT_PREEXISTING means the unpatched baseline failed the same "
        f"tests with the same output: the failure is UNCONFIRMED, not proven pre-existing, and cannot clear your "
        f"decision on its own): {json.dumps(det_qa.get('existing_test_smoke_labels') or [], ensure_ascii=False)}\n"
        f"- Submission Gate Warnings (non-blocking, not test passes): {json.dumps(det_qa.get('submission_gate_warnings') or [], ensure_ascii=False)}\n"
        f"- Last Observed Cargo Test Failure (observed_revision may precede current revision {getattr(state, '_revision', None)}): {json.dumps(latest_test_failure, ensure_ascii=False) if latest_test_failure else 'None observed'}\n"
        f"- Non-Regression Tests Executed After Last Edit: {json.dumps(det_qa.get('executed_test_targets') or [], ensure_ascii=False) or 'None'}\n"
        f"- Untouched Integration Test Targets in Modified Crates: {det_qa.get('untested_integration_targets') or 'None'}\n"
        f"- Issue-Referenced Source Files Untouched by Diff: {det_qa.get('untouched_referenced_files') or 'None'}\n"
        f"- Static Cargo.toml-Only Regression Test Warning: {det_qa.get('static_manifest_tests') or 'None'}\n"
        f"- Inline #[cfg(test)]/#[test] Blocks Added to Source Diff: {added_inline_tests or 'None'}\n"
        f"- Pre-existing Inline mod tests Modified in Source Diff: {det_qa.get('modified_inline_tests') or 'None'}\n"
        f"- Missing Requested APIs Heuristic: {missing_apis or 'None detected'}\n"
        f"{proxy_cov_line}"
        f"- Doctest Line-Shift Warnings (WARNING: cargo test --doc passing does NOT prevent exact line-number PASS_TO_PASS benchmark failures): {hygiene.get('doctest_shifts') or 'None'}\n"
        f"- Doc-test Check Status: {det_qa.get('doctest_status')}\n\n"
        f"Temporary Regression Test(s):\n{repro_contents}\n\n"
        f"{red_lock_block}"
        f"{sibling_section}"
        f"{impact_block}"
        f"{qa_hist_block}"
        f"Architect RepairPlan (actual plan used by Fixer):\n{json.dumps(verifier_plan, ensure_ascii=False, indent=2) if verifier_plan else 'No Architect plan: fast path.'}\n\n"
        f"Candidate Unified Diff (complete):\n{verifier_patch}\n\n"
        "Compare the plan's arbitration_rationale, ordered_edits, and forbidden_edits with each other and the diff; "
        "report concrete conflicts with field names and evidence. If plan_consistency='conflict', include an explanation "
        "grounded in the issue contract plus concrete test, caller, API, compile, or runtime evidence. "
        "Audit the issue and tests (using search_code/read_file/read_lines as needed to check sibling impls and consumer match arms), then call submit_qa_decision."
    )


def _should_escalate_qa_to_architect(
    det_qa: dict[str, Any],
    qa_history: list[dict[str, Any]],
    qa_rounds_completed: int,
) -> tuple[bool, str]:
    """Escalate stalled Fixer<->Verifier loops to Architect when contract/scope disputes repeat."""
    if qa_rounds_completed < 1 or not qa_history:
        return False, ""
    latest = qa_history[-1]
    if latest.get("decision") != "REJECT":
        return False, ""
    if det_qa.get("passed_on_original"):
        return True, "QA rejected while temporary regression test still passes on the original code."
    if det_qa.get("untouched_referenced_files"):
        return True, (
            "Repeated QA rejection while issue-referenced source files remain untouched: "
            + ", ".join(det_qa["untouched_referenced_files"])
        )
    if len(qa_history) >= 2 and qa_history[-2].get("decision") == "REJECT":
        prev = qa_history[-2]
        prev_reqs = {str(x).strip().lower() for x in (prev.get("missing_requirements") or []) if str(x).strip()}
        curr_reqs = {str(x).strip().lower() for x in (latest.get("missing_requirements") or []) if str(x).strip()}
        if prev_reqs & curr_reqs:
            return True, "Repeated QA rejection on the same missing requirement(s): " + "; ".join(sorted(prev_reqs & curr_reqs))[:300]
        prev_objs = {str(x).strip().lower()[:100] for x in (prev.get("objections") or []) if str(x).strip()}
        curr_objs = {str(x).strip().lower()[:100] for x in (latest.get("objections") or []) if str(x).strip()}
        if prev_objs & curr_objs:
            return True, "Repeated QA rejection on the same objection(s): " + "; ".join(sorted(prev_objs & curr_objs))[:300]
    return False, ""


_NAV_ALLOWED = [
    "read_file", "read_lines", "search_code", "cargo_workspace",
    "rust_symbols", "inspect_rust_neighborhood", "write_regression_test",
    "revert_file", "cargo_test", "cargo_check", "rust_error_explain",
]
_SUG_ALLOWED = [
    "read_file", "read_lines", "search_code",
    "cargo_workspace", "rust_symbols", "inspect_rust_neighborhood",
]
_FIX_ALLOWED = [
    "read_file", "read_lines", "search_code", "cargo_workspace",
    "rust_symbols", "inspect_rust_neighborhood", "edit_file", "write_file",
    "write_regression_test", "revert_file", "checkpoint",
    "cargo_check", "cargo_test", "rust_error_explain",
]


def _init_orchestration_ctx(
    repo: Path,
    issue: str,
    model: str,
    api_key: str,
    base_url: str | None,
    *,
    tools: list[Any],
    state: Any,
    client: Any,
    trace: Any,
    max_api_calls: int | None,
    compress_fn: Any,
    create_agent_fn: Any,
    chat_model_factory: Any,
    prompts_mod: Any,
    role_callback_cls: Any,
    topology: str = "staged",
) -> SimpleNamespace:
    tools_by_name = {t.name: t for t in tools}
    budgets = allocate_stage_budgets(max_api_calls)
    issue_contract = extract_issue_contract(issue)
    blackboard: dict[str, Any] = {
        "issue_contract": issue_contract,
        "localization": None,
        "repair_plan": None,
        "escalations": [],
        "cognitive_ledger": [],
        "unlocked_regressions": [],
        "qa_history": [],
        "shared_read_excerpts": {},
        "shared_ckg_notes": {},
        "sibling_contracts": {"sibling_impls": [], "test_contracts": []},
        "sibling_contracts_text": "",
        "milestones": {k: False for k in _MILESTONE_KEYS},
        "milestone_credits": {},
        "milestone_history": [],
        "coord_stats": {"downstream_reads": 0, "duplicate_reads": 0, "total_transitions": 0, "zero_delta_handoffs": 0},
        "replan_count": 0,
        "zero_progress_replans": 0,
        "pruned_replans": 0,
        "last_replan_score": None,
        "last_plan_expectation": "",
        "environment_blocker": False,
    }
    qa_repair_loop_enabled = os.getenv("RUSTBENCH_QA_REPAIR_LOOP", "1").strip().lower() not in (
        "0", "false", "off", "no"
    )
    if trace:
        trace.record(
            "multi_agent_start",
            topology=topology,
            budgets=budgets,
            issue_contract=issue_contract,
            qa_repair_loop_enabled=qa_repair_loop_enabled,
        )
        _record_handoff_snapshot(
            trace,
            stage="issue_contract",
            from_role="issue",
            to_role="navigator",
            attempt=1,
            features=_build_issue_contract_features(issue, issue_contract),
            extra={"budgets": budgets, "topology": topology},
        )
    return SimpleNamespace(
        repo=repo,
        issue=issue,
        issue_contract=issue_contract,
        model=model,
        api_key=api_key,
        base_url=base_url,
        tools_by_name=tools_by_name,
        state=state,
        client=client,
        trace=trace,
        max_api_calls=max_api_calls,
        budgets=budgets,
        compress_fn=compress_fn,
        create_agent_fn=create_agent_fn,
        chat_model_factory=chat_model_factory,
        prompts_mod=prompts_mod,
        role_callback_cls=role_callback_cls,
        blackboard=blackboard,
        qa_repair_loop_enabled=qa_repair_loop_enabled,
        max_escalations=max(2, (max_api_calls or 70) // 25),
        max_qa_rounds=max(2, (max_api_calls or 70) // 15),
        escalations_used=0,
        qa_rounds=0,
        qa_escalation_sigs=set(),
        last_qa_feedback="",
        rescue_attempted=False,
        route="planned",
        fix_session_history=None,
        nav_activity={"files_read": [], "symbols_queried": [], "calls_remaining_in_stage": budgets["navigator_cap"]},
        candidate_policy=None,
        unit_coordinator=None,
        _last_progress_sig=None,
    )


def _record_stage_transition(ctx: SimpleNamespace, from_role: str, to_role: str) -> None:
    cstats = ctx.blackboard.setdefault("coord_stats", {})
    cstats["total_transitions"] = int(cstats.get("total_transitions", 0)) + 1
    if from_role in ("fixer", "verifier") and to_role not in ("submission", "fallback"):
        try:
            sig_now = _repair_progress_signature(ctx.state)
            if ctx._last_progress_sig is not None and sig_now == ctx._last_progress_sig:
                cstats["zero_delta_handoffs"] = int(cstats.get("zero_delta_handoffs", 0)) + 1
            ctx._last_progress_sig = sig_now
        except Exception:
            pass


def _flush_suggester_unlocks(ctx: SimpleNamespace, sug_activity: dict[str, Any], sug_payload: dict[str, Any] | None) -> list[dict[str, str]]:
    items = list(sug_activity.get("unlocked_regressions") or [])
    if sug_payload and sug_payload.get("action") == "unlock":
        items.append({
            "name": sug_payload.get("unlocked", ""),
            "path": sug_payload.get("path", ""),
            "reason": sug_payload.get("reason", ""),
        })
    added: list[dict[str, str]] = []
    existing = ctx.blackboard.setdefault("unlocked_regressions", [])
    for item in items:
        if item and item not in existing:
            existing.append(item)
            added.append(item)
            if ctx.trace:
                ctx.trace.record(
                    "red_lock_release",
                    name=item.get("name", ""),
                    path=item.get("path", ""),
                    reason=item.get("reason", ""),
                )
    return added


def _exec_navigator_initial(ctx: SimpleNamespace) -> tuple[str, list[str], str, bool]:
    nav_handoff, nav_holder = _make_handoff_tools("navigator", ctx.state, ctx.blackboard, ctx.nav_activity)
    nav_calls_before = _client_calls_made(ctx.client)
    nav_tools = _wrap_tools_for_role(
        "navigator", ctx.tools_by_name, _NAV_ALLOWED, nav_handoff, nav_holder,
        ctx.state, ctx.client, nav_calls_before, ctx.budgets["navigator_cap"], ctx.nav_activity,
        blackboard=ctx.blackboard,
    )
    checklist = ctx.issue_contract.get("requirement_checklist") or []
    checklist_note = (
        "\nNotice: requirement_checklist contains multiple distinct sub-requirements. "
        "Inside your ONE temporary regression test file, write a separate #[test] fn test_req_1_..., #[test] fn test_req_2_... "
        "covering EVERY checklist requirement rather than only a single happy-path assertion.\n"
        if len(checklist) >= 2
        else ""
    )
    nav_prompt = (
        f"Issue:\n{ctx.issue}\n\n"
        f"Pre-extracted Issue Contract (verify exact names/crates):\n"
        f"{json.dumps(ctx.issue_contract, ensure_ascii=False, indent=2)}\n\n"
        f"{checklist_note}"
        "Locate root cause / candidate fault chains, reproduce the issue with write_regression_test(run_now=True), "
        "then call submit_localization_bundle."
    )
    ctx.state._submit_hint_override = "Call submit_localization_bundle(...) now to hand off findings."
    nav_payload, nav_outcome = _run_role_agent(
        "navigator", ctx.prompts_mod.NAVIGATOR_PROMPT, nav_prompt, nav_tools, nav_holder,
        model=ctx.model, api_key=ctx.api_key, base_url=ctx.base_url, client=ctx.client, trace=ctx.trace,
        compress_fn=ctx.compress_fn, create_agent_fn=ctx.create_agent_fn,
        chat_model_factory=ctx.chat_model_factory, role_callback_cls=ctx.role_callback_cls,
        stage_cap=ctx.budgets["navigator_cap"], reserve_floor=ctx.budgets["verifier_reserve"],
        max_continuations=1,
        continuation_fn=lambda r, s: "Wrap up exploration and call submit_localization_bundle(...) immediately.",
        handoff_only_tools=nav_handoff,
        wrapup_prompt="Exploration budget reached. Call submit_localization_bundle(...) IMMEDIATELY using the files and test outputs already observed.",
    )
    ctx.state._submit_hint_override = None
    _merge_activity_into_blackboard(ctx.blackboard, ctx.nav_activity)
    _prune_unreproduced_regression_targets(ctx.state, trace=ctx.trace)

    # Roll half of unused Navigator stage calls into Fixer budget
    nav_used = max(0, _client_calls_made(ctx.client) - nav_calls_before)
    nav_cap = ctx.budgets.get("navigator_cap")
    if nav_cap is not None and ctx.budgets.get("fixer_attempt_cap") is not None:
        unused_nav = max(0, int(nav_cap) - nav_used)
        rollover = unused_nav // 2
        if rollover > 0:
            ctx.budgets["fixer_attempt_cap"] = int(ctx.budgets["fixer_attempt_cap"]) + rollover
            if ctx.trace:
                ctx.trace.record(
                    "budget_rollover",
                    from_role="navigator",
                    to_role="fixer",
                    nav_used=nav_used,
                    nav_cap=int(nav_cap),
                    rollover=rollover,
                    fixer_attempt_cap=ctx.budgets["fixer_attempt_cap"],
                )

    repro_status, repro_meta = _format_repro_status_for_blackboard(ctx.state)
    nav_auto_synthesized = nav_payload is None
    if nav_payload is None:
        fallback_chains = []
        if getattr(ctx.state, "ckg", None) is not None:
            try:
                fallback_chains = ctx.state.ckg.build_fault_chains(
                    ctx.issue_contract.get("requested_symbols", []),
                    seed_files=ctx.nav_activity["files_read"][:4],
                    max_chains=3,
                )
            except Exception:
                fallback_chains = []
        nav_payload = {
            "summary": f"Navigator budget finished. Files inspected: {ctx.nav_activity['files_read'][:6]}.",
            "candidate_chains": fallback_chains,
            "reproduction_status": repro_status,
            "requested_apis": ctx.issue_contract.get("requested_symbols", []),
            "needs_architecture_planning": True,
            "notes": "Auto-synthesized bundle after stage cap.",
        }
    else:
        if repro_status != "not_reproduced":
            nav_payload["reproduction_status"] = repro_status
        if len(nav_payload.get("candidate_chains", [])) <= 1 and getattr(ctx.state, "ckg", None) is not None and nav_payload.get("needs_architecture_planning"):
            try:
                seeds = list(ctx.issue_contract.get("requested_symbols", [])) + ctx.nav_activity["symbols_queried"][:3]
                extra = ctx.state.ckg.build_fault_chains(seeds, seed_files=ctx.nav_activity["files_read"][:3], max_chains=3)
                if len(extra) > len(nav_payload.get("candidate_chains", [])):
                    nav_payload.setdefault("ckg_suggested_chains", extra)
            except Exception:
                pass
    nav_payload["repro_meta"] = repro_meta
    ctx.blackboard["localization"] = nav_payload
    _refresh_sibling_contracts(ctx)
    _update_milestones_and_kpi(ctx, "navigator")

    ctx.state.save_checkpoint("pre_fix")
    route, route_reasons = route_after_navigation(ctx.blackboard, ctx.state)
    ctx.route = route
    ctx.fix_session_history = nav_holder.get("history") if route == "fast" else None

    ctx.candidate_policy = _repair_candidates.CandidatePolicy(ctx.blackboard, ctx.state, ctx.trace)
    if nav_payload.get("unresolved_chain_ids"):
        ctx.candidate_policy.ambiguity(nav_payload["unresolved_chain_ids"], nav_payload.get("ambiguity_reason", ""))
    ctx.unit_coordinator = _repair_units.RepairUnitCoordinator(
        ctx.state, ctx.tools_by_name, ctx.blackboard, ctx.trace, candidate_policy=ctx.candidate_policy,
    )

    if ctx.trace:
        ctx.trace.record("adaptive_route", route=route, reasons=route_reasons)
        _record_handoff_snapshot(
            ctx.trace,
            stage="navigator_handoff",
            from_role="navigator",
            to_role=("suggester" if route == "planned" else "fixer"),
            attempt=1,
            features=_build_localization_features(
                ctx.blackboard,
                ctx.state,
                repro_contents=_collect_regression_test_contents(ctx.state, max_chars=2000),
            ),
            extra={
                "outcome": nav_outcome,
                "auto_synthesized": nav_auto_synthesized,
                "route": route,
                "route_reasons": route_reasons,
                "files_read_count": len(ctx.nav_activity.get("files_read", [])),
                "symbols_queried_count": len(ctx.nav_activity.get("symbols_queried", [])),
            },
        )
    return route, route_reasons, nav_outcome, nav_auto_synthesized


def _exec_navigator_relocalize(
    ctx: SimpleNamespace,
    reason: str,
    *,
    attempt: int,
    allow_fast_return: bool = False,
) -> tuple[str, list[str], str, Any]:
    nav_handoff2, nav_holder2 = _make_handoff_tools("navigator", ctx.state, ctx.blackboard, ctx.nav_activity)
    nav_tools2 = _wrap_tools_for_role(
        "navigator", ctx.tools_by_name, _NAV_ALLOWED, nav_handoff2, nav_holder2,
        ctx.state, ctx.client, _client_calls_made(ctx.client), max(4, ctx.budgets["navigator_cap"] // 2), ctx.nav_activity,
        blackboard=ctx.blackboard,
    )
    reloc_prompt = (
        f"Issue:\n{ctx.issue}\n\nPrevious localization was insufficient: {reason}.\n"
        "Search more broadly for the true implementation/trait/module and call submit_localization_bundle."
    )
    new_nav, reloc_nav_outcome = _run_role_agent(
        "navigator", ctx.prompts_mod.NAVIGATOR_PROMPT, reloc_prompt, nav_tools2, nav_holder2,
        model=ctx.model, api_key=ctx.api_key, base_url=ctx.base_url, client=ctx.client, trace=ctx.trace,
        compress_fn=ctx.compress_fn, create_agent_fn=ctx.create_agent_fn,
        chat_model_factory=ctx.chat_model_factory, role_callback_cls=ctx.role_callback_cls,
        stage_cap=max(4, ctx.budgets["navigator_cap"] // 2), reserve_floor=ctx.budgets["verifier_reserve"],
        attempt=attempt, max_continuations=1,
        handoff_only_tools=nav_handoff2,
        wrapup_prompt="Relocalization budget reached. Call submit_localization_bundle(...) IMMEDIATELY.",
    )
    _merge_activity_into_blackboard(ctx.blackboard, ctx.nav_activity)
    if new_nav:
        reloc_status, reloc_meta = _format_repro_status_for_blackboard(ctx.state)
        if reloc_status != "not_reproduced":
            new_nav["reproduction_status"] = reloc_status
        new_nav["repro_meta"] = reloc_meta
        ctx.blackboard["localization"] = new_nav
    reloc_route, reloc_reasons = route_after_navigation(ctx.blackboard, ctx.state)
    to_role = "fixer" if (allow_fast_return and reloc_route == "fast" and ctx.blackboard.get("repair_plan") is None) else "suggester"
    _refresh_sibling_contracts(ctx)
    _update_milestones_and_kpi(ctx, "navigator")
    if ctx.trace:
        _record_handoff_snapshot(
            ctx.trace,
            stage="navigator_handoff",
            from_role="navigator",
            to_role=to_role,
            attempt=attempt,
            features=_build_localization_features(
                ctx.blackboard,
                ctx.state,
                repro_contents=_collect_regression_test_contents(ctx.state, max_chars=2000),
            ),
            extra={
                "outcome": reloc_nav_outcome,
                "auto_synthesized": new_nav is None,
                "relocalized": True,
                "route": reloc_route,
                "route_reasons": reloc_reasons,
                "files_read_count": len(ctx.nav_activity.get("files_read", [])),
                "symbols_queried_count": len(ctx.nav_activity.get("symbols_queried", [])),
            },
        )
    ctx.candidate_policy.relocalized()
    return to_role, reloc_reasons, reloc_nav_outcome, new_nav


def _exec_suggester_step(ctx: SimpleNamespace, *, attempt: int) -> tuple[str, dict[str, Any] | None, str, dict[str, int]]:
    sug_activity: dict[str, Any] = {"files_read": [], "symbols_queried": [], "candidate_policy": ctx.candidate_policy}
    sug_handoff, sug_holder = _make_handoff_tools("suggester", ctx.state, ctx.blackboard, sug_activity)
    sug_tools = _wrap_tools_for_role(
        "suggester", ctx.tools_by_name, _SUG_ALLOWED, sug_handoff, sug_holder,
        ctx.state, ctx.client, _client_calls_made(ctx.client), ctx.budgets["suggester_cap"], sug_activity,
        blackboard=ctx.blackboard,
    )
    chains_for_ctx = ctx.blackboard["localization"].get("candidate_chains") or ctx.blackboard["localization"].get("ckg_suggested_chains") or []
    shared_ctx = _build_shared_context(ctx.state.root, ctx.blackboard, chains_for_ctx, max_chars=10000)
    repro_tests_txt = _collect_regression_test_contents(ctx.state, max_chars=2000)
    red_lock_txt = _build_red_lock_prompt_block(ctx.state)
    sibling_txt = (ctx.blackboard.get("sibling_contracts_text") or "").strip()
    sibling_block = f"{sibling_txt}\n\n" if sibling_txt else ""
    ledger_txt = _format_cognitive_ledger(ctx.blackboard)
    escalation_ctx = ""
    if ctx.blackboard["escalations"] or ledger_txt:
        latest_esc_obj = ctx.blackboard["escalations"][-1] if ctx.blackboard["escalations"] else {}
        latest_esc = (
            "\nLatest Escalation Detail:\n"
            + json.dumps(latest_esc_obj, ensure_ascii=False, indent=2)
            if ctx.blackboard["escalations"] else ""
        )
        grad_block = ""
        coord = getattr(ctx, "unit_coordinator", None)
        if coord and getattr(coord, "units", None):
            last_qa = (ctx.blackboard.get("qa_history") or [{}])[-1]
            grad = coord.textual_gradient_summary(
                last_error=str(latest_esc_obj.get("reason") or ctx.last_qa_feedback or ""),
                missing_requirements=list(last_qa.get("missing_requirements") or []),
                candidate_paths=list(ctx.issue_contract.get("referenced_files") or []),
            )
            grad_block = (
                "\nTextual Gradient Summary (freeze/keep verified units, revise only faulty units, add unplanned files as new units):\n"
                + json.dumps(grad, ensure_ascii=False, indent=2)
            )
        escalation_ctx = (
            "\n\nPREVIOUS ATTEMPT FAILED & ESCALATED:\n"
            + (f"{ledger_txt}\n" if ledger_txt else "")
            + latest_esc
            + grad_block
            + "\nDistinguish wrong implementation/test oracle/environment from evidence refuting the mechanism. "
            "Keep frozen_verified_units intact with stable IDs unless directly refuted; focus revisions on faulty_units and add any missing_subtask_candidates as new repair_units. "
            "If a candidate comparison is pending, compare brief mechanisms first; keep the current mechanism unless contract/execution evidence warrants another. "
            "Otherwise, if failure_route is navigator, call request_relocalization and re-map the fault before planning; if failure_route is architect, revise the trait/type/API plan."
        )
    sug_issue_contract_json = json.dumps(ctx.issue_contract, ensure_ascii=False, indent=2)
    sug_loc_json = json.dumps(ctx.blackboard["localization"], ensure_ascii=False, indent=2)
    sug_prompt = (
        f"Issue:\n{ctx.issue}\n\n"
        f"Issue Contract:\n{sug_issue_contract_json}\n\n"
        f"LocalizationBundle:\n{sug_loc_json}\n\n"
        f"Current Temporary Regression Test(s):\n{repro_tests_txt}\n\n"
        f"{red_lock_txt}"
        f"{sibling_block}"
        f"Pre-loaded Source Excerpts & CKG Context:\n{shared_ctx}"
        f"{escalation_ctx}\n\n"
        "Use the pre-loaded excerpts first (at most 1-3 extra read/search calls if needed), verify Rust contracts & Expected Result, and call submit_repair_plan."
        " Provide repair_units with edit_indices, prerequisites, and targeted verification. "
        "Retain stable IDs and definitions of already verified units unless evidence requires changing them.\n"
        + json.dumps(ctx.blackboard.get("repair_units", {}), ensure_ascii=False)
        + ctx.candidate_policy.prompt()
    )
    sug_prompt_breakdown = {
        "issue_chars": len(ctx.issue or ""),
        "issue_contract_chars": len(sug_issue_contract_json),
        "localization_chars": len(sug_loc_json),
        "repro_tests_chars": len(repro_tests_txt),
        "shared_ctx_chars": len(shared_ctx),
        "escalation_chars": len(escalation_ctx),
        "total_chars": len(sug_prompt),
    }
    ctx.state._submit_hint_override = "Call submit_repair_plan(...) now."
    sug_payload, sug_outcome = _run_role_agent(
        "suggester", ctx.prompts_mod.SUGGESTER_PROMPT, sug_prompt, sug_tools, sug_holder,
        model=ctx.model, api_key=ctx.api_key, base_url=ctx.base_url, client=ctx.client, trace=ctx.trace,
        compress_fn=ctx.compress_fn, create_agent_fn=ctx.create_agent_fn,
        chat_model_factory=ctx.chat_model_factory, role_callback_cls=ctx.role_callback_cls,
        stage_cap=ctx.budgets["suggester_cap"], reserve_floor=ctx.budgets["verifier_reserve"],
        attempt=attempt, max_continuations=1,
        continuation_fn=lambda r, s: "Call submit_repair_plan(...) immediately with your best plan.",
        handoff_only_tools=sug_handoff,
        wrapup_prompt="Planning exploration complete. Call submit_repair_plan(...) (or request_relocalization if chains are wrong) IMMEDIATELY with concrete file/symbol/line edits.",
    )
    ctx.state._submit_hint_override = None
    _merge_activity_into_blackboard(ctx.blackboard, sug_activity)
    _flush_suggester_unlocks(ctx, sug_activity, sug_payload)

    if sug_payload and sug_payload.get("action") == "unlock":
        rem_after_unlock = _client_remaining_calls(ctx.client)
        if rem_after_unlock is None or rem_after_unlock > ctx.budgets["verifier_reserve"] + 1:
            sug_holder.update(done=False, payload=None)
            plan_followup = (
                "The Red-phase regression lock has been released. Now call submit_repair_plan(...) "
                "IMMEDIATELY with concrete ordered_edits and the corrected test oracle expectation for Fixer."
            )
            sug_payload2, sug_outcome2 = _run_role_agent(
                "suggester", ctx.prompts_mod.SUGGESTER_PROMPT, plan_followup, sug_tools, sug_holder,
                model=ctx.model, api_key=ctx.api_key, base_url=ctx.base_url, client=ctx.client, trace=ctx.trace,
                compress_fn=ctx.compress_fn, create_agent_fn=ctx.create_agent_fn,
                chat_model_factory=ctx.chat_model_factory, role_callback_cls=ctx.role_callback_cls,
                stage_cap=max(2, ctx.budgets["suggester_cap"] // 2), reserve_floor=ctx.budgets["verifier_reserve"],
                history=sug_holder.get("history"), attempt=attempt, max_continuations=0,
                handoff_only_tools=sug_handoff,
                wrapup_prompt="Call submit_repair_plan(...) IMMEDIATELY after unlocking the regression test.",
            )
            _merge_activity_into_blackboard(ctx.blackboard, sug_activity)
            _flush_suggester_unlocks(ctx, sug_activity, sug_payload2)
            if sug_payload2:
                sug_payload = sug_payload2
                sug_outcome = sug_outcome2

    if sug_payload and sug_payload.get("action") == "relocalize" and ctx.escalations_used < ctx.max_escalations:
        ctx.escalations_used += 1
        if ctx.trace:
            ctx.trace.record("escalation", from_role="suggester", to_role="navigator", reason=sug_payload.get("reason", ""))
            _record_handoff_snapshot(
                ctx.trace,
                stage="suggester_handoff",
                from_role="suggester",
                to_role="navigator",
                attempt=ctx.escalations_used,
                features=_build_repair_plan_features(ctx.blackboard.get("repair_plan"), ctx.blackboard["escalations"]),
                extra={
                    "action": "relocalize",
                    "outcome": sug_outcome,
                    "relocalize_reason": str(sug_payload.get("reason", ""))[:240],
                    "sug_prompt_breakdown": sug_prompt_breakdown,
                },
            )
        return "relocalize", sug_payload, sug_outcome, sug_prompt_breakdown

    if sug_payload and sug_payload.get("action") == "plan":
        ctx.blackboard["repair_plan"] = sug_payload
        ctx.fix_session_history = None
    elif sug_payload and sug_payload.get("action") == "unlock":
        pass
    else:
        prior_plan = ctx.candidate_policy.fallback_plan(ctx.blackboard.get("repair_plan"))
        synth_edits = _synthesize_ordered_edits_from_chains(chains_for_ctx)
        ctx.blackboard["repair_plan"] = prior_plan or {
            "action": "plan",
            "selected_chain_id": (chains_for_ctx[0].get("chain_id", "C1") if chains_for_ctx else "C1"),
            "arbitration_rationale": (
                "Architect budget ended before submitting a plan; start directly from these candidate chain files/symbols "
                "and apply minimal edits within 3-5 steps instead of re-reading the repository."
            ),
            "ordered_edits": synth_edits,
            "rust_invariants": ["Keep edits minimal and preserve unrelated doctest line counts."],
            "api_checklist": ctx.issue_contract.get("new_api_candidates", []),
            "_synthesized": True,
        }
        ctx.fix_session_history = None
    plan_now = ctx.blackboard.get("repair_plan") or {}
    if attempt > 1 or ctx.blackboard.get("escalations"):
        ctx.blackboard["replan_count"] = int(ctx.blackboard.get("replan_count") or 0) + 1
        ctx.blackboard["last_replan_score"] = _compute_milestone_state(ctx.state)["progress_score"]
    ctx.blackboard["last_plan_expectation"] = _compact_handoff_text(
        plan_now.get("arbitration_rationale") or ", ".join(str(e.get("description") or "") for e in (plan_now.get("ordered_edits") or [])[:3]),
        180,
    )
    _update_milestones_and_kpi(ctx, "suggester")
    if ctx.trace:
        plan_now = ctx.blackboard.get("repair_plan") or {}
        _record_handoff_snapshot(
            ctx.trace,
            stage="suggester_handoff",
            from_role="suggester",
            to_role="fixer",
            attempt=attempt,
            features=_build_repair_plan_features(ctx.blackboard.get("repair_plan"), ctx.blackboard["escalations"]),
            extra={
                "action": "synthesized_fallback" if plan_now.get("_synthesized") else (sug_payload or {}).get("action", "plan"),
                "outcome": sug_outcome,
                "synthesized": bool(plan_now.get("_synthesized")),
                "sug_prompt_breakdown": sug_prompt_breakdown,
            },
        )
    return "plan_ready", sug_payload, sug_outcome, sug_prompt_breakdown


def _exec_fixer_step(ctx: SimpleNamespace, *, attempt: int) -> dict[str, Any]:
    rem_for_fixer = _client_remaining_calls(ctx.client)
    in_rescue_pass = bool(
        rem_for_fixer is not None
        and rem_for_fixer <= ctx.budgets["verifier_reserve"]
        and not ctx.rescue_attempted
        and rem_for_fixer >= 2
        and _extract_selected_source_paths(ctx.state)
    )
    if in_rescue_pass:
        ctx.rescue_attempted = True
        fixer_cap_now = max(1, rem_for_fixer - 1)
        fixer_reserve_now = 0
        fixer_continuations = 0
        allow_escalate_now = False
    else:
        fixer_cap_now = ctx.budgets["fixer_attempt_cap"]
        fixer_reserve_now = ctx.budgets["verifier_reserve"]
        fixer_continuations = 2
        allow_escalate_now = (ctx.escalations_used < ctx.max_escalations) and not ctx.blackboard.get("environment_blocker")

    unit_paths = [e["file"] for e in _synthesize_ordered_edits_from_chains(
        ctx.blackboard["localization"].get("candidate_chains") or ctx.blackboard["localization"].get("ckg_suggested_chains") or []
    ) if e.get("file")]
    if ctx.escalations_used:
        ctx.state.reset_stuck_counters()
    ctx.unit_coordinator.configure(ctx.blackboard.get("repair_plan"), unit_paths)
    ctx.unit_coordinator.select()
    fix_activity: dict[str, Any] = {
        "files_read": [],
        "symbols_queried": [],
        "unit_coordinator": ctx.unit_coordinator,
        "candidate_policy": ctx.candidate_policy,
    }
    fix_handoff, fix_holder = _make_handoff_tools(
        "fixer", ctx.state, ctx.blackboard, fix_activity, allow_escalate=allow_escalate_now,
    )
    if "generate_patch" in ctx.tools_by_name:
        gp_orig = ctx.tools_by_name["generate_patch"].func

        @tool
        def preview_patch() -> str:
            """Preview current diff without submitting. Call submit_for_qa when verification passes."""
            return gp_orig(paths=None, submit=False)

        extra_fixer_tools = fix_handoff + [preview_patch]
    else:
        extra_fixer_tools = fix_handoff

    fix_tools = _wrap_tools_for_role(
        "fixer", ctx.tools_by_name, _FIX_ALLOWED, extra_fixer_tools, fix_holder,
        ctx.state, ctx.client, _client_calls_made(ctx.client), fixer_cap_now, fix_activity,
        auto_escalate_on_stuck=allow_escalate_now,
        blackboard=ctx.blackboard,
    )
    chains_for_fix = ctx.blackboard["localization"].get("candidate_chains") or ctx.blackboard["localization"].get("ckg_suggested_chains") or []
    shared_fix_ctx = _build_shared_context(ctx.state.root, ctx.blackboard, chains_for_fix, max_chars=1800)
    repro_fix_txt = _collect_regression_test_contents(ctx.state, max_chars=1800)
    impact_paths = _extract_selected_source_paths(ctx.state)
    if not impact_paths:
        impact_paths = list(dict.fromkeys(
            Path(n["path"])
            for ch in chains_for_fix
            for n in (ch.get("nodes") or [])
            if n.get("path") and not _is_non_prod_path(n["path"])
        ))[:4]
    impact_fix_txt = _build_impact_prompt_block(ctx.state, impact_paths)[:1800]
    if impact_fix_txt and not impact_fix_txt.endswith("\n\n"):
        impact_fix_txt = impact_fix_txt.rstrip() + "\n\n"
    selected_for_diff = _extract_selected_source_paths(ctx.state)
    cur_diff_txt = ""
    if selected_for_diff and (ctx.last_qa_feedback or ctx.blackboard.get("escalations")):
        raw_cur_diff = ctx.state.build_patch(selected_for_diff)
        if raw_cur_diff:
            cur_diff_txt = f"Current Candidate Unified Diff (latest edits):\n{raw_cur_diff[:4500]}\n\n"
    failure_route = str(ctx.blackboard.get("last_failure_route") or "")
    handoff_card = build_handoff_card(
        ctx.issue_contract,
        ctx.blackboard,
        qa_feedback=ctx.last_qa_feedback,
        failure_route=failure_route,
        active_unit=ctx.unit_coordinator._unit(ctx.unit_coordinator.active) if ctx.unit_coordinator.active else None,
    )
    downstream_note = (
        f"Workspace Dependents (verify post-strip compilation): {ctx.blackboard['downstream_dependents']}\n\n"
        if ctx.blackboard.get("downstream_dependents")
        else ""
    )
    qa_block = f"PREVIOUS QA OBJECTIONS (MUST FIX):\n{ctx.last_qa_feedback}\n\n" if ctx.last_qa_feedback else ""
    sibling_fix = (ctx.blackboard.get("sibling_contracts_text") or "").strip()[:1000]
    sibling_fix_block = f"{sibling_fix}\n\n" if sibling_fix else ""
    ledger_fix = _format_cognitive_ledger(ctx.blackboard, max_entries=3) if (ctx.blackboard.get("escalations") or ctx.last_qa_feedback) else ""
    ledger_fix_block = f"{ledger_fix[:700]}\n\n" if ledger_fix else ""
    unlocked = ctx.blackboard.get("unlocked_regressions") or []
    unlock_block = ""
    if unlocked:
        lines = [
            "RED-PHASE LOCK RELEASED BY THE ARCHITECT (you may correct these tests):",
        ]
        for item in unlocked[-3:]:
            lines.append(
                f"  * {item.get('name')} ({item.get('path')}): {item.get('reason')}"
            )
        lines.append(
            "Correct the released test with write_regression_test and re-verify the Red->Green pair."
        )
        unlock_block = "\n".join(lines) + "\n\n"
    last_test_failure = ctx.state.stuck_status().get("last_test_failure") or {}
    observed_failures = list(getattr(ctx.state, "_existing_test_failures", []) or [])
    failure_evidence = (
        "Observed Cargo test failure evidence (check observed_revision; a later edit may have changed it):\n"
        + json.dumps({"current_revision": getattr(ctx.state, "_revision", None),
                      "last_failure": last_test_failure,
                      "existing_test_failures": observed_failures}, ensure_ascii=False, indent=2)
        + "\n\n"
    ) if last_test_failure or observed_failures else ""
    esc_note = (
        "If stuck, call escalate_for_replan WITHOUT restoring pre_fix; Coordinator preserves verified unrelated units; follow the handoff card's failure_route."
        if allow_escalate_now
        else "Final repair phase (escalation limit reached): resolve remaining test/edit issues directly and call submit_for_qa."
    )
    fix_prompt = (
        f"Issue:\n{ctx.issue}\n\n"
        f"Repair Handoff Card (compact; use targeted reads for missing details):\n{handoff_card}\n\n"
        f"Temporary Regression Test(s):\n{repro_fix_txt}\n\n"
        f"Targeted Source Excerpts:\n{shared_fix_ctx}\n\n"
        f"{impact_fix_txt}"
        f"{sibling_fix_block}"
        f"{cur_diff_txt}"
        f"{downstream_note}"
        f"{unlock_block}"
        f"{ledger_fix_block}"
        f"{qa_block}"
        f"{failure_evidence}"
        "Implement the fix with minimal edits, pass all temporary regression tests + relevant existing tests, "
        f"and call submit_for_qa(summary=...). {esc_note}"
    )
    fix_prompt_breakdown = {
        "issue_chars": len(ctx.issue or ""),
        "handoff_card_chars": len(handoff_card),
        "repro_tests_chars": len(repro_fix_txt),
        "shared_ctx_chars": len(shared_fix_ctx),
        "impact_chars": len(impact_fix_txt),
        "cur_diff_chars": len(cur_diff_txt),
        "downstream_chars": len(downstream_note),
        "qa_feedback_chars": len(qa_block),
        "test_failure_evidence_chars": len(failure_evidence),
        "total_chars": len(fix_prompt),
    }
    ctx.state._submit_hint_override = _fixer_budget_hint
    fix_payload, fix_outcome = _run_scheduled_fixer(
        ctx.prompts_mod.FIXER_PROMPT, fix_prompt, fix_tools, fix_holder,
        coordinator=ctx.unit_coordinator, allow_escalate=allow_escalate_now,
        model=ctx.model, api_key=ctx.api_key, base_url=ctx.base_url, client=ctx.client, trace=ctx.trace,
        compress_fn=ctx.compress_fn, create_agent_fn=ctx.create_agent_fn,
        chat_model_factory=ctx.chat_model_factory, role_callback_cls=ctx.role_callback_cls,
        stage_cap=fixer_cap_now, reserve_floor=fixer_reserve_now,
        history=ctx.fix_session_history,
        attempt=attempt, max_continuations=fixer_continuations,
        continuation_fn=lambda r, s: (
            "Verification passed - call submit_for_qa(summary=...) now!"
            if ctx.state.verified() and _extract_selected_source_paths(ctx.state)
            else f"Continue repair and verification ({ctx.state.missing_verification_steps()}), then call submit_for_qa."
        ),
    )
    ctx.fix_session_history = fix_holder.get("history") or ctx.fix_session_history
    ctx.state._submit_hint_override = None
    _merge_activity_into_blackboard(ctx.blackboard, fix_activity)
    return {
        "fix_payload": fix_payload,
        "fix_outcome": fix_outcome,
        "in_rescue_pass": in_rescue_pass,
        "fix_prompt_breakdown": fix_prompt_breakdown,
    }


def _record_escalation(
    ctx: SimpleNamespace,
    *,
    from_role: str,
    to_role: str,
    failure_route: str,
    reason: str,
    handoff_stage: str | None = None,
    handoff_to_role: str | None = None,
    handoff_extra: dict[str, Any] | None = None,
) -> None:
    ctx.escalations_used += 1
    ctx.blackboard["last_failure_route"] = failure_route
    ctx.blackboard["escalations"].append({
        "attempt": ctx.escalations_used,
        "reason": reason,
        "repair_units": ctx.unit_coordinator.status(),
        "failure_route": failure_route,
        "failed_plan": ctx.blackboard.get("repair_plan"),
        "stuck_status": ctx.state.stuck_status(),
        "existing_test_failures": list(getattr(ctx.state, "_existing_test_failures", [])),
    })
    _append_cognitive_entry(
        ctx,
        source=from_role,
        actual=reason,
        falsified=str((ctx.blackboard.get("repair_plan") or {}).get("selected_chain_id") or ""),
    )
    if ctx.trace:
        ctx.trace.record(
            "escalation",
            from_role=from_role,
            to_role=to_role,
            reason=reason,
            attempt=ctx.escalations_used,
            failure_route=failure_route,
        )
        if handoff_stage:
            _record_handoff_snapshot(
                ctx.trace,
                stage=handoff_stage,
                from_role=from_role,
                to_role=handoff_to_role or to_role,
                attempt=ctx.escalations_used + ctx.qa_rounds,
                features=_build_fixer_handoff_features(ctx.state, ctx.blackboard, det_qa=None),
                extra=handoff_extra or {},
            )
    ctx.state.reset_stuck_counters()
    ctx.fix_session_history = None


def _exec_det_qa_step(
    ctx: SimpleNamespace,
    *,
    fix_payload: dict[str, Any] | None,
    fix_outcome: str,
    in_rescue_pass: bool,
    fix_prompt_breakdown: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    _prune_unreproduced_regression_targets(ctx.state, trace=ctx.trace)
    unreproduced_reason = (fix_payload or {}).get("unreproduced_reason", "")
    det_qa = run_deterministic_qa_checks(
        ctx.state, ctx.issue_contract, run_doctest_on_shift=True,
        unreproduced_reason=unreproduced_reason,
    )
    if not det_qa["ok"] and "Pending regression verification" in det_qa.get("reason", ""):
        rem_prune = _client_remaining_calls(ctx.client)
        if in_rescue_pass or fix_outcome in ("reserve_reached", "stage_cap_reached") or (rem_prune is not None and rem_prune <= ctx.budgets["verifier_reserve"] + 2):
            if _prune_secondary_unverified_regression_targets(ctx.state, trace=ctx.trace):
                det_qa = run_deterministic_qa_checks(
                    ctx.state, ctx.issue_contract, run_doctest_on_shift=True,
                    unreproduced_reason=unreproduced_reason,
                )

    if (
        det_qa.get("ok")
        and not det_qa.get("passed_on_original")
        and hasattr(ctx.state, "verified")
        and ctx.state.verified()
        and hasattr(ctx.state, "_capture_snapshot")
    ):
        ctx.state._best_verified_snapshot = ctx.state._capture_snapshot()
    _update_milestones_and_kpi(ctx, "fixer", det_qa=det_qa)

    if ctx.trace:
        _record_handoff_snapshot(
            ctx.trace,
            stage="fixer_handoff",
            from_role="fixer",
            to_role="qa",
            attempt=attempt,
            features=_build_fixer_handoff_features(ctx.state, ctx.blackboard, det_qa=det_qa),
            extra={
                "action": (fix_payload or {}).get("action", "incomplete"),
                "outcome": fix_outcome,
                "in_rescue_pass": in_rescue_pass,
                "fix_prompt_breakdown": fix_prompt_breakdown,
            },
        )
    return det_qa


def _handle_low_budget_det_qa_bypass(ctx: SimpleNamespace, det_qa: dict[str, Any]) -> str:
    selected = det_qa["selected"]
    hygiene = det_qa["hygiene"]
    missing_apis = det_qa["missing_contract_apis"]
    ctx.state.patch = hygiene["patch"] or ctx.state.build_patch(selected)
    if ctx.trace:
        consistency_pending = bool(ctx.blackboard.get("_qa_repair_pending"))
        consistency_unchecked = bool(ctx.blackboard.get("repair_plan")) and not consistency_pending
        if consistency_pending or consistency_unchecked:
            _record_consistency_blocked(
                ctx.trace, ctx.blackboard, ctx.state.patch, ctx.blackboard.get("_last_qa_payload"),
                reason=(
                    "Verifier rejection was followed by budget exhaustion before a repair could be re-verified."
                    if consistency_pending else
                    "Verifier consistency audit was bypassed because the remaining API budget was exhausted."
                ),
                attempt=ctx.qa_rounds + 1,
                decision="REJECT" if consistency_pending else "BYPASS_LOW_BUDGET",
            )
            ctx.trace.record("qa_gate", deterministic_ok=True, llm_decision="CONSISTENCY_BLOCKED", missing_apis=missing_apis)
        else:
            ctx.trace.record("qa_gate", deterministic_ok=True, llm_decision="BYPASS_LOW_BUDGET", missing_apis=missing_apis)
        _record_plan_consistency_audit(
            ctx.trace, ctx.blackboard, ctx.state.patch, None,
            attempt=ctx.qa_rounds + 1, outcome="bypassed_low_budget",
        )
        _record_handoff_snapshot(
            ctx.trace,
            stage="verifier_handoff",
            from_role="verifier",
            to_role="submission",
            attempt=ctx.qa_rounds + 1,
            features=_build_verifier_handoff_features(ctx.blackboard.get("_last_qa_payload") if consistency_pending else None),
            extra={
                "decision": "CONSISTENCY_BLOCKED" if (consistency_pending or consistency_unchecked) else "BYPASS_LOW_BUDGET",
                "outcome": "blocked" if (consistency_pending or consistency_unchecked) else "bypassed",
                "ver_prompt_chars": 0,
            },
        )
    return ctx.state.patch


def _exec_verifier_step(ctx: SimpleNamespace, det_qa: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    selected = det_qa["selected"]
    hygiene = det_qa["hygiene"]
    ver_activity: dict[str, Any] = {"files_read": [], "symbols_queried": []}
    ver_handoff, ver_holder = _make_handoff_tools("verifier", ctx.state, ctx.blackboard, ver_activity)
    ver_tools = _wrap_tools_for_role(
        "verifier", ctx.tools_by_name, list(_VERIFIER_TOOLS),
        ver_handoff, ver_holder, ctx.state, ctx.client, _client_calls_made(ctx.client),
        ctx.budgets["verifier_cap"], ver_activity,
        blackboard=ctx.blackboard,
    )
    verifier_patch = hygiene.get("patch") or ctx.state.build_patch(selected)
    ver_prompt = _build_verifier_prompt(ctx.issue, ctx.issue_contract, det_qa, ctx.state, ctx.blackboard)
    ver_payload, ver_outcome = _run_role_agent(
        "verifier", ctx.prompts_mod.VERIFIER_PROMPT, ver_prompt, ver_tools, ver_holder,
        model=ctx.model, api_key=ctx.api_key, base_url=ctx.base_url, client=ctx.client, trace=ctx.trace,
        compress_fn=ctx.compress_fn, create_agent_fn=ctx.create_agent_fn,
        chat_model_factory=ctx.chat_model_factory, role_callback_cls=ctx.role_callback_cls,
        stage_cap=ctx.budgets["verifier_cap"], reserve_floor=0,
        attempt=ctx.qa_rounds + 1, max_continuations=1,
        continuation_fn=lambda r, s: "Call submit_qa_decision('APPROVE' or 'REJECT', ...); include an issue/evidence explanation when reporting plan_consistency='conflict'.",
        handoff_only_tools=ver_handoff,
        wrapup_prompt="Verification cap reached. Call submit_qa_decision('APPROVE' or 'REJECT', ...) IMMEDIATELY; include an issue/evidence explanation for any plan conflict.",
    )
    if not ver_payload:
        inconclusive_clean = bool(
            det_qa.get("ok")
            and not ctx.blackboard.get("_qa_repair_pending")
            and not det_qa.get("passed_on_original")
            and not det_qa.get("missing_contract_apis")
            and not det_qa.get("added_inline_tests")
            and not det_qa.get("comment_doctest_shifts")
            and not det_qa.get("static_manifest_tests")
        )
        if inconclusive_clean:
            ver_payload = {
                "decision": "INCONCLUSIVE_FALLBACK",
                "summary": "Verifier did not return a structured QA decision before its stage budget ended; deterministic QA passed cleanly.",
                "missing_requirements": [],
                "objections": [],
                "plan_consistency": "not_checked",
                "plan_conflicts": [],
                "_inconclusive": True,
            }
        else:
            ver_payload = {
                "decision": "REJECT",
                "summary": "Verifier did not return a structured QA decision before its stage budget ended.",
                "missing_requirements": [],
                "objections": ["No structured Verifier decision was returned; the patch requires another audit before submission."],
                "plan_consistency": "not_checked",
                "plan_conflicts": [],
            }
    decision = str(ver_payload.get("decision", "REJECT")).strip().upper()
    if decision not in ("APPROVE", "REJECT", "INCONCLUSIVE_FALLBACK"):
        decision = "REJECT"
        ver_payload["decision"] = decision
    if ver_payload:
        ctx.blackboard["qa_history"].append(dict(ver_payload))
    if ctx.trace:
        ctx.trace.record(
            "qa_gate", deterministic_ok=True, llm_decision=decision, payload=ver_payload,
            qa_round=ctx.qa_rounds + 1, reproduction=det_qa.get("reproduction"),
            unreproduced_reason=det_qa.get("unreproduced_reason"),
        )
        _record_plan_consistency_audit(
            ctx.trace, ctx.blackboard, verifier_patch, ver_payload,
            attempt=ctx.qa_rounds + 1,
            outcome=("inconclusive_fallback" if decision == "INCONCLUSIVE_FALLBACK" else ver_outcome),
        )
        _record_handoff_snapshot(
            ctx.trace,
            stage="verifier_handoff",
            from_role="verifier",
            to_role=("submission" if decision in ("APPROVE", "INCONCLUSIVE_FALLBACK") else "fixer"),
            attempt=ctx.qa_rounds + 1,
            features=_build_verifier_handoff_features(ver_payload),
            extra={
                "decision": decision,
                "outcome": ver_outcome,
                "ver_prompt_chars": len(ver_prompt),
            },
        )
    return decision, ver_payload, ver_outcome


def _resolve_fallback_patch(ctx: SimpleNamespace, *, error_message: str = "Model API call limit reached") -> str:
    fallback = _clean_and_audit_fallback_patch(
        ctx.state, ctx.issue_contract, ctx.state.auto_submit_verified_patch(allow_snapshot_fallback=True), trace=ctx.trace,
    )
    if fallback:
        if ctx.trace:
            ctx.trace.record("auto_submit_verified", patch=fallback, stage="deterministic_qa_last_resort")
        return fallback
    degraded = _try_degraded_patch_fallback(
        ctx.state, trace=ctx.trace, allow_unverified_compiling=True,
        issue_contract=ctx.issue_contract, blackboard=ctx.blackboard,
    )
    if degraded:
        return degraded
    raise RuntimeError(error_message)

def run_multi_agent(
    repo: Path,
    issue: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    *,
    tools: list[Any],
    state: Any,
    client: Any,
    trace: Any = None,
    max_api_calls: int | None = 70,
    compress_fn: Any,
    create_agent_fn: Any,
    chat_model_factory: Any,
    prompts_mod: Any,
    role_callback_cls: Any = None,
) -> str:
    """Execute the Adaptive Locate-Suggest-Fix-Verify multi-agent pipeline."""
    ctx = _init_orchestration_ctx(
        repo, issue, model, api_key, base_url,
        tools=tools, state=state, client=client, trace=trace, max_api_calls=max_api_calls,
        compress_fn=compress_fn, create_agent_fn=create_agent_fn,
        chat_model_factory=chat_model_factory, prompts_mod=prompts_mod,
        role_callback_cls=role_callback_cls, topology="staged",
    )

    try:
        # ------------------------------------------------------------------
        # Stage 1: Navigator & Reproducer
        # ------------------------------------------------------------------
        rem_init = _client_remaining_calls(ctx.client)
        if rem_init is not None and rem_init <= 0:
            return _resolve_fallback_patch(ctx)
        _route, _reasons, _nav_out, _nav_auto = _exec_navigator_initial(ctx)
        _record_stage_transition(ctx, "navigator", "suggester" if ctx.route == "planned" else "fixer")

        while True:
            if ctx.escalations_used:
                ctx.state.reset_stuck_counters()
            rem_before_stage = _client_remaining_calls(ctx.client)
            if rem_before_stage is not None and rem_before_stage <= 0:
                _record_stage_transition(ctx, "stage", "fallback")
                return _resolve_fallback_patch(ctx)

            # --------------------------------------------------------------
            # Stage 2: Rust Architect / Suggester (if Planned Path or Escalated)
            # --------------------------------------------------------------
            if ctx.route == "planned" and (ctx.blackboard["repair_plan"] is None or ctx.blackboard.get("_need_replan")):
                while True:
                    rem_sug = _client_remaining_calls(ctx.client)
                    if rem_sug is not None and rem_sug <= 0:
                        _record_stage_transition(ctx, "suggester", "fallback")
                        return _resolve_fallback_patch(ctx)
                    if rem_sug is not None and rem_sug <= ctx.budgets["verifier_reserve"]:
                        ctx.blackboard["_need_replan"] = False
                        break
                    ctx.blackboard["_need_replan"] = False
                    outcome_kind, sug_payload, _sug_out, _bd = _exec_suggester_step(
                        ctx, attempt=ctx.escalations_used + 1,
                    )
                    if outcome_kind == "relocalize":
                        _record_stage_transition(ctx, "suggester", "navigator")
                        reloc_reason = str((sug_payload or {}).get("reason", "")) or "Relocalization requested"
                        next_to_role, _reloc_reasons, _reloc_out, _new_nav = _exec_navigator_relocalize(
                            ctx, reloc_reason, attempt=ctx.escalations_used + 1, allow_fast_return=True,
                        )
                        _record_stage_transition(ctx, "navigator", next_to_role)
                        if next_to_role == "fixer":
                            ctx.route = "fast"
                            ctx.blackboard["_need_replan"] = False
                            break
                        ctx.route = "planned"
                        ctx.blackboard["_need_replan"] = True
                        continue
                    _record_stage_transition(ctx, "suggester", "fixer")
                    break

            # --------------------------------------------------------------
            # Stage 3: Rust Fixer
            # --------------------------------------------------------------
            fix_res = _exec_fixer_step(ctx, attempt=ctx.escalations_used + ctx.qa_rounds + 1)
            fix_payload = fix_res["fix_payload"]
            fix_outcome = fix_res["fix_outcome"]
            in_rescue_pass = fix_res["in_rescue_pass"]
            fix_prompt_breakdown = fix_res["fix_prompt_breakdown"]

            pruned_from_fixer = False
            if fix_payload and fix_payload.get("action") == "escalate" and ctx.escalations_used < ctx.max_escalations:
                reason = str(fix_payload.get("reason", "stuck"))
                if _is_env_dependency_blocker(reason):
                    ctx.blackboard["environment_blocker"] = True
                stuck = ctx.state.stuck_status()
                failure_route = (
                    "architect"
                    if fix_payload.get("candidate_comparison")
                    else _classify_failure_route(reason, stuck)
                )
                to_role = "navigator" if failure_route == "navigator" else "suggester"
                _record_escalation(
                    ctx,
                    from_role="fixer",
                    to_role=to_role,
                    failure_route=failure_route,
                    reason=reason,
                    handoff_stage="fixer_handoff",
                    handoff_extra={
                        "action": "escalate",
                        "outcome": fix_outcome,
                        "escalate_reason": reason[:240],
                        "failure_route": failure_route,
                        "in_rescue_pass": in_rescue_pass,
                        "fix_prompt_breakdown": fix_prompt_breakdown,
                    },
                )
                ctx.last_qa_feedback = ""
                if to_role == "navigator":
                    _record_stage_transition(ctx, "fixer", "navigator")
                    next_to_role, _reloc_reasons, _reloc_out, _new_nav = _exec_navigator_relocalize(
                        ctx, reason, attempt=ctx.escalations_used + 1, allow_fast_return=True,
                    )
                    _record_stage_transition(ctx, "navigator", next_to_role)
                    if next_to_role == "fixer":
                        ctx.route = "fast"
                        ctx.blackboard["_need_replan"] = False
                    else:
                        ctx.route = "planned"
                        ctx.blackboard["_need_replan"] = True
                    continue
                if not fix_payload.get("candidate_comparison"):
                    prune, prune_reason = _should_prune_suggester_replan(ctx)
                    if prune:
                        ctx.blackboard["pruned_replans"] = int(ctx.blackboard.get("pruned_replans") or 0) + 1
                        if ctx.trace:
                            ctx.trace.record("stage_edge_pruned", from_role="fixer", to_role="suggester", reason=prune_reason)
                        if not _extract_selected_source_paths(ctx.state):
                            _record_stage_transition(ctx, "fixer", "fallback")
                            return _resolve_fallback_patch(ctx, error_message=prune_reason)
                        pruned_from_fixer = True
                if not pruned_from_fixer:
                    _record_stage_transition(ctx, "fixer", "suggester")
                    ctx.route = "planned"
                    ctx.blackboard["_need_replan"] = True
                    continue

            if not pruned_from_fixer:
                lock_report = ctx.state.locked_regression_report() if hasattr(ctx.state, "locked_regression_report") else {}
                rem_after_fix = _client_remaining_calls(ctx.client)
                cur_rev = getattr(ctx.state, "_revision", None)
                if (
                    not ctx.state.verified()
                    and lock_report.get("violations")
                    and lock_report.get("locked")
                    and ctx.blackboard.get("_red_lock_escalated_rev") != cur_rev
                    and ctx.escalations_used < ctx.max_escalations
                    and (rem_after_fix is None or rem_after_fix > ctx.budgets["verifier_reserve"] + 5)
                ):
                    ctx.blackboard["_red_lock_escalated_rev"] = cur_rev
                    v_summary = "; ".join(
                        f"{v.get('path')}({v.get('action')} x{v.get('count', 1)})"
                        for v in (lock_report.get("violations") or [])[:3]
                    )
                    reason = f"Fixer hit Red-phase lock violation(s) on {v_summary}; inspect locked oracle and release/replan if wrong"
                    _record_escalation(
                        ctx,
                        from_role="fixer",
                        to_role="suggester",
                        failure_route="architect",
                        reason=reason,
                        handoff_stage="fixer_handoff",
                        handoff_extra={
                            "action": "red_lock_escalate",
                            "outcome": fix_outcome,
                            "escalate_reason": reason[:240],
                            "failure_route": "architect",
                            "in_rescue_pass": in_rescue_pass,
                            "fix_prompt_breakdown": fix_prompt_breakdown,
                        },
                    )
                    _record_stage_transition(ctx, "fixer", "suggester")
                    ctx.last_qa_feedback = ""
                    ctx.route = "planned"
                    ctx.blackboard["_need_replan"] = True
                    continue

            # --------------------------------------------------------------
            # Stage 4: Patch QA & Verifier (Deterministic Gate + LLM Audit)
            # --------------------------------------------------------------
            _record_stage_transition(ctx, "fixer", "deterministic_qa")
            det_qa = _exec_det_qa_step(
                ctx,
                fix_payload=fix_payload,
                fix_outcome=fix_outcome,
                in_rescue_pass=in_rescue_pass,
                fix_prompt_breakdown=fix_prompt_breakdown,
                attempt=ctx.escalations_used + ctx.qa_rounds + 1,
            )
            rem = _client_remaining_calls(ctx.client)

            if not det_qa["ok"]:
                if (rem is not None and rem <= 1) or fix_outcome in ("reserve_reached", "stage_cap_reached") or in_rescue_pass or pruned_from_fixer:
                    raw_fallback = ctx.state.auto_submit_verified_patch(allow_snapshot_fallback=True)
                    fallback = _clean_and_audit_fallback_patch(ctx.state, ctx.issue_contract, raw_fallback, trace=ctx.trace)
                    if fallback:
                        if rem is not None and rem >= 3 and not in_rescue_pass and not pruned_from_fixer:
                            det_qa = run_deterministic_qa_checks(ctx.state, ctx.issue_contract, run_doctest_on_shift=False)
                        if not det_qa["ok"] or rem is None or rem <= 2 or in_rescue_pass or pruned_from_fixer:
                            if ctx.trace:
                                ctx.trace.record("auto_submit_verified", patch=fallback, stage="deterministic_qa_fallback")
                            _record_stage_transition(ctx, "deterministic_qa", "submission")
                            return fallback
                    if pruned_from_fixer:
                        _record_stage_transition(ctx, "deterministic_qa", "fallback")
                        return _resolve_fallback_patch(ctx, error_message=f"Pruned zero-progress replanning: {det_qa['reason']}")

            if not det_qa["ok"]:
                if fix_outcome == "reserve_reached" or (rem is not None and rem <= ctx.budgets["verifier_reserve"]):
                    if not ctx.rescue_attempted and (rem is None or rem >= 2) and _extract_selected_source_paths(ctx.state):
                        ctx.last_qa_feedback = (
                            f"LOW BUDGET RESCUE ({rem} API calls remaining): Deterministic QA failed with: {det_qa['reason']}. "
                            "Fix ONLY this verification blocker (e.g., run the missing cargo_test or fix/remove a broken temporary regression test) "
                            "and call submit_for_qa immediately."
                        )
                        if ctx.trace:
                            ctx.trace.record("qa_gate", deterministic_ok=False, reason=det_qa["reason"], qa_round=ctx.qa_rounds + 1, rescue=True)
                        _record_stage_transition(ctx, "deterministic_qa", "fixer")
                        continue
                    _record_stage_transition(ctx, "deterministic_qa", "fallback")
                    return _resolve_fallback_patch(ctx, error_message=f"Agent failed deterministic QA verification: {det_qa['reason']}")

                stuck_now = ctx.state.stuck_status()
                if _is_env_dependency_blocker(det_qa.get("reason", "")):
                    ctx.blackboard["environment_blocker"] = True
                should_escalate_to_arch = (
                    not pruned_from_fixer
                    and not ctx.state.verified()
                    and not ctx.blackboard.get("environment_blocker")
                    and ctx.escalations_used < ctx.max_escalations
                    and (rem is None or rem > ctx.budgets["verifier_reserve"] + 6)
                    and (
                        fix_outcome == "stage_cap_reached"
                        or stuck_now["compile_repeats"] >= 2
                        or stuck_now["test_repeats"] >= 2
                    )
                )
                if should_escalate_to_arch:
                    failure_route = _classify_failure_route(det_qa.get("reason", ""), stuck_now)
                    to_role = "navigator" if failure_route == "navigator" else "suggester"
                    reason = f"Fixer ended without passing verification ({fix_outcome}): {det_qa['reason']}"
                    _record_escalation(
                        ctx,
                        from_role="fixer",
                        to_role=to_role,
                        failure_route=failure_route,
                        reason=reason,
                    )
                    ctx.last_qa_feedback = ""
                    if to_role == "navigator":
                        _record_stage_transition(ctx, "deterministic_qa", "navigator")
                        next_to_role, _reloc_reasons, _reloc_out, _new_nav = _exec_navigator_relocalize(
                            ctx, reason, attempt=ctx.escalations_used + 1, allow_fast_return=True,
                        )
                        _record_stage_transition(ctx, "navigator", next_to_role)
                        if next_to_role == "fixer":
                            ctx.route = "fast"
                            ctx.blackboard["_need_replan"] = False
                        else:
                            ctx.route = "planned"
                            ctx.blackboard["_need_replan"] = True
                        continue
                    prune, prune_reason = _should_prune_suggester_replan(ctx)
                    if prune:
                        ctx.blackboard["pruned_replans"] = int(ctx.blackboard.get("pruned_replans") or 0) + 1
                        if ctx.trace:
                            ctx.trace.record("stage_edge_pruned", from_role="deterministic_qa", to_role="suggester", reason=prune_reason)
                        _record_stage_transition(ctx, "deterministic_qa", "fallback")
                        return _resolve_fallback_patch(ctx, error_message=f"{prune_reason}: {det_qa['reason']}")
                    _record_stage_transition(ctx, "deterministic_qa", "suggester")
                    ctx.route = "planned"
                    ctx.blackboard["_need_replan"] = True
                    continue

                if ctx.qa_rounds < ctx.max_qa_rounds and (rem is None or rem > ctx.budgets["verifier_reserve"] + 2):
                    ctx.qa_rounds += 1
                    ctx.last_qa_feedback = f"Deterministic QA gate failed: {det_qa['reason']}"
                    if ctx.trace:
                        ctx.trace.record("qa_gate", deterministic_ok=False, reason=det_qa["reason"], qa_round=ctx.qa_rounds)
                    _record_stage_transition(ctx, "deterministic_qa", "fixer")
                    continue

                _record_stage_transition(ctx, "deterministic_qa", "fallback")
                return _resolve_fallback_patch(ctx, error_message=f"Agent failed deterministic QA verification: {det_qa['reason']}")

            selected = det_qa["selected"]
            hygiene = det_qa["hygiene"]
            rem = _client_remaining_calls(ctx.client)

            if rem is not None and rem <= 2:
                _record_stage_transition(ctx, "deterministic_qa", "submission")
                return _handle_low_budget_det_qa_bypass(ctx, det_qa)

            unreproduced_reason = (fix_payload or {}).get("unreproduced_reason", "")
            if (
                det_qa.get("passed_on_original")
                and not _documentation_only_patch(ctx.state, ctx.issue_contract)
                and not _is_valid_external_unreproduced_reason(
                    unreproduced_reason, ctx.issue_contract, state=ctx.state, blackboard=ctx.blackboard, issue_text=ctx.issue,
                )
                and ctx.qa_rounds < ctx.max_qa_rounds
                and (rem is None or rem > ctx.budgets["verifier_reserve"] + 6)
            ):
                ctx.qa_rounds += 1
                ctx.last_qa_feedback = (
                    "Deterministic QA rejected submission: your temporary regression test PASSED on the unpatched original checkout, "
                    "so it did not reproduce the bug. Construct a real Red-phase test that fails on original code and passes with your fix, "
                    "or call escalate_for_replan."
                )
                if ctx.trace:
                    ctx.trace.record("qa_gate", deterministic_ok=False, reason="Temporary regression test passed on unpatched original code.", qa_round=ctx.qa_rounds)
                if ctx.qa_rounds >= 2 and ctx.escalations_used < ctx.max_escalations:
                    _record_escalation(
                        ctx,
                        from_role="deterministic_qa",
                        to_role="suggester",
                        failure_route="architect",
                        reason=ctx.last_qa_feedback,
                    )
                    _record_stage_transition(ctx, "deterministic_qa", "suggester")
                    ctx.route = "planned"
                    ctx.blackboard["_need_replan"] = True
                else:
                    _record_stage_transition(ctx, "deterministic_qa", "fixer")
                continue

            _record_stage_transition(ctx, "deterministic_qa", "verifier")
            decision, ver_payload, _ver_outcome = _exec_verifier_step(ctx, det_qa)
            _update_milestones_and_kpi(ctx, "verifier", det_qa=det_qa, ver_payload=ver_payload)

            if decision in ("APPROVE", "INCONCLUSIVE_FALLBACK"):
                if decision == "APPROVE":
                    ctx.blackboard["_qa_repair_pending"] = False
                    ctx.blackboard.pop("_last_qa_payload", None)
                ctx.state.patch = hygiene["patch"] or ctx.state.build_patch(selected)
                _record_stage_transition(ctx, "verifier", "submission")
                return ctx.state.patch

            ctx.blackboard["_qa_repair_pending"] = True
            ctx.blackboard["_last_qa_payload"] = dict(ver_payload)
            _reconcile_plan_with_qa_rejection(ctx.blackboard, ver_payload)
            _append_cognitive_entry(
                ctx,
                source="verifier",
                actual="; ".join(
                    [str(x) for x in (ver_payload.get("missing_requirements") or [])]
                    + [str(x) for x in (ver_payload.get("objections") or [])]
                ) or str(ver_payload.get("summary") or "REJECT"),
            )
            repair_feedback = _format_qa_repair_feedback(ver_payload, round_number=ctx.qa_rounds + 1)
            rem_after_qa = _client_remaining_calls(ctx.client)
            if not ctx.qa_repair_loop_enabled:
                ctx.state.patch = hygiene["patch"] or ctx.state.build_patch(selected)
                if ctx.trace:
                    ctx.trace.record(
                        "qa_repair_cycle",
                        outcome="bypassed",
                        reason="RUSTBENCH_QA_REPAIR_LOOP is disabled for the control arm.",
                        qa_round=ctx.qa_rounds + 1,
                        plan_consistency=ver_payload.get("plan_consistency", "not_checked"),
                        plan_conflicts=list(ver_payload.get("plan_conflicts") or []),
                        patch=_feature_meta(ctx.state.patch),
                    )
                _record_stage_transition(ctx, "verifier", "submission")
                return ctx.state.patch
            if ctx.qa_rounds >= ctx.max_qa_rounds or (rem_after_qa is not None and rem_after_qa <= 4):
                ctx.state.patch = hygiene["patch"] or ctx.state.build_patch(selected)
                _record_consistency_blocked(
                    ctx.trace, ctx.blackboard, ctx.state.patch, ver_payload,
                    reason=(
                        "Verifier rejected the patch but no repair cycle remained "
                        f"(qa_rounds={ctx.qa_rounds}, remaining_calls={rem_after_qa})."
                    ),
                    attempt=ctx.qa_rounds + 1,
                )
                _record_stage_transition(ctx, "verifier", "submission")
                return ctx.state.patch

            ctx.qa_rounds += 1
            ctx.last_qa_feedback = repair_feedback
            escalate_qa, esc_reason = _should_escalate_qa_to_architect(det_qa, ctx.blackboard["qa_history"], ctx.qa_rounds - 1)
            if (
                escalate_qa
                and ctx.escalations_used < ctx.max_escalations
                and (rem_after_qa is None or rem_after_qa > ctx.budgets["verifier_reserve"] + 6)
            ):
                cur_sig = _repair_progress_signature(ctx.state)
                if cur_sig in ctx.qa_escalation_sigs:
                    best_raw = ctx.state.auto_submit_verified_patch(allow_snapshot_fallback=True)
                    best_clean = _clean_and_audit_fallback_patch(ctx.state, ctx.issue_contract, best_raw, trace=ctx.trace) if best_raw else ""
                    ctx.state.patch = best_clean or hygiene["patch"] or ctx.state.build_patch(selected)
                    _record_consistency_blocked(
                        ctx.trace, ctx.blackboard, ctx.state.patch, ver_payload,
                        reason=f"QA escalation halted after zero progress since previous escalation: {esc_reason}",
                        attempt=ctx.qa_rounds + 1,
                    )
                    if ctx.trace:
                        ctx.trace.record("qa_stall_stop", reason=esc_reason)
                    _record_stage_transition(ctx, "verifier", "submission")
                    return ctx.state.patch
                ctx.qa_escalation_sigs.add(cur_sig)
                failure_route = _classify_failure_route(esc_reason + " " + repair_feedback, ctx.state.stuck_status())
                to_role = "navigator" if failure_route == "navigator" else "suggester"
                _record_escalation(
                    ctx,
                    from_role="verifier",
                    to_role=to_role,
                    failure_route=failure_route,
                    reason=f"Repeated Verifier rejection ({esc_reason}): {repair_feedback[:600]}",
                )
                if ctx.trace:
                    ctx.trace.record(
                        "qa_repair_cycle",
                        outcome="scheduled",
                        qa_round=ctx.qa_rounds,
                        plan_consistency=ver_payload.get("plan_consistency", "not_checked"),
                        plan_conflicts=list(ver_payload.get("plan_conflicts") or []),
                        feedback=_feature_meta(repair_feedback),
                    )
                if to_role == "navigator":
                    _record_stage_transition(ctx, "verifier", "navigator")
                    next_to_role, _reloc_reasons, _reloc_out, _new_nav = _exec_navigator_relocalize(
                        ctx, esc_reason, attempt=ctx.escalations_used + 1, allow_fast_return=True,
                    )
                    _record_stage_transition(ctx, "navigator", next_to_role)
                    if next_to_role == "fixer":
                        ctx.route = "fast"
                        ctx.blackboard["_need_replan"] = False
                    else:
                        ctx.route = "planned"
                        ctx.blackboard["_need_replan"] = True
                    continue
                prune, prune_reason = _should_prune_suggester_replan(ctx)
                if prune:
                    ctx.blackboard["pruned_replans"] = int(ctx.blackboard.get("pruned_replans") or 0) + 1
                    if ctx.trace:
                        ctx.trace.record("stage_edge_pruned", from_role="verifier", to_role="suggester", reason=prune_reason)
                    best_raw = ctx.state.auto_submit_verified_patch(allow_snapshot_fallback=True)
                    best_clean = _clean_and_audit_fallback_patch(ctx.state, ctx.issue_contract, best_raw, trace=ctx.trace) if best_raw else ""
                    ctx.state.patch = best_clean or hygiene["patch"] or ctx.state.build_patch(selected)
                    _record_consistency_blocked(
                        ctx.trace, ctx.blackboard, ctx.state.patch, ver_payload,
                        reason=f"{prune_reason}: {esc_reason}",
                        attempt=ctx.qa_rounds + 1,
                    )
                    _record_stage_transition(ctx, "verifier", "submission")
                    return ctx.state.patch
                _record_stage_transition(ctx, "verifier", "suggester")
                ctx.route = "planned"
                ctx.blackboard["_need_replan"] = True
                continue

            if ctx.trace:
                ctx.trace.record(
                    "qa_repair_cycle",
                    outcome="scheduled",
                    qa_round=ctx.qa_rounds,
                    plan_consistency=ver_payload.get("plan_consistency", "not_checked"),
                    plan_conflicts=list(ver_payload.get("plan_conflicts") or []),
                    feedback=_feature_meta(repair_feedback),
                )
            if ctx.fix_session_history and callable(ctx.compress_fn):
                try:
                    ctx.fix_session_history = ctx.compress_fn(ctx.fix_session_history)
                except Exception:
                    pass
            _record_stage_transition(ctx, "verifier", "fixer")

    except Exception as exc:
        if getattr(exc, "is_api_server_error", False):
            raise
        fallback = _try_degraded_patch_fallback(
            ctx.state, trace=ctx.trace, allow_unverified_compiling=True,
            issue_contract=ctx.issue_contract, blackboard=ctx.blackboard,
        )
        if fallback:
            if ctx.trace:
                ctx.trace.record("auto_submit_after_error", error=str(exc), error_type=type(exc).__name__)
            return fallback
        raise
    finally:
        _emit_coordination_metrics(ctx)
