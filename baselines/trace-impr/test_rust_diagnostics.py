import importlib.util
import json
import shutil
from types import SimpleNamespace
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("rust_diagnostics_test", Path(__file__).with_name("rust_diagnostics.py"))
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


@pytest.mark.parametrize("code,category", [
    ("E0382", "ownership_move"), ("E0499", "borrow_conflict"),
    ("E0597", "lifetime"), ("E0277", "trait_bound"),
    ("E0308", "type_mismatch"), ("E0425", "other"),
])
def test_rust_error_codes(code, category):
    assert diagnostics.classify_diagnostic({"code": {"code": code}, "message": "problem"}) == category


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo unavailable")
def test_cargo_check_reports_real_diagnostics(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "diagnostics_example"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text('fn main() { let _: i32 = "wrong"; }\n', encoding="utf-8")
    def resolve(path):
        candidate = (tmp_path / path).resolve()
        if not candidate.is_relative_to(tmp_path.resolve()):
            raise ValueError("outside repository")
        return candidate
    check = diagnostics.make_cargo_check_tool(tmp_path, resolve)
    output = json.loads(check.invoke({}))
    error = next(d for d in output["diagnostics"] if d["code"] == "E0308")
    assert output["exit_code"] != 0
    assert error["category"] == "type_mismatch"
    assert any(s["file_name"].endswith("main.rs") and s["line_start"] == 1 for s in error["spans"])
    assert "Error" in check.invoke({"manifest_path": "../Cargo.toml"})

@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo unavailable")
def test_cargo_check_multiple_rust_error_categories(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "error_categories"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text('''
fn moved() { let a = String::new(); let _b = a; println!("{}", a); }
fn borrowed() { let mut a = 0; let b = &mut a; let c = &mut a; *b += *c; }
fn lifetime() -> &'static str { let s = String::new(); &s }
fn requires_copy<T: Copy>(_: T) {}
fn traits() { requires_copy(String::new()); }
fn types() { let _: i32 = "wrong"; }
''', encoding="utf-8")
    check = diagnostics.make_cargo_check_tool(tmp_path, lambda path: (tmp_path / path).resolve())
    result = json.loads(check.invoke({}))
    assert result["exit_code"] != 0
    assert {"ownership_move", "borrow_conflict", "lifetime", "trait_bound", "type_mismatch"} <= {d["category"] for d in result["diagnostics"]}


@pytest.mark.parametrize("code", ["E0381", "E0282", "E0284", "E0599"])
def test_ambiguous_codes_are_not_misclassified(code):
    assert diagnostics.classify_diagnostic({"code": {"code": code}, "message": "problem"}) == "other"

@pytest.mark.parametrize("failure,expected", [
    (FileNotFoundError("cargo missing"), "cargo missing"),
    (diagnostics.subprocess.TimeoutExpired("cargo", 120), "timed out"),
])
def test_cargo_check_reports_execution_failures(tmp_path, monkeypatch, failure, expected):
    (tmp_path / "Cargo.toml").write_text("", encoding="utf-8")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(diagnostics.subprocess, "run", fail)
    check = diagnostics.make_cargo_check_tool(tmp_path, lambda path: tmp_path / path)
    assert expected in check.invoke({})


def test_check_passes_requested_cargo_options(tmp_path, monkeypatch):
    (tmp_path / "Cargo.toml").write_text("", encoding="utf-8")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    check = diagnostics.make_cargo_check_tool(tmp_path, lambda path: tmp_path / path)
    result = json.loads(check.invoke({"package": "example", "features": ["one", "two"],
                                     "all_features": True, "no_default_features": True,
                                     "target": "wasm32-unknown-unknown", "tests": True}))
    assert result["exit_code"] == 0
    assert {"--package=example", "--features=one,two", "--all-features",
            "--no-default-features", "--target=wasm32-unknown-unknown", "--tests"} <= set(calls[0])
    assert "Error" in check.invoke({"tests": True, "all_targets": True})
    assert len(calls) == 1
    check.invoke({"all_targets": True})
    assert "--all-targets" in calls[1]


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo unavailable")
def test_check_feature_and_test_target(tmp_path):
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "feature_test"\nversion = "0.1.0"\nedition = "2021"\n'
        '[features]\nextra = []\n', encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text(
        '#[cfg(all(test, feature = "extra"))]\nmod tests { #[test] fn broken() { let _: i32 = "wrong"; } }\n',
        encoding="utf-8")
    check = diagnostics.make_cargo_check_tool(tmp_path, lambda path: tmp_path / path)
    assert json.loads(check.invoke({}))["exit_code"] == 0
    result = json.loads(check.invoke({"features": ["extra"], "tests": True}))
    assert result["exit_code"] != 0
    assert any(d["category"] == "type_mismatch" for d in result["diagnostics"])


def test_compact_diagnostics_prioritizes_errors_and_caps_warnings():
    warnings = [{"level": "warning", "code": "unused_variables", "message": f"warn {i}", "spans": [{"file_name": "a.rs"}]} for i in range(15)]
    errors = [{"level": "error", "code": "E0308", "message": "mismatched types", "spans": [{"file_name": "a.rs"}]}]
    summary = [{"level": "error", "code": None, "message": "aborting due to 1 previous error; 15 warnings emitted", "spans": []}]
    kept, omitted = diagnostics.compact_diagnostics(warnings + errors + summary)
    assert any(d["code"] == "E0308" for d in kept)
    assert sum(1 for d in kept if d["level"] == "warning") == 5
    assert omitted == 10
    assert not any("aborting due to" in d["message"] for d in kept)


def test_remediation_hint_and_suggested_fix_extraction():
    raw_msg = {
        "level": "error",
        "code": {"code": "E0382"},
        "message": "use of moved value: `x`",
        "spans": [{"file_name": "src/main.rs", "line_start": 4, "is_primary": True, "suggested_replacement": None}],
        "children": [
            {"level": "help", "message": "consider cloning the value", "spans": [{"file_name": "src/main.rs", "line_start": 3, "suggested_replacement": "x.clone()"}]}
        ],
    }
    simplified = diagnostics._simplify(raw_msg)
    assert simplified["category"] == "ownership_move"
    assert "remediation_hint" in simplified
    assert "borrowing" in simplified["remediation_hint"].lower() or "cloning" in simplified["remediation_hint"].lower()
    assert simplified["suggested_fix"] == "x.clone()"



def test_compact_diagnostics_clusters_cascading_errors_and_deduplicates_hints():
    errors = [
        {
            "category": "type_mismatch",
            "level": "error",
            "code": "E0308",
            "message": "mismatched types",
            "remediation_hint": "Types do not match.",
            "spans": [{"file_name": "src/lib.rs", "line_start": 10 + i, "column_start": 5, "is_primary": True}],
            "children": [],
        }
        for i in range(5)
    ]
    kept, omitted = diagnostics.compact_diagnostics(errors)
    assert len(kept) == 2
    assert omitted == 3
    assert kept[0]["remediation_hint"] == "Types do not match."
    assert "remediation_hint" not in kept[1]
    assert kept[0]["also_at"] == ["src/lib.rs:12", "src/lib.rs:13", "src/lib.rs:14"]


def test_attach_diagnostic_snippets_inlines_repo_files_and_skips_outside(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("line1\nline2\nlet x: i32 = \"bad\";\nline4\nline5\n", encoding="utf-8")
    outside = tmp_path.parent / "outside.rs"
    outside.write_text("secret\n", encoding="utf-8")

    diags = [{
        "level": "error",
        "code": "E0308",
        "message": "mismatched types",
        "spans": [
            {"file_name": "src/lib.rs", "line_start": 3, "line_end": 3, "is_primary": True},
            {"file_name": "../outside.rs", "line_start": 1, "line_end": 1},
        ],
        "children": [],
    }]
    diagnostics.attach_diagnostic_snippets(diags, tmp_path)
    primary_snippet = diags[0]["spans"][0]["snippet"]
    assert ">    3: let x: i32 = \"bad\";" in primary_snippet
    assert "     2: line2" in primary_snippet
    assert "snippet" not in diags[0]["spans"][1]


def test_clean_cargo_stderr_strips_progress_lines():
    raw = "   Compiling libc v0.2.150\n   Checking foo v0.1.0\nerror: build failed\n"
    cleaned = diagnostics.clean_cargo_stderr(raw)
    assert "Compiling libc" not in cleaned
    assert "error: build failed" in cleaned


def test_enrich_diagnostic_symbol_lookup(tmp_path):
    root = tmp_path.resolve()
    (root / "src").mkdir()
    (root / "src" / "service.rs").write_text('''
pub trait Compute {
    fn calculate(&self) -> i32;
}

pub struct Engine;

impl Compute for Engine {
    fn calculate(&self) -> i32 { 42 }
}

pub(crate) struct InternalWorker;
''', encoding="utf-8")
    (root / "src" / "lib.rs").write_text('''
pub mod service;
pub use crate::service::Engine;
''', encoding="utf-8")

    # 1. Test E0599
    diag_e0599 = [{
        "level": "error",
        "code": "E0599",
        "message": "no method named `calculate` found for struct `Engine` in the current scope",
    }]
    diagnostics.enrich_diagnostic_symbol_lookup(diag_e0599, root)
    assert "auto_lookup" in diag_e0599[0]
    assert "calculate" in diag_e0599[0]["auto_lookup"]
    assert "Compute" in diag_e0599[0]["auto_lookup"]
    assert "src/service.rs" in diag_e0599[0]["auto_lookup"]

    # 2. Test E0412 with pub(crate) notice
    diag_e0412 = [{
        "level": "error",
        "code": "E0412",
        "message": "cannot find type `InternalWorker` in this scope",
    }]
    diagnostics.enrich_diagnostic_symbol_lookup(diag_e0412, root)
    assert "auto_lookup" in diag_e0412[0]
    assert "InternalWorker" in diag_e0412[0]["auto_lookup"]
    assert "pub(crate)" in diag_e0412[0]["auto_lookup"]

    # 3. Test E0277
    diag_e0277 = [{
        "level": "error",
        "code": "E0277",
        "message": "the trait bound `OtherType: Compute` is not satisfied",
    }]
    diagnostics.enrich_diagnostic_symbol_lookup(diag_e0277, root)
    assert "auto_lookup" in diag_e0277[0]
    assert "Compute" in diag_e0277[0]["auto_lookup"]
    assert "src/service.rs" in diag_e0277[0]["auto_lookup"]
