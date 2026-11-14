"""Behavioral tests of Cargo test execution on a real Rust project."""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    root = tmp_path.resolve()
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "Cargo.toml").write_text('[package]\nname = "example"\nversion = "0.1.0"\nedition = "2021"\n[features]\nextra = []\n', encoding="utf-8")
    (root / "src" / "lib.rs").write_text('pub fn value() -> i32 { 1 }\n', encoding="utf-8")
    (root / "tests" / "sample.rs").write_text('''
#[test] fn works() { assert_eq!(example::value(), 1); }
#[test] fn fails() { assert_eq!(example::value(), 2); }
#[cfg(feature = "extra")]
#[test] fn feature_test() { assert_eq!(example::value(), 1); }
''', encoding="utf-8")
    def resolve(path):
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("outside working copy")
        return candidate
    test = load("cargo_testing").make_cargo_test_tool(root, resolve, load("rust_diagnostics")._simplify)
    return root, test


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_filter_target_features_and_failed_tests(project):
    _, tool = project
    selected = json.loads(tool.invoke({"package": "example", "test_target": "sample", "test_filter": "feature_test", "features": ["extra"]}))
    assert selected["status"] == "passed" and selected["exit_code"] == 0
    assert "1 passed" in selected["test_output"]
    failed = json.loads(tool.invoke({"test_target": "sample", "test_filter": "fails"}))
    assert failed["status"] == "failed" and failed["exit_code"] != 0
    assert failed["failed_tests"] == ["fails"]
    assert "assertion" in failed["test_output"]
    library = json.loads(tool.invoke({"lib": True}))
    assert library["status"] == "passed" and "running 0 tests" in library["test_output"]


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_reports_compile_diagnostics(project):
    root, tool = project
    (root / "src" / "lib.rs").write_text('pub fn value() -> i32 { "bad" }\n', encoding="utf-8")
    result = json.loads(tool.invoke({}))
    assert result["exit_code"] != 0
    assert any(item["category"] == "type_mismatch" for item in result["diagnostics"])


def test_rejects_invalid_arguments(project):
    _, tool = project
    assert json.loads(tool.invoke({"manifest_path": "../Cargo.toml"}))["status"] == "error"
    assert json.loads(tool.invoke({"timeout": 0}))["status"] == "error"
    assert json.loads(tool.invoke({"test_filter": "--help"}))["status"] == "error"
    assert json.loads(tool.invoke({"lib": True, "test_target": "sample"}))["status"] == "error"


@pytest.mark.parametrize("error,status", [
    (FileNotFoundError("cargo missing"), "error"),
    (subprocess.TimeoutExpired("cargo", 1), "timeout"),
])
def test_execution_errors_are_distinct(project, monkeypatch, error, status):
    _, tool = project
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(subprocess, "run", fail)
    result = json.loads(tool.invoke({"timeout": 1}))
    assert result["status"] == status and result["exit_code"] is None


def test_long_output_preserves_early_failure_details(project, monkeypatch):
    _, tool = project
    early_failure = (
        "running 500 tests\n"
        "test early_case ... FAILED\n"
        "failures:\n\n"
        "---- early_case stdout ----\n"
        "thread 'early_case' panicked at 'important root cause'\n\n"
        "failures:\n    early_case\n"
        "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    trailing_noise = "\n".join(f"test passing_case_{i} ... ok" for i in range(1200))
    full_stdout = early_failure + "\n" + trailing_noise + "\ntest result: ok. 1200 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
    assert len(full_stdout) > 24000
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 101, full_stdout, ""))
    result = json.loads(tool.invoke({}))
    assert result["test_output_truncated"] is True
    assert "important root cause" in result["test_output"]
    assert "passing_case_1199" in result["test_output"]


def test_cargo_test_tests_flag_and_mutex(project, monkeypatch):
    _, tool = project
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    res = json.loads(tool.invoke({"tests": True}))
    assert res["status"] == "passed"
    assert "--tests" in calls[0]
    # Mutex check: tests cannot be combined with lib or test_target
    assert json.loads(tool.invoke({"tests": True, "lib": True}))["status"] == "error"
    assert json.loads(tool.invoke({"tests": True, "test_target": "works"}))["status"] == "error"


def test_passing_long_test_output_compacts_to_summary(project, monkeypatch):
    _, tool = project
    lines = ["running 50 tests"] + [f"test test_case_with_long_name_{i} ... ok" for i in range(50)]
    lines.append("test result: ok. 50 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out")
    stdout = "\n".join(lines) + "\n"
    assert len(stdout) > 1000
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, stdout, "Compiling bytes\n"))
    res = json.loads(tool.invoke({}))
    assert res["status"] == "passed"
    assert res["tests_executed"] == 50
    assert res["test_output_truncated"] is True
    assert "50 passed" in res["test_output"]
    assert "test_case_with_long_name_0" not in res["test_output"]


def test_cargo_test_appends_no_fail_fast_on_test_target(project, monkeypatch):
    _, tool = project
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    res = json.loads(tool.invoke({"test_target": "my_regression"}))
    assert res["status"] == "passed"
    cmd = calls[0]
    assert "--test=my_regression" in cmd
    assert "--no-fail-fast" in cmd



def test_cargo_test_passes_features_to_on_success_and_compile_errors_to_on_failure(project, monkeypatch):
    root, _ = project
    seen_success = []
    seen_failure = []

    def on_success(token, **kwargs):
        seen_success.append(kwargs)

    def on_failure(token, **kwargs):
        seen_failure.append(kwargs)

    def resolve(path):
        return (root / path).resolve()

    diag_mod = load("rust_diagnostics")
    test_tool = load("cargo_testing").make_cargo_test_tool(
        root, resolve, diag_mod._simplify,
        on_start=lambda: "tok",
        on_success=on_success,
        on_failure=on_failure,
        compact_fn=diag_mod.compact_diagnostics,
        enrich_diagnostics=diag_mod.attach_diagnostic_snippets,
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a[0], 0, "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n", "   Compiling ex\n"
        ),
    )
    res_ok = json.loads(test_tool.invoke({"test_target": "sample", "features": ["extra"]}))
    assert res_ok["status"] == "passed"
    assert seen_success[0]["features"] == ["extra"]

    err_json = json.dumps({
        "reason": "compiler-message",
        "message": {"level": "error", "code": {"code": "E0308"}, "message": "mismatched types", "spans": [{"file_name": "src/lib.rs", "line_start": 1, "is_primary": True}], "children": []},
    })
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 101, err_json + "\n", "   Compiling ex\nerror: could not compile\n"),
    )
    res_fail = json.loads(test_tool.invoke({"test_target": "sample"}))
    assert res_fail["status"] == "failed"
    assert len(seen_failure[0]["compile_errors"]) == 1
    assert "snippet" in res_fail["diagnostics"][0]["spans"][0]



def test_cargo_test_supports_no_run_flag(project, monkeypatch):
    root, tool = project
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "   Compiling example v0.1.0\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = json.loads(tool.invoke({"tests": True, "no_run": True}))
    assert res["status"] == "passed"
    assert "--no-run" in calls[0]
    assert "--tests" in calls[0]
    assert res["tests_executed"] == 0


def test_cargo_test_supports_doc_flag_and_mutex(project, monkeypatch):
    _, tool = project
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = json.loads(tool.invoke({"doc": True}))
    assert res["status"] == "passed"
    assert res["tests_executed"] == 2
    assert "--doc" in calls[0]
    assert json.loads(tool.invoke({"doc": True, "tests": True}))["status"] == "error"
    assert json.loads(tool.invoke({"doc": True, "lib": True}))["status"] == "error"


def test_extract_project_backtrace():
    from cargo_testing import _extract_project_backtrace
    sample = '''thread 'main' panicked at 'explicit panic', src/lib.rs:10:5
stack backtrace:
   0: rust_begin_unwind
   1: core::panicking::panic_fmt
   2: example::value
             at ./src/lib.rs:10:5
   3: core::ops::function::FnOnce::call_once
test result: FAILED. 0 passed; 1 failed; 0 ignored;
'''
    res = _extract_project_backtrace(sample)
    assert "[Panic backtrace in project code]" in res
    assert "./src/lib.rs:10:5" in res
