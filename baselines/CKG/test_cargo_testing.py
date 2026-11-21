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
