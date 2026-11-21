"""Tests for Rust source exploration and Cargo metadata."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


@pytest.fixture
def rust_tools(tmp_path):
    root = tmp_path.resolve()
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text("pub fn first() {}\n// needle\npub fn second() {}\n", encoding="utf-8")
    (root / "target").mkdir()
    (root / "target" / "generated.rs").write_text("needle", encoding="utf-8")
    (root / "Cargo.toml").write_text('[package]\nname = "example"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    spec = importlib.util.spec_from_file_location("rust_tools_test", Path(__file__).with_name("rust_tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def resolve(path):
        target = (root / path).resolve()
        if not target.is_relative_to(root) or ".git" in target.relative_to(root).parts:
            raise ValueError("outside repository")
        return target
    return root, {t.name: t for t in module.make_rust_tools(root, resolve)}


def test_read_and_search(rust_tools):
    root, tools = rust_tools
    assert tools["read_lines"].invoke({"path": "src/lib.rs", "start_line": 2, "end_line": 3}) == (
        "[File: src/lib.rs (3 lines total)]\n"
        "(1 lines above)\n"
        "2: // needle\n"
        "3: pub fn second() {}"
    )
    assert "Error" in tools["read_lines"].invoke({"path": "../outside"})
    assert "Error" in tools["read_lines"].invoke({"path": "src/lib.rs", "start_line": 0})
    assert tools["search_code"].invoke({"query": "needle"}) == "src/lib.rs:2: // needle"
    with_ctx = tools["search_code"].invoke({"query": "needle", "context_lines": 1})
    assert "src/lib.rs:1- pub fn first() {}" in with_ctx
    assert "src/lib.rs:2: // needle" in with_ctx
    assert "src/lib.rs:3- pub fn second() {}" in with_ctx
    assert "Error" in tools["search_code"].invoke({"query": "needle", "context_lines": -1})
    assert "truncated" in tools["search_code"].invoke({"query": "fn (first|second)", "regex": True, "max_results": 1})
    assert "Error" in tools["search_code"].invoke({"query": "[", "regex": True})


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_workspace_metadata(rust_tools):
    _, tools = rust_tools
    result = json.loads(tools["cargo_workspace"].invoke({}))
    assert result["workspace_members"][0]["name"] == "example"
    assert result["workspace_members"][0]["targets"][0]["name"] == "example"
    assert "Error" in tools["cargo_workspace"].invoke({"manifest_path": "../Cargo.toml"})


def test_read_lines_defaults_to_100_lines_and_lists_dirs(rust_tools):
    root, tools = rust_tools
    (root / "src" / "medium.rs").write_text("\n".join(f"line {i}" for i in range(1, 151)) + "\n", encoding="utf-8")
    out = tools["read_lines"].invoke({"path": "src/medium.rs"})
    assert "[File: src/medium.rs (150 lines total)]" in out
    assert "100: line 100" in out
    assert "101: line 101" not in out
    assert "(50 lines below)" in out
    dir_out = tools["read_lines"].invoke({"path": "src"})
    assert "lib.rs" in dir_out
