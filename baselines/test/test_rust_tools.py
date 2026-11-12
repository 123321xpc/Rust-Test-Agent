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
    assert tools["read_lines"].invoke({"path": "src/lib.rs", "start_line": 2, "end_line": 3}) == "2: // needle\n3: pub fn second() {}"
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


def test_cargo_workspace_falls_back_without_offline(rust_tools, monkeypatch):
    import subprocess
    _, tools = rust_tools
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "--offline" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "error: Found argument '--offline' which wasn't expected")
        payload = {"workspace_root": str(Path(".").resolve()), "workspace_members": ["pkg#0.1.0"], "packages": []}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = json.loads(tools["cargo_workspace"].invoke({}))
    assert res["workspace_members"] == []
    assert len(calls) == 2
    assert "--offline" in calls[0]
    assert "--offline" not in calls[1]

def test_search_code_multiline_signatures(rust_tools):
    root, tools = rust_tools
    sig_text = 'pub fn compute_sum(' + chr(10) + '    first: i32,' + chr(10) + '    second: i32,' + chr(10) + ') -> i32 {' + chr(10) + '    first + second' + chr(10) + '}' + chr(10)
    (root / 'src' / 'multi_sig.rs').write_text(sig_text, encoding='utf-8')
    assert tools['search_code'].invoke({'query': 'pub fn compute_sum(first: i32, second: i32)'}) == '(no matches)'
    found = tools['search_code'].invoke({'query': 'pub fn compute_sum(first: i32, second: i32)', 'multiline': True})
    assert 'src/multi_sig.rs:1:' in found
    with_ctx = tools['search_code'].invoke({'query': 'pub fn compute_sum(first: i32, second: i32)', 'multiline': True, 'context_lines': 1})
    assert 'src/multi_sig.rs:1: pub fn compute_sum(' in with_ctx


def test_read_lines_clamps_range_over_500_lines(rust_tools):
    root, tools = rust_tools
    (root / "src" / "long.rs").write_text("\n".join(f"// line {i}" for i in range(1, 650)) + "\n", encoding="utf-8")
    out = tools["read_lines"].invoke({"path": "src/long.rs", "start_line": 10, "end_line": 600})
    assert "10: // line 10" in out
    assert "509: // line 509" in out
    assert "510: // line 510" not in out
    assert "clamped to 10-509" in out
