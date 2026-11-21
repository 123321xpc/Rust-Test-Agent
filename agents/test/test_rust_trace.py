"""Tests for type-safe hybrid dynamic tracing (rust_trace.py)."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("rust_trace_mod", Path(__file__).with_name("rust_trace.py"))
trace_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace_mod)


def test_instrument_source_for_trace():
    src = """
pub fn foo(x: i32) -> i32 {
    x + 1
}

fn bar() {
    //! inner comment
    println!("hello");
}
"""
    out, count = trace_mod.instrument_source_for_trace(src, {"foo"}, "src/lib.rs")
    assert count == 1
    assert "[RUST_TRACE_ENTER]" in out
    assert "[RUST_TRACE_EXIT]" in out
    assert "stringify!(foo)" in out
    assert "stringify!(bar)" not in out


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo unavailable")
def test_rust_trace_tool_end_to_end_and_mtime_preserved(tmp_path):
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "trace_demo"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (root / "src").mkdir()
    lib_path = root / "src" / "lib.rs"
    lib_path.write_text("""
pub fn compute(a: i32, b: i32) -> i32 {
    add(a, b)
}

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
""", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_math.rs").write_text("""
#[test]
fn test_compute() {
    assert_eq!(trace_demo::compute(20, 22), 42);
}
""", encoding="utf-8")

    orig_content = lib_path.read_text(encoding="utf-8")
    orig_mtime_ns = lib_path.stat().st_mtime_ns

    tool = trace_mod.make_rust_trace_tool(root, lambda p: (root / p).resolve())
    res = tool.invoke({
        "instrument": ["src/lib.rs:compute,add"],
        "test_target": "test_math",
    })
    assert "Rust Dynamic Trace Report" in res
    assert "compute" in res
    assert "add" in res
    assert "compute::a=20" in res
    assert "Total traced function calls: 2" in res
    assert lib_path.read_text(encoding="utf-8") == orig_content
    assert lib_path.stat().st_mtime_ns == orig_mtime_ns
