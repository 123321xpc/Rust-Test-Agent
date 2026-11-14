import importlib.util
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
    # bar should not be instrumented
    assert "stringify!(bar)" not in out


def test_format_trace_call_tree():
    raw = """
[RUST_TRACE_ENTER] src/main.rs:10 main
[RUST_TRACE_ENTER] src/parser.rs:45 parse_expr
[RUST_TRACE_ENTER] src/lexer.rs:102 next_token
[RUST_TRACE_EXIT] src/lexer.rs:102 next_token elapsed=15us
[RUST_TRACE_EXIT] src/parser.rs:45 parse_expr elapsed=120us
[RUST_TRACE_EXIT] src/main.rs:10 main elapsed=1ms
"""
    tree, count = trace_mod.format_trace_call_tree(raw)
    assert count == 3
    assert "main (src/main.rs:10)" in tree
    assert "parse_expr (src/parser.rs:45)" in tree
    assert "next_token (src/lexer.rs:102)" in tree


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo unavailable")
def test_rust_trace_tool_end_to_end(tmp_path):
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "trace_demo"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text("""
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

    orig_content = (root / "src" / "lib.rs").read_text(encoding="utf-8")
    def resolve(path):
        return (root / path).resolve()

    tool = trace_mod.make_rust_trace_tool(root, resolve)
    res = tool.invoke({
        "instrument": ["src/lib.rs:compute,add"],
        "test_target": "test_math",
    })
    assert "Rust Dynamic Trace Report" in res
    assert "compute" in res
    assert "add" in res
    assert "Total traced function calls: 2" in res
    # Check that file was cleanly restored without leftovers
    assert (root / "src" / "lib.rs").read_text(encoding="utf-8") == orig_content


def test_instrument_skips_tests_and_const_fn_and_handles_where_clause():
    src = """
pub const fn compile_time(x: i32) -> i32 {
    x + 1
}

pub fn runtime_fn<T>(v: T) -> T
where
    T: Clone,
{
    v.clone()
}

#[cfg(test)]
mod tests {
    #[test]
    fn unit_case() {
        assert_eq!(super::compile_time(1), 2);
    }
}
"""
    out, count = trace_mod.instrument_source_for_trace(src, set(), "src/lib.rs", watch_vars=["v"])
    assert count == 1
    assert "stringify!(runtime_fn)" in out
    assert "[RUST_TRACE_VAR] runtime_fn::v={:?}" in out
    assert "stringify!(compile_time)" not in out
    assert "stringify!(unit_case)" not in out


def test_rust_trace_watch_vars_e0277_fallback(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "t"\nversion = "0.1.0"\n', encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text("pub fn work(x: i32) -> i32 { x + 1 }\n", encoding="utf-8")

    calls = []

    def fake_run(cmd, **kwargs):
        src_now = (root / "src" / "lib.rs").read_text(encoding="utf-8")
        calls.append("[RUST_TRACE_VAR]" in src_now)
        if "[RUST_TRACE_VAR]" in src_now:
            return subprocess.CompletedProcess(cmd, 101, "", "error[E0277]: `NoDbg` doesn't implement `Debug`")
        return subprocess.CompletedProcess(
            cmd,
            0,
            "running 1 test\n",
            "[RUST_TRACE_ENTER] src/lib.rs:1 work\n[RUST_TRACE_EXIT] src/lib.rs:1 work elapsed=5us\n",
        )

    monkeypatch.setattr(trace_mod.subprocess, "run", fake_run)
    tool = trace_mod.make_rust_trace_tool(root, lambda p: (root / p).resolve())
    res = tool.invoke({"instrument": ["src/lib.rs:work"], "lib": True, "watch_vars": ["x"]})
    assert calls == [True, False]
    assert "automatically fell back" in res
    assert "Total traced function calls: 1" in res
    assert "[RUST_TRACE_ENTER]" not in (root / "src" / "lib.rs").read_text(encoding="utf-8")
