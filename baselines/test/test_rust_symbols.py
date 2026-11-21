import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def symbols(tmp_path):
    root = tmp_path.resolve()
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text('''// fn hidden() {}
/* struct Comment; /* trait Nested {} */ */
const EXAMPLE: &str = r##"fn hidden_raw() {}"##;
static mut COUNTER: usize = 0;
macro_rules! make_widget { () => {}; }
pub mod inner {
    pub struct Widget;
    pub struct WidgetBuilder;
    pub enum Choice { One }
    pub trait Draw { fn draw(&self); }
    impl<'a> Draw for Widget {
        fn draw(&self) {}
    }
    pub async fn run() {}
}
''', encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "integration.rs").write_text("struct TestOnlyWidget;\n", encoding="utf-8")
    (root / "target").mkdir()
    (root / "target" / "generated.rs").write_text("struct Generated;", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("rust_symbols_test", Path(__file__).with_name("rust_symbols.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def resolve(path):
        p = (root / path).resolve()
        if not p.is_relative_to(root):
            raise ValueError("outside working copy")
        return p
    return module.make_rust_symbols_tool(root, resolve)


def test_definitions_context_and_comments(symbols):
    result = json.loads(symbols.invoke({}))
    assert result["method"] == "text_approximation"
    entries = result["results"]
    assert {(e["kind"], e["name"]) for e in entries} >= {
        ("const", "EXAMPLE"),
        ("static", "COUNTER"),
        ("macro", "make_widget"),
        ("mod", "inner"),
        ("struct", "Widget"),
        ("enum", "Choice"),
        ("trait", "Draw"),
        ("fn", "run"),
        ("impl", "impl<'a> Draw for Widget"),
    }
    assert not any("hidden" in e["name"] or e["name"] in {"Comment", "Generated"} for e in entries)
    widget = next(e for e in entries if e["name"] == "Widget")
    assert widget["path"] == "src/lib.rs" and widget["line"] == 7 and "struct Widget" in widget["context"]


def test_callers_and_references_modes(tmp_path, symbols):
    root = tmp_path.resolve()
    (root / "src" / "caller.rs").write_text(
        '// run() in comment\n'
        'use crate::inner::{Widget, run};\n'
        'pub fn invoke() {\n'
        '    let _s = "run() in string";\n'
        '    run();\n'
        '    let _w = Widget;\n'
        '}\n',
        encoding="utf-8",
    )
    caller_res = json.loads(symbols.invoke({"name": "run", "callers": True}))
    assert caller_res["mode"] == "callers"
    assert [(r["path"], r["line"]) for r in caller_res["results"]] == [("src/caller.rs", 5)]

    ref_res = json.loads(symbols.invoke({"name": "Widget", "references": True, "exclude_tests": True}))
    assert ref_res["mode"] == "references"
    paths_lines = {(r["path"], r["line"]) for r in ref_res["results"]}
    assert ("src/caller.rs", 2) in paths_lines
    assert ("src/caller.rs", 6) in paths_lines
    assert not any(r["path"].startswith("tests/") for r in ref_res["results"])

    assert "error" in json.loads(symbols.invoke({"name": "run", "callers": True, "references": True}))
    assert "error" in json.loads(symbols.invoke({"callers": True}))
