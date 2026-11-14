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


def test_filters_and_limits(symbols):
    matches = json.loads(symbols.invoke({"kind": "fn", "name": "draw", "max_results": 1}))
    assert len(matches["results"]) == 1 and matches["truncated"]
    exact_matches = json.loads(symbols.invoke({"kind": "struct", "name": "Widget", "exact": True}))["results"]
    assert [m["name"] for m in exact_matches] == ["Widget"]
    no_tests = json.loads(symbols.invoke({"name": "Widget", "exclude_tests": True}))["results"]
    assert not any(m["name"] == "TestOnlyWidget" for m in no_tests)
    assert "error" in json.loads(symbols.invoke({"path": "../outside"}))
    assert "error" in json.loads(symbols.invoke({"kind": "invalid"}))

def test_outline_mode_and_multiline_signatures(tmp_path, symbols):
    root = tmp_path.resolve()
    sig_text = 'pub fn long_sig(' + chr(10) + '    a: usize,' + chr(10) + '    b: usize,' + chr(10) + ') -> usize {' + chr(10) + '    a + b' + chr(10) + '}' + chr(10)
    (root / 'src' / 'multi.rs').write_text(sig_text, encoding='utf-8')
    out = json.loads(symbols.invoke({'path': 'src/multi.rs', 'outline': True}))
    assert 'outline' in out
    assert len(out['outline']) == 1
    entry = out['outline'][0]
    assert entry['name'] == 'long_sig'
    assert 'pub fn long_sig( a: usize, b: usize, ) -> usize' in entry['context']



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


def test_visibility_and_enclosing(symbols):
    res = json.loads(symbols.invoke({}))
    entries = {e["name"]: e for e in res["results"]}
    assert entries["Widget"]["vis"] == "pub"
    assert entries["COUNTER"]["vis"] == "private"
    draw_impl = next(e for e in res["results"] if e["name"] == "draw" and e.get("enclosing") == "impl<'a> Draw for Widget")
    assert draw_impl["enclosing"] == "impl<'a> Draw for Widget"


def test_attrs_and_symbol_details_helpers(tmp_path):
    root = tmp_path.resolve()
    (root / "src").mkdir(exist_ok=True)
    (root / "src" / "types.rs").write_text(
        '#[derive(Debug, Clone, PartialEq)]\n'
        '#[cfg(feature = "extra")]\n'
        'pub struct CustomItem {\n'
        '    pub id: u32,\n'
        '}\n\n'
        'pub trait CustomTrait {\n'
        '    fn process(&self);\n'
        '}\n\n'
        'impl CustomTrait for CustomItem {\n'
        '    fn process(&self) {}\n'
        '}\n',
        encoding="utf-8"
    )
    (root / "src" / "lib.rs").write_text(
        'pub mod types;\n'
        'pub use crate::types::CustomItem;\n',
        encoding="utf-8"
    )
    import importlib.util
    spec = importlib.util.spec_from_file_location("rust_symbols_mod", Path('baselines/test/rust_symbols.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 1. Test lookup_symbol_details
    details = mod.lookup_symbol_details(root, name="CustomItem", exact=True)
    assert len(details) >= 1
    assert details[0]["path"] == "src/types.rs"
    assert details[0]["vis"] == "pub"
    assert "attrs" in details[0]
    assert any("derive(Debug, Clone, PartialEq)" in a for a in details[0]["attrs"])
    assert any('cfg(feature = "extra")' in a for a in details[0]["attrs"])

    # 2. Test find_trait_implementations
    impls = mod.find_trait_implementations(root, trait_name="CustomTrait")
    assert len(impls) >= 1
    assert impls[0]["path"] == "src/types.rs"
    assert "CustomTrait for CustomItem" in impls[0]["header"]

    # 3. Test find_reexports
    reexports = mod.find_reexports(root, symbol_name="CustomItem")
    assert len(reexports) >= 1
    assert reexports[0]["path"] == "src/lib.rs"
    assert "pub use crate::types::CustomItem;" in reexports[0]["reexport"]
