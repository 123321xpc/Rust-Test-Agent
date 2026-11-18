"""Tests for Rust token-aware CKG parser, neighborhood lookup, and fault propagation chains."""

import importlib.util
import json
import time
from pathlib import Path


def _load_ckg():
    spec = importlib.util.spec_from_file_location("rust_ckg_test", Path(__file__).with_name("rust_ckg.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tokenize_rust_handles_comments_raw_strings_and_lifetimes():
    ckg = _load_ckg()
    src = """
// fn fake_in_comment() {}
/* outer /* nested fn fake() {} */ still comment */
pub fn parse<'a>(s: &'a str, c: char) -> &'a str {
    let raw = r#"fn not_a_fn() { 'x' }"#;
    let ch = '\\'';
    let r#match = s;
    r#match
}
"""
    parsed = ckg.parse_rust_file(src, "src/lib.rs")
    fn_names = [it["name"] for it in parsed.items if it["kind"] == "fn"]
    assert fn_names == ["parse"]
    assert parsed.items[0]["has_body"] is True


def test_ckg_extracts_reexports_derives_traits_and_call_edges(tmp_path):
    ckg = _load_ckg()
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "my-crate"\nversion = "0.1.0"\n', encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "inner.rs").write_text("""
pub trait FormatValue {
    fn format_it(&self) -> String;
}

#[derive(Debug, Clone)]
pub struct Worker {
    pub id: u32,
}

impl FormatValue for Worker {
    fn format_it(&self) -> String {
        format_core(self.id)
    }
}

fn format_core(id: u32) -> String {
    id.to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_worker() {
        let _ = super::format_core(1);
    }
}
""", encoding="utf-8")
    (root / "src" / "lib.rs").write_text("""
pub mod inner;
pub use crate::inner::{Worker, FormatValue};

pub fn entry(w: &Worker) -> String {
    w.format_it()
}
""", encoding="utf-8")

    tool, graph = ckg.make_rust_ckg_tool(root)
    nb_worker = json.loads(tool.invoke({"target": "Worker", "hops": 1}))
    assert nb_worker["definitions"][0]["vis"] == "pub"
    assert "my_crate::Worker" in nb_worker["definitions"][0]["public_paths"]
    traits = {t["trait"] for t in nb_worker["trait_impls"]}
    assert {"Debug", "Clone", "FormatValue"} <= traits

    nb_core = json.loads(tool.invoke({"target": "format_core", "hops": 2}))
    caller_names = {c["caller"] for c in nb_core["callers"]}
    assert "format_it" in caller_names
    assert "test_worker" not in caller_names  # excluded test function

    chains = graph.build_fault_chains(["entry"], max_chains=3, max_depth=3)
    assert chains
    symbols_in_chain = [n["symbol"] for n in chains[0]["nodes"]]
    assert "entry" in symbols_in_chain
    assert "format_it" in symbols_in_chain
    assert "format_core" in symbols_in_chain


def test_ckg_incremental_cache_updates_on_file_edit(tmp_path):
    ckg = _load_ckg()
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "inc"\nversion = "0.1.0"\n', encoding="utf-8")
    (root / "src").mkdir()
    lib = root / "src" / "lib.rs"
    lib.write_text("pub fn alpha() {}\n", encoding="utf-8")
    graph = ckg.RustCodeGraph(root)
    res1 = graph.query_neighborhood("alpha")
    assert len(res1["definitions"]) == 1
    time.sleep(0.02)
    lib.write_text("pub fn alpha() {}\npub fn beta() { alpha(); }\n", encoding="utf-8")
    res2 = graph.query_neighborhood("alpha")
    assert any(c["caller"] == "beta" for c in res2["callers"])


def test_query_neighborhood_resolves_symbol_inside_a_file(tmp_path):
    """`file.rs:symbol` must return the symbol's definition, not an empty result."""
    ckg = _load_ckg()
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "symfile"\nversion = "0.1.0"\n', encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text(
        "pub fn consume_string() -> u32 {\n    1\n}\n\npub fn other() -> u32 {\n    2\n}\n",
        encoding="utf-8",
    )
    graph = ckg.RustCodeGraph(root)
    res = graph.query_neighborhood("src/lib.rs:consume_string")
    assert res.get("mode") == "file_symbol"
    assert not res.get("error")
    assert [it["name"] for it in res["items"]] == ["consume_string"]


def test_query_neighborhood_reports_unknown_symbol_in_file(tmp_path):
    ckg = _load_ckg()
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "symfile2"\nversion = "0.1.0"\n', encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text("pub fn present() -> u32 {\n    1\n}\n", encoding="utf-8")
    graph = ckg.RustCodeGraph(root)
    res = graph.query_neighborhood("src/lib.rs:absent")
    assert res.get("error")
    assert "present" in res.get("available_symbols", [])


def test_query_neighborhood_line_form_still_works(tmp_path):
    ckg = _load_ckg()
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "symfile3"\nversion = "0.1.0"\n', encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text("pub fn present() -> u32 {\n    1\n}\n", encoding="utf-8")
    graph = ckg.RustCodeGraph(root)
    res = graph.query_neighborhood("src/lib.rs:2")
    assert res.get("mode") == "file_line"
    assert res["items"]


def test_ckg_extracts_enum_variants_wildcard_matches_cfg_features_and_macros(tmp_path):
    ckg = _load_ckg()
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "ckg_ext"\nversion = "0.1.0"\n[features]\nserde = []\n', encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text("""
pub enum Token {
    Ident,
    Bogus,
}

#[cfg(feature = "serde")]
pub mod ser_support {
    pub fn encode() {}
}

pub fn handle(tok: Token) -> i32 {
    match tok {
        Token::Ident => 1,
        _ => 0,
    }
}

impl_table!(HashMap, PerCpuHashMap);
""", encoding="utf-8")
    graph = ckg.RustCodeGraph(root)
    parsed = graph.get_parsed_file(root / "src" / "lib.rs")
    enum_item = next(it for it in parsed.items if it["kind"] == "enum")
    assert enum_item["variants"] == ["Ident", "Bogus"]
    assert parsed.wildcard_matches and parsed.wildcard_matches[0]["fn"] == "handle"
    assert "Token" in parsed.wildcard_matches[0]["types"]
    assert graph.features_for_file_lines("src/lib.rs", [9]) == ["serde"]
    nb_macro = graph.query_neighborhood("PerCpuHashMap")
    assert nb_macro.get("macro_invocations")


def test_ckg_surfaces_static_table_and_literal_references(tmp_path):
    ckg = _load_ckg()
    root = tmp_path.resolve()
    (root / "Cargo.toml").write_text('[package]\nname = "tbl"\nversion = "0.1.0"\n', encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text(
        'pub struct Vtable { pub is_unique: fn() -> bool }\n'
        'pub static SHARED_VTABLE: Vtable = Vtable {\n'
        '    is_unique: shared_is_unique,\n'
        '};\n'
        'pub const PROG_SEC: &str = "cgroup_skb";\n'
        'fn shared_is_unique() -> bool { false }\n',
        encoding="utf-8",
    )
    graph = ckg.RustCodeGraph(root)
    nb_static = graph.query_neighborhood("is_unique")
    assert nb_static.get("static_references") and nb_static["static_references"][0]["item"] == "SHARED_VTABLE"
    nb_lit = graph.query_neighborhood("cgroup_skb")
    assert nb_lit.get("literal_references") and "cgroup_skb" in nb_lit["literal_references"][0]["text"]

