"""Behavioral tests for the CKG-derived impact surface."""

import importlib.util
from pathlib import Path


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("rust_impact.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ckg(tmp_path):
    ckg_spec = importlib.util.spec_from_file_location(
        "impact_ckg", Path(__file__).with_name("rust_ckg.py")
    )
    module = importlib.util.module_from_spec(ckg_spec)
    ckg_spec.loader.exec_module(module)
    return module.RustCodeGraph(tmp_path)


def test_impact_surface_reports_cross_file_consumers(tmp_path):
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text(
        "pub fn produce() -> u32 {\n    1\n}\n",
        encoding="utf-8",
    )
    (src / "consumer.rs").write_text(
        "pub fn consume() -> u32 {\n    crate::produce()\n}\n",
        encoding="utf-8",
    )
    impact = _load("impact_module")
    surface = impact.build_impact_surface(_ckg(tmp_path), ["src/lib.rs"])
    assert surface["files"], "the changed file must be reported"
    assert surface["cross_file"], "a consumer in another file must be detected"
    consumer_paths = {c["path"] for c in surface["consumers"]}
    assert "src/consumer.rs" in consumer_paths
    text = impact.format_impact_surface(surface)
    assert "src/lib.rs" in text
    assert "produce" in text
    assert "Cross-file impact detected" in text


def test_impact_surface_is_empty_without_changes(tmp_path):
    impact = _load("impact_module")
    surface = impact.build_impact_surface(_ckg(tmp_path), [])
    assert surface["files"] == []
    assert not surface["cross_file"]
    assert impact.format_impact_surface(surface) == ""


def test_detect_impact_triggers_lists_consumer_paths(tmp_path):
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text("pub fn produce() -> u32 {\n    1\n}\n", encoding="utf-8")
    (src / "consumer.rs").write_text(
        "pub fn consume() -> u32 {\n    crate::produce()\n}\n", encoding="utf-8"
    )
    impact = _load("impact_module")
    surface = impact.build_impact_surface(_ckg(tmp_path), ["src/lib.rs"])
    triggers = impact.detect_impact_triggers(surface)
    assert any("src/consumer.rs" in t for t in triggers)


def test_changed_lines_restrict_symbols_to_edited_ones(tmp_path):
    """Only the symbols enclosing the edited lines are reported, not the file's first items."""
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text(
        "pub fn unrelated() -> u32 {\n    1\n}\n\npub fn edited() -> u32 {\n    2\n}\n",
        encoding="utf-8",
    )
    impact = _load("impact_module")
    surface = impact.build_impact_surface(
        _ckg(tmp_path), ["src/lib.rs"], changed_lines={"src/lib.rs": [5]}
    )
    names = {sym["name"] for sym in surface["files"][0]["symbols"]}
    assert "edited" in names
    assert "unrelated" not in names


def test_changed_symbols_for_lines_ignores_test_items(tmp_path):
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text(
        "pub fn real() -> u32 {\n    1\n}\n\n#[cfg(test)]\nmod tests {\n    fn helper() {}\n}\n",
        encoding="utf-8",
    )
    impact = _load("impact_module")
    ckg = _ckg(tmp_path)
    symbols = impact._changed_symbols_for_lines(ckg, "src/lib.rs", [1, 2, 5, 6, 7])
    names = {s["name"] for s in symbols}
    assert "real" in names
    assert "helper" not in names


def test_consumer_scan_uses_a_single_repository_pass(tmp_path):
    """Several changed symbols are resolved with one scan, not one scan per symbol."""
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text(
        "pub fn alpha() -> u32 {\n    1\n}\n\npub fn beta() -> u32 {\n    2\n}\n",
        encoding="utf-8",
    )
    (src / "consumer.rs").write_text(
        "pub fn use_alpha() -> u32 {\n    crate::alpha()\n}\n\n"
        "pub fn use_beta() -> u32 {\n    crate::beta()\n}\n",
        encoding="utf-8",
    )
    impact = _load("impact_module")
    ckg = _ckg(tmp_path)
    calls = {"count": 0}
    original = ckg._all_parsed

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    ckg._all_parsed = counting
    consumers = impact._consumer_symbols(ckg, ["alpha", "beta"], exclude_paths=set(), limit=12)
    assert calls["count"] == 1, "one repository scan must cover every changed symbol"
    assert {c["path"] for c in consumers} == {"src/consumer.rs"}
