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
