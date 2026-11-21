"""Tests for Stage 1 rustdoc JSON definition graph, Stage 2 MIR call edges, and lexical fallback."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest


def _load(name: str):
    mod_name = f"test_sem_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_shape_crate(root: Path) -> None:
    (root / "Cargo.toml").write_text(
        '[package]\nname = "demo-shapes"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (root / "src").mkdir()
    (root / "src" / "geometry.rs").write_text(
        "pub trait Shape {\n"
        "    fn area(&self) -> f64;\n"
        "}\n"
        "pub struct Sq(pub f64);\n"
        "impl Shape for Sq {\n"
        "    fn area(&self) -> f64 { self.0 * self.0 }\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "src" / "shapes.rs").write_text(
        "pub trait Shape {\n"
        "    fn area(&self) -> f64;\n"
        "}\n"
        "pub struct Ci(pub f64);\n"
        "impl Shape for Ci {\n"
        "    fn area(&self) -> f64 { 3.14 * self.0 * self.0 }\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "src" / "callers.rs").write_text(
        "use crate::geometry::{Shape as GeoShape, Sq};\n"
        "use crate::shapes::{Shape as ShpShape, Ci};\n"
        "use crate::Sq2;\n"
        "\n"
        "pub fn call_sq(s: &Sq) -> f64 {\n"
        "    GeoShape::area(s)\n"
        "}\n"
        "pub fn call_ci(c: &Ci) -> f64 {\n"
        "    c.area()\n"
        "}\n"
        "pub fn call_sq2(x: &Sq2) -> f64 {\n"
        "    x.area()\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "src" / "lib.rs").write_text(
        "pub mod geometry;\n"
        "pub mod shapes;\n"
        "pub mod callers;\n"
        "\n"
        "pub struct Sq2(pub f64);\n"
        "impl Sq2 {\n"
        "    pub fn area(&self) -> f64 { self.0 * self.0 }\n"
        "}\n"
        "fn priv_helper(s: &geometry::Sq) -> f64 {\n"
        "    use geometry::Shape;\n"
        "    s.area()\n"
        "}\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo unavailable")
def test_stage1_isolates_same_name_traits_and_includes_private_items(tmp_path):
    ckg_mod = _load("rust_ckg")
    impact_mod = _load("rust_impact")
    sem_mod = _load("rust_semantics")
    root = tmp_path.resolve()
    _write_shape_crate(root)

    ws = sem_mod.build_workspace_graph(root)
    assert ws.precise is True
    assert ws.mode == "compiler"
    assert "priv_helper" in ws.items_by_name

    # Exact trait -> impl lookup separates geometry::Shape from shapes::Shape
    geo_impls = ws.dependents_of_trait_any(["demo_shapes::geometry::Shape"])
    assert len(geo_impls) == 1
    assert geo_impls[0].type_full_path == "demo_shapes::geometry::Sq"

    graph = ckg_mod.RustCodeGraph(root)
    nb_geo = graph.query_neighborhood("geometry::Shape")
    assert nb_geo["precision"] in ("compiler", "compiler_mir")
    assert len(nb_geo["trait_impls"]) == 1
    assert nb_geo["trait_impls"][0]["for_type"] == "Sq"

    # Bare name query keeps both traits visible but marks precision as compiler_ambiguous
    nb_bare = graph.query_neighborhood("Shape")
    assert nb_bare["precision"] == "compiler_ambiguous"
    assert {ti["for_type"] for ti in nb_bare["trait_impls"]} == {"Sq", "Ci"}

    # Editing geometry::Shape line 1 only flags geometry::Shape for Sq, never shapes::Shape for Ci
    surface = impact_mod.build_impact_surface(
        graph, ["src/geometry.rs"], changed_lines={"src/geometry.rs": [1]},
    )
    assert len(surface["trait_impls"]) == 1
    assert surface["trait_impls"][0]["path"] == "src/geometry.rs"
    assert "Sq" in surface["trait_impls"][0]["for_type"]


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo unavailable")
def test_stage2_mir_disambiguates_method_callers_impact_and_fault_chains(tmp_path):
    ckg_mod = _load("rust_ckg")
    impact_mod = _load("rust_impact")
    root = tmp_path.resolve()
    _write_shape_crate(root)

    graph = ckg_mod.RustCodeGraph(root)
    nb = graph.query_neighborhood("Sq::area")
    assert nb["precision"] == "compiler_mir"
    caller_names = {c["caller"] for c in nb["callers"]}
    assert "call_sq" in caller_names
    assert "priv_helper" in caller_names
    assert "call_ci" not in caller_names
    assert "call_sq2" not in caller_names

    # Editing geometry::Sq::area (line 6) must only report cross-file callers of Sq::area (call_sq & priv_helper), not call_ci/call_sq2
    surface = impact_mod.build_impact_surface(
        graph, ["src/geometry.rs"], changed_lines={"src/geometry.rs": [6]},
    )
    assert surface["precision"] == "compiler_mir"
    consumer_callers = {c["caller"] for c in surface["consumers"]}
    assert consumer_callers == {"call_sq", "priv_helper"}

    # Fault chains from call_ci vs call_sq2 resolve to Ci::area vs Sq2::area respectively
    chains = graph.build_fault_chains(["call_ci", "call_sq2"], max_chains=2, max_depth=2)
    assert len(chains) == 2
    ci_paths = [n["path"] for n in chains[0]["nodes"]]
    sq2_paths = [n["path"] for n in chains[1]["nodes"]]
    assert ci_paths == ["src/callers.rs", "src/shapes.rs"]
    assert sq2_paths == ["src/callers.rs", "src/lib.rs"]
