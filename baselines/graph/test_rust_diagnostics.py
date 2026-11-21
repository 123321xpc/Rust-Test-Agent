import importlib.util
import json
import shutil
from types import SimpleNamespace
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("rust_diagnostics_test", Path(__file__).with_name("rust_diagnostics.py"))
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo unavailable")
def test_cargo_check_reports_real_diagnostics(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "diagnostics_example"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text('fn main() { let _: i32 = "wrong"; }\n', encoding="utf-8")
    def resolve(path):
        candidate = (tmp_path / path).resolve()
        if not candidate.is_relative_to(tmp_path.resolve()):
            raise ValueError("outside repository")
        return candidate
    check = diagnostics.make_cargo_check_tool(tmp_path, resolve)
    output = json.loads(check.invoke({}))
    error = next(d for d in output["diagnostics"] if d["code"] == "E0308")
    assert output["exit_code"] != 0
    assert error["category"] == "type_mismatch"
    assert any(s["file_name"].endswith("main.rs") and s["line_start"] == 1 for s in error["spans"])
    assert "Error" in check.invoke({"manifest_path": "../Cargo.toml"})

def test_compact_diagnostics_prioritizes_errors_and_caps_warnings():
    warnings = [{"level": "warning", "code": "unused_variables", "message": f"warn {i}", "spans": [{"file_name": "a.rs"}]} for i in range(15)]
    errors = [{"level": "error", "code": "E0308", "message": "mismatched types", "spans": [{"file_name": "a.rs"}]}]
    summary = [{"level": "error", "code": None, "message": "aborting due to 1 previous error; 15 warnings emitted", "spans": []}]
    kept, omitted = diagnostics.compact_diagnostics(warnings + errors + summary)
    assert any(d["code"] == "E0308" for d in kept)
    assert sum(1 for d in kept if d["level"] == "warning") == 5
    assert omitted == 10
    assert not any("aborting due to" in d["message"] for d in kept)


def test_enrich_diagnostic_symbol_lookup(tmp_path):
    root = tmp_path.resolve()
    (root / "src").mkdir()
    (root / "src" / "service.rs").write_text('''
pub trait Compute {
    fn calculate(&self) -> i32;
}

pub struct Engine;

impl Compute for Engine {
    fn calculate(&self) -> i32 { 42 }
}

pub(crate) struct InternalWorker;
''', encoding="utf-8")
    (root / "src" / "lib.rs").write_text('''
pub mod service;
pub use crate::service::Engine;
''', encoding="utf-8")

    # 1. Test E0599
    diag_e0599 = [{
        "level": "error",
        "code": "E0599",
        "message": "no method named `calculate` found for struct `Engine` in the current scope",
    }]
    diagnostics.enrich_diagnostic_symbol_lookup(diag_e0599, root)
    assert "auto_lookup" in diag_e0599[0]
    assert "calculate" in diag_e0599[0]["auto_lookup"]
    assert "Compute" in diag_e0599[0]["auto_lookup"]
    assert "src/service.rs" in diag_e0599[0]["auto_lookup"]

    # 2. Test E0412 with pub(crate) notice
    diag_e0412 = [{
        "level": "error",
        "code": "E0412",
        "message": "cannot find type `InternalWorker` in this scope",
    }]
    diagnostics.enrich_diagnostic_symbol_lookup(diag_e0412, root)
    assert "auto_lookup" in diag_e0412[0]
    assert "InternalWorker" in diag_e0412[0]["auto_lookup"]
    assert "pub(crate)" in diag_e0412[0]["auto_lookup"]

    # 3. Test E0277
    diag_e0277 = [{
        "level": "error",
        "code": "E0277",
        "message": "the trait bound `OtherType: Compute` is not satisfied",
    }]
    diagnostics.enrich_diagnostic_symbol_lookup(diag_e0277, root)
    assert "auto_lookup" in diag_e0277[0]
    assert "Compute" in diag_e0277[0]["auto_lookup"]
    assert "src/service.rs" in diag_e0277[0]["auto_lookup"]
