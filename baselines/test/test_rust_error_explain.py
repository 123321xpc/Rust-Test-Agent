"""Behavioral tests for rustc error explanations."""
import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def explanation(tmp_path):
    spec = importlib.util.spec_from_file_location("test_rust_error_explain", Path(__file__).with_name("rust_error_explain.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.make_rust_error_explain_tool(tmp_path)


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc unavailable")
def test_real_rust_error_explanation(explanation):
    _, tool = explanation
    output = tool.invoke({"code": "e0382"})
    assert "moved" in output.lower() and len(output) > 100
    assert "Error explaining E9999" in tool.invoke({"code": "E9999"})


def test_rejects_invalid_input_without_running_compiler(explanation, monkeypatch):
    module, tool = explanation
    def fail(*args, **kwargs):
        pytest.fail("rustc should not be invoked")
    monkeypatch.setattr(module.subprocess, "run", fail)
    for code in ("E382", "E0382 --help", "E０３８２", ""):
        assert "Error" in tool.invoke({"code": code})
