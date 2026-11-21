import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("test_rust_environment_test", Path(__file__).with_name("rust_environment.py"))
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


def test_dataset_uses_harness_test_spec(tmp_path, monkeypatch):
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps({"instance_id": "aya-rs__aya-774", "updated_at": "2023-09-01"}) + "\n", encoding="utf-8")
    seen = []
    def make_spec(instance):
        seen.append(instance)
        return SimpleNamespace(rust_version="1.72.0")
    monkeypatch.setattr(environment.importlib, "import_module", lambda name: SimpleNamespace(make_test_spec=make_spec))
    assert environment.benchmark_toolchain(dataset, "aya-rs__aya-774", tmp_path) == "1.72.0"
    assert seen[0]["updated_at"] == "2023-09-01"
    with pytest.raises(ValueError, match="exactly one"):
        environment.benchmark_toolchain(dataset, "missing", tmp_path)


def test_unavailable_toolchain_fails_before_agent(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="toolchain not installed")
    monkeypatch.setattr(environment.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="toolchain not installed"):
        environment.prepare_rust_environment(root, "1.72.0")


def test_shared_cargo_cache_configuration_and_seed(tmp_path, monkeypatch):
    cache_root = tmp_path / "shared_cache"
    monkeypatch.setenv("RUSTBENCH_CARGO_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)

    repo = tmp_path / "bytes_repo"
    (repo / "target" / "debug").mkdir(parents=True)
    (repo / "target" / "debug" / "libbytes.rlib").write_bytes(b"seed-artifact")

    env = {}
    info = environment.configure_cargo_cache(env, repo_path=repo, instance_id="tokio-rs__bytes-253")
    expected_target = cache_root / "tokio-rs__bytes"
    assert env["CARGO_TARGET_DIR"] == str(expected_target)
    assert info["cargo_target_dir"] == str(expected_target)
    assert info["seeded"] is True
    assert (expected_target / "debug" / "libbytes.rlib").read_bytes() == b"seed-artifact"
