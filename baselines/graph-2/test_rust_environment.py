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


def test_old_harness_extracts_version_from_setup(tmp_path, monkeypatch):
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text('{"instance_id": "id"}\n', encoding="utf-8")
    harness = SimpleNamespace(make_test_spec=lambda instance: SimpleNamespace(repo_script_list=["rustup default 1.72.0"]))
    monkeypatch.setattr(environment.importlib, "import_module", lambda name: harness)
    assert environment.benchmark_toolchain(dataset, "id", tmp_path) == "1.72.0"


@pytest.mark.parametrize("pinned", [False, True])
def test_verify_toolchain_without_modifying_repo(tmp_path, monkeypatch, pinned):
    root = tmp_path / "checkout"
    root.mkdir()
    if pinned:
        (root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.70.0"\n', encoding="utf-8")
    monkeypatch.setenv("RUSTUP_TOOLCHAIN", "wrong")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"{command[0]} 1.72.0\n")
    monkeypatch.setattr(environment.subprocess, "run", run)
    env, info = environment.prepare_rust_environment(root, "1.72.0")
    if pinned:
        assert "RUSTUP_TOOLCHAIN" not in env
        assert info["source"] == "rust-toolchain.toml"
        assert info["selected"] is None
    else:
        assert "RUSTUP_TOOLCHAIN" not in env
        assert info["source"] == "benchmark"
        assert info["selected"] == "1.72.0"
    assert [call[0] for call in calls] == [["rustc", "--version"], ["cargo", "--version"]]
    assert all(call[1]["env"] == env for call in calls)
    assert sorted(path.name for path in root.iterdir()) == (["rust-toolchain.toml"] if pinned else [])
    assert (tmp_path / "rust-toolchain").exists() is not pinned


def test_unavailable_toolchain_fails_before_agent(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="toolchain not installed")
    monkeypatch.setattr(environment.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="toolchain not installed"):
        environment.prepare_rust_environment(root, "1.72.0")


def test_mismatched_override_stops_run(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(environment.subprocess, "run", lambda command, **kwargs:
                        SimpleNamespace(stdout=f"{command[0]} 1.99.1\n"))
    with pytest.raises(RuntimeError, match="Expected rustc 1.72.0"):
        environment.prepare_rust_environment(root, "1.72.0")


def test_evaluation_override_takes_precedence_over_repo_pin(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "stable"\n', encoding="utf-8")
    monkeypatch.setenv("RUSTBENCH_EVAL_TOOLCHAIN_OVERRIDE", "1.72.0")
    monkeypatch.setattr(environment.subprocess, "run", lambda command, **kwargs:
                        SimpleNamespace(stdout=f"{command[0]} 1.72.0\n"))
    env, info = environment.prepare_rust_environment(root, "1.80.0")
    assert env["RUSTUP_TOOLCHAIN"] == "1.72.0"
    assert info["source"] == "evaluation_override"


def test_invalid_explicit_override_is_rejected_even_with_repo_pin(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "stable"\n', encoding="utf-8")
    monkeypatch.setenv("RUSTBENCH_EVAL_TOOLCHAIN_OVERRIDE", "1.72.0; other")
    with pytest.raises(ValueError, match="Invalid benchmark Rust toolchain"):
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



def test_cargo_cache_isolates_by_toolchain(tmp_path, monkeypatch):
    cache_root = tmp_path / "shared_cache"
    monkeypatch.setenv("RUSTBENCH_CARGO_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    env1, env2 = {}, {}
    info1 = environment.configure_cargo_cache(env1, repo_path=repo, instance_id="tokio-rs__bytes-253", toolchain="1.72.0")
    info2 = environment.configure_cargo_cache(env2, repo_path=repo, instance_id="tokio-rs__bytes-253", toolchain="1.80.0")
    assert info1["cargo_target_dir"].endswith("tokio-rs__bytes__1.72.0")
    assert info2["cargo_target_dir"].endswith("tokio-rs__bytes__1.80.0")
    assert info1["cargo_target_dir"] != info2["cargo_target_dir"]


def test_clean_crate_artifacts_removes_current_crate_and_keeps_deps(tmp_path):
    target = tmp_path / "cargo_target"
    for profile in ("debug", "release"):
        (target / profile / ".fingerprint" / "my_crate-1111").mkdir(parents=True)
        (target / profile / ".fingerprint" / "other_dep-2222").mkdir(parents=True)
        (target / profile / "deps").mkdir(parents=True)
        (target / profile / "deps" / "libmy_crate-1111.rlib").write_bytes(b"bin")
        (target / profile / "deps" / "libother_dep-2222.rlib").write_bytes(b"bin")
        (target / profile / "my_crate-1111").write_bytes(b"exe")

    environment.clean_crate_artifacts(target, ["my-crate"])
    for profile in ("debug", "release"):
        assert not (target / profile / ".fingerprint" / "my_crate-1111").exists()
        assert not (target / profile / "deps" / "libmy_crate-1111.rlib").exists()
        assert not (target / profile / "my_crate-1111").exists()
        # third-party deps preserved
        assert (target / profile / ".fingerprint" / "other_dep-2222").exists()
        assert (target / profile / "deps" / "libother_dep-2222.rlib").exists()


def test_find_repo_crate_names(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "root_pkg"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "crates" / "sub").mkdir(parents=True)
    (repo / "crates" / "sub" / "Cargo.toml").write_text('[package]\nname = "sub_pkg"\nversion = "0.1.0"\n', encoding="utf-8")
    names = environment.find_repo_crate_names(repo)
    assert set(names) == {"root_pkg", "sub_pkg"}


def test_prepare_rust_environment_retries_transient_rustup_download(tmp_path, monkeypatch):
    repo = tmp_path / "work" / "repo"
    repo.mkdir(parents=True)
    calls = {"rustc": 0}
    monkeypatch.setattr(environment.time, "sleep", lambda _s: None)

    def fake_run(cmd, **kwargs):
        exe = cmd[0]
        if exe == "rustc":
            calls["rustc"] += 1
            if calls["rustc"] < 2:
                err = subprocess.CalledProcessError(1, cmd, stderr="error: could not download file from rsproxy.cn: 504")
                raise err
            return SimpleNamespace(stdout="rustc 1.80.0\n", stderr="")
        return SimpleNamespace(stdout="cargo 1.80.0\n", stderr="")

    monkeypatch.setattr(environment.subprocess, "run", fake_run)
    env, info = environment.prepare_rust_environment(repo, "1.80.0", instance_id="demo__demo-1")
    assert calls["rustc"] == 2
    assert info["rustc"] == "rustc 1.80.0"
    assert "CARGO_TARGET_DIR" in env
