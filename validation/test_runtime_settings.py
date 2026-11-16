import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Rust-bench"))
from swebench.harness import test_spec


spec = importlib.util.spec_from_file_location("validation_runner_runtime", ROOT / "validation/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("group,expected_features", [("default", True), ("test", False)])
def test_test_agent_receives_evaluation_runtime_flags(tmp_path, monkeypatch, group, expected_features):
    harness = tmp_path / "harness"
    config_dir = harness / "swebench/harness/logs"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({group: ["org__repo-1"]}), encoding="utf-8")
    monkeypatch.setattr(runner, "HARNESS", harness)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "docker_base", lambda task, env: ["docker", "run"])
    monkeypatch.setattr(test_spec, "evaluation_toolchain_override", lambda row: "1.72.0")
    captured = {}

    def run_logged(args, log_path, cwd, env):
        captured["args"] = args
        return 0, 0

    monkeypatch.setattr(runner, "run_logged", run_logged)
    task = {"instance_id": "org__repo-1", "repo": "org/repo", "image": "task-image", "rust_toolchain": "1.70.0"}
    runner.run_baseline("test", task, {"problem_statement": "issue"}, tmp_path, {})
    args = captured["args"]
    assert "RUSTBENCH_EVAL_TOOLCHAIN_OVERRIDE=1.72.0" in args
    assert ("RUSTBENCH_ALL_FEATURES=0" in args) is not expected_features
    assert any("--rust-toolchain" in arg for arg in args)
    assert "1.70.0" in args
