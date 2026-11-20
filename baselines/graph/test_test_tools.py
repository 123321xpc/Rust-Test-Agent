"""Behavioral tests for the minimal baseline tools."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def normal(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "edit.rs").write_bytes(b"before\n")
    (repo / "delete.rs").write_bytes(b"delete me\n")
    spec = importlib.util.spec_from_file_location("test_tools_test", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools, state = module.make_tools(repo)
    yield repo, {tool.name: tool for tool in tools}, state
    state.close()


def test_edits_produce_applicable_complete_patch_without_mutating_checkout(normal, tmp_path):
    repo, tools, state = normal
    assert set(tools) == {"read_file", "write_file", "write_regression_test", "edit_file", "revert_file",
                          "checkpoint", "generate_patch", "read_lines", "search_code", "cargo_workspace",
                          "cargo_check", "cargo_test", "rust_symbols", "inspect_rust_neighborhood", "rust_error_explain",
                          "verifier_probe"}
    write = tools["write_file"]
    write.invoke({"path": "edit.rs", "content": "intermediate\n"})
    write.invoke({"path": "edit.rs", "content": "after\n"})
    write.invoke({"path": "new/file.rs", "content": "no newline"})
    write.invoke({"path": "delete.rs", "content": None})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_fix.rs", "content": "test"})
    assert tools["read_file"].invoke({"path": "edit.rs"}) == "after\n"
    state.pass_test(state.start_test(), test_target="normal_regression_fix")
    state.pass_test(state.start_test())
    preview = tools["generate_patch"].invoke({})
    assert "Unsubmitted preview" in preview and not state.patch
    patch = tools["generate_patch"].invoke({"paths": ["edit.rs", "new/file.rs", "delete.rs"]})
    assert state.patch == patch and "No newline at end of file" in patch
    assert (repo / "edit.rs").read_bytes() == b"before\n"
    assert (repo / "delete.rs").exists() and not (repo / "new").exists()
    copy = tmp_path / "verify"
    shutil.copytree(repo, copy)
    subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-"], cwd=copy, input=patch.encode(), check=True, capture_output=True)
    assert (copy / "edit.rs").read_bytes() == b"after\n"
    assert (copy / "new/file.rs").read_bytes() == b"no newline"
    assert not (copy / "delete.rs").exists()
    write.invoke({"path": "edit.rs", "content": "again\n"})
    assert not state.patch
    state.pass_test(state.start_test(), test_target="normal_regression_fix")
    state.pass_test(state.start_test())
    patch = tools["generate_patch"].invoke({"paths": ["edit.rs", "new/file.rs", "delete.rs"]})
    assert "+again\n" in patch


def test_revert_file_restores_modified_deleted_created_and_regression_files(normal):
    _, tools, state = normal
    write, revert = tools["write_file"], tools["revert_file"]
    assert "No changes to revert" in revert.invoke({"path": "edit.rs"})
    assert "Error" in revert.invoke({"path": "nonexistent.rs"})
    write.invoke({"path": "edit.rs", "content": "changed\n"})
    write.invoke({"path": "delete.rs", "content": None})
    write.invoke({"path": "scratch.rs", "content": "temp\n"})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_tmp.rs", "content": "test\n"})
    assert len(state.changed_paths()) == 4

    assert "original state" in revert.invoke({"path": "edit.rs"})
    assert (state.root / "edit.rs").read_bytes() == b"before\n"
    assert "original state" in revert.invoke({"path": "delete.rs"})
    assert (state.root / "delete.rs").read_bytes() == b"delete me\n"
    assert "removed newly created file" in revert.invoke({"path": "scratch.rs"})
    assert not (state.root / "scratch.rs").exists()
    assert "removed newly created file" in revert.invoke({"path": "tests/normal_regression_tmp.rs"})
    assert not state.regression_tests
    assert state.changed_paths() == []
    assert tools["generate_patch"].invoke({}) == "No changes to submit."


def test_red_phase_regression_is_locked_after_confirmed_failure(normal):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text(
        "[package]\nname = \"lockdemo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        encoding="utf-8",
    )
    write_test = tools["write_regression_test"]
    write_test.invoke({"path": "tests/normal_regression_lock.rs", "content": "fn main() {}\n"})
    key = ("Cargo.toml", "normal_regression_lock")
    state.fail_test(
        state.start_test(), manifest_path="Cargo.toml", test_target="normal_regression_lock",
        executed=1, failures=["normal_regression_lock"], returncode=101,
    )
    report = state.locked_regression_report()
    assert report["locked"], "a confirmed Red-phase regression must be locked"
    assert report["locked"][0]["name"] == "normal_regression_lock"
    assert report["locked"][0]["status"] == "intact"
    assert not report["tampered"]
    assert key in state._locked_regressions


def test_locked_regression_cannot_be_rewritten_or_reverted(normal):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text(
        "[package]\nname = \"lockdemo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        encoding="utf-8",
    )
    write_test = tools["write_regression_test"]
    write_test.invoke({"path": "tests/normal_regression_lock2.rs", "content": "fn main() {}\n"})
    state.fail_test(
        state.start_test(), manifest_path="Cargo.toml", test_target="normal_regression_lock2",
        executed=1, failures=["normal_regression_lock2"], returncode=101,
    )
    assert state._locked_regressions
    rewrite = write_test.invoke({
        "path": "tests/normal_regression_lock2.rs",
        "content": "fn main() { /* weakened */ }\n",
    })
    assert "locked" in rewrite.lower() and "Error" in rewrite
    revert = tools["revert_file"].invoke({"path": "tests/normal_regression_lock2.rs"})
    assert "locked" in revert.lower() and "Error" in revert
    assert not state.reproduced(), "a blocked revert must clear the Red-phase evidence"
    violations = state.locked_regression_report()["violations"]
    assert any(v["action"] == "rewrite_locked_regression" for v in violations)
    assert any(v["action"] == "revert_locked_regression" for v in violations)


def test_locked_regression_detects_tampering_in_report(normal):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text(
        "[package]\nname = \"lockdemo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        encoding="utf-8",
    )
    tools["write_regression_test"].invoke({
        "path": "tests/normal_regression_tamper.rs", "content": "fn main() {}\n",
    })
    state.fail_test(
        state.start_test(), manifest_path="Cargo.toml", test_target="normal_regression_tamper",
        executed=1, failures=["normal_regression_tamper"], returncode=101,
    )
    assert not state.locked_regression_report()["tampered"]
    target = state.root / "tests" / "normal_regression_tamper.rs"
    target.write_text("fn main() { assert!(false); }\n", encoding="utf-8")
    report = state.locked_regression_report()
    assert report["tampered"]
    assert report["tampered_entries"][0]["status"] == "modified"


def test_verifier_probe_rejects_unknown_probe_type(normal):
    _, tools, _state = normal
    out = tools["verifier_probe"].invoke({"probe_type": "nonsense"})
    assert "probe_type must be" in out


@pytest.mark.parametrize(
    ("assertion_code", "expected_status", "expected_passed", "expected_failed"),
    [
        ("assert_eq!(probe_demo::value(), 1);", "passed", 1, 0),
        ("assert_eq!(probe_demo::value(), 2);", "failed", 0, 1),
        ("#[test]\nfn complete_probe_test() { assert_eq!(probe_demo::value(), 1); }", "passed", 1, 0),
        ("#[test]\nfn complete_probe_test() { assert_eq!(probe_demo::value(), 2); }", "failed", 0, 1),
        ("#[test]\n#[ignore]\nfn ignored_probe_test() {}", "no_tests", 0, 0),
    ],
)
def test_verifier_probe_executes_assertions_and_cleans_temp_test(
    normal, assertion_code, expected_status, expected_passed, expected_failed,
):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text(
        "[package]\nname = \"probe_demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        encoding="utf-8",
    )
    (state.root / "src").mkdir(parents=True, exist_ok=True)
    (state.root / "src" / "lib.rs").write_text("pub fn value() -> i32 { 1 }\n", encoding="utf-8")
    probe_file = state.root / "tests" / "test_verifier_probe_tmp.rs"

    result = tools["verifier_probe"].invoke({
        "probe_type": "assertion",
        "manifest_path": "Cargo.toml",
        "assertion_code": assertion_code,
        "timeout": 60,
    })
    payload = json.loads(result)
    assert payload["status"] == expected_status
    assert payload["tests_passed"] == expected_passed
    assert payload["tests_failed"] == expected_failed
    assert not probe_file.exists()


def test_verifier_probe_preserves_existing_target_file(normal):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text("[package]\nname = 'probe_demo'\nversion = '0.1.0'\n", encoding="utf-8")
    probe_file = state.root / "tests" / "test_verifier_probe_tmp.rs"
    probe_file.parent.mkdir(parents=True, exist_ok=True)
    probe_file.write_text("existing test", encoding="utf-8")

    result = tools["verifier_probe"].invoke({"probe_type": "assertion", "assertion_code": "assert!(true);"})

    assert "already active" in result
    assert probe_file.read_text(encoding="utf-8") == "existing test"


def test_write_file_rejects_erasing_existing_rust_source_without_force(normal):
    _, tools, state = normal
    (state.root / "src").mkdir()
    (state.root / "src" / "lib.rs").write_text("pub fn keep() -> i32 { 1 }\n", encoding="utf-8")
    try:
        result = tools["write_file"].invoke({"path": "src/lib.rs", "content": "//! temporary note\n"})
        assert "Refusing to replace an existing Rust source" in result
        assert "pub fn keep" in (state.root / "src" / "lib.rs").read_text(encoding="utf-8")
        result = tools["write_file"].invoke({
            "path": "src/lib.rs", "content": "//! intentional reset\n", "force": True,
        })
        assert result.startswith("Updated src/lib.rs")
    finally:
        state.close()


def test_revert_preserves_recovery_checkpoint_when_dropping_last_substantive_change(normal):
    _, tools, state = normal
    source = state.root / "src" / "lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("pub fn keep() -> i32 { 1 }\n", encoding="utf-8")
    stat_result = source.stat()
    state.originals[Path("src/lib.rs")] = (source.read_bytes(), stat_result.st_mode, stat_result.st_mtime_ns)
    try:
        tools["write_file"].invoke({"path": "src/lib.rs", "content": "pub fn keep() -> i32 { 2 }\n"})
        result = tools["revert_file"].invoke({"path": "src/lib.rs"})
        assert "Preserved the prior working copy as checkpoint" in result
        assert state.changed_paths() == []
        recovery = [name for name in state._checkpoints if name.startswith("__recovery_before_revert_")]
        assert recovery
        state.restore_checkpoint(recovery[0])
        assert "{ 2 }" in (state.root / "src/lib.rs").read_text(encoding="utf-8")
    finally:
        state.close()


def test_checkpoint_restore_preserves_only_substantive_change(normal):
    _, tools, state = normal
    source = state.root / "src" / "lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("pub fn keep() -> i32 { 1 }\n", encoding="utf-8")
    stat_result = source.stat()
    state.originals[Path("src/lib.rs")] = (source.read_bytes(), stat_result.st_mode, stat_result.st_mtime_ns)
    try:
        assert "Saved checkpoint 'baseline'" in state.save_checkpoint("baseline")
        tools["write_file"].invoke({"path": "src/lib.rs", "content": "pub fn keep() -> i32 { 2 }\n"})
        result = tools["checkpoint"].invoke({"action": "restore", "name": "baseline"})
        assert "Preserved the prior working copy as checkpoint" in result
        recovery = [name for name in state._checkpoints if name.startswith("__recovery_before_restore_")]
        assert recovery
        assert "{ 1 }" in source.read_text(encoding="utf-8")
        state.restore_checkpoint(recovery[0])
        assert "{ 2 }" in source.read_text(encoding="utf-8")
    finally:
        state.close()


def test_existing_test_smoke_checks_multiple_targets_and_compares_baseline(tmp_path, monkeypatch):
    repo = tmp_path / "smoke_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    (repo / "tests" / "cli.rs").write_text("#[test] fn cli() {}\n", encoding="utf-8")
    (repo / "tests" / "integration.rs").write_text("#[test] fn integration() {}\n", encoding="utf-8")
    (repo / "Cargo.toml").write_text(
        '[package]\nname = "smoke_repo"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location("tools_smoke_compare", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools_list, state = module.make_tools(repo, post_strip_check=False)
    tools = {item.name: item for item in tools_list}
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/cargo")
    try:
        tools["write_file"].invoke({"path": "src/main.rs", "content": "fn main() { let _patched = true; }\n"})
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            source = (state.root / "src/main.rs").read_text(encoding="utf-8")
            if "_patched" in source:
                return SimpleNamespace(returncode=1, stdout="test result: FAILED. 0 passed; 1 failed\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="test result: ok. 1 passed; 0 failed\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        results = state.run_existing_test_smoke([Path("src/main.rs")], timeout=3)
        assert {item["target"] for item in results} == {"test:cli", "test:integration"}
        assert all(item["status"] == "failed" for item in results)
        assert all(item["classification"] == "code" for item in results)
        assert all(item["baseline_status"] == "passed" for item in results)
        assert len(calls) == 4
        assert "_patched" in (state.root / "src/main.rs").read_text(encoding="utf-8")
    finally:
        state.close()


def test_existing_test_smoke_does_not_attribute_baseline_failure_to_patch(tmp_path, monkeypatch):
    repo = tmp_path / "smoke_baseline_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    (repo / "tests" / "integration.rs").write_text("#[test] fn integration() {}\n", encoding="utf-8")
    (repo / "Cargo.toml").write_text(
        '[package]\nname = "smoke_baseline_repo"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location("tools_smoke_baseline", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools_list, state = module.make_tools(repo, post_strip_check=False)
    tools = {item.name: item for item in tools_list}
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/cargo")
    try:
        tools["write_file"].invoke({"path": "src/main.rs", "content": "fn main() { let _patched = true; }\n"})
        calls = iter([
            SimpleNamespace(returncode=1, stdout="test result: FAILED. 0 passed; 1 failed\n", stderr=""),
            SimpleNamespace(returncode=1, stdout="test result: FAILED. 0 passed; 1 failed\n", stderr=""),
        ])
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(calls))
        results = state.run_existing_test_smoke([Path("src/main.rs")], timeout=3)
        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert results[0]["classification"] == "unknown"
        assert results[0]["baseline_status"] == "failed"
        assert results[0]["baseline_label"] == "BASELINE_UNCONFIRMED_NOT_PREEXISTING"
    finally:
        state.close()


def test_existing_test_smoke_flags_snapshot_change_under_same_test_name(tmp_path, monkeypatch):
    """Identical failing test names with different snapshot output must be attributed to the patch."""
    repo = tmp_path / "smoke_snapshot_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    (repo / "tests" / "integration.rs").write_text("#[test] fn integration() {}\n", encoding="utf-8")
    (repo / "Cargo.toml").write_text(
        '[package]\nname = "smoke_snapshot_repo"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location("tools_smoke_snapshot", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools_list, state = module.make_tools(repo, post_strip_check=False)
    tools = {item.name: item for item in tools_list}
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/cargo")
    try:
        tools["write_file"].invoke({"path": "src/main.rs", "content": "fn main() { let _patched = true; }\n"})
        patched = SimpleNamespace(
            returncode=1,
            stdout=(
                "test integration ... FAILED\n"
                "---- integration stdout ----\n"
                "Snapshot Summary\n"
                "- old rendered value\n"
                "+ new rendered value\n"
                "test result: FAILED. 0 passed; 1 failed\n"
            ),
            stderr="",
        )
        baseline = SimpleNamespace(
            returncode=1,
            stdout=(
                "test integration ... FAILED\n"
                "---- integration stdout ----\n"
                "Snapshot Summary\n"
                "- old rendered value\n"
                "test result: FAILED. 0 passed; 1 failed\n"
            ),
            stderr="",
        )
        calls = iter([patched, baseline])
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(calls))
        results = state.run_existing_test_smoke([Path("src/main.rs")], timeout=3)
        assert len(results) == 1
        assert results[0]["classification"] == "code"
        assert results[0]["baseline_status"] == "failed"
        assert results[0]["new_failed_tests"] == ["integration"]
    finally:
        state.close()


def test_unlocked_regression_can_be_rewritten_but_only_once(normal):
    """An Architect-released oracle may be corrected; the release works only once."""
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text(
        "[package]\nname = \"lockdemo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        encoding="utf-8",
    )
    write_test = tools["write_regression_test"]
    write_test.invoke({"path": "tests/normal_regression_unlock.rs", "content": "fn main() {}\n"})
    state.fail_test(
        state.start_test(), manifest_path="Cargo.toml", test_target="normal_regression_unlock",
        executed=1, failures=["normal_regression_unlock"], returncode=101,
    )
    assert state._locked_regressions

    refused = state.unlock_regression("normal_regression_unlock", "too short")
    assert not refused["ok"]
    assert "20 chars" in refused["error"]

    released = state.unlock_regression(
        "normal_regression_unlock",
        "The locked test asserted the wrong expected value for the parsed date.",
        approved_by="architect",
    )
    assert released["ok"]
    assert released["unlocked"] == "normal_regression_unlock"
    assert not state._locked_regressions

    rewrite = write_test.invoke({
        "path": "tests/normal_regression_unlock.rs",
        "content": "fn main() { /* corrected oracle */ }\n",
    })
    assert rewrite.startswith("Updated") or "locked" not in rewrite.lower()

    again = state.unlock_regression(
        "normal_regression_unlock",
        "Trying to release the same oracle a second time for extra latitude.",
    )
    assert not again["ok"]
    assert "already unlocked" in again["error"]


def test_changed_line_numbers_reports_only_modified_lines():
    spec = importlib.util.spec_from_file_location("tools_changed_lines", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = "a\nb\nc\n"
    new = "a\nB\nc\n"
    assert module._changed_line_numbers(old, new) == [2]
    assert module._changed_line_numbers(old, old) == []
    assert module._changed_line_numbers("", "x\ny\n") == [1, 2]


def test_budget_notice_injected_when_remaining_calls_low(normal):
    _, tools, state = normal
    client = SimpleNamespace(max_calls=10, remaining_calls=2)
    state.set_budget(client, warning_threshold=3)
    out = tools["write_file"].invoke({"path": "edit.rs", "content": "after\n"})
    assert "Budget notice: 2 API call(s) remaining" in out
    symbols_out = json.loads(tools["rust_symbols"].invoke({}))
    assert "Budget notice: 2 API call(s) remaining" in symbols_out["budget_notice"]
    assert tools["read_file"].invoke({"path": "edit.rs"}) == "after\n"


def test_patch_selection_excludes_scratch_but_includes_new_files(normal, tmp_path):
    repo, tools, state = normal
    write = tools["write_file"]
    write.invoke({"path": "edit.rs", "content": "fixed\n"})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_fix.rs", "content": "test\n"})
    write.invoke({"path": "macro_check.rs", "content": "scratch\n"})
    state.pass_test(state.start_test(), test_target="normal_regression_fix")
    state.pass_test(state.start_test())

    preview = tools["generate_patch"].invoke({})
    assert "macro_check.rs" in preview and not state.patch
    assert "Include all other changed files" in tools["generate_patch"].invoke({"paths": ["edit.rs"]})
    write.invoke({"path": "macro_check.rs", "content": None})
    state.pass_test(state.start_test(), test_target="normal_regression_fix")
    state.pass_test(state.start_test())
    patch = tools["generate_patch"].invoke({"paths": ["edit.rs"]})
    assert state.patch == patch and "normal_regression_fix.rs" not in patch
    copy = tmp_path / "selected"
    shutil.copytree(repo, copy)
    subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-"], cwd=copy,
                   input=patch.encode(), check=True, capture_output=True)
    assert (copy / "edit.rs").read_bytes() == b"fixed\n"
    assert not (copy / "tests/normal_regression_fix.rs").exists()
    assert not (copy / "macro_check.rs").exists()

    write.invoke({"path": "macro_check.rs", "content": "updated scratch\n"})
    assert not state.patch
    state.pass_test(state.start_test(), test_target="normal_regression_fix")
    state.pass_test(state.start_test())
    assert "Not a changed file" in tools["generate_patch"].invoke({"paths": ["missing.rs"]})
    assert not state.patch


def test_paths_and_lifetime(normal, tmp_path):
    repo, tools, state = normal
    assert "Error" in tools["read_file"].invoke({"path": "../outside"})
    assert "Error" in tools["write_file"].invoke({"path": "../outside", "content": "bad"})
    assert "Error" in tools["write_file"].invoke({"path": ".git/config", "content": "bad"})
    assert "Error" in tools["write_file"].invoke({"path": "missing", "content": None})
    assert tools["generate_patch"].invoke({}) == "No changes to submit."
    temp_root = state.root
    state.close()
    assert not temp_root.exists()


def test_patch_submission_preserves_verification_evidence_after_last_write(normal, monkeypatch):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text('[package]\nname = "sample"\nversion = "0.1.0"\n')
    tools["write_file"].invoke({"path": "edit.rs", "content": "fixed\n"})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_fix.rs", "content": "test"})
    generate = tools["generate_patch"]
    cargo_test = tools["cargo_test"]
    assert "Pending verification: pass regression tests" in generate.invoke({})
    assert generate.invoke({"paths": ["edit.rs"]}).startswith("diff --git")
    assert not state.verified()

    real_run = subprocess.run

    def result(returncode, output):
        def run(command, *args, **kwargs):
            if command[0] == "cargo":
                return SimpleNamespace(returncode=returncode, stdout=output, stderr="")
            return real_run(command, *args, **kwargs)
        monkeypatch.setattr(subprocess, "run", run)

    result(1, "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n")
    assert json.loads(cargo_test.invoke({}))["status"] == "failed"
    assert "Pending verification: pass regression tests" in generate.invoke({})
    result(0, "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")
    assert json.loads(cargo_test.invoke({}))["tests_executed"] == 0
    assert "Pending verification: pass regression tests" in generate.invoke({})
    def timeout_cargo(command, *args, **kwargs):
        if command[0] == "cargo":
            raise subprocess.TimeoutExpired("cargo", 1)
        return real_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", timeout_cargo)
    assert json.loads(cargo_test.invoke({}))["status"] == "timeout"
    assert "Pending verification: pass regression tests" in generate.invoke({})
    result(0, "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")
    assert json.loads(cargo_test.invoke({}))["tests_executed"] == 1
    assert "Pending verification: pass regression tests" in generate.invoke({})
    cargo_test.invoke({"test_target": "normal_regression_fix"})
    monkeypatch.undo()
    assert "Unsubmitted preview" in generate.invoke({})
    patch = generate.invoke({"paths": ["edit.rs"]})
    assert state.patch == patch and patch
    assert json.loads(cargo_test.invoke({"timeout": 0}))["status"] == "error"
    assert not state.patch and "Unsubmitted preview" in generate.invoke({})
    result(0, "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")
    cargo_test.invoke({})
    cargo_test.invoke({"test_target": "normal_regression_fix"})
    monkeypatch.undo()
    assert generate.invoke({"paths": ["edit.rs"]}).startswith("diff --git")
    tools["write_file"].invoke({"path": "edit.rs", "content": "changed again\n"})
    assert not state.patch and "Pending verification: pass regression tests" in generate.invoke({})


def test_gate_requires_matching_package_regression(normal):
    _, tools, state = normal
    tools["write_file"].invoke({"path": "edit.rs", "content": "fixed\n"})
    tools["write_regression_test"].invoke({"path": "member/tests/normal_regression_issue.rs", "content": "test"})
    generate = tools["generate_patch"]
    state.pass_test(state.start_test(), test_target="normal_regression_issue")
    state.pass_test(state.start_test())
    assert not state.verified()
    state.pass_test(state.start_test(), manifest_path="member/Cargo.toml", package="other", test_target="normal_regression_issue")
    state.pass_test(state.start_test(), manifest_path="member/Cargo.toml", package="other")
    assert not state.verified()
    state.pass_test(state.start_test(), manifest_path="member/Cargo.toml", test_target="normal_regression_issue", test_filter="one")
    state.pass_test(state.start_test(), manifest_path="member/Cargo.toml", lib=True)
    assert not state.verified()
    state.pass_test(state.start_test(), manifest_path="member/Cargo.toml", test_target="normal_regression_issue")
    assert state.verified()
    state.pass_test(state.start_test(), manifest_path="member/Cargo.toml")
    assert state.verified()
    assert "Temporary regression test" in generate.invoke({"paths": ["edit.rs", "member/tests/normal_regression_issue.rs"]})
    assert generate.invoke({"paths": ["edit.rs"]}).startswith("diff --git")
    token = state.start_test()
    state.fail_test(token, manifest_path="member/Cargo.toml", test_target="normal_regression_issue")
    state.pass_test(state.start_test(), manifest_path="member/Cargo.toml")
    assert not state.verified()


def test_filtered_standalone_regression_counts_only_when_target_was_fully_run(normal):
    _, tools, state = normal
    state.require_reproduction = True
    tools["write_regression_test"].invoke({
        "path": "tests/normal_regression_filtered.rs",
        "content": "#[test] fn reproduces_issue() {}\n#[test] fn another_case() {}",
    })
    target = "normal_regression_filtered"

    state.fail_test(
        state.start_test(),
        test_target=target,
        test_filter="reproduces_issue",
        failures=["reproduces_issue"],
        filtered_tests=1,
        executed=1,
        returncode=101,
    )
    assert not state.reproduced_cleanly()

    state.fail_test(
        state.start_test(),
        test_target=target,
        test_filter="reproduces_issue",
        failures=["reproduces_issue"],
        filtered_tests=0,
        executed=1,
        returncode=101,
    )
    assert state.reproduced_cleanly()

    tools["write_file"].invoke({"path": "src/lib.rs", "content": "pub fn fixed() {}\n"})
    state.pass_test(
        state.start_test(),
        test_target=target,
        test_filter="reproduces_issue",
        passed_tests=["reproduces_issue"],
        filtered_tests=1,
        executed=1,
    )
    assert not state.verified()
    assert (Path("Cargo.toml"), target) not in state._passed_regressions

    state.pass_test(
        state.start_test(),
        test_target=target,
        test_filter="reproduces_issue",
        passed_tests=["reproduces_issue"],
        filtered_tests=0,
        executed=2,
    )
    assert state.verified()

    state.fail_test(
        state.start_test(),
        test_target=target,
        test_filter="another_case",
        failures=["another_case"],
        filtered_tests=1,
        executed=1,
        returncode=101,
    )
    assert not state.verified()
    assert (Path("Cargo.toml"), target) not in state._passed_regressions


def test_cargo_toml_edit_reports_only_locked_lockfile_update_errors(normal, monkeypatch):
    _, tools, state = normal
    manifest = state.root / "Cargo.toml"
    lockfile = state.root / "Cargo.lock"
    lockfile.write_text("# existing lockfile\n", encoding="utf-8")
    calls = []

    def cargo_metadata(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                command,
                101,
                "",
                "error: cannot update the lock file because --locked was passed to prevent this\n",
            )
        return subprocess.CompletedProcess(command, 101, "", "error: no matching package found offline\n")

    monkeypatch.setattr(shutil, "which", lambda _: "cargo")
    monkeypatch.setattr(subprocess, "run", cargo_metadata)

    warning = tools["write_file"].invoke({
        "path": "Cargo.toml",
        "content": "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    })
    assert "Cargo.lock constraint" in warning
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "--offline" in command and "--locked" in command
    assert "--no-deps" not in command
    assert kwargs["timeout"] == 20

    second = tools["write_file"].invoke({
        "path": "Cargo.toml",
        "content": "[package]\nname = \"demo\"\nversion = \"0.2.0\"\nedition = \"2021\"\n",
    })
    assert "Cargo.lock constraint" not in second
    assert len(calls) == 2


def test_regression_tool_rejects_existing_files_and_other_writes(normal):
    repo, tools, state = normal
    assert "Error" in tools["write_regression_test"].invoke({"path": "edit.rs", "content": "test"})
    assert "tests/test_regression_" in tools["write_regression_test"].invoke({"path": "tests/regression.rs", "content": "test"})
    assert "Updated" in tools["write_regression_test"].invoke({"path": "tests/normal_regression_issue.rs", "content": "test"})
    assert "Use write_regression_test" in tools["write_file"].invoke({"path": "tests/normal_regression_issue.rs", "content": "other"})


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_registered_rust_tools_use_working_copy(normal):
    repo, tools, _ = normal
    tools["write_file"].invoke({"path": "Cargo.toml", "content": '[package]\nname = "example"\nversion = "0.1.0"\nedition = "2021"\n'})
    tools["write_file"].invoke({"path": "src/lib.rs", "content": 'pub fn value() -> i32 { "wrong" }\n'})
    assert "1: pub fn value" in tools["read_lines"].invoke({"path": "src/lib.rs"})
    assert "src/lib.rs:1:" in tools["search_code"].invoke({"query": "value"})
    assert json.loads(tools["cargo_workspace"].invoke({}))["workspace_members"][0]["name"] == "example"
    result = json.loads(tools["cargo_check"].invoke({}))
    assert any(d["category"] == "type_mismatch" for d in result["diagnostics"])
    assert not (repo / "Cargo.toml").exists()


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_edit_and_test_repair_in_working_copy(normal):
    repo, tools, state = normal
    tools["write_file"].invoke({"path": "Cargo.toml", "content": '[package]\nname = "repair"\nversion = "0.1.0"\nedition = "2021"\n'})
    tools["write_file"].invoke({"path": "src/lib.rs", "content": '#[test]\nfn regression() { assert_eq!(1, 2); }\n'})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_fix.rs", "content": '#[test]\nfn works() { assert!(true); }\n'})
    assert json.loads(tools["cargo_test"].invoke({"test_filter": "regression"}))["status"] == "failed"
    assert "Updated" in tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "assert_eq!(1, 2)", "new_text": "assert_eq!(1, 1)"})
    assert json.loads(tools["cargo_test"].invoke({"test_filter": "regression"}))["status"] == "passed"
    assert json.loads(tools["cargo_test"].invoke({"test_target": "normal_regression_fix"}))["status"] == "passed"
    assert json.loads(tools["cargo_test"].invoke({}))["status"] == "passed"
    patch = tools["generate_patch"].invoke({"paths": ["Cargo.toml", "src/lib.rs"]})
    assert "assert_eq!(1, 1)" in patch
    assert not (repo / "src").exists()


def test_registered_symbol_tool_sees_working_copy_edits(normal):
    repo, tools, _ = normal
    source = 'pub struct Widget;\nfn use_widget(_: Widget) {}\n'
    tools["write_file"].invoke({"path": "src/lib.rs", "content": source})
    definitions = json.loads(tools["rust_symbols"].invoke({"name": "Widget"}))
    assert any(item["kind"] == "struct" and item["path"] == "src/lib.rs" for item in definitions["results"])
    assert not (repo / "src").exists()


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc is not installed")
def test_registered_rust_error_explanation(normal):
    _, tools, _ = normal
    assert "moved" in tools["rust_error_explain"].invoke({"code": "E0382"}).lower()


def test_red_phase_reproduction_tracking(normal):
    _, tools, state = normal
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_bug.rs", "content": "test"})
    assert not state.reproduced()
    token = state.start_test()
    # Simulate failed run on regression test (executed 1 test, 1 failure)
    state.fail_test(token, test_target="normal_regression_bug", executed=1, failures=["bug_fails"], returncode=101)
    assert state.reproduced()
    info = state._reproduced_regressions[(Path("Cargo.toml"), "normal_regression_bug")]
    assert info["clean_red"] is True
    assert info["runtime_failure"] is True

    # Rewriting regression test resets reproduction
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_bug.rs", "content": "test v2"})
    assert not state.reproduced()

    # Reproduce again
    token = state.start_test()
    state.fail_test(token, test_target="normal_regression_bug", executed=1, failures=["bug_fails"], returncode=101)
    assert state.reproduced()

    # Reverting regression test clears reproduction
    tools["revert_file"].invoke({"path": "tests/normal_regression_bug.rs"})
    assert not state.reproduced()


def test_require_reproduction_reports_pending_verification(normal):
    _, tools, state = normal
    state.require_reproduction = True
    tools["write_file"].invoke({"path": "edit.rs", "content": "fixed\n"})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_bug.rs", "content": "test"})
    token = state.start_test()
    # If test passes directly without prior reproduction:
    state.pass_test(token, test_target="normal_regression_bug")
    state.pass_test(state.start_test())
    assert not state.verified()
    preview = tools["generate_patch"].invoke({})
    assert "Unsubmitted preview" in preview
    assert "confirm failure (Red phase)" in preview

    # Now simulate a prior reproduction run
    token = state.start_test()
    state.fail_test(token, test_target="normal_regression_bug", executed=1, failures=["fails"])
    assert state.reproduced()

    # Now pass the test on fixed code
    token = state.start_test()
    state.pass_test(token, test_target="normal_regression_bug")
    state.pass_test(state.start_test())
    assert state.verified()


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_workspace_package_normalization_and_tests_suite(normal):
    _, tools, state = normal
    # Create a workspace with a subcrate
    tools["write_file"].invoke({"path": "Cargo.toml", "content": '[workspace]\nmembers = ["crates/sub"]\nresolver = "2"\n'})
    tools["write_file"].invoke({"path": "crates/sub/Cargo.toml", "content": '[package]\nname = "sub_crate"\nversion = "0.1.0"\nedition = "2021"\n'})
    tools["write_file"].invoke({"path": "crates/sub/src/lib.rs", "content": 'pub fn value() -> i32 { 10 }\n#[test]\nfn unit() { assert_eq!(value(), 10); }\n'})
    tools["write_regression_test"].invoke({"path": "crates/sub/tests/normal_regression_sub.rs", "content": '#[test]\nfn reg() { assert_eq!(sub_crate::value(), 10); }\n'})

    # Agent runs from workspace root specifying package="sub_crate"
    reg_run = json.loads(tools["cargo_test"].invoke({"manifest_path": "Cargo.toml", "package": "sub_crate", "test_target": "normal_regression_sub"}))
    assert reg_run["status"] == "passed"
    suite_run = json.loads(tools["cargo_test"].invoke({"manifest_path": "Cargo.toml", "package": "sub_crate", "tests": True}))
    assert suite_run["status"] == "passed"

    assert state.verified()
    patch = tools["generate_patch"].invoke({"paths": ["Cargo.toml", "crates/sub/Cargo.toml", "crates/sub/src/lib.rs"]})
    assert "sub_crate" in patch and "normal_regression_sub.rs" not in patch


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_post_strip_check_catches_unstripped_dependency(normal):
    _, tools, state = normal
    tools["write_file"].invoke({"path": "Cargo.toml", "content": '[package]\nname = "broken"\nversion = "0.1.0"\nedition = "2021"\n'})
    # src/lib.rs contains code with a syntax/compiler error
    tools["write_file"].invoke({"path": "src/lib.rs", "content": 'pub fn broken() { let _: i32 = "wrong"; }\n'})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_check.rs", "content": '#[test]\nfn dummy() {}\n'})

    # Mock passes for the gate to reach generate_patch submission
    state.pass_test(state.start_test(), test_target="normal_regression_check")
    state.pass_test(state.start_test())
    assert state.verified()

    # Attempt to submit patch with broken src/lib.rs
    out = tools["generate_patch"].invoke({"paths": ["Cargo.toml", "src/lib.rs"]})
    assert out.startswith("diff --git")
    assert state.patch == out
    assert "Post-strip verification failed" in state._last_submit_error


def test_feedback_precision_displays_runnable_commands(normal):
    _, tools, state = normal
    tools["write_file"].invoke({"path": "edit.rs", "content": "modified\n"})
    tools["write_regression_test"].invoke({"path": "crates/pkg/tests/normal_regression_xyz.rs", "content": "test"})
    msg = tools["generate_patch"].invoke({})
    assert "cargo_test(manifest_path='crates/pkg/Cargo.toml', test_target='normal_regression_xyz')" in msg
    assert "tests=True" not in msg


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_old_test_failure_does_not_block_all_generated_tests(normal):
    _, tools, state = normal
    tools["write_file"].invoke({"path": "Cargo.toml", "content": '[package]\nname = "gate_example"\nversion = "0.1.0"\nedition = "2021"\n'})
    tools["write_file"].invoke({"path": "src/lib.rs", "content": 'pub fn value() -> u32 { 1 }\n'})
    (state.root / "tests").mkdir(parents=True, exist_ok=True)
    (state.root / "tests" / "old.rs").write_text('#[test] fn old_behavior() { assert_eq!(gate_example::value(), 0); }\n', encoding="utf-8")
    for name in ("first", "second"):
        tools["write_regression_test"].invoke({"path": f"tests/normal_regression_{name}.rs", "content": f'#[test] fn {name}() {{ assert_eq!(gate_example::value(), 1); }}\n'})
    test = tools["cargo_test"]
    assert json.loads(test.invoke({"test_target": "normal_regression_first"}))["status"] == "passed"
    assert not state.verified()
    assert json.loads(test.invoke({"test_target": "normal_regression_second"}))["status"] == "passed"
    assert state.verified()
    old = json.loads(test.invoke({"tests": True}))
    assert old["status"] == "failed" and "old_behavior" in old["failed_tests"]
    assert state.verified()
    patch = tools["generate_patch"].invoke({"paths": ["Cargo.toml", "src/lib.rs"]})
    assert patch.startswith("diff --git") and "normal_regression_" not in patch
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_second.rs", "content": '#[test] fn second() { assert_eq!(1, 0); }\n'})
    assert not state.verified()
    assert json.loads(test.invoke({"test_target": "normal_regression_second"}))["status"] == "failed"
    assert not state.verified()


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_ignored_generated_test_does_not_pass_gate(normal):
    _, tools, state = normal
    tools["write_file"].invoke({"path": "Cargo.toml", "content": '[package]\nname = "ignored_example"\nversion = "0.1.0"\nedition = "2021"\n'})
    tools["write_file"].invoke({"path": "src/lib.rs", "content": 'pub fn value() {}\n'})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_ignored.rs", "content": '#[test] fn runs() {}\n#[test] #[ignore] fn skipped() {}\n'})
    result = json.loads(tools["cargo_test"].invoke({"test_target": "normal_regression_ignored"}))
    assert result["status"] == "passed" and result["tests_ignored"] == 1
    assert not state.verified()


def test_git_worktree_isolation_and_direct_diff(tmp_path):
    repo = tmp_path / "git_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)
    (repo / "edit.rs").write_bytes(b"before\n")
    (repo / "delete.rs").write_bytes(b"delete me\n")
    (repo / "keep.rs").write_bytes(b"keep me\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    supports_worktree = subprocess.run(
        ["git", "worktree", "list"], cwd=repo, capture_output=True, text=True
    ).returncode == 0

    spec = importlib.util.spec_from_file_location("test_tools_git_test", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools, state = module.make_tools(repo, instance_id="tokio-rs__bytes-253")
    tool_map = {t.name: t for t in tools}
    try:
        assert state._is_git is True
        assert state._is_worktree is supports_worktree
        assert (state.root / ".git").exists()
        assert "tokio-rs__bytes" in state.rust_env["CARGO_TARGET_DIR"]

        tool_map["write_file"].invoke({"path": "edit.rs", "content": "after\n"})
        tool_map["write_file"].invoke({"path": "new/file.rs", "content": "pub fn new_fn() {}\n"})
        tool_map["write_file"].invoke({"path": "delete.rs", "content": None})
        tool_map["write_regression_test"].invoke({"path": "tests/normal_regression_fix.rs", "content": "test"})
        state.pass_test(state.start_test(), test_target="normal_regression_fix")
        state.pass_test(state.start_test())

        patch = tool_map["generate_patch"].invoke({"paths": ["edit.rs", "new/file.rs", "delete.rs"]})
        assert state.patch == patch
        assert "normal_regression_fix" not in patch
        assert (repo / "edit.rs").read_bytes() == b"before\n"
        assert (repo / "delete.rs").exists()

        verify = tmp_path / "verify_clone"
        subprocess.run(["git", "-c", "core.autocrlf=false", "clone", str(repo), str(verify)], check=True, capture_output=True)
        subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-"], cwd=verify, input=patch.encode("utf-8"), check=True, capture_output=True)
        assert (verify / "edit.rs").read_bytes() == b"after\n"
        assert (verify / "new" / "file.rs").read_bytes() == b"pub fn new_fn() {}\n"
        assert not (verify / "delete.rs").exists()
    finally:
        state.close()

    if supports_worktree:
        wt_list = subprocess.run(["git", "worktree", "list"], cwd=repo, check=True, capture_output=True, text=True).stdout
        assert "test-agent-" not in wt_list
    else:
        assert not state.root.exists()


def test_patch_state_uses_deterministic_workspace_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.rs").write_text("pub fn v() -> i32 { 1 }\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("tools_mod", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools, state = m.make_tools(repo, instance_id="tokio-rs__bytes-253")
    try:
        assert "test-agent-workspaces" in str(state.root)
        assert "tokio-rs__bytes-253" in str(state.root)
        assert state._custom_workspace is True
        assert (state.root / "lib.rs").is_file()
    finally:
        state.close()
    assert not state.root.exists()



def test_fail_test_reproduction_rules(normal):
    _, tools, state = normal
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_edge.rs", "content": "test"})
    token = state.start_test()

    # Timeout or OSError (returncode is None, executed 0) -> rejected
    state.fail_test(token, test_target="normal_regression_edge", executed=0, failures=[], returncode=None)
    assert not state.reproduced()

    # Executed 0 with non-API compile error -> rejected
    state.fail_test(token, test_target="normal_regression_edge", executed=0, returncode=101, compile_errors=[{"code": "E0000"}])
    assert not state.reproduced()

    # Executed 0 on clean base with valid API compile error code (e.g. E0425, E0599) -> accepted as compile_reproduction
    state.fail_test(token, test_target="normal_regression_edge", executed=0, returncode=101, compile_errors=[{"code": "E0599"}])
    assert state.reproduced()
    assert state._reproduced_regressions[(Path("Cargo.toml"), "normal_regression_edge")]["compile_reproduction"] is True

    # Runtime failure (executed > 0) -> accepted
    state.fail_test(token, test_target="normal_regression_edge", executed=1, failures=["fail"], returncode=101)
    assert state.reproduced()
    assert state._reproduced_regressions[(Path("Cargo.toml"), "normal_regression_edge")]["runtime_failure"] is True


def test_exploration_notice_after_repeated_reads(normal):
    _, tools, state = normal
    for _ in range(15):
        tools["read_lines"].invoke({"path": "edit.rs", "start_line": 1, "end_line": 1})
    notice = state.budget_notice()
    assert notice is not None
    assert "Exploration notice: 15 read/search calls" in notice

    # Writing a test resets the counter
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_t.rs", "content": "test"})
    assert state.budget_notice() is None


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_paired_reproduction_catches_fake_test_and_accepts_real_bug_test(tmp_path):
    repo = tmp_path / "cargo_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "paired_test"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn answer() -> i32 { 1 }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_paired", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools, state = m.make_tools(repo, require_reproduction=True)
    tool_map = {t.name: t for t in tools}
    try:
        # Agent edits source FIRST (answer -> 2)
        tool_map["edit_file"].invoke({"path": "src/lib.rs", "old_text": "1", "new_text": "2"})

        # Case A: Agent writes a FAKE test that asserts 1 == 1 (passes on both original and patched code)
        tool_map["write_regression_test"].invoke({
            "path": "tests/normal_regression_fake.rs",
            "content": "#[test] fn fake() { assert_eq!(1, 1); }\n",
        })
        run_res = json.loads(tool_map["cargo_test"].invoke({"test_target": "normal_regression_fake"}))
        assert run_res["status"] == "passed"

        preview = tool_map["generate_patch"].invoke({})
        assert "Unsubmitted preview" in preview
        assert "Paired reproduction check failed" in preview
        assert not state.patch
        assert not state.reproduced_cleanly()

        # Case B: Agent replaces with a REAL regression test asserting answer() == 2 (fails on original code 1, passes on fixed code 2)
        tool_map["write_regression_test"].invoke({
            "path": "tests/normal_regression_fake.rs",
            "content": "#[test] fn real() { assert_eq!(paired_test::answer(), 2); }\n",
        })
        run_res2 = json.loads(tool_map["cargo_test"].invoke({"test_target": "normal_regression_fake"}))
        assert run_res2["status"] == "passed"

        # Paired check should now confirm it fails on original code (clean_red verified!) and allow submission
        preview2 = tool_map["generate_patch"].invoke({})
        assert "Unsubmitted preview" in preview2
        patch = tool_map["generate_patch"].invoke({"paths": ["src/lib.rs"]})
        assert "pub fn answer() -> i32 { 2 }" in patch
        assert state.patch == patch
    finally:
        state.close()



def test_checkpoint_tool_save_restore_and_list(normal):
    repo, tools, state = normal
    edit, cp, wf = tools["edit_file"], tools["checkpoint"], tools["write_file"]

    # Initial edit
    wf.invoke({"path": "src/lib.rs", "content": "pub fn a() -> i32 { 1 }\n"})
    assert "Saved checkpoint 'v1'" in cp.invoke({"action": "save", "name": "v1"})
    assert len(state.changed_paths()) == 1

    # Speculative edits: edit src/lib.rs and add scratch.rs
    edit.invoke({"path": "src/lib.rs", "old_text": "1", "new_text": "99"})
    wf.invoke({"path": "scratch.rs", "content": "temp\n"})
    assert len(state.changed_paths()) == 2
    assert "99" in tools["read_file"].invoke({"path": "src/lib.rs"})

    # Check listing
    c_list = json.loads(cp.invoke({"action": "list"}))
    assert any(c["name"] == "v1" for c in c_list)

    # Restore v1: scratch.rs must be deleted, src/lib.rs restored to 1!
    res = cp.invoke({"action": "restore", "name": "v1"})
    assert "Restored checkpoint 'v1'" in res
    assert not (state.root / "scratch.rs").exists()
    assert "1" in tools["read_file"].invoke({"path": "src/lib.rs"})
    assert "99" not in tools["read_file"].invoke({"path": "src/lib.rs"})
    assert "scratch.rs" not in [p.as_posix() for p in state.changed_paths()]

    # Unknown checkpoint
    assert "Error in checkpoint: Unknown checkpoint" in cp.invoke({"action": "restore", "name": "missing"})


def test_edit_file_emits_signature_change_impact_notice(tmp_path):
    repo = tmp_path / "sig_repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn compute(x: i32) -> i32 { x + 1 }\n", encoding="utf-8")
    (repo / "src" / "caller.rs").write_text("use crate::compute;\npub fn run() { compute(10); }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_sig", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools, state = m.make_tools(repo)
    tool_map = {t.name: t for t in tools}
    try:
        # Edit signature from (x: i32) to (x: &i32)
        res = tool_map["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "pub fn compute(x: i32) -> i32",
            "new_text": "pub fn compute(x: &i32) -> i32",
        })
        assert "Updated src/lib.rs" in res
        assert "Impact notice: signature modified for `compute`" in res
        assert "src/caller.rs:2" in res
    finally:
        state.close()


def test_failure_tracker_injects_strategy_warning(normal, monkeypatch):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text('[package]\nname = "test_tracker"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    check_tool = tools["cargo_check"]

    # First check error
    err_json = json.dumps({
        "reason": "compiler-message",
        "message": {
            "level": "error",
            "code": {"code": "E0277"},
            "message": "the trait bound `T: Display` is not satisfied",
            "spans": [{"file_name": "src/lib.rs", "line_start": 10, "is_primary": True}],
            "children": [],
        },
    })
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 101, err_json + "\n", ""))

    res1 = json.loads(check_tool.invoke({}))
    assert "strategy_warning" not in res1

    # Edit file to advance revision, then check again with same error
    tools["write_file"].invoke({"path": "src/lib.rs", "content": "dummy\n"})
    res2 = json.loads(check_tool.invoke({}))
    assert "strategy_warning" in res2
    assert "Strategy warning: the same compiler error persisted across 2 edit revisions" in res2["strategy_warning"]



def test_auto_submit_falls_back_to_best_verified_snapshot(normal):
    _, tools, state = normal
    tools["write_file"].invoke({"path": "edit.rs", "content": "good_fix\n"})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_fix.rs", "content": "test"})
    state.pass_test(state.start_test(), test_target="normal_regression_fix")
    assert state.verified()
    assert state._best_verified_snapshot is not None

    # Agent makes a broken speculative edit afterwards without re-verifying
    tools["write_file"].invoke({"path": "edit.rs", "content": "broken_cleanup\n"})
    assert not state.verified()

    # auto_submit_verified_patch must fall back to the verified snapshot!
    patch = state.auto_submit_verified_patch()
    assert "+good_fix" in patch
    assert "broken_cleanup" not in patch


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_post_strip_check_catches_downstream_workspace_breakage(tmp_path):
    repo = tmp_path / "ws_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[workspace]\nmembers = ["core_lib", "app_bin"]\nresolver = "2"\n', encoding="utf-8")
    (repo / "core_lib" / "src").mkdir(parents=True)
    (repo / "core_lib" / "Cargo.toml").write_text('[package]\nname = "core_lib"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (repo / "core_lib" / "src" / "lib.rs").write_text("pub fn helper(x: i32) -> i32 { x + 1 }\n", encoding="utf-8")

    (repo / "app_bin" / "src").mkdir(parents=True)
    (repo / "app_bin" / "Cargo.toml").write_text('[package]\nname = "app_bin"\nversion = "0.1.0"\nedition = "2021"\n[dependencies]\ncore_lib = { path = "../core_lib" }\n', encoding="utf-8")
    (repo / "app_bin" / "src" / "lib.rs").write_text("pub fn run() -> i32 { core_lib::helper(5) }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_ws", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools, state = m.make_tools(repo)
    tool_map = {t.name: t for t in tools}
    try:
        # Break core_lib::helper signature without updating app_bin
        tool_map["edit_file"].invoke({
            "path": "core_lib/src/lib.rs",
            "old_text": "pub fn helper(x: i32) -> i32 { x + 1 }",
            "new_text": "pub fn helper(x: &i32) -> i32 { *x + 1 }",
        })
        tool_map["write_regression_test"].invoke({
            "path": "core_lib/tests/normal_regression_core.rs",
            "content": "#[test] fn ok() { assert_eq!(core_lib::helper(&5), 6); }\n",
        })
        res = json.loads(tool_map["cargo_test"].invoke({
            "manifest_path": "core_lib/Cargo.toml",
            "test_target": "normal_regression_core",
        }))
        assert res["status"] == "passed"

        submit_out = tool_map["generate_patch"].invoke({"paths": ["core_lib/src/lib.rs"]})
        assert submit_out.startswith("diff --git")
        assert state.patch == submit_out
        assert "downstream workspace check failed" in state._last_submit_error
        assert "app_bin" in state._last_submit_error
    finally:
        state.close()


def test_read_file_truncates_large_files(normal):
    _, tools, state = normal
    large_content = "\n".join(f"pub fn line_{i}() {{}}" for i in range(1, 301)) + "\n"
    (state.root / "src").mkdir(parents=True, exist_ok=True)
    (state.root / "src" / "large.rs").write_text(large_content, encoding="utf-8")
    out = tools["read_file"].invoke({"path": "src/large.rs"})
    assert "[File: src/large.rs (300 lines total)]" in out
    assert "File has 300 lines" in out
    assert "1: pub fn line_1()" in out
    assert "120: pub fn line_120()" in out
    assert "pub fn line_200()" not in out
    assert "300: pub fn line_300()" in out
    assert "read_lines" in out


def test_write_and_edit_file_prohibit_test_and_bench_files(normal):
    _, tools, state = normal
    (state.root / "tests").mkdir(parents=True, exist_ok=True)
    (state.root / "tests" / "existing.rs").write_text("#[test] fn t() {}\n", encoding="utf-8")
    (state.root / "crates" / "sub" / "tests").mkdir(parents=True, exist_ok=True)
    (state.root / "crates" / "sub" / "tests" / "nested.rs").write_text("#[test] fn n() {}\n", encoding="utf-8")
    err1 = tools["write_file"].invoke({"path": "tests/existing.rs", "content": "changed\n"})
    assert "prohibited" in err1
    err2 = tools["edit_file"].invoke({"path": "tests/existing.rs", "old_text": "fn t()", "new_text": "fn x()"})
    assert "prohibited" in err2
    err3 = tools["write_file"].invoke({"path": "benches/bench.rs", "content": "fn b() {}\n"})
    assert "prohibited" in err3
    err4 = tools["write_file"].invoke({"path": "crates/sub/tests/nested.rs", "content": "changed\n"})
    assert "prohibited" in err4


def test_unreproduced_patch_submission_at_high_and_low_budget(tmp_path):
    repo = tmp_path / "soft_repro_repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "soft_repro"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (repo / "src" / "lib.rs").write_text("pub fn val() -> i32 { 1 }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_soft", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools_list, state = m.make_tools(repo, require_reproduction=True)
    tools = {t.name: t for t in tools_list}
    try:
        class FakeClient:
            def __init__(self, rem):
                self.max_calls = 150
                self.remaining_calls = rem
        tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "1", "new_text": "2"})
        tools["write_regression_test"].invoke({
            "path": "tests/normal_regression_check.rs",
            "content": "#[test] fn always_passes() { assert!(soft_repro::val() >= 1); }\n",
        })
        res = json.loads(tools["cargo_test"].invoke({"test_target": "normal_regression_check"}))
        assert res["status"] == "passed"

        state.set_budget(FakeClient(50))
        out_high = tools["generate_patch"].invoke({"paths": ["src/lib.rs"]})
        assert out_high.startswith("diff --git")
        assert state.patch == out_high
        assert "Paired reproduction check failed" in state._last_submit_error
        assert not state.reproduced_cleanly()

        state.set_budget(FakeClient(15))
        out_low = tools["generate_patch"].invoke({"paths": ["src/lib.rs"]})
        assert out_low.startswith("diff --git")
        assert state.patch == out_low
        assert not state.reproduced_cleanly()
    finally:
        state.close()


def test_auto_submit_requires_clean_red_reproduction(tmp_path):
    repo = tmp_path / "auto_soft_repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "auto_soft"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (repo / "src" / "lib.rs").write_text("pub fn val() -> i32 { 1 }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_auto_soft", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools_list, state = m.make_tools(repo, require_reproduction=True)
    tools = {t.name: t for t in tools_list}
    try:
        class FakeClient:
            max_calls = 150
            remaining_calls = 50
        state.set_budget(FakeClient())
        tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "1", "new_text": "2"})
        tools["write_regression_test"].invoke({
            "path": "tests/normal_regression_check.rs",
            "content": "#[test] fn passes() { assert!(auto_soft::val() >= 1); }\n",
        })
        assert json.loads(tools["cargo_test"].invoke({"test_target": "normal_regression_check"}))["status"] == "passed"

        # Without fallback (premature idle): does not bypass strict red check
        assert state.auto_submit_verified_patch(allow_snapshot_fallback=False) == ""
        # A passing temporary test on the original code never becomes a soft Red proof.
        assert state.auto_submit_verified_patch(allow_snapshot_fallback=True) == ""
    finally:
        state.close()


def test_edit_file_emits_impact_notice_for_renamed_method(tmp_path):
    repo = tmp_path / "rename_repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub trait Buf { fn bytes(&self) -> &[u8]; }\n", encoding="utf-8")
    (repo / "src" / "impls.rs").write_text("use crate::Buf;\nfn f<T: Buf>(b: &T) -> &[u8] { b.bytes() }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_rename", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools_list, state = m.make_tools(repo)
    tools = {t.name: t for t in tools_list}
    try:
        res = tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "fn bytes(&self) -> &[u8];",
            "new_text": "fn chunk(&self) -> &[u8];",
        })
        assert "Impact notice: signature modified for `bytes`" in res
        assert "src/impls.rs:2" in res
    finally:
        state.close()


def test_init_symlinks_cargo_directory_without_recursive_copy(tmp_path):
    repo = tmp_path / "cargo_repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn foo() {}\n", encoding="utf-8")
    (repo / ".cargo").mkdir()
    (repo / ".cargo" / "config").write_text("config_data\n", encoding="utf-8")
    (repo / ".cargo" / "vendor_dir").mkdir()
    (repo / ".cargo" / "vendor_dir" / "pkg").write_bytes(b"vendor-artifact")

    spec = importlib.util.spec_from_file_location("tools_cargo_link", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools_list, state = m.make_tools(repo)
    try:
        dest_cargo = state.root / ".cargo"
        assert dest_cargo.is_dir()
        assert (dest_cargo / "config").read_text(encoding="utf-8") == "config_data\n"
        # vendor_dir was symlinked if OS permitted, or present
        assert (dest_cargo / "vendor_dir" / "pkg").exists()
    finally:
        state.close()


def test_rust_syntax_guardrail_rejects_unbalanced_braces_and_e0753(normal):
    repo, tools, state = normal
    # Unbalanced closing brace
    bad_brace = "pub fn broken() {\n    let x = 1;\n"
    res = tools["write_file"].invoke({"path": "src/broken.rs", "content": bad_brace})
    assert "Syntax guardrail error" in res
    assert "unclosed opening delimiter '{'" in res

    # Extra closing brace
    extra_close = "pub fn broken() {\n    let x = 1;\n}}\n"
    res2 = tools["write_file"].invoke({"path": "src/broken2.rs", "content": extra_close})
    assert "Syntax guardrail error" in res2
    assert "unexpected closing delimiter '}'" in res2

    # E0753: inner doc comment after code item
    e0753_code = "use std::collections::HashMap;\n//! Inner doc after use\npub fn ok() {}\n"
    res3 = tools["write_file"].invoke({"path": "src/e0753.rs", "content": e0753_code})
    assert "Syntax guardrail error" in res3
    assert "E0753" in res3


def test_edit_file_provides_region_hint_on_missing_old_text(normal):
    repo, tools, state = normal
    tools["write_file"].invoke({"path": "src/hint.rs", "content": "fn one() {\n    let a = 1;\n    let b = 2;\n}\n"})
    out = tools["edit_file"].invoke({"path": "src/hint.rs", "old_text": "let missing = 0;", "new_text": "let c = 3;", "start_line": 2, "end_line": 3})
    assert "old_text was not found" in out
    assert "Region hint" in out
    assert "let a = 1;" in out


def test_generate_patch_submit_flag_one_step(normal):
    repo, tools, state = normal
    tools["write_file"].invoke({"path": "src/lib.rs", "content": "pub fn val() -> i32 { 42 }\n"})
    tools["write_regression_test"].invoke({"path": "tests/test_regression_v.rs", "content": "test"})
    state.pass_test(state.start_test(), test_target="test_regression_v")
    state.pass_test(state.start_test())

    # submit=True with paths=None
    patch = tools["generate_patch"].invoke({"submit": True})
    assert state.patch == patch
    assert "+pub fn val() -> i32 { 42 }" in patch


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is not installed")
def test_in_module_ephemeral_regression_test_lifecycle(tmp_path):
    repo = tmp_path / "in_mod_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "in_mod_test"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("fn secret_calc() -> i32 { 10 }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_in_mod", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools, state = m.make_tools(repo, require_reproduction=True)
    tool_map = {t.name: t for t in tools}

    # 1. Mount in-module regression test testing private fn
    res = tool_map["write_regression_test"].invoke({
        "path": "test_regression_secret",
        "target_src": "src/lib.rs",
        "content": "#[test] fn test_secret() { assert_eq!(secret_calc(), 20); }\n",
        "run_now": True,
    })
    assert "Mounted temporary in-module regression test" in res
    assert "[Auto-run result for __test_regression_secret]" in res

    # 2. Check that working copy contains the test block but changed_paths() does not count it as prod change
    cur_text = (state.root / "src" / "lib.rs").read_text(encoding="utf-8")
    assert "// __TEST_REGRESSION_START: __test_regression_secret__" in cur_text
    assert "mod __test_regression_secret" in cur_text
    assert state.changed_paths() == []

    # 3. Agent fixes production code: 10 -> 20
    tool_map["edit_file"].invoke({"path": "src/lib.rs", "old_text": "10", "new_text": "20"})
    assert "src/lib.rs" in [p.as_posix() for p in state.changed_paths()]

    # 4. Run cargo test --lib to pass the test
    pass_res = json.loads(tool_map["cargo_test"].invoke({"lib": True, "test_filter": "__test_regression_secret"}))
    assert pass_res["status"] == "passed"
    assert state.verified()

    # 5. Generate patch: verify test block is completely stripped from submitted patch!
    patch = tool_map["generate_patch"].invoke({"submit": True})
    assert "secret_calc() -> i32 { 20 }" in patch
    assert "__TEST_REGRESSION_START" not in patch
    assert "mod __test_regression_secret" not in patch

    # 6. Revert file: verify file is restored cleanly
    tool_map["revert_file"].invoke({"path": "src/lib.rs"})
    reverted_text = (state.root / "src" / "lib.rs").read_text(encoding="utf-8")
    assert "__TEST_REGRESSION_START" not in reverted_text
    assert "fn secret_calc() -> i32 { 10 }" in reverted_text


def test_package_manifest_scan_handles_arrays_before_name(tmp_path):
    repo = tmp_path / "pkg_array_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('''[package]
authors = ["alice <alice@example.com>"]
categories = ["science"]
name = "array_pkg"
version = "0.1.0"
edition = "2021"
''', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn test() {}\\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_array_pkg", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools, state = m.make_tools(repo)
    found = state._get_package_manifest("array_pkg")
    assert found is not None
    assert found.name == "Cargo.toml"


def test_lockfile_modification_is_prohibited(tmp_path):
    repo = tmp_path / "lock_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\\nname = "lock_test"\\nversion = "0.1.0"\\nedition = "2021"\\n', encoding="utf-8")
    (repo / "Cargo.lock").write_text("# lock content\\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn f() {}\\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_lock", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools_list, state = m.make_tools(repo)
    tools = {t.name: t for t in tools_list}
    res = tools["write_file"].invoke({"path": "Cargo.lock", "content": "corrupted lock"})
    assert "Modifying Cargo.lock is prohibited" in res


def test_diff_hygiene_and_stuck_reset(tmp_path):
    repo = tmp_path / "hygiene_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "hygiene"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn compute(x: i32) -> i32 { x }\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("tools_hygiene", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tools_list, state = m.make_tools(repo)
    tools = {t.name: t for t in tools_list}
    try:
        tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "pub fn compute(x: i32) -> i32 { x }",
            "new_text": "pub fn compute(x: i32) -> i32 { dbg!(x) }",
        })
        rep = state.diff_hygiene_report()
        assert rep["added_dbg_lines"]
        state.record_check_result(101, [{"level": "error", "code": "E0308", "message": "bad", "spans": [{"file_name": "src/lib.rs"}]}])
        state.invalidate_write()
        state.record_check_result(101, [{"level": "error", "code": "E0308", "message": "bad", "spans": [{"file_name": "src/lib.rs"}]}])
        assert state.stuck_status()["compile_repeats"] == 2
        state.reset_stuck_counters()
        assert state.stuck_status()["compile_repeats"] == 0
    finally:
        state.close()


def test_fail_test_preserves_clean_red_on_dirty_checkout_and_bin_target_in_module(tmp_path):
    repo = tmp_path / "bin_red_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "mybin"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "main.rs").write_text("fn helper() -> i32 { 0 }\nfn main() {}\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_bin_red", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tools_list, state = m.make_tools(repo, require_reproduction=True)
    tools = {t.name: t for t in tools_list}
    try:
        tools["write_regression_test"].invoke({
            "path": "src/main.rs",
            "target_src": "src/main.rs",
            "content": "#[test] fn check_helper() { assert_eq!(helper(), 1); }",
        })
        # Clean Red fails at runtime via bin_target
        state.fail_test(
            state.start_test(),
            bin_target="mybin",
            test_filter="__test_regression_main",
            executed=1,
            failures=["__test_regression_main::check_helper"],
            returncode=101,
        )
        assert state.reproduced_cleanly()

        # Fixer edits src/main.rs; a transient compile/Cargo error (executed=0) must NOT erase clean Red
        tools["edit_file"].invoke({
            "path": "src/main.rs",
            "old_text": "fn helper() -> i32 { 0 }",
            "new_text": "fn helper() -> i32 { 1 }",
        })
        state.fail_test(
            state.start_test(),
            bin_target="mybin",
            test_filter="check_helper",
            executed=0,
            failures=[],
            returncode=101,
        )
        assert state.reproduced_cleanly()

        # Passing via bin_target + inner test_filter marks the in-module regression test verified!
        state.pass_test(
            state.start_test(),
            bin_target="mybin",
            test_filter="check_helper",
            passed_tests=["__test_regression_main::check_helper"],
        )
        assert state.verified()

        # Callable _submit_hint_override uses truthful state
        state._submit_hint_override = lambda st, rem: "custom pending" if not st.verified() else "custom ready"
        state.set_budget(SimpleNamespace(max_calls=20, remaining_calls=5), warning_threshold=6)
        assert "custom ready" in state.budget_notice()
    finally:
        state.close()



def test_restore_snapshot_preserves_existing_test_failures(tmp_path):
    repo = tmp_path / "snap_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "snap"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn f() -> i32 { 0 }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_snap", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tools_list, state = m.make_tools(repo)
    tools = {t.name: t for t in tools_list}
    try:
        tools["edit_file"].invoke({"path": "src/lib.rs", "old_text": "0", "new_text": "1"})
        state.fail_test(state.start_test(), test_target="existing_t", executed=1, failures=["old_case"], returncode=101)
        assert state._existing_test_failures
        state.save_checkpoint("with_fail")
        state._existing_test_failures = []
        state.restore_checkpoint("with_fail")
        assert len(state._existing_test_failures) == 1
        assert state._existing_test_failures[0]["failures"] == ["old_case"]
    finally:
        state.close()


def test_failure_evidence_survives_checkpoint_and_unrelated_success(normal, tmp_path):
    _, _, state = normal
    detail = {"test": "check_message", "kind": "panic_message_mismatch",
              "actual_panic_message": "actual visible text", "expected_panic_substring": "expected visible text"}
    scope = {"package": "", "filter": "", "lib": False, "bin_target": "", "tests": False, "doc": False}
    state.record_test_result("Cargo.toml", "visible", 101, ["check_message"], 1, [],
                             failure_details=[detail], selection=scope)
    state.dump_checkpoint_dir(tmp_path / "checkpoint")
    state.record_test_result("Cargo.toml", "other", 0, [], 1, [], selection=scope)
    assert state.stuck_status()["last_test_failure"]["failure_details"] == [detail]
    state.record_test_result("Cargo.toml", "visible", 0, [], 1, [], selection=scope)
    assert not state.stuck_status()["last_test_failure"]
    state.load_checkpoint_dir(tmp_path / "checkpoint")
    assert state.stuck_status()["last_test_failure"]["failure_details"] == [detail]


def test_write_regression_test_rejects_virtual_workspace_and_rolls_back_missing_target(tmp_path, monkeypatch):
    repo = tmp_path / "ws_reg_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/sub"]\n', encoding="utf-8")
    (repo / "crates" / "sub" / "src").mkdir(parents=True)
    (repo / "crates" / "sub" / "Cargo.toml").write_text('[package]\nname = "sub_pkg"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "crates" / "sub" / "src" / "lib.rs").write_text("pub fn ok() -> i32 { 1 }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_ws_reg", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tools_list, state = m.make_tools(repo, require_reproduction=True)
    tools = {t.name: t for t in tools_list}
    try:
        err = tools["write_regression_test"].invoke({
            "path": "tests/test_regression_root.rs",
            "content": "#[test] fn t() {}\n",
            "run_now": False,
        })
        assert "virtual Cargo [workspace]" in err
        assert "crates/sub/tests/test_regression_root.rs" in err
        assert not (state.root / "tests" / "test_regression_root.rs").exists()
        assert state.get_regression_targets() == {}

        # If a standalone test inside a crate fails with "no test target named", run_now=True rolls it back
        import os
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=101,
                stdout="",
                stderr="error: no test target named `test_regression_ghost`",
            ),
        )
        out = tools["write_regression_test"].invoke({
            "path": "crates/sub/tests/test_regression_ghost.rs",
            "content": "#[test] fn t() {}\n",
            "run_now": True,
        })
        assert "was not kept" in out
        assert not (state.root / "crates" / "sub" / "tests" / "test_regression_ghost.rs").exists()
        assert state.get_regression_targets() == {}
    finally:
        state.close()


def test_non_rs_template_edit_touches_crate_entry_mtime_and_hygiene_flags_scratch(tmp_path):
    import os
    repo = tmp_path / "tpl_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname = "tpl_pkg"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text('pub fn tpl() -> &\'static str { include_str!("../templates/msg.txt") }\n', encoding="utf-8")
    (repo / "templates").mkdir()
    (repo / "templates" / "msg.txt").write_text("hello\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_tpl", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tools_list, state = m.make_tools(repo)
    tools = {t.name: t for t in tools_list}
    try:
        lib_rs = state.root / "src" / "lib.rs"
        os.utime(lib_rs, (1_000_000, 1_000_000))
        tools["edit_file"].invoke({"path": "templates/msg.txt", "old_text": "hello", "new_text": "world"})
        assert lib_rs.stat().st_mtime > 1_000_000
        assert [p.as_posix() for p in state.changed_paths()] == ["templates/msg.txt"]

        tools["write_file"].invoke({"path": "_splice_snaps.py", "content": "print(1)\n"})
        rep = state.diff_hygiene_report()
        assert "_splice_snaps.py" in rep["scratch_files"]
    finally:
        state.close()



def test_bin_only_in_module_verification_aborted_red_and_snapshot_red_preservation(tmp_path):
    repo = tmp_path / "bin_only_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text(
        '[package]\nauthors = ["a"]\nname = "mycli"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (repo / "src").mkdir()
    (repo / "src" / "main.rs").write_text("fn run() -> i32 { 0 }\nfn main() {}\n", encoding="utf-8")
    (repo / "src" / "other.rs").write_text("pub fn helper() -> i32 { 1 }\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tools_bin_only", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tools_list, state = m.make_tools(repo, require_reproduction=True, post_strip_check=False)
    tools = {t.name: t for t in tools_list}
    try:
        tools["write_regression_test"].invoke({
            "path": "test_regression_cli",
            "target_src": "src/other.rs",
            "content": "#[test] fn t() { assert_eq!(helper(), 2); }",
        })
        missing_msg = state.missing_verification_steps()
        assert "bin_target='mycli'" in missing_msg
        assert "lib=True" not in missing_msg

        # Moving the same regression module name to src/main.rs strips the stale block from src/other.rs
        tools["write_regression_test"].invoke({
            "path": "test_regression_cli",
            "target_src": "src/main.rs",
            "content": "#[test] fn t() { assert_eq!(run(), 1); }",
        })
        assert "__TEST_REGRESSION_START" not in (state.root / "src" / "other.rs").read_text(encoding="utf-8")
        assert len(state.get_regression_targets()) == 1

        # Guard against copying compression placeholders or editing ephemeral test blocks directly
        bad_wrt = tools["write_regression_test"].invoke({
            "path": "tests/test_regression_bad.rs",
            "content": "fn x() {\n... [earlier write omitted (10 lines)] ...\n}",
        })
        assert "history-compression omission placeholder" in bad_wrt
        bad_edit = tools["edit_file"].invoke({
            "path": "src/main.rs",
            "old_text": "assert_eq!(run(), 1);",
            "new_text": "assert_eq!(run(), 2);",
        })
        assert "Do not modify temporary '// __TEST_REGRESSION_START' blocks via edit_file" in bad_edit

        # Save pre_fix, edit source, save fix, restore pre_fix, reproduce Red via tests_started=1 & exit_code=1, restore fix
        state.save_checkpoint("pre_fix")
        tools["edit_file"].invoke({"path": "src/main.rs", "old_text": "fn run() -> i32 { 0 }", "new_text": "fn run() -> i32 { 1 }"})
        state.save_checkpoint("fix")
        state.restore_checkpoint("pre_fix")
        state.fail_test(
            state.start_test(),
            bin_target="mycli",
            test_filter="t",
            executed=0,
            tests_started=1,
            failures=[],
            returncode=1,
        )
        assert state.reproduced_cleanly()
        state.restore_checkpoint("fix")
        assert state.reproduced_cleanly()
    finally:
        state.close()


def test_smoke_ignores_snapshot_template_text_and_keeps_snapshot_diff(tmp_path, monkeypatch):
    repo = tmp_path / "insta_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    (repo / "tests" / "integration.rs").write_text("#[test] fn snap() {}\n", encoding="utf-8")
    (repo / "Cargo.toml").write_text(
        '[package]\nname = "insta_repo"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location("tools_insta_smoke", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools_list, state = module.make_tools(repo, post_strip_check=False)
    tools = {item.name: item for item in tools_list}
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/cargo")
    try:
        tools["write_file"].invoke({"path": "src/main.rs", "content": "fn main() { let _patched = true; }\n"})
        insta_stdout = (
            "running 1 test\n"
            "test err_snap ... FAILED\n\n"
            "failures:\n\n"
            "---- err_snap stdout ----\n"
            "Snapshot Summary\n"
            "Snapshot: err_snap\n"
            "-old snapshot\n"
            "+new results\n"
            "  10 │+  err \"failed to download\"\n"
            "  11 │+  × Expected an array, an object, or a literal\n"
            "To update snapshots run `cargo insta review`\n"
            "thread 'err_snap' panicked at insta/src/runtime.rs:563:9\n\n"
            "failures:\n    err_snap\n\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n"
            + "\n".join(f"stored new snapshot /tmp/repo/tests/s_{i}.snap.new" for i in range(30))
        )
        calls = iter([
            SimpleNamespace(returncode=101, stdout=insta_stdout, stderr=""),
            SimpleNamespace(returncode=0, stdout="test result: ok. 1 passed; 0 failed\n", stderr=""),
        ])
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(calls))
        results = state.run_existing_test_smoke([Path("src/main.rs")], timeout=3)
        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert results[0]["classification"] == "code"
        assert "Expected an array, an object, or a literal" in results[0]["details"]
        assert "stored new snapshot /tmp/repo/tests/s_29.snap.new" not in results[0]["details"]

        assert module._smoke_environment_failure(
            "thread 'aka_musl' panicked: Connection timed out\n",
            "fatal: unable to access 'https://github.com/x/y': Failed to connect to github.com",
        )
    finally:
        state.close()


def test_compile_reproduction_does_not_hard_lock_and_bin_only_rejects_crate_import(normal):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text(
        '[package]\nname = "bin-only-app"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (state.root / "src").mkdir(exist_ok=True)
    (state.root / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")

    bad_bin = tools["write_regression_test"].invoke({
        "path": "tests/test_regression_bin.rs",
        "content": "use bin_only_app::opts;\n#[test] fn t() {}\n",
    })
    assert "Error" in bad_bin and "binary target" in bad_bin

    (state.root / "src" / "lib.rs").write_text("pub fn exists() {}\n", encoding="utf-8")
    write_res = tools["write_regression_test"].invoke({
        "path": "tests/test_regression_api.rs",
        "content": "use demo::missing_fn;\n#[test] fn t() { missing_fn(); }\n",
    })
    assert write_res.startswith("Updated")
    state.fail_test(
        state.start_test(),
        manifest_path="Cargo.toml",
        test_target="test_regression_api",
        executed=0,
        returncode=101,
        compile_errors=[{"code": "E0432"}],
    )
    assert state.reproduced()
    assert not state._locked_regressions
    rewrite = tools["write_regression_test"].invoke({
        "path": "tests/test_regression_api.rs",
        "content": "#[test] fn t() { demo::exists(); }\n",
    })
    assert rewrite.startswith("Updated")


def test_edit_file_warns_inline_test_isolation_top_level_use_and_doctest_placement(tmp_path):
    repo = tmp_path / "hint_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text('[package]\nname = "hint_repo"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
    (repo / "src" / "lib.rs").write_text(
        "use rhai::{Dynamic, Module};\n\n"
        "/// Example:\n"
        "/// ```\n"
        "/// assert_eq!(1, 1);\n"
        "/// ```\n"
        "pub fn entry(val: Dynamic) -> Dynamic { val }\n\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    use super::*;\n"
        "    #[test]\n"
        "    fn t() { let _ = entry(Dynamic::UNIT); }\n"
        "}\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("tools_hint_test", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools_list, state = module.make_tools(repo, post_strip_check=False)
    tools = {item.name: item for item in tools_list}
    try:
        out1 = tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "/// Example:",
            "new_text": "fn helper(_a: rhai::Array) {}\n\n/// Example:",
        })
        assert "Doctest Line-Shift Notice" in out1 and "below existing doctests" in out1
        assert "Rust Test-Scope Hint" in out1 and "Array" in out1
        assert "Inline-Test Isolation Warning" not in out1

        out2 = tools["edit_file"].invoke({
            "path": "src/lib.rs",
            "old_text": "    fn t() { let _ = entry(Dynamic::UNIT); }",
            "new_text": "    fn t() { let _ = entry(Dynamic::UNIT); /* touched */ }",
        })
        assert "Inline-Test Isolation Warning" in out2
    finally:
        state.close()


def test_guardrail_atomicity_unclosed_literals_and_recovery_context(normal):
    _, tools, state = normal
    tools["write_file"].invoke({"path": "src/lib.rs", "content": "pub fn ok() -> i32 { 1 }\n"})
    tools["write_regression_test"].invoke({"path": "tests/test_regression_valid.rs", "content": "#[test] fn t() {}\n"})
    state.pass_test(state.start_test(), test_target="test_regression_valid")
    assert state.verified()

    # Syntax rejection in write_regression_test must NOT pollute regression_tests or invalidate prior verification
    bad_reg = tools["write_regression_test"].invoke({
        "path": "tests/test_regression_broken.rs",
        "content": '#[test] fn broken() { let s = "unclosed;\n}\n',
    })
    assert "Syntax guardrail error" in bad_reg
    assert "unclosed string literal" in bad_reg
    assert all("broken" not in p.as_posix() for p in state.regression_tests)
    assert state.verified()

    # edit_file rejection shows Proposed + Original code context and tracks consecutive edit failures
    bad_edit = tools["edit_file"].invoke({
        "path": "src/lib.rs",
        "old_text": "pub fn ok() -> i32 { 1 }",
        "new_text": "pub fn ok() -> i32 { /* unclosed comment\n",
    })
    assert "Syntax guardrail error" in bad_edit
    assert "Proposed code around line" in bad_edit
    assert "Original code around line" in bad_edit
    assert "Recovery notice" not in bad_edit

    bad_edit_2 = tools["edit_file"].invoke({
        "path": "src/lib.rs",
        "old_text": "missing_symbol",
        "new_text": "x",
    })
    assert "Recovery notice: 2 consecutive edit failures" in bad_edit_2


def test_e0753_guardrail_respects_module_scopes_and_block_comments_and_duplicate_edit_reports_lines(normal):
    _, tools, _ = normal
    # Valid inner doc inside nested `pub mod error` and `#![deny(missing_docs)]` after block comment without `*`
    valid_nested = (
        "/*\nLicense header without leading stars\n*/\n"
        "#![deny(missing_docs)]\n"
        "pub struct Receiver;\n"
        "pub mod error {\n"
        "    //! Broadcast error types\n"
        "    pub struct RecvError;\n"
        "}\n"
    )
    ok_res = tools["write_file"].invoke({"path": "src/nested_mod.rs", "content": valid_nested})
    assert "Updated src/nested_mod.rs" in ok_res

    # In-module regression test mounting must also succeed on files with nested `//!` or header `#![...]`
    reg_res = tools["write_regression_test"].invoke({
        "path": "test_nested",
        "target_src": "src/nested_mod.rs",
        "content": "#[test] fn ok() {}",
    })
    assert "Mounted temporary in-module regression test" in reg_res

    # Invalid top-level `//!` after code declaration still rejects as E0753
    bad_nested = "pub struct S;\npub mod m {\n    //! ok\n}\n//! bad top-level doc\n"
    bad_res = tools["write_file"].invoke({"path": "src/bad_e0753.rs", "content": bad_nested})
    assert "E0753" in bad_res

    # Duplicate matches report line numbers
    tools["write_file"].invoke({"path": "src/dup.rs", "content": "fn a() {\n    let x = 1;\n}\nfn b() {\n    let x = 1;\n}\n"})
    dup_err = tools["edit_file"].invoke({"path": "src/dup.rs", "old_text": "let x = 1;", "new_text": "let x = 2;"})
    assert "more than once" in dup_err and "2" in dup_err and "5" in dup_err


def test_replace_broken_lock_reason_allows_compile_broken_or_setup_panic_but_not_assertions(normal, monkeypatch):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text('[package]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8")
    (state.root / "src").mkdir(exist_ok=True)
    (state.root / "src" / "lib.rs").write_text("pub fn ok() -> i32 { 1 }\n", encoding="utf-8")

    tools["write_regression_test"].invoke({
        "path": "tests/test_regression_assert.rs",
        "content": "#[test] fn assert_case() { assert_eq!(1, 2); }\n",
    })
    tok = state.start_test()
    state.fail_test(
        tok,
        manifest_path="Cargo.toml",
        test_target="test_regression_assert",
        executed=1,
        failures=["assert_case"],
        returncode=101,
    )
    assert state._locked_regressions
    blocked = tools["write_regression_test"].invoke({
        "path": "tests/test_regression_assert.rs",
        "content": "#[test] fn assert_case() { assert_eq!(1, 1); }\n",
        "replace_broken_lock_reason": "trying to weaken a failing assertion test",
    })
    assert "cannot self-unlock" in blocked

    tools["write_regression_test"].invoke({
        "path": "tests/test_regression_compile.rs",
        "content": "#[test] fn compile_case() { assert_eq!(1, 2); }\n",
    })
    tok2 = state.start_test()
    state.fail_test(
        tok2,
        manifest_path="Cargo.toml",
        test_target="test_regression_compile",
        executed=1,
        failures=["compile_case"],
        failure_details=[{"name": "compile_case", "kind": "unexpected_panic", "output": "No such file or directory"}],
        returncode=101,
    )
    replaced = tools["write_regression_test"].invoke({
        "path": "tests/test_regression_compile.rs",
        "content": "#[test] fn compile_case() { assert_eq!(1, 1); }\n",
        "replace_broken_lock_reason": "locked oracle panicked in setup with missing file path",
    })
    assert "Error" not in replaced
    assert "1, 1" in (state.root / "tests" / "test_regression_compile.rs").read_text(encoding="utf-8")

    # Smoke target matching splits underscored/hyphenated stems cleanly
    tests_dir = state.root / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "parser_v2.rs").write_text("#[test] fn t() {}\n", encoding="utf-8")
    seen_targets = []
    import subprocess as sp_mod
    orig_sp_run = sp_mod.run
    monkeypatch.setattr(
        sp_mod, "run",
        lambda cmd, **kw: (
            (seen_targets.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
            if cmd and cmd[0] == "cargo"
            else orig_sp_run(cmd, **kw)
        ),
    )
    state.run_existing_test_smoke([Path("src/parser_extra.rs")])
    assert any("parser_v2" in " ".join(c) for c in seen_targets)

