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
                          "cargo_check", "cargo_test", "rust_trace", "rust_symbols", "inspect_rust_neighborhood",
                          "rust_error_explain", "verifier_probe"}
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


def test_self_unlock_skewed_assertion_requires_non_weakening_and_red_on_base(normal, monkeypatch):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text('[package]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8")
    (state.root / "src").mkdir(exist_ok=True)
    (state.root / "src" / "lib.rs").write_text("pub fn compute() -> i32 { 0 }\n", encoding="utf-8")

    orig_test = "#[test] fn compute_case() { assert_eq!(demo::compute(), 42); }\n"
    tools["write_regression_test"].invoke({"path": "tests/test_regression_skewed.rs", "content": orig_test})
    state.fail_test(
        state.start_test(),
        manifest_path="Cargo.toml",
        test_target="test_regression_skewed",
        executed=1,
        failures=["compute_case"],
        returncode=101,
    )
    assert ("Cargo.toml", "test_regression_skewed") in state._locked_regressions

    # 1. Downgrading assert_eq! to assert!(is_ok()) or dropping assertions must be rejected before running cargo
    weakened = tools["write_regression_test"].invoke({
        "path": "tests/test_regression_skewed.rs",
        "content": "#[test] fn compute_case() { assert!(demo::compute() >= 0); }\n",
        "replace_broken_lock_reason": "Trying to replace assert_eq with weaker boolean check",
    })
    assert "cannot self-unlock" in weakened
    assert ("Cargo.toml", "test_regression_skewed") in state._locked_regressions

    import shutil as sh_mod
    import subprocess as sp_mod
    orig_sp_run = sp_mod.run
    monkeypatch.setattr(sh_mod, "which", lambda cmd: "/usr/bin/cargo" if cmd == "cargo" else None)

    # 2. If rewritten test PASSES on unpatched original code, rollback and keep the original lock intact
    monkeypatch.setattr(
        sp_mod, "run",
        lambda cmd, **kw: (
            SimpleNamespace(returncode=0, stdout="running 1 test\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n", stderr="")
            if cmd and cmd[0] == "cargo"
            else orig_sp_run(cmd, **kw)
        ),
    )
    pass_on_base = tools["write_regression_test"].invoke({
        "path": "tests/test_regression_skewed.rs",
        "content": "#[test] fn compute_case() { assert_eq!(demo::compute(), 0); }\n",
        "replace_broken_lock_reason": "Updated expected value to 0 after inspecting issue",
    })
    assert "cannot self-unlock" in pass_on_base
    assert ("Cargo.toml", "test_regression_skewed") in state._locked_regressions
    assert "42" in (state.root / "tests" / "test_regression_skewed.rs").read_text(encoding="utf-8")

    # 3. If rewritten test preserves assertion strength AND FAILS on unpatched original code, allow self-unlock once
    monkeypatch.setattr(
        sp_mod, "run",
        lambda cmd, **kw: (
            SimpleNamespace(
                returncode=101,
                stdout="running 1 test\ntest compute_case ... FAILED\ntest result: FAILED. 0 passed; 1 failed; 0 ignored;\n",
                stderr="",
            )
            if cmd and cmd[0] == "cargo"
            else orig_sp_run(cmd, **kw)
        ),
    )
    valid_rewrite = tools["write_regression_test"].invoke({
        "path": "tests/test_regression_skewed.rs",
        "content": "#[test] fn compute_case() { assert_eq!(demo::compute(), 7); }\n",
        "replace_broken_lock_reason": "Issue specifies return value 7 instead of 42 for compute()",
    })
    assert "Error" not in valid_rewrite
    assert ("Cargo.toml", "test_regression_skewed") not in state._locked_regressions
    assert ("Cargo.toml", "test_regression_skewed") in state._unlocked_regressions
    assert "7" in (state.root / "tests" / "test_regression_skewed.rs").read_text(encoding="utf-8")
