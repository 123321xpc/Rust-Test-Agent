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
                          "cargo_check", "cargo_test", "rust_symbols", "rust_error_explain"}
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


def test_patch_requires_successful_nonempty_test_after_last_write(normal, monkeypatch):
    _, tools, state = normal
    (state.root / "Cargo.toml").write_text('[package]\nname = "sample"\nversion = "0.1.0"\n')
    tools["write_file"].invoke({"path": "edit.rs", "content": "fixed\n"})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_fix.rs", "content": "test"})
    generate = tools["generate_patch"]
    cargo_test = tools["cargo_test"]
    assert "After the last write" in generate.invoke({})
    assert "After the last write" in generate.invoke({"paths": ["edit.rs"]})

    def result(returncode, output):
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode, stdout=output, stderr=""))

    result(1, "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n")
    assert json.loads(cargo_test.invoke({}))["status"] == "failed"
    assert "After the last write" in generate.invoke({})
    result(0, "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")
    assert json.loads(cargo_test.invoke({}))["tests_executed"] == 0
    assert "After the last write" in generate.invoke({})
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("cargo", 1)))
    assert json.loads(cargo_test.invoke({}))["status"] == "timeout"
    assert "After the last write" in generate.invoke({})
    result(0, "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")
    assert json.loads(cargo_test.invoke({}))["tests_executed"] == 1
    assert "After the last write" in generate.invoke({})
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
    assert not state.patch and "After the last write" in generate.invoke({})


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


def test_require_reproduction_gate_enforcement(normal):
    _, tools, state = normal
    state.require_reproduction = True
    tools["write_file"].invoke({"path": "edit.rs", "content": "fixed\n"})
    tools["write_regression_test"].invoke({"path": "tests/normal_regression_bug.rs", "content": "test"})
    token = state.start_test()
    # If test passes directly without prior reproduction:
    state.pass_test(token, test_target="normal_regression_bug")
    state.pass_test(state.start_test())
    assert not state.verified()
    assert "confirm failure (Red phase)" in tools["generate_patch"].invoke({})

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
    assert "Post-strip verification failed" in out
    assert not state.patch


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

    spec = importlib.util.spec_from_file_location("test_tools_git_test", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools, state = module.make_tools(repo, instance_id="tokio-rs__bytes-253")
    tool_map = {t.name: t for t in tools}
    try:
        assert state._is_git is True
        assert state._is_worktree is True
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

    wt_list = subprocess.run(["git", "worktree", "list"], cwd=repo, check=True, capture_output=True, text=True).stdout
    assert "test-agent-" not in wt_list


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

        # generate_patch must trigger paired check against originals and REJECT the fake test
        preview = tool_map["generate_patch"].invoke({})
        assert "Paired reproduction check failed" in preview
        assert not state.patch

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

        # Submission must fail post-strip check because downstream workspace crate app_bin broke!
        submit_out = tool_map["generate_patch"].invoke({"paths": ["core_lib/src/lib.rs"]})
        assert "downstream workspace check failed" in submit_out
        assert "app_bin" in submit_out
        assert not state.patch
    finally:
        state.close()


def test_read_file_truncates_large_files(normal):
    _, tools, state = normal
    large_content = "\n".join(f"pub fn line_{i}() {{}}" for i in range(1, 301)) + "\n"
    (state.root / "src").mkdir(parents=True, exist_ok=True)
    (state.root / "src" / "large.rs").write_text(large_content, encoding="utf-8")
    out = tools["read_file"].invoke({"path": "src/large.rs"})
    assert "File has 300 lines" in out
    assert "pub fn line_1()" in out
    assert "pub fn line_120()" in out
    assert "pub fn line_200()" not in out
    assert "pub fn line_300()" in out
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


def test_soft_reproduction_fallback_on_low_budget_and_auto_submit(tmp_path):
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
        assert "Paired reproduction check failed" in out_high
        assert not state.patch

        state.set_budget(FakeClient(15))
        out_low = tools["generate_patch"].invoke({"paths": ["src/lib.rs"]})
        assert out_low.startswith("diff --git")
        assert state.patch == out_low
    finally:
        state.close()


def test_auto_submit_soft_fallback_respects_allow_snapshot_fallback(tmp_path):
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
        # With fallback (budget limit / repeated idle): submits verified source patch
        patch = state.auto_submit_verified_patch(allow_snapshot_fallback=True)
        assert patch.startswith("diff --git")
        assert "+pub fn val() -> i32 { 2 }" in patch
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
