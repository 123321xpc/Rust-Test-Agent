"""Behavioral tests for the generic normal baseline tools."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def normal(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "edit.rs").write_bytes(b"before\n")
    (repo / "delete.rs").write_bytes(b"delete me\n")
    spec = importlib.util.spec_from_file_location("normal_tools_test", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools, state = module.make_tools(repo)
    yield repo, {tool.name: tool for tool in tools}, state
    state.close()


def test_edits_produce_applicable_complete_patch_without_mutating_checkout(normal, tmp_path):
    repo, tools, state = normal
    assert set(tools) == {"read_file", "search_code", "write_file", "edit_file", "run_command", "generate_patch"}
    write = tools["write_file"]
    write.invoke({"path": "edit.rs", "content": "intermediate\n"})
    write.invoke({"path": "edit.rs", "content": "after\n"})
    write.invoke({"path": "new/file.rs", "content": "no newline"})
    write.invoke({"path": "delete.rs", "content": None})
    assert tools["read_file"].invoke({"path": "edit.rs"}) == "1: after"

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
    patch = tools["generate_patch"].invoke({"paths": ["edit.rs", "new/file.rs", "delete.rs"]})
    assert "+again\n" in patch


def test_read_file_lists_dirs_reads_ranges_and_clamps_over_500_lines(normal):
    _, tools, state = normal
    read = tools["read_file"]
    assert "edit.rs" in read.invoke({"path": "."})
    (state.root / "empty.txt").write_text("", encoding="utf-8")
    assert read.invoke({"path": "empty.txt"}) == "(empty file)"

    (state.root / "src").mkdir(parents=True, exist_ok=True)
    (state.root / "src" / "long.rs").write_text("\n".join(f"// line {i}" for i in range(1, 650)) + "\n", encoding="utf-8")
    out = read.invoke({"path": "src/long.rs", "start_line": 10, "end_line": 600})
    assert "10: // line 10" in out
    assert "509: // line 509" in out
    assert "510: // line 510" not in out
    assert "clamped to 10-509" in out
    assert "Error" in read.invoke({"path": "src/long.rs", "start_line": 0})
    assert read.invoke({"path": "src/long.rs", "start_line": 700, "end_line": 710}) == "(no lines in range)"


def test_search_code_literal_regex_and_default_all_files(normal):
    _, tools, state = normal
    search = tools["search_code"]
    (state.root / "src").mkdir(parents=True, exist_ok=True)
    (state.root / "src" / "lib.rs").write_text("pub fn first() {}\n// needle\npub fn second() {}\n", encoding="utf-8")
    (state.root / "notes.txt").write_text("needle in text\n", encoding="utf-8")
    (state.root / "target").mkdir(exist_ok=True)
    (state.root / "target" / "ignored.rs").write_text("needle in target\n", encoding="utf-8")

    all_matches = search.invoke({"query": "needle"})
    assert "src/lib.rs:2: // needle" in all_matches
    assert "notes.txt:1: needle in text" in all_matches
    assert "target/" not in all_matches

    rs_only = search.invoke({"query": "needle", "file_glob": "*.rs"})
    assert "src/lib.rs:2:" in rs_only and "notes.txt" not in rs_only

    assert "truncated" in search.invoke({"query": "fn (first|second)", "regex": True, "max_results": 1})
    assert "Error" in search.invoke({"query": "[", "regex": True})


def test_write_and_edit_allow_test_paths_and_enforce_sandbox(normal):
    _, tools, state = normal
    tools["write_file"].invoke({"path": "tests/helper.rs", "content": "#[test] fn t() {}\n"})
    res = tools["edit_file"].invoke({"path": "tests/helper.rs", "old_text": "fn t()", "new_text": "fn t2()"})
    assert "Updated tests/helper.rs" in res
    assert "Error" in tools["read_file"].invoke({"path": "../outside"})
    assert "Error" in tools["write_file"].invoke({"path": "../outside", "content": "bad"})
    assert "Error" in tools["write_file"].invoke({"path": ".git/config", "content": "bad"})
    assert "Error" in tools["write_file"].invoke({"path": "missing", "content": None})
    temp_root = state.root
    state.close()
    assert not temp_root.exists()


def test_generate_patch_allows_subset_selection_and_rejects_unknown(normal, tmp_path):
    repo, tools, state = normal
    write = tools["write_file"]
    write.invoke({"path": "edit.rs", "content": "fixed\n"})
    write.invoke({"path": "scratch.txt", "content": "temp note\n"})

    preview = tools["generate_patch"].invoke({})
    assert "edit.rs" in preview and "scratch.txt" in preview and not state.patch

    patch = tools["generate_patch"].invoke({"paths": ["edit.rs"]})
    assert state.patch == patch and "scratch.txt" not in patch

    copy = tmp_path / "selected"
    shutil.copytree(repo, copy)
    subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-"], cwd=copy, input=patch.encode(), check=True, capture_output=True)
    assert (copy / "edit.rs").read_bytes() == b"fixed\n"
    assert not (copy / "scratch.txt").exists()

    assert "Not a changed file" in tools["generate_patch"].invoke({"paths": ["missing.rs"]})
    assert not state.patch


def test_run_command_executes_tracks_changes_and_handles_timeouts(normal, monkeypatch):
    repo, tools, state = normal
    py = json.dumps(sys.executable)
    out = tools["run_command"].invoke({
        "command": f'{py} -c "from pathlib import Path; Path(\'edit.rs\').write_text(\'cmd_edited\\n\'); print(\'ok\')"'
    })
    assert "Exit code: 0" in out and "ok" in out
    assert Path("edit.rs") in state.changed_paths()
    patch = tools["generate_patch"].invoke({"paths": ["edit.rs"]})
    assert "+cmd_edited" in patch
    assert (repo / "edit.rs").read_bytes() == b"before\n"

    # File changes via run_command invalidate previously submitted patch
    tools["run_command"].invoke({
        "command": f'{py} -c "from pathlib import Path; Path(\'edit.rs\').write_text(\'cmd_again\\n\')"'
    })
    assert not state.patch

    assert "Error running command" in tools["run_command"].invoke({"command": ""})
    assert "Error running command" in tools["run_command"].invoke({"command": "echo hi", "timeout": 0})
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("cmd", 1)))
    assert "timed out" in tools["run_command"].invoke({"command": "sleep 5", "timeout": 1})


def test_budget_notice_injected_when_remaining_calls_low(normal):
    _, tools, state = normal
    client = SimpleNamespace(max_calls=10, remaining_calls=2)
    state.set_budget(client, warning_threshold=3)
    out = tools["write_file"].invoke({"path": "edit.rs", "content": "after\n"})
    assert "Budget notice: 2 API call(s) remaining" in out
    read_out = tools["read_file"].invoke({"path": "edit.rs"})
    assert "Budget notice: 2 API call(s) remaining" in read_out
    submitted = tools["generate_patch"].invoke({"paths": ["edit.rs"]})
    assert "Budget notice" not in submitted


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

    spec = importlib.util.spec_from_file_location("normal_tools_git_test", Path(__file__).with_name("tools.py"))
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

        patch = tool_map["generate_patch"].invoke({"paths": ["edit.rs", "new/file.rs", "delete.rs"]})
        assert state.patch == patch
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
    assert "normal-agent-" not in wt_list


def test_patch_state_uses_deterministic_workspace_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.rs").write_text("pub fn v() -> i32 { 1 }\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("tools_mod", Path(__file__).with_name("tools.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    _, state = m.make_tools(repo, instance_id="tokio-rs__bytes-253")
    try:
        assert "normal-agent-workspaces" in str(state.root)
        assert "tokio-rs__bytes-253" in str(state.root)
        assert state._custom_workspace is True
        assert (state.root / "lib.rs").is_file()
    finally:
        state.close()
    assert not state.root.exists()


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

    _, state = m.make_tools(repo)
    try:
        dest_cargo = state.root / ".cargo"
        assert dest_cargo.is_dir()
        assert (dest_cargo / "config").read_text(encoding="utf-8") == "config_data\n"
        assert (dest_cargo / "vendor_dir" / "pkg").exists()
        assert state.changed_paths() == []
    finally:
        state.close()
