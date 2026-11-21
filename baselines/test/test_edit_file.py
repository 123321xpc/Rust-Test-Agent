"""Behavioral checks for exact, isolated edits and patch generation."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.rs").write_bytes(b"fn a() { 1; }\r\nfn b() { 1; }\r\n")
    spec = importlib.util.spec_from_file_location("edit_tools_test", Path(__file__).with_name("tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools, state = module.make_tools(repo)
    yield repo, {tool.name: tool for tool in tools}, state
    state.close()


def test_edit_is_unique_and_integrates_with_write_and_patch(workspace, tmp_path):
    repo, tools, state = workspace
    edit, write = tools["edit_file"], tools["write_file"]
    assert "more than once" in edit.invoke({"path": "lib.rs", "old_text": "1;", "new_text": "2;"})
    assert "not found" in edit.invoke({"path": "lib.rs", "old_text": "missing", "new_text": "2;"})
    assert "empty" in edit.invoke({"path": "lib.rs", "old_text": "", "new_text": "2;"})
    assert not state.patch
    out = edit.invoke({"path": "lib.rs", "old_text": "fn a() { 1; }", "new_text": "fn a() { 2; }"})
    assert "Updated" in out and "1: fn a() { 2; }" in out
    write.invoke({"path": "extra.rs", "content": "before\n"})
    edit.invoke({"path": "extra.rs", "old_text": "before", "new_text": "after"})
    assert (state.root / "lib.rs").read_bytes() == b"fn a() { 2; }\r\nfn b() { 1; }\r\n"
    assert (repo / "lib.rs").read_bytes() == b"fn a() { 1; }\r\nfn b() { 1; }\r\n"
    tools["write_regression_test"].invoke({"path": "tests/test_regression_fix.rs", "content": "test"})
    state.pass_test(state.start_test(), test_target="test_regression_fix")
    state.pass_test(state.start_test())
    patch = tools["generate_patch"].invoke({"paths": ["lib.rs", "extra.rs"]})
    copy = tmp_path / "verify"
    shutil.copytree(repo, copy)
    subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-"], cwd=copy, input=patch.encode(), check=True, capture_output=True)
    assert (copy / "lib.rs").read_bytes() == b"fn a() { 2; }\r\nfn b() { 1; }\r\n"
    assert (copy / "extra.rs").read_bytes() == b"after\n"


def test_edit_supports_multi_hunk_edits_and_crlf_matching(workspace):
    _, tools, state = workspace
    edit = tools["edit_file"]
    res = edit.invoke({
        "path": "lib.rs",
        "old_text": "fn a() { 1; }\nfn b() { 1; }",
        "new_text": "fn a() { 10; }\nfn b() { 20; }",
    })
    assert "Updated" in res and "1: fn a() { 10; }" in res and "2: fn b() { 20; }" in res
    assert (state.root / "lib.rs").read_bytes() == b"fn a() { 10; }\r\nfn b() { 20; }\r\n"

    batch_res = edit.invoke({
        "path": "lib.rs",
        "edits": [
            {"old_text": "fn a() { 10; }", "new_text": "fn a() { 100; }"},
            {"old_text": "fn b() { 20; }", "new_text": "fn b() { 200; }"},
        ],
    })
    assert "Updated" in batch_res and "1: fn a() { 100; }" in batch_res and "2: fn b() { 200; }" in batch_res
    assert (state.root / "lib.rs").read_bytes() == b"fn a() { 100; }\r\nfn b() { 200; }\r\n"
    assert "Error" in edit.invoke({"path": "lib.rs", "old_text": "a", "new_text": "b", "edits": [{"old_text": "a", "new_text": "b"}]})
