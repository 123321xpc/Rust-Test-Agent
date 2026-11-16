import subprocess

from swebench.harness.test_spec import make_test_patch_apply_command


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def setup_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    source = repo / "lib.rs"
    original = [f"line {number}\n" for number in range(1, 16)]
    source.write_text("".join(original), encoding="utf-8")
    git(repo, "add", "lib.rs")
    git(repo, "commit", "-qm", "base")
    return repo, source, original


def test_nearby_nonoverlapping_changes_merge(tmp_path):
    repo, source, original = setup_repo(tmp_path)
    theirs = original.copy()
    theirs[8] = "test patch line\n"
    source.write_text("".join(theirs), encoding="utf-8")
    test_patch = git(repo, "diff", "--", "lib.rs")
    ours = original.copy()
    ours[6] = "model patch line\n"
    source.write_text("".join(ours), encoding="utf-8")

    result = subprocess.run(["bash", "-c", make_test_patch_apply_command(test_patch)],
                            cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "trying a three-way merge" in result.stdout, result.stderr
    merged = source.read_text(encoding="utf-8")
    assert "model patch line\n" in merged
    assert "test patch line\n" in merged, result.stdout + result.stderr


def test_overlapping_changes_fail_without_dropping_either_side(tmp_path):
    repo, source, original = setup_repo(tmp_path)
    theirs = original.copy()
    theirs[3] = "test patch line\n"
    source.write_text("".join(theirs), encoding="utf-8")
    test_patch = git(repo, "diff", "--", "lib.rs")
    ours = original.copy()
    ours[3] = "model patch line\n"
    source.write_text("".join(ours), encoding="utf-8")

    result = subprocess.run(["bash", "-c", make_test_patch_apply_command(test_patch)],
                            cwd=repo, capture_output=True, text=True)
    assert result.returncode != 0
    assert "ERROR: failed to apply test patch: three-way merge conflicted" in result.stdout
    merged = source.read_text(encoding="utf-8")
    assert "model patch line\n" in merged
    assert "test patch line\n" in merged
