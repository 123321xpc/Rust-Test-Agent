"""Generic file, search, command execution, and patch tools for the normal baseline."""

from functools import wraps
import fnmatch
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from langchain_core.tools import tool

_ORIGINAL_SUBPROCESS_RUN = subprocess.run


def _format_edit_snippet(content: str, line_ranges: list[tuple[int, int]], context: int = 3) -> str:
    lines = content.splitlines()
    if not lines or not line_ranges:
        return ""
    total = len(lines)
    windows = []
    for start, end in sorted(line_ranges):
        w_start = max(1, start - context)
        w_end = min(total, max(start, end) + context)
        if windows and w_start <= windows[-1][1] + 1:
            windows[-1][1] = max(windows[-1][1], w_end)
        else:
            windows.append([w_start, w_end])
    blocks = []
    for w_start, w_end in windows:
        blocks.append("\n".join(f"{lineno}: {lines[lineno - 1]}" for lineno in range(w_start, w_end + 1)))
    return "\n--\n".join(blocks)


def _apply_exact_replacement(content: str, old_text: str, new_text: str) -> tuple[str, tuple[int, int]]:
    if not isinstance(old_text, str) or not old_text:
        raise ValueError("old_text must not be empty")
    if not isinstance(new_text, str):
        raise ValueError("new_text must be a string")

    actual_old = old_text
    actual_new = new_text
    position = content.find(actual_old)
    if position < 0 and "\r\n" in content and "\r\n" not in old_text and "\n" in old_text:
        crlf_old = old_text.replace("\n", "\r\n")
        if content.find(crlf_old) >= 0:
            actual_old = crlf_old
            actual_new = new_text.replace("\r\n", "\n").replace("\n", "\r\n")
            position = content.find(actual_old)

    if position < 0:
        raise ValueError("old_text was not found")
    if content.find(actual_old, position + 1) >= 0:
        raise ValueError("old_text matches more than once; provide more context")

    start_line = content.count("\n", 0, position) + 1
    end_line = start_line + actual_new.count("\n")
    updated = content[:position] + actual_new + content[position + len(actual_old):]
    return updated, (start_line, end_line)


def _truncate_output(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    head_len = limit // 3
    marker = f"\n... [truncated {len(text) - limit} chars] ...\n"
    tail_len = max(0, limit - head_len - len(marker))
    return text[:head_len] + marker + (text[-tail_len:] if tail_len else "")


def _run_shell_command(command: str, cwd: Path, timeout: int, env: dict[str, str] | None) -> subprocess.CompletedProcess:
    if subprocess.run is not _ORIGINAL_SUBPROCESS_RUN:
        return subprocess.run(
            command, cwd=cwd, shell=True, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
        )
    if sys.platform != "win32":
        proc = subprocess.Popen(
            command, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", env=env,
            start_new_session=True,
        )
        try:
            stdout_text, stderr_text = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(command, proc.returncode, stdout_text, stderr_text)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            raise exc
    return subprocess.run(
        command, cwd=cwd, shell=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
    )


def _remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


class PatchState:
    def __init__(self, repo, require_reproduction: bool = False, post_strip_check: bool = False, instance_id: str | None = None):
        self._temp = None
        self._custom_workspace = False
        if instance_id and instance_id.strip():
            sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id.strip()).strip("_")
            custom_root = Path(os.environ.get("RUSTBENCH_WORKSPACE_ROOT") or (Path(tempfile.gettempdir()) / "normal-agent-workspaces"))
            base_dir = custom_root / sanitized
            try:
                if base_dir.exists():
                    shutil.rmtree(base_dir, onerror=_remove_readonly)
                base_dir.mkdir(parents=True, exist_ok=True)
                self._workspace_dir = base_dir
                self.root = base_dir / "checkout"
                self._custom_workspace = True
            except Exception:
                self._temp = tempfile.TemporaryDirectory(prefix="normal-agent-")
                self.root = Path(self._temp.name) / "checkout"
                self._custom_workspace = False
        else:
            self._temp = tempfile.TemporaryDirectory(prefix="normal-agent-")
            self.root = Path(self._temp.name) / "checkout"

        self._repo = Path(repo).resolve()
        self._is_worktree = False
        self._is_git = False

        if (self._repo / ".git").exists() and shutil.which("git"):
            try:
                subprocess.run(
                    ["git", "rev-parse", "--verify", "HEAD"],
                    cwd=self._repo, capture_output=True, check=True,
                )
                wt_res = subprocess.run(
                    ["git", "worktree", "add", "--detach", str(self.root), "HEAD"],
                    cwd=self._repo, capture_output=True, text=True,
                )
                if wt_res.returncode == 0:
                    self._is_worktree = True
                    self._is_git = True
                else:
                    clone_res = subprocess.run(
                        ["git", "-c", "core.autocrlf=false", "clone", "--local", "--no-hardlinks", str(self._repo), str(self.root)],
                        capture_output=True, text=True,
                    )
                    if clone_res.returncode == 0:
                        self._is_git = True
            except Exception:
                pass

        if not self._is_git:
            try:
                shutil.copytree(
                    repo,
                    self.root,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(".git", "target", ".cargo"),
                )
                cargo_dir = self._repo / ".cargo"
                if cargo_dir.is_dir():
                    dest_cargo = self.root / ".cargo"
                    dest_cargo.mkdir(parents=True, exist_ok=True)
                    for item in cargo_dir.iterdir():
                        dest_item = dest_cargo / item.name
                        if not dest_item.exists() and not dest_item.is_symlink():
                            if item.is_dir():
                                try:
                                    dest_item.symlink_to(item.resolve())
                                except OSError:
                                    pass
                            else:
                                try:
                                    dest_item.write_bytes(item.read_bytes())
                                except OSError:
                                    pass
            except Exception:
                if self._temp is not None:
                    self._temp.cleanup()
                raise

            if shutil.which("git"):
                try:
                    subprocess.run(["git", "init", "-q"], cwd=self.root, capture_output=True, check=True)
                    exclude_file = self.root / ".git" / "info" / "exclude"
                    exclude_file.parent.mkdir(parents=True, exist_ok=True)
                    existing_exclude = exclude_file.read_text(encoding="utf-8", errors="replace") if exclude_file.exists() else ""
                    exclude_file.write_text(existing_exclude + "\ntarget/\n.cargo/\n", encoding="utf-8")
                    for key, val in (
                        ("core.autocrlf", "false"),
                        ("core.quotepath", "false"),
                        ("user.email", "normal@example.com"),
                        ("user.name", "Normal"),
                    ):
                        subprocess.run(["git", "config", key, val], cwd=self.root, capture_output=True, check=True)
                    subprocess.run(["git", "add", "--all"], cwd=self.root, capture_output=True, check=True)
                    subprocess.run(["git", "commit", "-qm", "base", "--allow-empty"], cwd=self.root, capture_output=True, check=True)
                    self._is_git = True
                except Exception:
                    shutil.rmtree(self.root / ".git", ignore_errors=True)
                    self._is_git = False

        self.originals = {}
        self.patch = ""
        self._lock = threading.RLock()
        self.rust_env = None
        self.rust_info = None
        self._budget_client = None
        self._budget_warning_threshold = 10

    def set_budget(self, client, warning_threshold: int | None = None):
        with self._lock:
            self._budget_client = client
            if warning_threshold is not None:
                self._budget_warning_threshold = warning_threshold
            elif getattr(client, "max_calls", None) is not None:
                max_calls = client.max_calls
                self._budget_warning_threshold = min(10, max(2, max_calls // 4)) if max_calls < 20 else 10

    def remaining_api_calls(self) -> int | None:
        client = self._budget_client
        if client is None or getattr(client, "max_calls", None) is None:
            return None
        remaining = getattr(client, "remaining_calls", None)
        if isinstance(remaining, int):
            return remaining
        calls = getattr(client, "_calls", 0)
        return max(0, client.max_calls - calls)

    def budget_notice(self) -> str | None:
        remaining = self.remaining_api_calls()
        if remaining is not None and remaining <= self._budget_warning_threshold:
            return f"[Budget notice: {remaining} API call(s) remaining. Call generate_patch when ready to submit.]"
        return None

    def changed_paths(self) -> list[Path]:
        with self._lock:
            root = self.root.resolve()
            if self._is_git:
                try:
                    raw = subprocess.run(
                        ["git", "-c", "core.quotepath=false", "status", "--porcelain", "-z", "-uall"],
                        cwd=root,
                        capture_output=True,
                        check=True,
                    ).stdout
                    tokens = raw.split(b"\x00")
                    paths = []
                    seen = set()
                    idx = 0
                    while idx < len(tokens):
                        entry = tokens[idx]
                        idx += 1
                        if len(entry) < 4:
                            continue
                        status = entry[:2].decode("ascii", errors="replace")
                        candidates = [entry[3:].decode("utf-8", errors="surrogateescape")]
                        if status[0] in ("R", "C") or status[1] in ("R", "C"):
                            if idx < len(tokens) and tokens[idx]:
                                candidates.append(tokens[idx].decode("utf-8", errors="surrogateescape"))
                                idx += 1
                        for item in candidates:
                            rel = Path(item)
                            if rel.parts and not any(p.lower() == ".git" for p in rel.parts) and rel not in seen:
                                seen.add(rel)
                                paths.append(rel)
                    return sorted(paths, key=lambda p: p.as_posix())
                except Exception:
                    pass

            changed = []
            for relative, original in self.originals.items():
                source = root / relative
                if source.is_file():
                    if original is None or source.read_bytes() != original[0]:
                        changed.append(relative)
                elif original is not None:
                    changed.append(relative)
            return sorted(changed, key=lambda p: p.as_posix())

    def build_patch(self, paths: list[Path]) -> str:
        if not paths:
            return ""
        root = self.root.resolve()
        if self._is_git:
            try:
                posix_paths = [p.as_posix() for p in paths]
                intent_added = []
                for relative in paths:
                    source = root / relative
                    if source.is_file():
                        tracked = subprocess.run(
                            ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
                            cwd=root,
                            capture_output=True,
                        ).returncode == 0
                        if not tracked:
                            subprocess.run(
                                ["git", "-c", "core.autocrlf=false", "add", "-N", "--force", "--", relative.as_posix()],
                                cwd=root,
                                capture_output=True,
                                check=True,
                            )
                            intent_added.append(relative.as_posix())
                try:
                    diff = subprocess.run(
                        ["git", "-c", "core.autocrlf=false", "-c", "core.quotepath=false",
                         "diff", "--no-ext-diff", "--no-color", "--binary", "HEAD", "--", *posix_paths],
                        cwd=root,
                        capture_output=True,
                        check=True,
                    ).stdout.decode("utf-8", errors="replace")
                finally:
                    if intent_added:
                        subprocess.run(
                            ["git", "reset", "-q", "--", *intent_added],
                            cwd=root,
                            capture_output=True,
                        )
                return diff
            except Exception:
                pass

        def git(directory, *args):
            return subprocess.run(
                ["git", "-c", "core.autocrlf=false", "-c", "core.quotepath=false", *args],
                cwd=directory,
                capture_output=True,
                check=True,
            ).stdout

        with tempfile.TemporaryDirectory(prefix="normal-patch-") as directory:
            work = Path(directory)
            git(work, "init", "-q")
            for relative in paths:
                original = self.originals.get(relative)
                if original is not None:
                    destination = work / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(original[0])
                    destination.chmod(original[1])
            git(work, "add", "--force", "--all")
            baseline = git(work, "write-tree").decode("ascii").strip()
            for relative in paths:
                original = self.originals.get(relative)
                source = root / relative
                destination = work / relative
                if not source.is_file():
                    if destination.exists():
                        destination.unlink()
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.read_bytes())
                    if original is not None:
                        destination.chmod(original[1])
            git(work, "add", "--force", "--all")
            return git(work, "diff", "--cached", "--no-ext-diff", "--no-color", "--binary", baseline).decode("utf-8", errors="replace")

    def auto_submit_patch(self) -> str:
        with self._lock:
            if self.patch:
                return self.patch
            selected = self.changed_paths()
            if not selected:
                return ""
            patch = self.build_patch(selected)
            if patch:
                self.patch = patch
            return self.patch

    def close(self):
        if self._is_worktree:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(self.root)],
                    cwd=self._repo,
                    capture_output=True,
                    timeout=30,
                )
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=self._repo,
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                pass
        if getattr(self, "_custom_workspace", False) and hasattr(self, "_workspace_dir"):
            try:
                shutil.rmtree(self._workspace_dir, onerror=_remove_readonly)
            except Exception:
                pass
        elif getattr(self, "_temp", None):
            try:
                self._temp.cleanup()
            except Exception:
                try:
                    shutil.rmtree(self._temp.name, onerror=_remove_readonly)
                except Exception:
                    pass


def make_tools(repo_path, benchmark_version=None, require_reproduction=False, post_strip_check=False, instance_id=None):
    """Return generic tools and state; call state.close() after the agent run."""
    repo = Path(repo_path).resolve(strict=True)
    if not repo.is_dir():
        raise ValueError(f"Not a directory: {repo}")
    state = PatchState(
        repo,
        require_reproduction=require_reproduction,
        post_strip_check=post_strip_check,
        instance_id=instance_id,
    )
    root = state.root.resolve()
    rust_env = None
    state.rust_info = None
    spec = importlib.util.spec_from_file_location("rust_environment", Path(__file__).with_name("rust_environment.py"))
    environment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(environment)
    if benchmark_version is not None:
        try:
            rust_env, state.rust_info = environment.prepare_rust_environment(root, benchmark_version, repo_path=repo, instance_id=instance_id)
            state.rust_env = rust_env
        except Exception:
            state.close()
            raise
    else:
        try:
            base_env = os.environ.copy()
            cache_info = environment.configure_cargo_cache(base_env, repo_path=repo, instance_id=instance_id)
            state.rust_env = base_env
            rust_env = base_env
            state.rust_info = cache_info
        except Exception:
            pass

    def synchronized(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with state._lock:
                return function(*args, **kwargs)
        return wrapper

    def resolve_path(path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise ValueError("Path must be a non-empty string")
        target = (root / path).resolve()
        if not target.is_relative_to(root) or any(p.lower() == ".git" for p in target.relative_to(root).parts):
            raise ValueError("Path must be inside the repository and outside .git")
        return target

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
        """List a directory or read UTF-8 file lines (1-based inclusive, at most 500 lines per call); paths are repository-relative."""
        try:
            target = resolve_path(path)
            if target.is_dir():
                return "\n".join(p.name + ("/" if p.is_dir() else "") for p in sorted(target.iterdir()) if p.name != ".git") or "(empty directory)"
            if not target.is_file():
                raise ValueError(f"File does not exist: {path}")
            if start_line < 1:
                raise ValueError("start_line must be at least 1")
            if end_line < start_line:
                raise ValueError("end_line must be greater than or equal to start_line")
            clamped_notice = ""
            if end_line - start_line >= 500:
                end_line = start_line + 499
                clamped_notice = f"\n[Notice: requested range exceeded 500 lines; clamped to {start_line}-{end_line}]"

            lines = target.read_bytes().decode("utf-8").splitlines()
            if not lines:
                return "(empty file)"
            total_lines = len(lines)
            if start_line > total_lines:
                return "(no lines in range)"
            actual_end = min(end_line, total_lines)
            formatted = "\n".join(f"{lineno}: {lines[lineno - 1]}" for lineno in range(start_line, actual_end + 1))
            if total_lines > actual_end and not clamped_notice:
                clamped_notice = f"\n[Showing lines {start_line}-{actual_end} of {total_lines}; pass start_line/end_line to read more]"
            elif total_lines > actual_end and clamped_notice:
                clamped_notice += f" (total {total_lines} lines)"
            return formatted + clamped_notice
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error reading {path}: {exc}"

    @tool
    def search_code(query: str, path: str = ".", file_glob: str = "*", regex: bool = False, max_results: int = 100) -> str:
        """Search UTF-8 files by literal text or regex, reporting repository-relative file and line."""
        try:
            if not query or not 1 <= max_results <= 500:
                raise ValueError("Provide a query and max_results between 1 and 500")
            target = resolve_path(path)
            if not target.exists():
                raise ValueError("Search path does not exist")
            if target.is_file():
                files = [target]
            else:
                files = []
                for directory, dirs, names in os.walk(target, followlinks=False):
                    dirs[:] = sorted(d for d in dirs if d not in (".git", "target") and not (Path(directory) / d).is_symlink())
                    files.extend(Path(directory) / name for name in sorted(names) if fnmatch.fnmatch(name, file_glob))

            pattern = re.compile(query if regex else re.escape(query))
            matches = []
            for file in files:
                resolved = file.resolve()
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    continue
                try:
                    with resolved.open(encoding="utf-8") as stream:
                        for number, line in enumerate(stream, 1):
                            if pattern.search(line):
                                matches.append(f"{file.relative_to(root).as_posix()}:{number}: {line.rstrip()}")
                                if len(matches) > max_results:
                                    return "\n".join(matches[:max_results]) + "\n(results truncated)"
                except (OSError, UnicodeError):
                    continue
            return "\n".join(matches) or "(no matches)"
        except (OSError, ValueError, re.error) as exc:
            return f"Error searching code: {exc}"

    @tool
    @synchronized
    def write_file(path: str, content: str | None) -> str:
        """Write UTF-8 contents in the working copy; null deletes a file."""
        try:
            target = resolve_path(path)
            relative = target.relative_to(root)
            if not relative.parts or (content is not None and not isinstance(content, str)):
                raise ValueError("Provide a file path and string or null content")
            if target.exists() and not target.is_file():
                raise ValueError("Path is not a regular file")
            if content is None and not target.is_file():
                raise ValueError("Cannot delete a missing file")
            if relative not in state.originals:
                state.originals[relative] = ((target.read_bytes(), target.stat().st_mode) if target.is_file() else None)
            state.patch = ""
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
            return f"Updated {relative.as_posix()} in the working copy."
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error writing {path}: {exc}"

    @tool
    @synchronized
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace one exact occurrence of old_text with new_text in an existing UTF-8 file."""
        try:
            target = resolve_path(path)
            relative = target.relative_to(root)
            if not target.is_file():
                raise ValueError(f"File does not exist: {path}")
            raw_bytes = target.read_bytes()
            content = raw_bytes.decode("utf-8")
            updated, line_range = _apply_exact_replacement(content, old_text, new_text)
            if relative not in state.originals:
                state.originals[relative] = (raw_bytes, target.stat().st_mode)
            state.patch = ""
            target.write_bytes(updated.encode("utf-8"))
            snippet = _format_edit_snippet(updated, [line_range], context=3)
            out_text = f"Updated {relative.as_posix()} in the working copy."
            return f"{out_text}\n{snippet}" if snippet else out_text
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error editing {path}: {exc}"

    @tool
    @synchronized
    def run_command(command: str, timeout: int = 120) -> str:
        """Run a shell command in the isolated working copy (e.g., build, test, or inspect files)."""
        try:
            if not isinstance(command, str) or not command.strip():
                raise ValueError("command must be a non-empty string")
            if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 600:
                raise ValueError("timeout must be an integer between 1 and 600")
            before_diff = state.build_patch(state.changed_paths()) if state.patch else None
            result = _run_shell_command(command, root, timeout, rust_env)
            if before_diff is not None and state.build_patch(state.changed_paths()) != before_diff:
                state.patch = ""
            stdout_text = _truncate_output(result.stdout or "", limit=8000)
            stderr_text = _truncate_output(result.stderr or "", limit=8000)
            return f"Exit code: {result.returncode}\nSTDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
        except subprocess.TimeoutExpired:
            return f"Error running command: timed out after {timeout} seconds."
        except (OSError, ValueError) as exc:
            return f"Error running command: {exc}"

    @tool
    @synchronized
    def generate_patch(paths: list[str] | None = None) -> str:
        """Preview changed files and diff (when paths is omitted) or submit the final patch for selected paths."""
        try:
            changed = state.changed_paths()
            if paths is None:
                if not changed:
                    return "No changes to submit."
                names = "\n".join(path.as_posix() for path in changed)
                hint = f"Call generate_patch(paths={json.dumps([p.as_posix() for p in changed])}) to submit."
                return f"Unsubmitted preview. Changed files:\n{names}\n{hint}\n\n{state.build_patch(changed)}"
            state.patch = ""
            if not paths:
                raise ValueError("Provide at least one changed file path")
            selected = []
            for path in paths:
                relative = resolve_path(path).relative_to(root)
                if relative not in changed:
                    raise ValueError(f"Not a changed file: {path}")
                if relative not in selected:
                    selected.append(relative)
            patch = state.build_patch(selected)
            state.patch = patch
            return patch or "No changes to submit."
        except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as exc:
            return f"Error generating patch: {exc}"

    all_tools = [read_file, search_code, write_file, edit_file, run_command, generate_patch]

    def attach_budget_notice(tool_obj):
        original_func = tool_obj.func

        @wraps(original_func)
        def wrapped(*args, **kwargs):
            output = original_func(*args, **kwargs)
            notice = state.budget_notice()
            if not notice or not isinstance(output, str):
                return output
            if tool_obj.name == "generate_patch" and state.patch:
                return output
            return f"{output}\n\n{notice}"

        tool_obj.func = wrapped
        return tool_obj

    return [attach_budget_notice(t) for t in all_tools], state
