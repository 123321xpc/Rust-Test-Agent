"""Run targeted Cargo tests in the isolated repair checkout."""

import json
import os
import re
import signal
import subprocess
import sys

from langchain_core.tools import tool

_ORIGINAL_SUBPROCESS_RUN = subprocess.run


def _run_test_process(command, cwd, timeout, env):
    """Execute cargo test in a dedicated process session on POSIX to kill orphaned children on timeout."""
    if subprocess.run is not _ORIGINAL_SUBPROCESS_RUN:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, env=env)
    if sys.platform != "win32":
        proc = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env)


def _compact_test_output(test_output: str, limit: int = 16000) -> tuple[str, bool]:
    """Keep failure stdout blocks and panic lines when truncating long test output."""
    if len(test_output) <= limit:
        return test_output, False
    tail_budget = limit // 2
    head_region = test_output[:-tail_budget]
    excerpts = []
    for match in re.finditer(r"(?ms)^failures:\r?\n.*?(?=^test result:|\Z)", head_region):
        block = match.group(0).strip()
        if block and block not in excerpts:
            excerpts.append(block)
    if not excerpts:
        lines = [
            line for line in head_region.splitlines()
            if ("FAILED" in line or "panicked at" in line or line.startswith("test result:"))
        ]
        if lines:
            excerpts.append("\n".join(lines))
    if not excerpts:
        return test_output[-limit:], True
    marker = "\n...[intermediate test output omitted]...\n"
    pinned_budget = limit - tail_budget - len(marker)
    pinned = "\n\n".join(excerpts)[:pinned_budget]
    remaining = max(0, limit - len(pinned) - len(marker))
    return pinned + marker + test_output[-remaining:], True


def _clean_cargo_stderr(stderr: str, max_chars: int = 2000) -> str:
    if not stderr:
        return ""
    lines = stderr.splitlines()
    filtered = [
        line for line in lines
        if not re.match(r"^\s*(?:Compiling|Checking|Downloading|Downloaded|Updating|Fresh|Locking)\s+", line)
    ]
    cleaned = "\n".join(filtered).strip()
    if not cleaned:
        cleaned = "\n".join(lines[-6:]).strip()
    return cleaned[-max_chars:]


def _compact_diagnostics(diagnostics, max_warnings=10, max_errors=12):
    non_summary = [
        d for d in diagnostics
        if d.get("code") or d.get("spans") or not any(
            p in (d.get("message") or "").lower()
            for p in ("aborting due to", "warning emitted", "warnings emitted", "previous error")
        )
    ]
    items = non_summary if non_summary else list(diagnostics)
    has_errors = any(d.get("level") == "error" for d in items)
    warning_budget = min(max_warnings, 5) if has_errors else max_warnings
    error_count = 0
    warning_count = 0
    kept = []
    omitted = 0
    seen_exact = set()
    for diag in items:
        spans = diag.get("spans") or []
        first_span = spans[0] if spans else {}
        exact_key = (
            diag.get("level"),
            diag.get("code"),
            diag.get("message") or "",
            first_span.get("file_name") or "",
            first_span.get("line_start"),
            first_span.get("column_start"),
        )
        if exact_key in seen_exact:
            omitted += 1
            continue
        seen_exact.add(exact_key)
        if diag.get("level") == "error":
            if error_count < max_errors:
                kept.append(diag)
                error_count += 1
            else:
                omitted += 1
        else:
            if warning_count < warning_budget:
                kept.append(diag)
                warning_count += 1
            else:
                omitted += 1
    return kept, omitted


def make_cargo_test_tool(
    root,
    resolve_path,
    simplify_diagnostic,
    on_start=None,
    on_success=None,
    synchronize=None,
    on_failure=None,
    env=None,
    compact_fn=None,
    enrich_diagnostics=None,
    on_test_feedback=None,
):
    if synchronize is None:
        synchronize = lambda function: function
    compact_diag = compact_fn or _compact_diagnostics

    def _invoke_on_success(token, **kwargs):
        if not on_success:
            return
        try:
            on_success(token, **kwargs)
        except TypeError:
            legacy_keys = (
                "manifest_path", "package", "test_target", "test_filter",
                "lib", "bin_target", "tests", "passed_tests", "ignored_tests",
            )
            try:
                on_success(token, **{k: kwargs[k] for k in legacy_keys if k in kwargs})
            except TypeError:
                on_success(
                    token,
                    kwargs["manifest_path"],
                    kwargs["package"],
                    kwargs["test_target"],
                    kwargs["test_filter"],
                    kwargs["lib"],
                    kwargs["bin_target"],
                )

    def _invoke_on_failure(token, **kwargs):
        if not on_failure:
            return
        try:
            on_failure(token, **kwargs)
        except TypeError:
            legacy_keys = (
                "manifest_path", "package", "test_target", "test_filter",
                "lib", "bin_target", "tests", "executed", "failures",
                "returncode", "failed_target",
            )
            try:
                on_failure(token, **{k: kwargs[k] for k in legacy_keys if k in kwargs})
            except TypeError:
                on_failure(token)

    @tool
    @synchronize
    def cargo_test(manifest_path: str = "Cargo.toml", package: str = "",
                   test_target: str = "", test_filter: str = "", features: list[str] | None = None,
                   all_features: bool = False, no_default_features: bool = False,
                   target: str = "", timeout: int = 300, lib: bool = False,
                   bin_target: str = "", tests: bool = False, no_run: bool = False,
                   doc: bool = False) -> str:
        """Run Cargo tests in the working copy. test_target selects an integration-test target;
        test_filter selects test names; lib or bin_target select a library or binary target;
        tests runs all library and integration test targets (skipping doc-tests);
        doc runs documentation tests (cargo test --doc).
        Set no_run=True to compile test targets without executing them (useful for fast verification).
        Features and compilation target are optional.
        timeout is in seconds (1 to 1800). Returns JSON, including test output and diagnostics.
        """
        token = on_start() if on_start else None
        try:
            manifest = resolve_path(manifest_path)
            if not manifest.is_file() or manifest.name != "Cargo.toml":
                raise ValueError("manifest_path must point to a Cargo.toml file in the working copy")
            if not 1 <= timeout <= 1800:
                raise ValueError("timeout must be between 1 and 1800 seconds")
            if sum(bool(value) for value in (test_target, lib, bin_target, tests, doc)) > 1:
                raise ValueError("select only one of test_target, lib, bin_target, tests, and doc")
            if any(not f or f.startswith('-') or ',' in f or any(c.isspace() for c in f) for f in (features or [])):
                raise ValueError("features must contain individual nonempty feature names")
            if test_filter.startswith('-'):
                raise ValueError("test_filter must not start with a dash")
            command = ["cargo", "test", "--message-format=json", "--manifest-path", str(manifest)]
            if package:
                command.append("--package=" + package)
            if test_target:
                command.append("--test=" + test_target)
                command.append("--no-fail-fast")
            if lib:
                command.append("--lib")
            if bin_target:
                command.append("--bin=" + bin_target)
            if tests:
                command.append("--tests")
                command.append("--no-fail-fast")
            if doc:
                command.append("--doc")
            if no_run:
                command.append("--no-run")
            if features:
                command.append("--features=" + ",".join(features))
            if all_features:
                command.append("--all-features")
            if no_default_features:
                command.append("--no-default-features")
            if target:
                command.append("--target=" + target)
            if test_filter:
                command.append(test_filter)
            result = _run_test_process(command, cwd=root, timeout=timeout, env=env)
            diagnostics, output = [], []
            for line in result.stdout.splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    output.append(line)
                    continue
                if isinstance(item, dict) and item.get("reason") == "compiler-message" and isinstance(item.get("message"), dict):
                    diagnostics.append(simplify_diagnostic(item["message"]))
                elif not isinstance(item, dict) or "reason" not in item:
                    output.append(line)
            test_output = "\n".join(output)
            failures = re.findall(r"^test (.+?) \.\.\. FAILED\s*$", test_output, re.MULTILINE)
            passed_tests = re.findall(r"^test (.+?) \.\.\. ok\s*$", test_output, re.MULTILINE)
            failed_target_match = re.search(r"--test(?:=|\s+)([\w-]+)", result.stderr)
            failed_target = failed_target_match.group(1) if failed_target_match else ""
            counts = re.findall(r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;", test_output, re.MULTILINE)
            executed = sum(int(passed) + int(failed) for passed, failed, ignored in counts)
            ignored_tests = sum(int(ignored) for _, _, ignored in counts)
            compile_errors = [d for d in diagnostics if d.get("level") == "error"]
            if no_run and result.returncode == 0:
                pass
            elif result.returncode == 0 and executed > 0:
                _invoke_on_success(
                    token,
                    manifest_path=manifest_path,
                    package=package,
                    test_target=test_target,
                    test_filter=test_filter,
                    lib=lib,
                    bin_target=bin_target,
                    tests=tests,
                    passed_tests=passed_tests,
                    ignored_tests=ignored_tests,
                    features=features,
                    all_features=all_features,
                    no_default_features=no_default_features,
                    target=target,
                )
            else:
                _invoke_on_failure(
                    token,
                    manifest_path=manifest_path,
                    package=package,
                    test_target=test_target,
                    test_filter=test_filter,
                    lib=lib,
                    bin_target=bin_target,
                    tests=tests,
                    executed=executed,
                    failures=failures,
                    returncode=result.returncode,
                    failed_target=failed_target,
                    compile_errors=compile_errors,
                )
            compacted_output, output_truncated = _compact_test_output(test_output, 16000)
            kept_diagnostics, omitted_diagnostics = compact_diag(diagnostics)
            if enrich_diagnostics and compile_errors:
                enrich_diagnostics(kept_diagnostics, root, manifest_dir=manifest.parent)
            if result.returncode == 0:
                cargo_output = _clean_cargo_stderr(result.stderr, max_chars=1000)
                cargo_truncated = len(result.stderr) > len(cargo_output)
                if executed > 0 and not failures and len(compacted_output) > 1000:
                    summary_lines = [
                        line for line in test_output.splitlines()
                        if line.startswith("running ") or line.startswith("test result:")
                    ]
                    if summary_lines:
                        compacted_output = "\n".join(summary_lines)
                        output_truncated = True
            elif compile_errors:
                cargo_output = _clean_cargo_stderr(result.stderr, max_chars=2000)
                cargo_truncated = len(result.stderr) > len(cargo_output)
            else:
                cargo_output = result.stderr[-8000:]
                cargo_truncated = len(result.stderr) > 8000

            payload = {
                "status": "passed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "diagnostics": kept_diagnostics,
                "tests_executed": executed,
                "tests_ignored": ignored_tests,
                "failed_tests": failures,
                "test_output": compacted_output,
                "test_output_truncated": output_truncated,
                "cargo_output": cargo_output,
                "cargo_output_truncated": cargo_truncated,
            }
            if omitted_diagnostics:
                payload["diagnostics_omitted"] = omitted_diagnostics
            if on_test_feedback and not no_run:
                strategy_warning = on_test_feedback(
                    manifest_path, test_target, result.returncode, failures, executed, compile_errors
                )
                if strategy_warning:
                    payload["strategy_warning"] = strategy_warning
            return json.dumps(payload, ensure_ascii=False)
        except subprocess.TimeoutExpired:
            _invoke_on_failure(
                token,
                manifest_path=manifest_path,
                package=package,
                test_target=test_target,
                test_filter=test_filter,
                lib=lib,
                bin_target=bin_target,
                tests=tests,
                executed=0,
                failures=[],
                returncode=None,
            )
            err_msg = f"cargo test timed out after {timeout} seconds."
            if tests or not (test_target or test_filter):
                err_msg += " For large crates, avoid running all test targets at once: try no_run=True, lib=True with test_filter='...', or test_target='...'."
            return json.dumps({"status": "timeout", "exit_code": None, "error": err_msg})
        except (OSError, ValueError) as exc:
            _invoke_on_failure(
                token,
                manifest_path=manifest_path,
                package=package,
                test_target=test_target,
                test_filter=test_filter,
                lib=lib,
                bin_target=bin_target,
                tests=tests,
                executed=0,
                failures=[],
                returncode=None,
            )
            return json.dumps({"status": "error", "exit_code": None, "error": str(exc)}, ensure_ascii=False)

    return cargo_test
