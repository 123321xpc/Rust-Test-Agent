# test: Rust-oriented LangChain baseline

`test` solves Rust repository issues using file editing, file reversion,
line-based reading, code search with optional context lines, Cargo workspace
inspection, and compiler diagnostic classification. Edits are made in an
isolated copy; the original checkout stays unchanged. The copy excludes
directories named `.git` and `target`, so files in those directories are
unavailable to the agent. The agent must submit a non-empty patch after its
last write to produce a prediction. Calling `generate_patch` without paths
previews all changed files and their diff; calling it with the
repository-relative paths for the final fix submits those files. Every changed
file must be selected, except temporary regression tests created with
`write_regression_test`, which are automatically excluded. New files and
deletions must also be selected.

The agent is instructed to analyze the issue, write a regression test, implement
the fix, and run the regression and relevant existing tests before generating a
patch. It writes temporary integration tests through `write_regression_test` as
new `tests/test_regression_<name>.rs` files, then submits source changes without
those tests. This reduces conflicts with the benchmark's hidden test patch; it
does not guarantee compatibility. After the last write, every temporary
regression test must pass as an unfiltered `cargo_test` integration target with
at least one test executed and no ignored tests. Existing test suites may be run
and their failures reported, but passing them is not a submission requirement.
A failed generated regression test invalidates its verification. Any write or
revert invalidates all regression verification. Gate checks accept either the
package manifest or a workspace manifest with `package="<name>"`. The agent must
not modify existing tests merely to make the suite pass. Before submission,
`generate_patch` temporarily removes the generated regression tests and runs
`cargo check` on affected packages. This checks ordinary build targets, not the
existing test targets. By default, the CLI runner enables `--require-reproduction` (use
`--no-require-reproduction` to disable), requiring paired Red-Green verification:
temporary regression tests must fail on the unpatched base checkout (verified
either during the Red phase before source edits or automatically against the
original files at submission time) and pass after the fix. If the model stops or
reaches the API call limit after verification, the runner may submit the source
patch automatically, subject to the same paired and post-strip checks (using the
package and feature configuration verified during `cargo_test`). Benchmark evaluation is still necessary.

The agent can use `read_lines` for numbered, inclusive line ranges, `search_code`
for literal or regular-expression searches with optional `context_lines` (Rust
files by default, prioritizing `src/` over examples/tests; supports `exclude_tests=True`,
`summarize=True` for per-file match counts, and empty `query=""` with `file_glob` to list matching files),
and `cargo_workspace` for workspace packages and targets (including `import_name`, `features`,
`default_features`, and `has_lib`).
Workspace inspection runs `cargo metadata --offline`, so dependencies needed for
resolution must be cached. `cargo_check` runs `cargo check` in the working copy
and returns JSON diagnostics, including categories for ownership moves, borrow
conflicts, lifetimes, trait bounds, and type mismatches. Error diagnostics are
prioritized, cascading identical errors are clustered with `also_at` locations,
local source snippets are inlined into primary/child spans, and repetitive
warnings and Cargo build-progress lines are compacted to preserve context budget. Cargo and a suitable Rust toolchain must be
installed. `cargo_check` accepts package, feature, target triple and
test/all-target selection options. `cargo_test` runs tests in the working copy with
`RUST_BACKTRACE=1` by default, automatically extracting repository-local stack frames on panic,
and accepts package, integration-test, library or binary target, tests (running all test targets while skipping doc-tests), test-name
filter, features, target triple and timeout; its JSON result includes the exit
status, test output (preserving early failure stdout blocks and panic backtraces when output is
truncated), failed test names, compacted compiler diagnostics, and the number of
tests executed. Local test results do not replace separate benchmark evaluation.
Both commands may download dependencies and execute build scripts; tests execute
project code. Use `edit_file` for a single unique exact-text replacement or a
batch of replacements (`edits=[...]`), returning 6 numbered context lines around
each edit; it includes write-time Rust syntax & guardrail verification (delimiter balance,
unclosed literals, E0753 inner doc/attr misplacement), provides region hints when `old_text` is not found,
and emits impact notices when function, trait, struct, or enum definitions are modified.
`write_regression_test` supports `run_now=True` for instant test execution, and
`generate_patch` supports `submit=True` for automatic one-step submission of all verified source changes.
Use `revert_file` to restore a modified, deleted, or newly created file to its original checkout state.

`rust_symbols` finds candidate Rust definitions (`fn`, `struct`, `enum`, `trait`,
`impl`, `mod`, `union`, `type`, `const`, `static`, `macro`) by name, kind, and
path, with optional `exact` identifier matching, `exclude_tests` filtering,
`callers=True` (call-site lookup), and `references=True` (identifier references).
`edit_file` automatically emits an impact notice when a function signature changes
and call sites exist. `checkpoint` (`save`/`restore`/`list`) snapshots and
rolls back exploratory working-copy changes, and `cargo_check`/`cargo_test`
include a failure tracker that flags consecutive identical failures across edits.
`cargo_test` also supports `no_run=True` for fast test compilation checks.
It uses lexical approximations, not semantic resolution: same-name identifiers,
macros and conditional compilation can make results incomplete or unrelated.
Review the surrounding code before editing. `rust_error_explain` retrieves the
full `rustc --explain` text for an error code using the checkout's toolchain.

Install the dependencies with `pip install -r baselines/test/requirements.txt`.
Set `API_KEY`, `MODEL`, and `BASE_URL` in the project-root `.env` (existing
environment variables take precedence). The same `.env` may also contain `GITHUB_TOKEN`. The model endpoint must support tool
calling. Git must be available. Check out the instance's base commit and save its
problem statement as a UTF-8 file, then run from the project root:

During repair, if `~/xpc/git-2.43.0/bin/git` exists, its directory is prepended
to `PATH` for the agent and its child processes. Otherwise, the existing `PATH`
is used. Benchmark evaluation continues to use the Git in its container.

```bash
python baselines/test/agent.py --repo /path/to/checkout \
  --instance-id owner__repo-123 --issue-file /path/to/issue.txt \
  --dataset /path/to/tasks.jsonl \
  --output patch/test.jsonl
```

The runner reads the selected task from the JSONL and uses the evaluation
harness's `make_test_spec` to determine its default Rust toolchain. Use
`--harness-root /path/to/Rust-bench` to point at the exact harness used for
evaluation. For a non-default evaluation configuration, or when metadata is not
available, pass its known toolchain with `--rust-toolchain 1.72.0` instead of
`--dataset`. If the base checkout has `rust-toolchain.toml` or `rust-toolchain`,
that repository pin takes precedence, as in the standard evaluation setup.
The runner verifies `rustc --version` and `cargo --version` in the isolated
checkout before calling the model and passes the same environment to all Rust
tools. For an unpinned checkout, the benchmark default is configured in its
temporary parent directory, allowing nested repository toolchain pins to work.
Rustup may install a missing toolchain; verification failure stops the run.
No repository file or global rustup default is changed. The actual
compiler and Cargo versions are recorded in the trace and provided to the agent.

The command appends a Rust-bench prediction with `model_name_or_path` set to
`test__<model>`. To evaluate predictions separately, use the Rust-bench
harness described in the project README. Use a fresh JSONL file for repeated
instances.

The CLI supports API call limits and a trace directory; run `--help` for details.
Set `API_LIMIT` to a positive integer in the root `.env` to limit model HTTP
requests per repair, including retries; it defaults to 70 when unset. The optional
`--max-api-calls` can set a lower limit but cannot exceed `API_LIMIT`. When
remaining API calls drop below the warning threshold, tool outputs include a
budget notice so the agent can finish verification and submit in time.
Each run writes a JSONL trace under `<output directory>/traces/<run-id>.jsonl`
(or the directory specified by `--trace-dir`). It contains model requests and
responses, tool inputs and outputs, the submitted patch, errors, and an
`agent_metrics` event with elapsed time, HTTP attempts (including retries),
input/output/total tokens, and the number of successful responses without usage
information. Token totals include only usage reported by the API and may be
incomplete when usage is absent. Traces include issue and source contents.
Authentication headers are omitted and the configured key is redacted.

For a comparison, use the same instance, base commit, model, endpoint, and
evaluation settings, and compare harness reports and trace metrics.
