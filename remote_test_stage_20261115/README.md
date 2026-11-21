# test: Rust-oriented repair agent

The default `continuous` mode uses one repair conversation for investigation,
editing and verification. After repeated failures, it can consult a read-only
Architect and then resume with its previous conversation and edits intact.
The Verifier reviews the candidate patch and the issue contract. Set
`RUSTBENCH_REPAIR_MODE=staged` to use the staged controller described below as
an experimental comparison. `--single-agent` selects the legacy single-agent
loop instead.

Both modes retain existing edits across replanning and allow explicit checkpoint
restores. Final QA runs the post-strip production check with the regression
test's Cargo configuration, then attempts bounded existing library and integration
tests. The recorded results distinguish passing, failing, timed-out and zero-test
runs. Doctest line shifts are reported for review; mixed code/doc hunks are not
discarded solely because their line counts changed.

`test` is backed by a deterministic Rust Code Knowledge Graph (CKG), compiler
diagnostic symbol lookup, and zero-dependency dynamic tracing.

## Staged comparison mode

1. **Layer 0 — Deterministic Rust Foundations**:
   - `rust_ckg.py` (`inspect_rust_neighborhood`): Zero-dependency token-aware Rust parser and incremental Code Knowledge Graph resolving `mod` trees, `pub use` re-export import paths, `impl Trait` / `#[derive]` relations, call edges, and candidate fault propagation chains.
   - `rust_diagnostics.py`: Classifies rustc diagnostics, surfaces `MachineApplicable` fixes, and automatically enriches `E0599`, `E0425`, `E0412`, `E0405`, `E0432`, `E0433`, and `E0277` compiler errors in both `cargo_check` and `cargo_test` with symbol/re-export/trait-impl lookups.
   - `rust_trace.py`: Token-aware non-destructive RAII function tracing (skipping `#[test]`, `#[cfg(test)]`, and `const fn`) with optional `watch_vars` and automatic fallback if `Debug` is not implemented.
2. **Agent 1 — Navigator & Reproducer** (`multi_agent.py`): Extracts exact requested API names/signals from the issue, maps 1–3 candidate fault propagation chains, creates a temporary regression test (`tests/test_regression_<name>.rs` or in-module `#[cfg(test)]`) to verify the Red phase, optionally runs `rust_trace`, and emits a structured `LocalizationBundle`.
3. **Adaptive Routing Gate**: Chooses **Fast Path** (skipping Architect for simple, localized single-file bugs) or **Planned Path** (invoking Architect for multi-file, multi-chain, trait/lifetime, macro, new-API, or downstream-dependent tasks).
4. **Agent 2 — Rust Architect / Suggester**: Runs in a clean context to arbitrate root cause vs. surface callers, audit trait/lifetime/API contracts, and emit a structured `RepairPlan`.
5. **Agent 3 — Rust Fixer**: Executes dependency-closed repair units in bounded clean-context slices. The deterministic Coordinator checks each unit, records partial-patch evidence, and rolls back only failed units and their dependents on escalation.
6. **Agent 4 — Patch QA & Verifier**: Enforces deterministic checks (paired Red-Green reproduction, post-strip `cargo check --tests` + downstream workspace checks, diff hygiene for `dbg!`/trace leftovers, doctest line-shift checks) followed by an independent LLM semantic audit against the issue contract before final submission.

## Dependency-aware repair units

Architect's `submit_repair_plan` accepts optional `repair_units`. Each unit has a
stable `unit_id`, `description`, zero-based `edit_indices` into `ordered_edits`,
`depends_on`, and optional `verification` (`cargo_check` / `cargo_test` plus their
structured tool arguments, never shell commands). For example:

```json
{
  "repair_units": [
    {"unit_id": "interface", "description": "Define the shared contract", "edit_indices": [0, 1], "depends_on": []},
    {"unit_id": "registry", "description": "Propagate registry state", "edit_indices": [2], "depends_on": ["interface"],
     "verification": [{"tool": "cargo_check", "arguments": {"manifest_path": "registry/Cargo.toml"}}]}
  ]
}
```

- Coordinator's ledger tracks current unit, prerequisites, modified paths,
  verification evidence (configuration/revision/output), and blockers. Ready
  units run sequentially; file-overlapping units and dependency cycles are joined
  for safe joint verification and file-level rollback. Group mutually dependent
  interface/caller hunks rather than checking an unbuildable half-interface.
- Fixer calls `complete_repair_unit(summary=...)`; only actual successful targeted
  checks close the unit. Default verification is ordinary `cargo_check` on the
  affected manifests, avoiding future-API regression tests in intermediate units.
  Explicit runtime test targets must execute at least one test; `no_run` is only
  compilation evidence. Intermediate checks do **not** replace final paired
  Red–Green, post-strip checks, existing-test review, or independent QA.
- Fixer retains its conversation across unit checks. Read-only tools do not
  force a context reset, and the global API budget still bounds the run.
- Escalation and replanning retain the current working changes. Replanning
  preserves evidence for units whose definitions and inputs are unchanged;
  QA edits invalidate affected dependent units.
  Global `checkpoint(action="restore")` is disabled during unit repair. A partial
  rollback clears global Green/suite evidence but retains matching clean Red
  evidence and unrelated source edits. Exact-config checks at the same revision
  are reused; compile success is never promoted to behavioral correctness.
- Fast Path and older plans without units use one coherent unit; no speculative
  dependency ordering is inferred. Trace events include `repair_unit_plan`,
  `repair_unit_start`, `repair_unit_edit`, `repair_unit_verification`,
  `repair_unit_invalidated`, and `repair_unit_rollback`.

## Demand-driven mechanism comparison

`repair_candidates.py` adds a deterministic gate within the existing roles and
budgets, **not a fixed set of Proposer agents**. The default remains one mechanism
(A), one detailed plan, and one implementation. Multiple localization chains alone
do not enable a second candidate. Brief comparison is available only for:

- Two distinct known root-cause chains that Navigator/Architect explicitly cannot
  distinguish (`unresolved_chain_ids` plus `ambiguity_reason`).
- A specific prediction contradicted by actual current-revision Cargo evidence,
  requested via `request_candidate_comparison(trigger="verification_counterevidence",
  reason=..., evidence_ids=[...])`. Repeated behavioral failures over edited
  revisions can also open the gate. Ordinary compilation errors do not
  automatically refute the mechanism.
- Sustained production-scope growth (two successive expansions or two distinct
  unplanned-file requests), not simply an initially large repair plan.

Architect calls `compare_repair_candidates(candidates_brief=[A, B], next_check=...)`
before submitting another detailed plan. Each short card contains a causal
mechanism, contract assessment, execution evidence IDs and interpretation,
dependency compatibility, estimated paths, and verification cost. No candidate
is implemented by the comparison itself. Contract conflicts, refuted mechanisms,
and incompatible dependencies are excluded. Remaining candidates are ranked by
contract completeness, execution support, then dependency compatibility; only
equivalent primary assessments use scope and verification cost as tiebreakers.
An uncertain, untested alternative does not displace an admissible current
mechanism speculatively. If neither is admissible, request relocalization.

Only the selected `candidate_id` may receive a detailed `submit_repair_plan`.
Keeping the current candidate preserves its partial Fast Path edits; selecting
the alternative rolls back affected repair units/dependents while retaining
unrelated verified units. Replanning/budget fallback cannot silently synthesize
an unimplemented alternative. Final Red–Green, post-strip, and independent QA
gates remain unchanged.

Cargo tool results expose `candidate_evidence_id`; regression auto-runs also
record evidence. The ledger records revision, candidate, episode and actual
execution status, rejects fabricated/obsolete evidence, and does not treat a
zero-test run as behavioral support or as resolving a failing regression.
Successful exact-config reruns supersede historical failures. Trace events
include `candidate_gate`, `candidate_evidence`, `candidate_comparison`,
`candidate_plan`, `candidate_comparison_deferred`, and `candidate_relocalized`.
No additional model role, fixed parallel candidates, or global budget increase
is introduced.

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
