# test: Rust-oriented repair agent

The default `graph` mode orchestrates the four roles (`Navigator`, `Architect`,
`Fixer`, `Verifier`) plus deterministic QA as a constrained directed Agent Graph
(`langgraph.graph.StateGraph` + `Command`), routing localized issues along the
Fast Path, allowing direct `Fixer -> Navigator` and `Verifier -> Navigator`
relocalization edges without Architect detours, and combining Red-phase lock
releases with RepairPlans. Set `RUSTBENCH_REPAIR_MODE=staged` for the legacy
linear staged controller or `RUSTBENCH_REPAIR_MODE=continuous` for the
single-conversation continuous repair mode.

Both modes retain existing edits across replanning and allow explicit checkpoint
restores. Final QA checks affected crates and the workspace with the evaluation
feature selection after removing temporary regression tests, then attempts bounded
existing library and integration tests. Failed compile checks are compared with
the unpatched checkout; a confirmed new compilation failure returns to repair,
while an existing failure or timeout is recorded as inconclusive. Doctest line
shifts are reported for review; mixed code/doc hunks are not discarded solely
because their line counts changed.

`test` is backed by a deterministic Rust Code Knowledge Graph (CKG), compiler
diagnostic symbol lookup, and zero-dependency dynamic tracing.

## Red-phase test locking, Verifier probes, and CKG impact injection

**Red-phase test locking.** When a temporary regression test is confirmed to
FAIL on the unpatched original code, `PatchState` freezes it as the issue's
contract oracle. `write_regression_test` then refuses to rewrite or rename it,
and `revert_file` refuses to delete it; either attempt records a lock violation
and clears that test's Red-phase evidence, so the patch must re-establish a
genuine reproduction before submission. This removes the self-grading loophole
where an agent weakens the very test that was supposed to falsify its patch.
The Verifier prompt receives a `Locked Red-phase Regression Test(s)` block with
each locked test's SHA-256 and tamper status.

**Verifier sandbox probe.** `verifier_probe` gives the Verifier read-only
execution evidence instead of pure prose reasoning:

- `probe_type='existing_test'` re-runs an existing crate test target and
  reports the exit code, pass/fail counts, failed test names, and output tails.
- `probe_type='assertion'` mounts a temporary assertion module, compiles and
  runs it against the crate, then removes it; the working copy is left
  untouched.
- `probe_type='snapshot'` re-runs a target and extracts insta snapshot diff
  hunks so a producer/consumer cascade can be inspected directly.

The Verifier prompt now requires **evidence-backed rejection**: a patch that
already passed paired Red-Green verification may only be rejected when the
verifier cites a probe result, a sibling implementation line, a compiler
diagnostic, a failing test name, or a snapshot diff hunk. Speculative concerns
must not trigger a rejection, and a probe that contradicts the objection
requires approval.

**CKG impact injection.** `rust_impact.py` derives a producer -> consumer
impact surface from the parsed code graph. Whenever production Rust is edited,
`edit_file` and `write_file` return the downstream consumers and trait/impl or
match-arm sites that reference the changed symbols. The same surface is
injected into the Verifier prompt and the Architect advisory prompt, so a
cross-layer change (for example a new lexer token missing a parser match arm)
is surfaced deterministically instead of being rediscovered by ad-hoc grep.

## `test-notest` comparison baseline

`baselines/test-notest` is a copy of `baselines/test` that intentionally omits
the pre-fix test-generation step, so the contribution of the Red-phase
regression test can be measured against the rest of the pipeline. It differs
from `test` in exactly four ways:

1. `write_regression_test` is removed from the Fixer and Navigator tool
   allow-lists in both `continuous` and `staged` modes.
2. The prompts direct the agents to analyze the code and rely on the
   repository's existing tests instead of generating a temporary test.
3. `--require-reproduction` defaults to disabled, because no generated
   regression test exists to reproduce against.
4. The Verifier prompt states that `Reproduction evidence: unverified` is
   expected and must not by itself cause a rejection.

Everything else — the CKG impact surface, the Verifier sandbox probe, the
Architect escalation paths, the submission gate, and the fallback handling —
is identical to `test`, so a difference in resolved tasks isolates the value
of the Red-phase test.

## Staged comparison mode

1. **Layer 0 — Deterministic Rust Foundations**:
   - `rust_ckg.py` (`inspect_rust_neighborhood`): Zero-dependency token-aware Rust parser and incremental Code Knowledge Graph resolving `mod` trees, `pub use` re-export import paths, `impl Trait` / `#[derive]` relations, call edges, and candidate fault propagation chains.
   - `rust_diagnostics.py`: Classifies rustc diagnostics, surfaces `MachineApplicable` fixes, and automatically enriches `E0599`, `E0425`, `E0412`, `E0405`, `E0432`, `E0433`, and `E0277` compiler errors in both `cargo_check` and `cargo_test` with symbol/re-export/trait-impl lookups.
   - `rust_trace.py`: Token-aware non-destructive RAII function tracing (skipping `#[test]`, `#[cfg(test)]`, and `const fn`) with optional `watch_vars` and automatic fallback if `Debug` is not implemented.
2. **Agent 1 — Navigator & Reproducer** (`multi_agent.py`): Extracts exact requested API names/signals from the issue, maps 1–3 candidate fault propagation chains, creates a temporary regression test (`tests/test_regression_<name>.rs` or in-module `#[cfg(test)]`) to verify the Red phase, optionally runs `rust_trace`, and emits a structured `LocalizationBundle`.
3. **Adaptive Routing Gate**: Chooses **Fast Path** (skipping Architect for simple, localized single-file bugs) or **Planned Path** (invoking Architect for multi-file, multi-chain, trait/lifetime, macro, new-API, or downstream-dependent tasks).
4. **Agent 2 — Rust Architect / Suggester**: Runs in a clean context to arbitrate root cause vs. surface callers, audit trait/lifetime/API contracts, and emit a structured `RepairPlan`.
5. **Agent 3 — Rust Fixer**: Executes dependency-closed repair units in bounded clean-context slices. The deterministic Coordinator checks each unit, records partial-patch evidence, and rolls back only failed units and their dependents on escalation.
6. **Agent 4 — Patch QA & Verifier**: Records deterministic checks (paired Red-Green reproduction, post-strip cargo checks, existing-test smoke, diff hygiene, and doctest line-shift findings) and runs an independent LLM semantic audit. Confirmed new compilation failures and test-file conflicts return the patch to repair; inconclusive checks are recorded, and the best available patch may still be submitted when the budget is exhausted.

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
- Fixer calls `complete_repair_unit(summary=...)`; failed checks close the unit
  as `completed_unverified`, without claiming they passed. Default verification is ordinary `cargo_check` on the
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

The selected `candidate_id` guides the detailed `submit_repair_plan`.
Keeping the current candidate preserves its partial Fast Path edits; selecting
the alternative rolls back affected repair units/dependents while retaining
unrelated verified units. Replanning/budget fallback cannot silently synthesize
an unimplemented alternative. Final Red–Green, post-strip, and independent QA
results are recorded as advisory evidence instead of submission gates.

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
the fix, and run relevant checks. Temporary integration tests created with
`write_regression_test` are excluded from the submitted source patch. Red-Green
reproduction, regression tests, existing-test smoke, post-strip compilation,
diff hygiene, and independent QA provide evidence and warnings rather than hard
submission gates. A failed or unrun check is not reported as a pass. Verifier
objections can guide another repair attempt; when no further attempt is available,
the current non-empty source diff can still be submitted. Repair units completed
without passing checks are recorded as `completed_unverified`. If execution
stops or the model request fails, the runner first attempts to preserve the
current source diff, then a verified snapshot. Benchmark evaluation remains
necessary to determine whether the patch resolves the task.

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
`--dataset`. A repository toolchain pin takes precedence unless the evaluation
script explicitly overrides it; the runner passes that override to the agent.
The runner verifies `rustc --version` and `cargo --version` in the isolated
checkout before calling the model and passes the same environment to all Rust
tools.
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
`--max-api-calls` can set a lower limit but cannot exceed `API_LIMIT`. Transient
HTTP and transport failures are counted separately from stage progress; the
global request limit still includes every retry. When
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
