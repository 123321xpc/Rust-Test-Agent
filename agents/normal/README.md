# normal: Generic LangChain ReAct Baseline

`normal` is a generic, language-agnostic LangChain ReAct baseline agent using
standard tool calling in an isolated working copy of the target repository.
The original checkout stays unchanged.

## Tools

`normal` exposes six general-purpose tools:

- `read_file(path, start_line=1, end_line=200)`: Lists a directory or reads
  1-based numbered lines from a UTF-8 file (clamped to 500 lines per call).
- `search_code(query, path=".", file_glob="*", regex=False, max_results=100)`:
  Literal or regular-expression text search across repository files.
- `write_file(path, content)`: Creates, overwrites, or deletes (`content=None`)
  a file in the isolated working copy.
- `edit_file(path, old_text, new_text)`: Performs an exact unique string
  replacement in an existing file and returns surrounding numbered lines.
- `run_command(command, timeout=120)`: Runs a shell command in the working copy
  with the configured benchmark environment and returns exit code, stdout, and
  stderr.
- `generate_patch(paths=None)`: Previews all changed files and unified diff when
  `paths` is omitted, or submits the diff for the selected `paths`.

## Usage

Install dependencies with `pip install -r baselines/normal/requirements.txt`.
Set `API_KEY`, `MODEL`, and `BASE_URL` (or `YJ_*` / `DS_*`) in the
project-root `.env` (existing environment variables take precedence).

```bash
python baselines/normal/agent.py --repo /path/to/checkout \
  --instance-id owner__repo-123 --issue-file /path/to/issue.txt \
  --dataset /path/to/tasks.jsonl \
  --output patch/normal.jsonl
```

The runner configures the benchmark toolchain environment (`--dataset` or
`--rust-toolchain`) for `run_command` execution, tracks API calls against
`API_LIMIT` (default `70`, or `--max-api-calls`), and records JSONL traces under
`<output directory>/traces/<run-id>.jsonl` (or `--trace-dir`).
