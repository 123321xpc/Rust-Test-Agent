#!/usr/bin/env bash
set -euo pipefail
cd /home/xupeicai/xpc
export PATH="$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/xpc/gcc-12/lib:$HOME/xpc/gcc-12/x86_64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/xpc/envs/rustbench/bin/python"
EXPERIMENT="$HOME/xpc/validation/runs/p1_bytes_20261114_matched_3x_retry"
"$PY" validation/run_bytes_p1.py replay --dir "$EXPERIMENT" --repeats 3 > "$EXPERIMENT/replay.log" 2>&1
"$PY" validation/run_bytes_p1.py report --dir "$EXPERIMENT" --repeats 3 > "$EXPERIMENT/report.log" 2>&1
