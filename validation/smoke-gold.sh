#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/xpc"
export LD_LIBRARY_PATH="$HOME/xpc/gcc-12/lib:$HOME/xpc/gcc-12/x86_64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}"
exec "$HOME/xpc/envs/rustbench/bin/python" "$HOME/xpc/validation/smoke_checks.py" gold
