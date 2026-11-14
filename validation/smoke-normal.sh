#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/xpc"
export API_LIMIT=1
exec "$HOME/xpc/validation/validate" --agent normal --instance chronotope__chrono-991
