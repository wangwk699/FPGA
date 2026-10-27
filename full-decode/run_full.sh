#!/usr/bin/env bash
set -euo pipefail
qwen_full_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$qwen_full_dir/../scripts/python_exec.sh" "$qwen_full_dir/scripts/run_full.py" "$@"
