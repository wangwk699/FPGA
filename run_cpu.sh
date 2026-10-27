#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$project_dir/scripts/python_exec.sh" "$project_dir/scripts/infer_local.py" "$@"
