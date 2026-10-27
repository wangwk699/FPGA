#!/usr/bin/env bash
set -euo pipefail
qwen_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
qwen_python="${QWEN_PYTHON:-}"
if [[ -z "$qwen_python" && -f "$qwen_root/deployment.json" ]]; then
    qwen_python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("python", ""))' "$qwen_root/deployment.json")"
fi
if [[ -z "$qwen_python" ]]; then
    qwen_python="$(command -v python3)"
fi
exec "$qwen_python" "$@"
