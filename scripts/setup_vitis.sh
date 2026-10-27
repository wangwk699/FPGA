#!/usr/bin/env bash
set -euo pipefail
qwen_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
qwen_installer="${QWEN_VITIS_INSTALLER:-$qwen_project/.deps/vitis-installer-2025.1/xsetup}"
qwen_install_config="${QWEN_VITIS_INSTALL_CONFIG:-$qwen_project/config/vitis-2025.1.local.txt}"
case "${1:-help}" in
    auth|install|add-hbm) ;;
    *)
        echo 'Usage: setup_vitis.sh auth | install | add-hbm'
        echo 'Override QWEN_VITIS_INSTALLER, QWEN_VITIS_INSTALL_CONFIG or QWEN_VITIS_ROOT as needed.'
        exit 0 ;;
esac
if [[ "${1}" == add-hbm ]]; then
    qwen_destination="${QWEN_VITIS_ROOT:-}"
    if [[ -z "$qwen_destination" && -f "$qwen_project/deployment.json" ]]; then
        qwen_destination="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("tool_installation",{}).get("destination",""))' "$qwen_project/deployment.json")"
    fi
    qwen_destination="${qwen_destination:-$HOME/amd}"
    qwen_installer="${QWEN_VITIS_INSTALLER:-$qwen_destination/.xinstall/2025.1/xsetup}"
fi
if [[ ! -x "$qwen_installer" ]]; then
    echo "AMD installer is missing: $qwen_installer" >&2
    exit 1
fi
if [[ "${1}" == auth ]]; then
    exec "$qwen_installer" -b AuthTokenGen
fi
if [[ ! -f "$qwen_install_config" && -f "$qwen_project/config/vitis-2025.1.txt" ]]; then
    qwen_install_config="$qwen_project/config/vitis-2025.1.txt"
fi
if [[ ! -f "$qwen_install_config" ]]; then
    echo 'Generate the local installer configuration with scripts/configure.py first.' >&2
    exit 1
fi
export DEBUG_ARGS="${DEBUG_ARGS:+$DEBUG_ARGS }-Dsun.net.client.defaultConnectTimeout=60000 -Dsun.net.client.defaultReadTimeout=300000"
qwen_action=Install
if [[ "${1}" == add-hbm ]]; then qwen_action=Add; fi
exec "$qwen_installer" -a XilinxEULA,3rdPartyEULA -b "$qwen_action" -c "$qwen_install_config"
