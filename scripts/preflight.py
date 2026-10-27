#!/usr/bin/env python3
"""Read-only environment inventory. Does not download, flash or reset anything."""
import argparse
import json
import os
import resource
import shutil
import subprocess
from pathlib import Path


def run(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/preflight.json"))
    args = parser.parse_args()
    result = {
        "os_release": Path("/etc/os-release").read_text(),
        "tools": {name: shutil.which(name) for name in ["g++", "make", "v++", "vitis_hls", "vitis-run", "xrt-smi", "xbutil"]},
        "devices": [str(p) for pattern in ["/dev/dri/renderD*", "/dev/xclmgmt*"] for p in Path("/").glob(pattern.lstrip("/"))],
        "memlock_bytes": resource.getrlimit(resource.RLIMIT_MEMLOCK),
        "xrt": run(["xrt-smi", "examine"]),
        "pcie": run(["lspci", "-Dnnk", "-s", "0000:01:00"]),
        "model_download_performed": False,
        "platform_environment": os.environ.get("PLATFORM"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
