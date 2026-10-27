#!/usr/bin/env python3
"""Select the unencrypted clock adapter already supplied in the AMD platform.

Only derived VPL build files are changed. The platform archive and shell remain
untouched. Vendor copyright notices are preserved by copying the complete file.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = "84a63bd868e1e7b4c3b5586fc7e269fc9b2973082ad2e29e2c77aa13268769b3"
OLD_HASH = "b9d91b98bf8800bfcd8a968365d5e6be15b6c03ac520d863e9131c5cab1148fb"
FILENAME = "clk_metadata_adapter_v1_0_vl_rfs.v"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=ROOT / "_x/link/vivado/vpl")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    build = args.build_root.resolve()
    # Restrict writes to the intended generated build directory.
    allowed = {(ROOT / "_x/link/vivado/vpl").resolve(),
               (ROOT / "full-decode/_x/link/vivado/vpl").resolve(),
               (ROOT / "full-decode-fast/_x/link/vivado/vpl").resolve(),
               (ROOT / "full-decode-stream/_x/link/vivado/vpl").resolve()}
    allowed.add((ROOT / "full-decode-stream/variants/Qwen3-1.7B/_x/link/vivado/vpl").resolve())
    if build not in allowed:
        parser.error("Only the hybrid or full-decoder generated VPL directory may be changed")
    project = build.parents[3]
    source = build / ".local/hw_platform/iprepo/ipdefs/other_ipdefs/clk_metadata_adapter_v1_0/hdl" / FILENAME
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_HASH:
        raise RuntimeError("Vendor clock adapter source does not match the independently validated file")
    candidates = sorted(build.rglob(FILENAME))
    changes = []
    for target in candidates:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == SOURCE_HASH:
            continue
        if digest != OLD_HASH or target.is_symlink():
            raise RuntimeError(f"Unexpected clock adapter source; refusing to modify {target}")
        changes.append(target)
    backup_root = project / "reports/clock-adapter-backups"
    # All candidates are checked before the first write.
    for target in changes:
        if args.apply:
            backup = backup_root / target.relative_to(build)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                if hashlib.sha256(backup.read_bytes()).hexdigest() != OLD_HASH:
                    raise RuntimeError(f"Unexpected backup contents: {backup}")
            else:
                shutil.copy2(target, backup)
            # Extracted vendor files can be read-only; replace atomically through
            # their writable generated directory, preserving the original mode.
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".clock-adapter-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            try:
                temporary.chmod(stat.S_IMODE(target.stat().st_mode))
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    report = {"updated_at": datetime.datetime.now().astimezone().isoformat(),
        "applied": args.apply, "vendor_source": str(source), "source_sha256": SOURCE_HASH,
        "previous_sha256": OLD_HASH, "changed_files": [str(p) for p in changes],
        "backup_root": str(backup_root), "platform_archive_modified": False,
        "shell_modified": False}
    if args.apply:
        (project / "reports").mkdir(exist_ok=True)
        (project / "reports/clock-adapter-selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
