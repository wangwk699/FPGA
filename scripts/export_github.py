#!/usr/bin/env python3
"""Export source-only GitHub material using an allowlist, never following data symlinks."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ["full-decode", "full-decode-fast", "full-decode-stream"]


def source_files():
    selected = set()
    for name in ["README.md", "Makefile", ".gitignore", "requirements-reference.txt", "requirements-reproduce.txt",
                 "run_cpu.sh", "run_hybrid.sh"]:
        selected.add(ROOT / name)
    for directory, suffixes in [("scripts", {".py", ".sh", ".tcl"}), ("tests", {".py", ".cpp"}),
                                ("hls", {".cpp", ".hpp", ".cfg"}), ("host", {".cpp", ".hpp"}),
                                ("docs", {".md"})]:
        selected.update(p for p in (ROOT / directory).rglob("*")
                        if p.is_file() and not p.is_symlink() and p.suffix in suffixes and "__pycache__" not in p.parts)
    selected.update((ROOT / "config").rglob("*.example.*"))
    evidence = ROOT / "docs/evidence/benchmark-summary.json"
    if evidence.is_file():
        selected.add(evidence)
    for name in PROJECTS:
        project = ROOT / name
        for filename in ["README.md", "Makefile", "run_full.sh"]:
            selected.add(project / filename)
        for directory, suffixes in [("hls", {".cpp", ".hpp", ".cfg"}),
                                    ("host", {".cpp", ".hpp"}), ("scripts", {".py"})]:
            selected.update(p for p in (project / directory).glob("*") if p.is_file() and p.suffix in suffixes)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/FPGA-github")
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if output.exists():
        parser.error("Output exists; use a new directory (existing files are never overwritten)")
    files = source_files()
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected regular source file: {path}")
        if path.stat().st_size > 2 * 1024**2:
            raise ValueError(f"Unexpected large source file: {path}")
    output.mkdir(parents=True)
    manifest = []
    for path in files:
        relative = path.relative_to(ROOT)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest.append({"path": str(relative), "bytes": target.stat().st_size,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (output / "SOURCE-MANIFEST.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n")
    archive = output.with_suffix(".tar.gz")
    if archive.exists():
        raise FileExistsError(archive)
    def clean_metadata(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(output, arcname=output.name, filter=clean_metadata)
    print(json.dumps({"directory": str(output), "archive": str(archive),
        "source_files": len(files), "source_bytes": sum(f["bytes"] for f in manifest),
        "model_downloaded": False, "git_push_performed": False}, indent=2))


if __name__ == "__main__":
    main()
