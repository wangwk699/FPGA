#!/usr/bin/env python3
"""Maintainer-only: move existing FPGA dependencies into a shared server directory.

No downloads, authentication files, user environments or firmware changes.
Moves are on the same filesystem and leave compatibility symlinks at old paths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--amd-root', type=Path, required=True)
    parser.add_argument('--shared-root', type=Path, default=Path('/home/share/fpga'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    source = args.source_project.expanduser().absolute()
    shared = args.shared_root.expanduser().absolute()
    amd = args.amd_root.expanduser().absolute() / '2025.1'
    latest = source / 'full-decode-stream'
    transfers = [
        (amd, shared / 'amd/2025.1'),
        (source / 'platforms/u55c', shared / 'platforms/u55c'),
        (source / '.deps/xrt-dev', shared / 'xrt-dev'),
        (latest / 'weights', shared / 'models/Qwen3-0.6B/tiled-bf16-v2'),
        (latest / 'reference/default', shared / 'reference/Qwen3-0.6B/default'),
        (latest / 'build/hw/qwen_full_decode.xclbin', shared / 'artifacts/qwen3-0.6b-stream/qwen_full_decode.xclbin'),
    ]
    for filename in ['FPGAs_AdaptiveSoCs_Unified_SDI_2025.1_0530_0145_Lin64.bin',
                     'xilinx-u55c-gen3x16-xdma-3-202210-1-dev_1-3514517_all.deb']:
        transfers.append((source / 'installers' / filename, shared / 'downloads' / filename))
    for pattern in ['libxrt-dev_*.deb', 'uuid-dev_*.deb', 'u55c-deployment.tar.gz']:
        transfers.extend((p, shared / 'downloads' / p.name) for p in sorted((source / '.deps').glob(pattern)))
    for old, new in transfers:
        if not old.exists():
            raise FileNotFoundError(old)
        if new.exists() or new.is_symlink():
            if not (old.is_symlink() and old.resolve() == new.resolve()):
                raise FileExistsError(f'Refusing to merge/overwrite: {new}')
        if old.stat().st_dev != shared.parent.stat().st_dev:
            raise RuntimeError('Same-filesystem move required; no automatic 104GB copy')
        if not new.resolve().is_relative_to(shared.resolve()):
            raise RuntimeError(f'Target escapes the shared directory: {new}')
    if not args.apply:
        print(json.dumps({'moves': [[str(a), str(b)] for a, b in transfers],
                          'private_auth_and_conda_excluded': True}, indent=2))
        return
    shared.mkdir(parents=True, exist_ok=True)
    log_dir = source / 'reports/shared-relocation'
    log_dir.mkdir(parents=True, exist_ok=True)
    # These are installer-generated settings, not arbitrary vendor source edits.
    settings = []
    for component in ['Vitis', 'Vivado', 'Model_Composer']:
        for path in (amd / component).iterdir():
            if (path.name.startswith(('settings64', '.settings64'))
                    and path.suffix in {'.sh', '.csh'} and path.is_file()
                    and str(amd) in path.read_text()):
                settings.append(path.relative_to(amd))
                backup = log_dir / 'settings-before' / path.relative_to(amd)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
    moved = []
    for old, new in transfers:
        if old.is_symlink() and old.resolve() == new.resolve():
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        old.symlink_to(new, target_is_directory=new.is_dir())
        moved.append([str(old), str(new)])
        print('Shared:', new, flush=True)
    new_amd = shared / 'amd/2025.1'
    for relative in settings:
        path = new_amd / relative
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
        path.write_text(path.read_text().replace(str(amd), str(new_amd)))
    changed_links = 0
    directories = [shared]
    file_count = 0
    for base, dirs, files in os.walk(shared, followlinks=False):
        base = Path(base)
        for name in dirs + files:
            path = base / name
            if path.is_symlink():
                target = os.readlink(path)
                if target.startswith(str(amd) + '/'):
                    path.unlink()
                    path.symlink_to(str(new_amd) + target[len(str(amd)):])
                    changed_links += 1
                continue
            mode = path.stat().st_mode
            tool = path.is_relative_to(shared / 'amd')
            if tool:
                # Other users can run the shared compiler but cannot modify it.
                path.chmod((stat.S_IMODE(mode) | 0o444 | (0o111 if path.is_dir() or mode & 0o111 else 0)) & ~0o022)
            else:
                path.chmod(stat.S_IMODE(mode) | 0o666 | (0o111 if path.is_dir() or mode & 0o111 else 0))
            if path.is_dir() and not tool:
                directories.append(path)
            else:
                file_count += 1
        if len(directories) >= 128:
            subprocess.run(['setfacl', '-m', 'd:u::rwx,d:g::rwx,d:o::rwx,d:m::rwx',
                            *map(str, directories)], check=True)
            directories.clear()
    shared.chmod(0o777)
    if directories:
        subprocess.run(['setfacl', '-m', 'd:u::rwx,d:g::rwx,d:o::rwx,d:m::rwx',
                        *map(str, directories)], check=True)
    config = json.loads((source / 'deployment.json').read_text())
    platform = shared / 'platforms/u55c/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm'
    weights = shared / 'models/Qwen3-0.6B/tiled-bf16-v2'
    reference = shared / 'reference/Qwen3-0.6B/default'
    xclbin = shared / 'artifacts/qwen3-0.6b-stream/qwen_full_decode.xclbin'
    record = {'schema_version': 1, 'model_id': config['model_id'], 'revision': config['revision'],
        'model_snapshot': config['model_snapshot'], 'platform': str(platform),
        'vitis_root': str(shared / 'amd'), 'vitis_settings': str(new_amd / 'Vitis/settings64.sh'),
        'xrt_include': str(shared / 'xrt-dev/usr/include'),
        'xrt_lib': '/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2',
        'packed_weights': str(weights), 'reference': str(reference), 'xclbin': str(xclbin),
        'kernel_source_sha256': hashlib.sha256((latest / 'hls/full_decode.cpp').read_bytes()
            + (latest / 'hls/full_decode.hpp').read_bytes()).hexdigest(),
        'quantized': False, 'source_dtype': 'bfloat16', 'compute_dtype': 'float32',
        'checksums': {}}
    targets = {'model': Path(record['model_snapshot']) / 'model.safetensors',
        'weights': weights / 'weights.bin', 'offsets': weights / 'offsets.bin',
        'inv_freq': weights / 'inv_freq.bin', 'xclbin': xclbin,
        'reference_json': reference / 'reference.json', 'reference_vectors': reference / 'vectors.npz',
        'platform': platform, 'vitis_settings': Path(record['vitis_settings'])}
    targets.update({'download:' + p.name: p for p in sorted((shared / 'downloads').iterdir()) if p.is_file()})
    for label, path in targets.items():
        record['checksums'][label] = {'path': str(path), 'sha256': digest(path)}
    signoff = json.loads((latest / 'reports/signoff.json').read_text())
    record['existing_bitstream_signoff'] = {key: value for key, value in signoff.items()
        if key not in {'timing_report', 'bus_skew_report'}}
    assert record['existing_bitstream_signoff']['xclbin_sha256'] == digest(xclbin)
    manifest = shared / 'resources.json'
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    manifest.chmod(0o666)
    log = {'moved': moved, 'settings_updated': list(map(str, settings)),
           'absolute_links_updated': changed_links, 'regular_files_shared': file_count,
           'private_auth_and_conda_excluded': True}
    (log_dir / 'migration.json').write_text(json.dumps(log, indent=2) + '\n')
    print(json.dumps({'shared_manifest': str(manifest), **log}, indent=2))


if __name__ == '__main__':
    main()
