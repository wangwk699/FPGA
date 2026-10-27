#!/usr/bin/env python3
"""Use already-installed resources on this shared server; never download/install."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-root', type=Path, default=Path('/home/share/fpga'))
    parser.add_argument('--device', default='0000:01:00.1')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    shared = args.shared_root.expanduser().resolve()
    record = json.loads((shared / 'resources.json').read_text())
    expected = json.loads((ROOT / 'config/shared-server.example.json').read_text())
    for key in ['schema_version', 'model_id', 'revision', 'kernel_source_sha256', 'quantized']:
        if record[key] != expected[key]:
            raise ValueError('Shared resource identity differs from the published record: ' + key)
    source_hash = hashlib.sha256((ROOT / 'full-decode-stream/hls/full_decode.cpp').read_bytes()
        + (ROOT / 'full-decode-stream/hls/full_decode.hpp').read_bytes()).hexdigest()
    if source_hash != record['kernel_source_sha256']:
        raise ValueError('Local kernel differs from the validated shared bitstream')
    for label, item in expected['checksums'].items():
        actual = record['checksums'][label]
        if actual['sha256'] != item['sha256'] or digest(Path(actual['path'])) != item['sha256']:
            raise ValueError('Shared resource SHA256 mismatch: ' + label)
    for key in ['vitis_settings', 'platform', 'xrt_lib']:
        if not Path(record[key]).is_file():
            raise FileNotFoundError(record[key])
    if not (Path(record['xrt_include']) / 'xrt/xrt_device.h').is_file():
        raise FileNotFoundError(record['xrt_include'])
    print('Shared model, tools, platform, weights, reference and bitstream verified.', flush=True)
    if args.check_only:
        return
    links = [(ROOT / 'full-decode-stream/weights', Path(record['packed_weights'])),
             (ROOT / 'full-decode-stream/reference/default', Path(record['reference']))]
    for target, data in links:
        if (target.exists() or target.is_symlink()) and target.resolve() != data.resolve():
            raise FileExistsError(f'Existing local data is preserved: {target}')
    subprocess.run([sys.executable, str(ROOT / 'scripts/configure.py'),
        '--model', record['model_snapshot'], '--python', sys.executable,
        '--device', args.device, '--platform', record['platform'],
        '--vitis-root', record['vitis_root'], '--vitis-settings', record['vitis_settings'],
        '--xrt-include', record['xrt_include'], '--xrt-lib', record['xrt_lib']], check=True)
    for target, data in links:
        if not target.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(data, target_is_directory=True)
    config_path = ROOT / 'deployment.json'
    config = json.loads(config_path.read_text())
    config['shared_resources'] = {'manifest': str(shared / 'resources.json'),
        'xclbin': record['xclbin'], 'xclbin_sha256': record['checksums']['xclbin']['sha256']}
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    print('Created only this user\'s local configuration and data links; no downloads or Conda creation.')
    print('Shared xclbin:', record['xclbin'])


if __name__ == '__main__':
    main()
