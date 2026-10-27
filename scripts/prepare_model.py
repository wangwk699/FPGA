#!/usr/bin/env python3
"""Prepare a model-specific build from existing local files, without downloading."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shlex
import subprocess
import sys

from model_profiles import PROFILES, canonical_model, config_path, expected_shapes, project_path, source_digest, weight_bytes
from export_weights import read_header

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=canonical_model, choices=PROFILES, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--shared',action='store_true',help='Use the published and physically verified shared model artifacts; build only local host libraries')
    args = parser.parse_args()
    if args.model == 'Qwen3-0.6B':
        parser.error('Use scripts/use_shared_resources.py for the existing 0.6B deployment')
    destination = config_path(ROOT, args.model)
    project = project_path(ROOT, args.model)
    if destination.exists() or project.exists():
        parser.error('Model workspace already exists; existing build/state is preserved')
    if not (ROOT/'deployment.json').is_file():
        parser.error('Prepare this user\'s shared base dependencies first: python scripts/use_shared_resources.py')
    base_deployment = json.loads((ROOT / 'deployment.json').read_text())
    profile = PROFILES[args.model]
    shared=None
    if args.shared:
        manifest=Path('/home/share/fpga')/f'resources-{args.model}.json'
        expected=ROOT/'config/models'/f'{args.model}.shared.example.json'
        shared=json.loads(manifest.read_text())
        published=json.loads(expected.read_text())
        for key in ['schema_version','model_id','revision','model_snapshot','model_geometry','packed_weights',
                    'reference','xclbin','kernel_source_sha256','xclbin_sha256','fpga_verified','quantized']:
            if shared[key]!=published[key]:raise ValueError('Shared model identity differs: '+key)
        if shared['model_id']!='Qwen/'+args.model or shared['fpga_verified'] is not True:
            raise ValueError('Shared model has not passed physical FPGA verification')
        for label,item in published['checksums'].items():
            entry=shared['checksums'][label]
            with Path(entry['path']).open('rb') as stream:
                actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if entry['sha256']!=item['sha256'] or actual!=item['sha256']:
                raise ValueError('Shared model resource SHA256 mismatch: '+label)
        args.snapshot=Path(shared['model_snapshot'])
    snapshot = (args.snapshot or Path('/home/share/hub') / ('models--Qwen--' + args.model)
                / 'snapshots' / profile['revision']).expanduser().resolve()
    config = json.loads((snapshot / 'config.json').read_text())
    shapes = expected_shapes(args.model, config)
    tensors, hashes = {}, {}
    for file in sorted(snapshot.glob('*.safetensors')):
        header, _ = read_header(file)
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            if name in tensors:
                raise ValueError('Duplicate source tensor: ' + name)
            tensors[name] = tensor
        with file.open('rb') as stream:
            hashes[file.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if set(tensors) - {'lm_head.weight'} != set(shapes):
        raise ValueError('Missing/unexpected checkpoint tensors')
    for name, shape in shapes.items():
        if tensors[name]['shape'] != shape or tensors[name]['dtype'] != 'BF16':
            raise ValueError('Expected original BF16 tensor: ' + name)
    for name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json']:
        if not (snapshot / name).is_file():
            raise FileNotFoundError(snapshot / name)
    base = ROOT / 'full-decode-stream'
    for folder in ['hls', 'host', 'build', 'reports']:
        (project / folder).mkdir(parents=True, exist_ok=True)
    (project / 'hls/full_decode.cpp').write_bytes((base / 'hls/full_decode.cpp').read_bytes())
    header = (base / 'hls/full_decode.hpp').read_text()
    for key, value in [('HIDDEN', profile['hidden_size']), ('INTERMEDIATE', profile['intermediate_size'])]:
        header, count = re.subn(rf'constexpr int {key} = \d+;', f'constexpr int {key} = {value};', header)
        if count != 1:
            raise ValueError('Unexpected base kernel declaration: ' + key)
    (project / 'hls/full_decode.hpp').write_text(header)
    host = (base / 'host/full_runtime.cpp').read_text()
    if host.count('WEIGHT_BYTES = 1192099840;') != 1:
        raise ValueError('Unexpected host weight allocation declaration')
    host = host.replace('WEIGHT_BYTES = 1192099840;', f'WEIGHT_BYTES = {weight_bytes(args.model)};')
    host = host.replace('Qwen3-0.6B', args.model)
    host += '\nextern "C" std::uint64_t qwen_full_model_signature() { return (std::uint64_t(HIDDEN) << 32) | INTERMEDIATE; }\n'
    (project / 'host/full_runtime.cpp').write_text(host)
    cfg = (base / 'hls/connectivity.cfg').read_text().replace('HBM[0:3]', 'HBM[0:7]')
    cfg = cfg.replace('HBM[4]', 'HBM[8]').replace('HBM[5]', 'HBM[9]').replace('HBM[6]', 'HBM[10]')
    (project / 'hls/connectivity.cfg').write_text(cfg)
    make = (base / 'Makefile').read_text().replace('ROOT := $(abspath ..)', f'ROOT := {ROOT}')
    (project / 'Makefile').write_text(make)
    wrapper = project / 'run_full.sh'
    wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '
        + shlex.quote(str(base / 'run_full.sh')) + ' --model ' + shlex.quote(args.model) + ' "$@"\n')
    wrapper.chmod(0o755)
    deployment={key:base_deployment[key] for key in ['fpga_device','platform','vitis_settings','xrt_include',
        'xrt_lib','tool_installation','batch_size','max_context_tokens','source_dtype','compute_dtype','quantized']
        if key in base_deployment}
    checkpoint_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    revision=snapshot.name if re.fullmatch(r'[0-9a-f]{40}',snapshot.name) else 'local-'+checkpoint_hash
    if shared and revision!=shared['revision']:
        raise ValueError('Shared revision and actual local checkpoint identity differ')
    deployment.update(model_id='Qwen/' + args.model, revision=revision,
        model_snapshot=str(snapshot), python=sys.executable, source_files_sha256=hashes,
        model_file_sha256=checkpoint_hash,
        model_hash_kind='sha256_of_sorted_shard_sha256_json',
        fpga_deployment_complete=False, full_decoder_on_fpga=False,
        model_geometry=config, weight_bytes=weight_bytes(args.model),
        kernel_source_sha256=source_digest(project), generated_project=str(project))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(deployment, indent=2) + '\n')
    if shared:
        if source_digest(project)!=shared['kernel_source_sha256']:
            raise ValueError('Generated kernel differs from the validated shared model')
        for relative,key in [('weights','packed_weights'),('reference/default','reference')]:
            target=project/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(shared[key],target_is_directory=True)
        deployment['shared_resources']={'manifest':str(manifest),'xclbin':shared['xclbin'],
            'xclbin_sha256':shared['xclbin_sha256']}
        destination.write_text(json.dumps(deployment,indent=2)+'\n')
        subprocess.run(['make','-C',str(project),'runtime'],check=True)
    print(json.dumps({'model': args.model, 'snapshot': str(snapshot), 'project': str(project),
        'config': str(destination), 'weight_bytes': weight_bytes(args.model),
        'hbm_weight_capacity_bytes': profile['weight_banks'] * 512 * 1024**2,
        'downloaded': False, 'fpga_verified': False}, indent=2))


if __name__ == '__main__':
    main()
