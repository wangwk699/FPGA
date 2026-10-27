#!/usr/bin/env python3
"""Publish an already physically verified model without changing its local build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from model_profiles import PROFILES,canonical_model,load_config,project_path,source_digest

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=canonical_model,choices=PROFILES,required=True)
    parser.add_argument('--shared-root',type=Path,default=Path('/home/share/fpga'))
    args=parser.parse_args()
    if args.model=='Qwen3-0.6B':parser.error('The existing 0.6B publication is preserved')
    project=project_path(ROOT,args.model);config=load_config(ROOT,args.model)
    final=json.loads((project/'reports/FINAL-DEPLOYMENT.json').read_text())
    inference=json.loads((project/'reports/full-fpga.json').read_text())
    if (final.get('model_id')!='Qwen/'+args.model or not final.get('full_decoder_on_fpga')
            or final.get('quantized') is not False or not inference.get('fpga_executed')
            or not inference.get('matches_reference_tokens')
            or len(inference.get('numerical_checks',[]))!=5
            or not all(c['passed'] for c in inference['numerical_checks'])):
        raise ValueError('Physical model verification has not passed')
    if source_digest(project)!=final['kernel_source_sha256']:
        raise ValueError('Kernel changed after verification')
    if (inference.get('model_id')!=final['model_id']
            or inference.get('kernel_source_sha256')!=final['kernel_source_sha256']
            or inference.get('xclbin_sha256')!=final['xclbin']['sha256']):
        raise ValueError('Inference evidence belongs to another model or build')
    local_bitstream=project/'build/hw/qwen_full_decode.xclbin'
    if digest(local_bitstream)!=final['xclbin']['sha256']:
        raise ValueError('Bitstream changed after physical verification')
    shared=args.shared_root.expanduser().resolve()
    weights=shared/'models'/args.model/'tiled-bf16-v2'
    reference=shared/'reference'/args.model/'default'
    artifact=shared/'artifacts'/(args.model.lower()+'-stream')
    manifest=shared/f'resources-{args.model}.json'
    for target in [weights,reference,artifact,manifest]:
        if target.exists() or target.is_symlink():raise FileExistsError(target)
    staging=Path(tempfile.mkdtemp(prefix=f'.publish-{args.model}-',dir=shared))
    try:
        shutil.copytree(project/'weights',staging/'weights')
        shutil.copytree(project/'reference/default',staging/'reference')
        (staging/'artifacts').mkdir()
        shutil.copy2(local_bitstream,staging/'artifacts/qwen_full_decode.xclbin')
        packed=json.loads((staging/'weights/manifest.json').read_text())
        for name,sha in packed['files_sha256'].items():
            if digest(staging/'weights'/name)!=sha:raise ValueError('Published weight digest mismatch')
        if digest(staging/'artifacts/qwen_full_decode.xclbin')!=final['xclbin']['sha256']:
            raise ValueError('Published bitstream digest mismatch')
        for target,folder in [(weights,'weights'),(reference,'reference'),(artifact,'artifacts')]:
            target.parent.mkdir(parents=True,exist_ok=True)
            (staging/folder).rename(target)
        record={'schema_version':1,'model_id':'Qwen/'+args.model,'revision':config['revision'],
            'model_snapshot':config['model_snapshot'],'model_geometry':config['model_geometry'],
            'kernel_source_sha256':final['kernel_source_sha256'],'fpga_verified':True,
            'verified_at':final['completed_at'],
            'source_dtype':'bfloat16','compute_dtype':'float32','quantized':False,
            'max_context_tokens':2048,'batch_size':1,'kv_cache_bytes':469762048,
            'packed_weights':str(weights),'reference':str(reference),
            'xclbin':str(artifact/'qwen_full_decode.xclbin'),'xclbin_sha256':final['xclbin']['sha256'],
            'xclbin_uuid':final['xclbin']['uuid'],'checksums':{},
            'signoff':{k:v for k,v in final['signoff'].items() if k not in {'timing_report','bus_skew_report'}},
            'validation':{k:inference[k] for k in ['model_id','fpga_executed','matches_reference_tokens','token_kernel_calls',
                'layer_evaluations','incremental_decode_tokens_per_second','generated_ids','answer']}}
        files={**{'checkpoint:'+name:Path(config['model_snapshot'])/name for name in config['source_files_sha256']},
            **{'weights:'+name:weights/name for name in packed['files_sha256']},
            'reference_json':reference/'reference.json','reference_vectors':reference/'vectors.npz',
            'xclbin':artifact/'qwen_full_decode.xclbin'}
        files.update({'model_metadata:'+p.name:p for p in Path(config['model_snapshot']).iterdir()
                      if p.is_file() and p.suffix in {'.json','.txt'}})
        for label,path in files.items():record['checksums'][label]={'path':str(path),'sha256':digest(path)}
        for folder in [weights,reference,artifact]:
            dirs=[folder]
            for base,children,names in os.walk(folder):
                p=Path(base);p.chmod(0o777)
                dirs.extend(p/name for name in children)
                for name in names:(p/name).chmod(0o666)
            subprocess.run(['setfacl','-m','d:u::rwx,d:g::rwx,d:o::rwx,d:m::rwx',*map(str,dirs)],check=True)
        text=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
        pending=shared/(manifest.name+'.tmp');pending.write_text(text);pending.chmod(0o666)
        pending.rename(manifest)
        public=ROOT/'config/models'/f'{args.model}.shared.example.json'
        public.parent.mkdir(parents=True,exist_ok=True);public.write_text(text)
        print(json.dumps({'model':args.model,'manifest':str(manifest),'fpga_verified':True,
            'original_build_preserved':True,'downloaded':False},indent=2))
    finally:
        shutil.rmtree(staging)


if __name__=='__main__':main()
