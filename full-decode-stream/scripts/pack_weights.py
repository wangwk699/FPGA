#!/usr/bin/env python3
"""Losslessly tile a local original safetensors checkpoint; verify every tensor."""
import argparse,hashlib,json,math,shutil,struct,sys,tempfile
from pathlib import Path
import numpy as np
PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent
sys.path.insert(0, str(ROOT/'scripts'))
from export_weights import expected_shapes, read_header

LAYER_NAMES = ['input_layernorm.weight', 'self_attn.q_proj.weight',
    'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.q_norm.weight',
    'self_attn.k_norm.weight', 'self_attn.o_proj.weight', 'post_attention_layernorm.weight',
    'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify(path):
    m=json.loads((path/'manifest.json').read_text())
    if m['format']!='qwen3-tiled-bf16-v2' or m['quantized']:raise ValueError('Invalid tiled manifest')
    for name,sha in m['files_sha256'].items():
        if digest(path/name)!=sha:raise ValueError('File digest mismatch: '+name)
    image=np.memmap(path/'weights.bin',mode='r',dtype='<u2')
    for record in m['tensors']:
        shape=record['shape']; start=record['offset_elements']; count=record['nbytes']//2
        data=image[start:start+count]
        if len(shape)==2:
            rows,cols=shape
            data=data.reshape(rows//16,cols//32,16,32).transpose(0,2,1,3).copy().reshape(shape)
        if hashlib.sha256(data.tobytes()).hexdigest()!=record['sha256']:
            raise ValueError('Tensor round-trip mismatch: '+record['name'])
    return m

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,help='Local original Qwen3-0.6B snapshot')
    parser.add_argument('--output',type=Path,default=PROJECT/'weights')
    args=parser.parse_args()
    if args.output.exists():verify(args.output);print('All 310 tiled tensors round-trip verified.');return
    deployment=json.loads((ROOT/'deployment.json').read_text()) if (ROOT/'deployment.json').exists() else {}
    if not args.model and not deployment.get('model_snapshot'):
        parser.error('Provide --model or create deployment.json with scripts/configure.py')
    model=(args.model or Path(deployment['model_snapshot'])).expanduser().resolve()
    config=json.loads((model/'config.json').read_text())
    shapes=expected_shapes(config)
    sf=model/'model.safetensors'
    header,payload=read_header(sf)
    order=['model.embed_tokens.weight','model.norm.weight'] + [
        f'model.layers.{layer}.{name}' for layer in range(28) for name in LAYER_NAMES]
    if set(header)-{'__metadata__','lm_head.weight'} != set(order):
        raise ValueError('Expected exactly 310 independent original Qwen3-0.6B tensors')
    for name in order:
        if header[name]['dtype']!='BF16' or header[name]['shape']!=shapes[name]:
            raise ValueError('Expected original BF16 shape: '+name)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='tiled-weights-',dir=args.output.parent))
    try:
        source=np.memmap(sf,mode='r',dtype='<u2',offset=payload)
        if 'lm_head.weight' in header:
            head=header['lm_head.weight']; embedding=header['model.embed_tokens.weight']
            if any(head[key]!=embedding[key] for key in ['dtype','shape']):
                raise ValueError('Tied lm_head has a different shape or dtype')
            def raw_digest(item):
                begin,end=item['data_offsets']
                return hashlib.sha256(source[begin//2:end//2]).digest()
            if raw_digest(head)!=raw_digest(embedding):
                raise ValueError('Tied lm_head differs from the embedding; refusing deduplication')
        records=[]
        with (tmp/'weights.bin').open('wb') as out:
            for name in order:
                item=header[name];begin,end=item['data_offsets']
                data=source[begin//2:end//2]
                raw_sha=hashlib.sha256(data.tobytes()).hexdigest()
                out.write(b'\0'*((-out.tell())%64))
                rec={'name':name,'offset_elements':out.tell()//2,'nbytes':end-begin,
                     'shape':item['shape'],'dtype':'BF16','sha256':raw_sha}
                if len(item['shape'])==2:
                    rows,cols=item['shape']
                    if rows%16 or cols%32:raise ValueError('Unsupported tile shape')
                    data=data.reshape(rows//16,16,cols//32,32).transpose(0,2,1,3).copy()
                data.tofile(out)
                records.append(rec)
        if (tmp/'weights.bin').stat().st_size!=1192099840:raise ValueError('Unexpected image size')
        (tmp/'offsets.bin').write_bytes(struct.pack('<310Q',*(x['offset_elements'] for x in records)))
        import torch
        exponents=torch.arange(0,128,2,dtype=torch.float32)/128
        (tmp/'inv_freq.bin').write_bytes((1.0/(float(config['rope_theta'])**exponents)).numpy().tobytes())
        m={'format':'qwen3-tiled-bf16-v2','matrix_layout':'row_tiles_16_column_blocks_32',
           'quantized':False,'source_model':str(model),'source_model_sha256':digest(sf),
           'revision':deployment.get('revision','c1899de289a04d12100db370d81485cdf75e47ca'),
           'weight_bytes':1192099840,'cache_bytes':469762048,'source_dtype':'bfloat16',
           'compute_dtype':'float32','max_context_tokens':2048,'config':config,'tensors':records}
        m['files_sha256']={name:digest(tmp/name) for name in ['weights.bin','offsets.bin','inv_freq.bin']}
        (tmp/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
        verify(tmp);tmp.rename(args.output)
        print('Packed 310 tensors without quantization; every reconstructed original tensor SHA256 matches.')
    except BaseException:shutil.rmtree(tmp);raise
if __name__=='__main__':main()
