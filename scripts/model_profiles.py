"""Model identities and isolated build locations; no network or hardware access."""
import hashlib
import json
import math
from pathlib import Path

PROFILES = {
    'Qwen3-0.6B': {'hidden_size': 1024, 'intermediate_size': 3072,
                   'revision': 'c1899de289a04d12100db370d81485cdf75e47ca', 'weight_banks': 4},
    'Qwen3-1.7B': {'hidden_size': 2048, 'intermediate_size': 6144,
                   'revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', 'weight_banks': 8},
}


def canonical_model(value):
    value = value.removeprefix('Qwen/')
    if value in {'0.6B', '1.7B'}:
        value = 'Qwen3-' + value
    if value not in PROFILES:
        raise ValueError('Supported models: ' + ', '.join(PROFILES))
    return value


def geometry(name):
    profile = PROFILES[canonical_model(name)]
    return {'model_type': 'qwen3', 'hidden_size': profile['hidden_size'],
        'intermediate_size': profile['intermediate_size'], 'num_hidden_layers': 28,
        'num_attention_heads': 16, 'num_key_value_heads': 8, 'head_dim': 128,
        'vocab_size': 151936, 'tie_word_embeddings': True}


def expected_shapes(name, config=None):
    g = geometry(name)
    if config is not None:
        for key, value in g.items():
            if config.get(key) != value:
                raise ValueError(f'{name} requires {key}={value}, got {config.get(key)}')
    h, inter = g['hidden_size'], g['intermediate_size']
    shapes = {'model.embed_tokens.weight': [151936, h], 'model.norm.weight': [h]}
    for layer in range(28):
        shapes.update({f'model.layers.{layer}.{key}': shape for key, shape in {
            'input_layernorm.weight': [h], 'post_attention_layernorm.weight': [h],
            'self_attn.q_norm.weight': [128], 'self_attn.k_norm.weight': [128],
            'self_attn.q_proj.weight': [2048, h], 'self_attn.k_proj.weight': [1024, h],
            'self_attn.v_proj.weight': [1024, h], 'self_attn.o_proj.weight': [h, 2048],
            'mlp.gate_proj.weight': [inter, h], 'mlp.up_proj.weight': [inter, h],
            'mlp.down_proj.weight': [h, inter]}.items()})
    return shapes


def weight_bytes(name):
    return 2 * sum(math.prod(shape) for shape in expected_shapes(name).values())


def project_path(root, name):
    name = canonical_model(name)
    base = Path(root) / 'full-decode-stream'
    return base if name == 'Qwen3-0.6B' else base / 'variants' / name


def config_path(root, name):
    name = canonical_model(name)
    return Path(root) / ('deployment.json' if name == 'Qwen3-0.6B' else f'config/models/{name}.local.json')


def load_config(root, name):
    name = canonical_model(name)
    path = config_path(root, name)
    if not path.is_file():
        raise FileNotFoundError(f'{name} is not prepared. Run: python scripts/prepare_model.py --model {name}')
    config = json.loads(path.read_text())
    if config.get('model_id', 'Qwen/' + name) != 'Qwen/' + name:
        raise ValueError('Selected model and deployment configuration differ')
    return config


def source_digest(project):
    p = Path(project) / 'hls'
    return hashlib.sha256((p / 'full_decode.cpp').read_bytes() + (p / 'full_decode.hpp').read_bytes()).hexdigest()


def verify_bitstream_binding(project, config, path):
    with Path(path).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    registered = []
    shared=config.get('shared_resources',{})
    expected_source=config.get('kernel_source_sha256')
    if not expected_source and shared.get('manifest') and Path(shared['manifest']).is_file():
        expected_source=json.loads(Path(shared['manifest']).read_text()).get('kernel_source_sha256')
    if expected_source and expected_source==source_digest(project):
        if config.get('verified_xclbin', {}).get('sha256'):
            registered.append(config['verified_xclbin']['sha256'])
        if shared.get('xclbin_sha256'):
            registered.append(shared['xclbin_sha256'])
    linked = Path(project) / 'reports/linked-source.json'
    if linked.is_file():
        record = json.loads(linked.read_text())
        if record['kernel_source_sha256'] == source_digest(project):
            registered.append(record['xclbin_sha256'])
    if actual not in registered:
        raise ValueError('Bitstream is not registered for the selected model; refusing to load another model\'s FPGA design')
