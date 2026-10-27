#!/usr/bin/env python3
"""Offline, lossless safetensors export. Python standard library only; no downloads."""
import argparse
import hashlib
import json
import math
import shutil
import struct
import tempfile
from pathlib import Path


def expected_shapes(config):
    expected = {
        "model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
        "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Expected Qwen3-0.6B {key}={value}, got {config.get(key)}")
    shapes = {"model.embed_tokens.weight": [151936, 1024], "model.norm.weight": [1024]}
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        shapes.update({prefix + name: shape for name, shape in {
            "input_layernorm.weight": [1024], "post_attention_layernorm.weight": [1024],
            "self_attn.q_norm.weight": [128], "self_attn.k_norm.weight": [128],
            "self_attn.q_proj.weight": [2048, 1024],
            "self_attn.k_proj.weight": [1024, 1024],
            "self_attn.v_proj.weight": [1024, 1024],
            "self_attn.o_proj.weight": [1024, 2048],
            "mlp.gate_proj.weight": [3072, 1024],
            "mlp.up_proj.weight": [3072, 1024],
            "mlp.down_proj.weight": [1024, 3072],
        }.items()})
    return shapes


def read_header(path):
    with path.open("rb") as f:
        size_bytes = f.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Truncated safetensors: {path}")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if not 2 <= header_size <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError(f"Invalid header size: {path}")
        header = json.loads(f.read(header_size))
    payload_start = 8 + header_size
    intervals = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        dtype = item["dtype"]
        if dtype not in {"BF16", "F32"}:
            raise ValueError(f"Unsupported dtype {dtype} in {name}; no implicit conversion")
        shape = item["shape"]
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f"Invalid shape for {name}")
        begin, end = item["data_offsets"]
        if type(begin) is not int or type(end) is not int:
            raise ValueError(f"Invalid offsets for {name}")
        if begin < 0 or end - begin != math.prod(shape) * (2 if dtype == "BF16" else 4):
            raise ValueError(f"Invalid data size for {name}")
        if payload_start + end > path.stat().st_size:
            raise ValueError(f"Truncated tensor {name}")
        intervals.append((begin, end, name))
    intervals.sort()
    for a, b in zip(intervals, intervals[1:]):
        if a[1] > b[0]:
            raise ValueError(f"Overlapping tensors: {a[2]}, {b[2]}")
    return header, payload_start


def export(model_dir, output, shard_bytes=480 * 1024**2, validate_model=True):
    model_dir, output = Path(model_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if shard_bytes <= 0:
        raise ValueError("shard size must be positive")
    config = json.loads((model_dir / "config.json").read_text())
    records = {}
    # All source paths are local. No Hub APIs or network fallbacks.
    for path in sorted(model_dir.glob("*.safetensors")):
        header, payload_start = read_header(path)
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if name in records:
                raise ValueError(f"Duplicate tensor {name}")
            records[name] = (path, payload_start, item)
    if not records:
        raise ValueError("No local .safetensors files found")
    if validate_model:
        shapes = expected_shapes(config)
        for name, shape in shapes.items():
            if name not in records or records[name][2]["shape"] != shape:
                raise ValueError(f"Missing or incorrect tensor {name}; expected {shape}")
        extras = set(records) - set(shapes) - {"lm_head.weight"}
        if extras:
            raise ValueError(f"Unexpected tensors: {sorted(extras)}")
    if config.get("tie_word_embeddings") and "lm_head.weight" in records:
        embedding = records.get("model.embed_tokens.weight")
        head = records["lm_head.weight"]
        if embedding is None or any(embedding[2][k] != head[2][k] for k in ["shape", "dtype"]):
            raise ValueError("Tied embedding and lm_head have different data")
        def source_hash(record):
            path, base, item = record
            start, end = item["data_offsets"]
            digest = hashlib.sha256()
            with path.open("rb") as f:
                f.seek(base + start)
                remaining = end - start
                while remaining:
                    data = f.read(min(4 * 1024**2, remaining))
                    if not data:
                        raise ValueError("Truncated tied weights")
                    digest.update(data)
                    remaining -= len(data)
            return digest.digest()
        if source_hash(embedding) != source_hash(head):
            raise ValueError("Tied embedding and lm_head have different data")
        del records["lm_head.weight"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        manifest = {"format_version": 1, "byte_order": "little", "layout": "row_major",
                    "quantized": False, "shard_limit_bytes": shard_bytes,
                    "config": config, "tensors": {}, "aliases": {}}
        shard = -1
        stream = None
        offset = 0
        try:
            for name, (path, base, item) in sorted(records.items()):
                begin, end = item["data_offsets"]
                length = end - begin
                if length > shard_bytes:
                    raise ValueError(f"Tensor {name} exceeds shard size")
                aligned = (offset + 63) // 64 * 64
                if stream is None or aligned + length > shard_bytes:
                    if stream:
                        stream.close()
                    shard += 1
                    filename = f"shard_{shard:02d}.bin"
                    stream = (temporary / filename).open("wb")
                    aligned = offset = 0
                stream.write(b"\0" * (aligned - offset))
                sha = hashlib.sha256()
                with path.open("rb") as source:
                    source.seek(base + begin)
                    remaining = length
                    while remaining:
                        data = source.read(min(4 * 1024**2, remaining))
                        if not data:
                            raise ValueError(f"Unexpected EOF reading {name}")
                        stream.write(data)
                        sha.update(data)
                        remaining -= len(data)
                manifest["tensors"][name] = {"shape": item["shape"], "dtype": item["dtype"],
                    "file": filename, "offset": aligned, "nbytes": length, "sha256": sha.hexdigest()}
                offset = aligned + length
        finally:
            if stream:
                stream.close()
        if config.get("tie_word_embeddings") and "model.embed_tokens.weight" in records:
            target = manifest["tensors"]["model.embed_tokens.weight"]
            head = manifest["tensors"].get("lm_head.weight")
            if head and any(head[k] != target[k] for k in ["shape", "dtype", "sha256"]):
                raise ValueError("Tied embedding and lm_head have different data")
            if head is None:
                manifest["aliases"]["lm_head.weight"] = "model.embed_tokens.weight"
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Verify every exported tensor by rereading bytes, including shard offsets.
        verify(temporary)
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    for name, item in manifest["tensors"].items():
        path = output / item["file"]
        if path.resolve().parent != output.resolve():
            raise ValueError("Shard path must stay inside the output directory")
        sha = hashlib.sha256()
        with path.open("rb") as f:
            f.seek(item["offset"])
            remaining = item["nbytes"]
            while remaining:
                data = f.read(min(4 * 1024**2, remaining))
                if not data:
                    raise ValueError(f"Truncated exported tensor {name}")
                sha.update(data)
                remaining -= len(data)
        if sha.hexdigest() != item["sha256"]:
            raise ValueError(f"Hash mismatch for {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output)
    else:
        if args.model is None:
            parser.error("--model is required for export")
        result = export(args.model, args.output)
        print(f"Exported {len(result['tensors'])} tensors without quantization")
