#!/usr/bin/env python3
"""Build a lossless BF16 image and fixed tensor descriptors from local files."""
import argparse
import hashlib
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / "scripts"))
from export_weights import expected_shapes, read_header

LAYER_NAMES = ["input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "self_attn.o_proj.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
    "mlp.up_proj.weight", "mlp.down_proj.weight"]


def tensor_order():
    return ["model.embed_tokens.weight", "model.norm.weight"] + [
        f"model.layers.{layer}.{name}" for layer in range(28) for name in LAYER_NAMES]


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["quantized"] is False and manifest["format"] == "qwen3-full-bf16-v1"
    for name, digest in manifest["files_sha256"].items():
        with (directory / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Packed file checksum mismatch: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "weights")
    args = parser.parse_args()
    if args.output.exists():
        verify(args.output)
        print("Existing lossless full-decoder image verified.")
        return
    deployment = json.loads((ROOT / "deployment.json").read_text())
    model = Path(deployment["model_snapshot"])
    config = json.loads((model / "config.json").read_text())
    shapes = expected_shapes(config)
    source = model / "model.safetensors"
    header, payload = read_header(source)
    previous = json.loads((ROOT / "weights/qwen3-0.6b/manifest.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="full-weights-", dir=args.output.parent))
    try:
        records = []
        with source.open("rb") as stream, (temporary / "weights.bin").open("wb") as output:
            for name in tensor_order():
                item = header[name]
                if item["dtype"] != "BF16" or item["shape"] != shapes[name]:
                    raise ValueError(f"Expected original BF16 tensor: {name}")
                padding = (-output.tell()) % 64
                output.write(b"\0" * padding)
                offset = output.tell() // 2
                begin, end = item["data_offsets"]
                stream.seek(payload + begin)
                digest = hashlib.sha256()
                remaining = end - begin
                while remaining:
                    data = stream.read(min(1024 * 1024, remaining))
                    if not data:
                        raise ValueError(f"Truncated tensor: {name}")
                    output.write(data)
                    digest.update(data)
                    remaining -= len(data)
                if digest.hexdigest() != previous["tensors"][name]["sha256"]:
                    raise ValueError(f"Tensor differs from validated original weights: {name}")
                records.append({"name": name, "offset_elements": offset, "shape": item["shape"],
                    "dtype": "BF16", "sha256": digest.hexdigest(), "nbytes": end - begin})
        if (temporary / "weights.bin").stat().st_size != 1192099840:
            raise ValueError("Unexpected weight image size")
        (temporary / "offsets.bin").write_bytes(struct.pack("<310Q", *(x["offset_elements"] for x in records)))
        # Constant model metadata, initialized exactly as Transformers default
        # Qwen3 RoPE. Runtime angle multiplication/sin/cos/rotation are on FPGA.
        import torch
        exponents = torch.arange(0, 128, 2, dtype=torch.float32) / 128
        inv_freq = 1.0 / (1000000.0 ** exponents)
        (temporary / "inv_freq.bin").write_bytes(inv_freq.numpy().tobytes())
        hashes = {}
        for name in ["weights.bin", "offsets.bin", "inv_freq.bin"]:
            with (temporary / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {"format": "qwen3-full-bf16-v1", "quantized": False,
            "revision": deployment["revision"], "source_model": str(model),
            "source_model_sha256": deployment["model_file_sha256"],
            "weight_bytes": 1192099840, "cache_bytes": 28 * 2 * 2048 * 1024 * 4,
            "source_dtype": "bfloat16", "compute_dtype": "float32", "max_context_tokens": 2048,
            "config": config, "tensors": records, "files_sha256": hashes}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(temporary)
        temporary.rename(args.output)
        print(f"Packed {len(records)} original BF16 tensors without quantization: {args.output}")
    except BaseException:
        shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
