import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from scripts.export_weights import export, verify, read_header, expected_shapes


def safetensors(path, tensors):
    header, payload = {}, bytearray()
    for name, dtype, shape, data in tensors:
        begin = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
        self.out = self.root / "weights"
        # Includes signed zero, subnormal, normal and large BF16 values; copied bit-for-bit.
        self.raw = struct.pack("<8H", 0, 0x8000, 1, 0x3f80, 0xbf80, 0x7f7f, 0x0080, 0x3fc0)
        safetensors(self.model / "model.safetensors", [
            ("model.embed_tokens.weight", "BF16", [2, 4], self.raw),
            ("model.norm.weight", "F32", [4], struct.pack("<4f", 1, 2, 3, 4)),
        ])

    def tearDown(self):
        self.temporary.cleanup()

    def test_lossless_and_bank_splitting(self):
        result = export(self.model, self.out, shard_bytes=64, validate_model=False)
        a = result["tensors"]["model.embed_tokens.weight"]
        b = result["tensors"]["model.norm.weight"]
        self.assertNotEqual(a["file"], b["file"])
        data = (self.out / a["file"]).read_bytes()[a["offset"]:a["offset"] + a["nbytes"]]
        self.assertEqual(data, self.raw)
        self.assertEqual(a["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(result["aliases"]["lm_head.weight"], "model.embed_tokens.weight")
        self.assertFalse(result["quantized"])
        verify(self.out)

    def test_corruption_detected(self):
        result = export(self.model, self.out, validate_model=False)
        item = next(iter(result["tensors"].values()))
        path = self.out / item["file"]
        data = bytearray(path.read_bytes())
        data[item["offset"]] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            verify(self.out)

    def test_refuses_overwrite(self):
        self.out.mkdir()
        marker = self.out / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            export(self.model, self.out, validate_model=False)
        self.assertEqual(marker.read_text(), "keep")

    def test_no_network_fallback(self):
        (self.model / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "No local"):
            export(self.model, self.out, validate_model=False)

    def test_rejects_integer_weights(self):
        safetensors(self.model / "model.safetensors", [("w", "I8", [2], b"\x01\x02")])
        with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
            export(self.model, self.out, validate_model=False)

    def test_rejects_truncated_data(self):
        path = self.model / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            read_header(path)

    def test_rejects_oversize_tensor_and_cleans_partial_output(self):
        with self.assertRaisesRegex(ValueError, "exceeds shard size"):
            export(self.model, self.out, shard_bytes=8, validate_model=False)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob("weights.tmp-*")), [])

    def test_rejects_inconsistent_tied_head(self):
        safetensors(self.model / "model.safetensors", [
            ("model.embed_tokens.weight", "BF16", [2, 4], self.raw),
            ("lm_head.weight", "BF16", [2, 4], b"\x00" * 16),
        ])
        with self.assertRaisesRegex(ValueError, "different data"):
            export(self.model, self.out, validate_model=False)
        self.assertFalse(self.out.exists())

    def test_deduplicates_explicit_tied_head(self):
        safetensors(self.model / "model.safetensors", [
            ("model.embed_tokens.weight", "BF16", [2, 4], self.raw),
            ("lm_head.weight", "BF16", [2, 4], self.raw),
        ])
        result = export(self.model, self.out, validate_model=False)
        self.assertNotIn("lm_head.weight", result["tensors"])
        self.assertEqual(result["aliases"]["lm_head.weight"], "model.embed_tokens.weight")

    def test_qwen_dimensions_use_explicit_head_dim(self):
        config = {"model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": True}
        shapes = expected_shapes(config)
        self.assertEqual(shapes["model.layers.0.self_attn.q_proj.weight"], [2048, 1024])
        self.assertEqual(shapes["model.layers.27.self_attn.o_proj.weight"], [1024, 2048])
        config["head_dim"] = 64
        with self.assertRaisesRegex(ValueError, "head_dim"):
            expected_shapes(config)


if __name__ == "__main__":
    unittest.main()
