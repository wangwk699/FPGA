#!/usr/bin/env python3
"""Compare native HLS source (C++ execution) to all Qwen3 linear layers' real vectors.

This is software validation, not HLS RTL simulation and not FPGA execution.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=Path("build/libqwen_gemv_csim.so"))
    parser.add_argument("--output", type=Path, default=Path("reports/real-gemv.json"))
    args = parser.parse_args()
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_num_threads(8)
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    vectors = reference.get("linear_vectors")
    if not vectors or len(vectors) != 197:
        raise ValueError("Expected 196 decoder linear layers plus lm_head; regenerate reference")
    library_path = args.library.resolve()
    library = ctypes.CDLL(str(library_path))
    gemv = library.qwen_gemv
    u16ptr = ctypes.POINTER(ctypes.c_uint16)
    f32ptr = ctypes.POINTER(ctypes.c_float)
    gemv.argtypes = [u16ptr, f32ptr, f32ptr, ctypes.c_int, ctypes.c_int]
    gemv.restype = None
    results = []
    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as weights:
        for name, vector in vectors.items():
            key = "model.embed_tokens.weight" if name == "lm_head" else name + ".weight"
            weight = weights.get_tensor(key)
            if weight.dtype != torch.bfloat16:
                raise ValueError(f"{key} is {weight.dtype}, expected original BF16")
            rows, cols = weight.shape
            if cols > 3072:
                raise ValueError(f"Kernel column limit exceeded by {key}")
            # Bit view only: no conversion or quantization.
            bits = weight.contiguous().view(torch.int16).numpy().view(np.uint16)
            x = vector["input"].float().contiguous().numpy()
            expected = vector["output"].float().contiguous().numpy()
            if x.size != cols or expected.size != rows:
                raise ValueError(f"Mismatched reference shape for {key}")
            actual = np.empty(rows, dtype=np.float32)
            for start in range(0, rows, 4096):
                count = min(rows - start, 4096)
                gemv(bits[start:].ctypes.data_as(u16ptr), x.ctypes.data_as(f32ptr),
                     actual[start:].ctypes.data_as(f32ptr), count, cols)
            difference = actual.astype(np.float64) - expected.astype(np.float64)
            max_abs = float(np.max(np.abs(difference)))
            relative_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-12))
            ok = bool(np.all(np.isfinite(actual)) and max_abs <= 0.005 and relative_l2 <= 1e-4)
            results.append({"name": name, "shape": [rows, cols], "max_abs_error": max_abs,
                            "relative_l2_error": relative_l2, "passed": ok})
    failed = [item["name"] for item in results if not item["passed"]]
    report = {
        "execution": "native_cpp_software_only", "fpga_executed": False,
        "quantized": False, "model": str(args.model.resolve()),
        "library_sha256": hashlib.file_digest(library_path.open("rb"), "sha256").hexdigest(),
        "cases": len(results), "failed": failed,
        "max_abs_error": max(item["max_abs_error"] for item in results),
        "max_relative_l2_error": max(item["relative_l2_error"] for item in results),
        "thresholds": {"max_abs_error": 0.005, "relative_l2_error": 1e-4},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(results) - len(failed)}/{len(results)} real-weight GEMV cases passed")
    print(f"max_abs_error={report['max_abs_error']:.8g}, max_relative_l2_error={report['max_relative_l2_error']:.8g}")
    if failed:
        raise SystemExit(f"Failed: {failed}")


if __name__ == "__main__":
    main()
