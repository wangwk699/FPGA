#!/usr/bin/env python3
"""Real FPGA regression for device-only uploads and row tiles; no model weights."""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xclbin", type=Path, default=root / "build/hw/qwen_gemv.xclbin")
    parser.add_argument("--report", type=Path, default=root / "reports/fpga-runtime-probe.json")
    args = parser.parse_args()
    if os.environ.get("XCL_EMULATION_MODE"):
        parser.error("Unset XCL_EMULATION_MODE; this test requires the physical FPGA")
    import torch
    from native_linear import Runtime
    torch.set_num_threads(8)
    config = json.loads((root / "deployment.json").read_text())
    results = []
    runtime = Runtime(root / "build/libqwen_runtime_fpga.so", args.xclbin, config["fpga_device"])
    try:
        # Distinct row values expose incorrect offsets beyond the 4096-row tile.
        for rows, cols in [(4103, 1024), (17, 3072)]:
            r = torch.arange(rows).reshape(-1, 1)
            c = torch.arange(cols).reshape(1, -1)
            weight = (((r * 7 + c * 3) % 31 - 15).float() / 32).to(torch.bfloat16).contiguous()
            source = ((torch.arange(cols) % 17 - 8).float() / 16).contiguous()
            expected = weight.float() @ source
            slot = runtime.add_weight(weight)
            started = time.perf_counter()
            actual = runtime.linear(slot, source, rows)
            error = (actual - expected).abs().max().item()
            passed = bool(torch.isfinite(actual).all() and error <= 1e-4)
            results.append({"rows": rows, "cols": cols, "max_abs_error": error,
                "passed": passed, "elapsed_seconds": time.perf_counter() - started})
            print(json.dumps(results[-1]), flush=True)
            if not passed:
                raise RuntimeError(f"FPGA runtime comparison failed for {rows}x{cols}")
    finally:
        runtime.close()
    report = {"execution": "fpga_synthetic_runtime", "fpga_executed": True,
        "synthetic_weights": True, "device": config["fpga_device"],
        "xclbin": str(args.xclbin.resolve()), "results": results,
        "passed": all(item["passed"] for item in results)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
