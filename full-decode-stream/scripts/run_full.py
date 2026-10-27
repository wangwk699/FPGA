#!/usr/bin/env python3
"""Complete token forward passes on FPGA; CPU only loads/tokenizes/controls.

Optional numerical checking reads device diagnostics but never supplies model
activations, KV entries, logits or token decisions to the inference algorithm.
The cpu-cpp backend is explicitly a software reference, not FPGA execution.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fpga", "cpu-cpp"], default="fpga")
    parser.add_argument("--prompt", default="请用一句话介绍杭州。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--weights", type=Path, default=PROJECT / "weights")
    parser.add_argument("--xclbin", type=Path, help="Local bitstream, or the configured shared bitstream when no local build exists")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    deployment = json.loads((PROJECT.parent / "deployment.json").read_text())
    shared = deployment.get("shared_resources", {})
    if args.xclbin is None:
        local = PROJECT / "build/hw/qwen_full_decode.xclbin"
        args.xclbin = local if local.is_file() else Path(shared.get("xclbin", str(local)))
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.backend == "fpga" and (not args.xclbin.is_file() or os.environ.get("XCL_EMULATION_MODE")):
        parser.error("A real full-decoder xclbin is required; unset XCL_EMULATION_MODE")
    if (args.backend == "fpga" and shared.get("xclbin")
            and args.xclbin.resolve() == Path(shared["xclbin"]).resolve()
            and hashlib.sha256(args.xclbin.read_bytes()).hexdigest() != shared["xclbin_sha256"]):
        parser.error("Shared bitstream SHA256 differs from the configured verified artifact")
    from transformers import AutoTokenizer
    from full_runtime import FullRuntime
    tokenizer = AutoTokenizer.from_pretrained(deployment["model_snapshot"], local_files_only=True)
    text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text)["input_ids"]
    if not ids or len(ids) + args.max_new_tokens > 2048:
        parser.error("Input plus output must fit the 2048-token context")
    eos_value = json.loads((Path(deployment["model_snapshot"]) / "generation_config.json").read_text())["eos_token_id"]
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    reference, vectors = None, None
    if args.verify:
        import numpy as np
        reference = json.loads((args.verify / "reference.json").read_text())
        if reference["input_ids"] != ids or reference["revision"] != deployment["revision"]:
            parser.error("Reference prompt/model differs from the inference input")
        with np.load(args.verify / "vectors.npz") as arrays:
            vectors = {key: arrays[key].copy() for key in arrays.files}
    checks = []
    def compare(name, actual):
        expected = vectors[name]
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        maximum = float(np.max(np.abs(difference)))
        relative = float(np.linalg.norm(difference) / max(float(np.linalg.norm(expected.astype(np.float64))), 1e-12))
        # Residual streams contain real outliers above 8000. Use elementwise
        # absolute + relative tolerance, plus a separate L2 check for every
        # layer, so a large residual cannot conceal errors in later RMSNorms.
        scaled = np.abs(difference) / (0.005 + 1e-4 * np.abs(expected.astype(np.float64)))
        passed = bool(np.isfinite(actual).all() and np.max(scaled) <= 1 and relative <= 1e-4)
        entry = {"name": name, "max_abs_error": maximum, "relative_l2_error": relative,
                 "max_scaled_error": float(np.max(scaled)), "atol": 0.005, "rtol": 1e-4}
        if "trace" in name:
            entry["per_slot_max_abs_error"] = np.max(np.abs(difference), axis=1).tolist()
            per_slot_relative = np.linalg.norm(difference, axis=1) / np.maximum(
                np.linalg.norm(expected.astype(np.float64), axis=1), 1e-12)
            entry["per_slot_relative_l2_error"] = per_slot_relative.tolist()
            passed = passed and bool(np.max(per_slot_relative) <= 1e-4)
        entry["passed"] = passed
        checks.append(entry)
        print(f"Numerical check {name}: max_abs={maximum:.8g}, relative_l2={relative:.8g}", file=sys.stderr, flush=True)
        if not passed:
            raise RuntimeError(f"Full decoder numerical comparison failed: {entry}")
    start = time.perf_counter()
    print(f"Loading complete model into {args.backend} runtime.", file=sys.stderr, flush=True)
    runtime = FullRuntime(args.backend, args.weights, args.xclbin, deployment["fpga_device"])
    load_seconds = time.perf_counter() - start
    diagnostic_reads = 0
    try:
        generated = []
        for position, token in enumerate(ids):
            last = position == len(ids) - 1
            capture = vectors is not None and position in {0, len(ids) - 1}
            next_id = runtime.step(token, emit_logits=last, capture_trace=capture)
            if capture:
                compare("prefill_trace_first" if position == 0 else "prefill_trace_last", runtime.trace())
                diagnostic_reads += 1
            if last:
                generated.append(next_id)
                if vectors is not None:
                    compare("prefill_logits", runtime.logits())
                    diagnostic_reads += 1
            print(f"{args.backend} full prefill {position + 1}/{len(ids)} complete.", file=sys.stderr, flush=True)
        prefill_seconds = time.perf_counter() - start
        decode_start = time.perf_counter()
        for step in range(args.max_new_tokens - 1):
            if generated[-1] in eos:
                break
            capture = vectors is not None and step == 0
            next_id = runtime.step(generated[-1], emit_logits=True, capture_trace=capture)
            if capture:
                compare("first_decode_trace", runtime.trace())
                compare("first_decode_logits", runtime.logits())
                diagnostic_reads += 2
            generated.append(next_id)
            print(f"{args.backend} full decode {step + 1}/{args.max_new_tokens - 1} complete; "
                  f"token kernel calls: {runtime.calls}.", file=sys.stderr, flush=True)
        calls = runtime.calls
        decode_seconds = time.perf_counter() - decode_start
    finally:
        runtime.close()
    matches = None if reference is None else generated == reference["generated_ids"][:args.max_new_tokens]
    report = {"backend": args.backend, "fpga_executed": args.backend == "fpga" and calls > 0,
        "full_decoder_on_fpga": args.backend == "fpga", "quantized": False,
        "source_dtype": "bfloat16", "compute_dtype": "float32", "layers": 28,
        "fpga_device": deployment["fpga_device"] if args.backend == "fpga" else None,
        "xclbin": str(args.xclbin.resolve()) if args.backend == "fpga" else None,
        "xclbin_sha256": hashlib.sha256(args.xclbin.read_bytes()).hexdigest() if args.backend == "fpga" else None,
        "cpu_model_forward_calls": 0, "layer_compute_roundtrips_to_cpu": 0,
        "diagnostic_tensor_readbacks": diagnostic_reads,
        "prefill_device": "fpga" if args.backend == "fpga" else "cpu_cpp",
        "attention_device": "fpga" if args.backend == "fpga" else "cpu_cpp",
        "kv_cache_device": "hbm" if args.backend == "fpga" else "host_memory",
        "token_selection_device": "fpga" if args.backend == "fpga" else "cpu_cpp",
        "kv_cache_bytes": 469762048, "prompt": args.prompt, "input_ids": ids,
        "native_library": str(runtime.library),
        "native_library_sha256": hashlib.sha256(runtime.library.read_bytes()).hexdigest(),
        "kernel_source_sha256": hashlib.sha256((PROJECT / "hls/full_decode.cpp").read_bytes()
            + (PROJECT / "hls/full_decode.hpp").read_bytes()).hexdigest(),
        "generated_ids": generated, "answer": tokenizer.decode(generated, skip_special_tokens=True),
        "token_kernel_calls": calls, "layer_evaluations": calls * 28,
        "matches_reference_tokens": matches, "numerical_checks": checks,
        "load_seconds": load_seconds, "prefill_seconds": prefill_seconds - load_seconds,
        "prefill_seconds_including_load": prefill_seconds, "decode_seconds": decode_seconds,
        "incremental_decode_tokens_per_second": (len(generated) - 1) / decode_seconds if decode_seconds else None,
        "elapsed_seconds": time.perf_counter() - start}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if matches is False:
        raise SystemExit("Full-decoder output token IDs differ from the CPU reference")


if __name__ == "__main__":
    main()
