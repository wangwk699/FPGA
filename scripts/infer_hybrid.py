#!/usr/bin/env python3
"""CPU prefill and attention, native BF16/FP32 linear layers during decode.

--backend csim executes C++ software only; --backend fpga requires a real xclbin.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["csim", "fpga"], default="fpga")
    parser.add_argument("--xclbin", type=Path, default=root / "build/hw/qwen_gemv.xclbin")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt", default="请用一句话介绍杭州。")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 2:
        parser.error("At least two output tokens are required to exercise native decode")
    if args.backend == "fpga" and not args.xclbin.is_file():
        parser.error(f"FPGA xclbin missing: {args.xclbin}. No CPU fallback is performed.")
    if args.backend == "fpga" and os.environ.get("XCL_EMULATION_MODE"):
        parser.error("Unset XCL_EMULATION_MODE to run the real FPGA backend")
    settings = json.loads((root / "deployment.json").read_text())
    model_path = args.model or Path(settings["model_snapshot"])
    library = root / f"build/libqwen_runtime_{'cpu' if args.backend == 'csim' else 'fpga'}.so"
    if not library.is_file():
        parser.error(f"Missing runtime library: {library}; run make runtime")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from native_linear import Runtime, replace_linears
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32,
        attn_implementation="eager", local_files_only=True).eval()
    print("Local model loaded; starting CPU prefill.", file=sys.stderr, flush=True)
    text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt")
    if inputs["input_ids"].shape[1] + args.max_new_tokens > 2048:
        parser.error("Input plus output exceeds 2048 tokens")
    runtime = None
    started = time.perf_counter()
    report = {"backend": args.backend, "fpga_executed": False, "quantized": False,
        "fpga_device": settings["fpga_device"] if args.backend == "fpga" else None,
        "xclbin": str(args.xclbin.resolve()) if args.backend == "fpga" else None,
        "native_library": str(library),
        "prefill_device": "cpu", "attention_device": "cpu", "prompt": args.prompt,
        "model": str(model_path), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__}
    with torch.inference_mode():
        prefill = model(**inputs, use_cache=True)
        cache = prefill.past_key_values
        generated = [int(prefill.logits[0, -1, :].argmax())]
        first = torch.tensor([[generated[0]]])
        # Teacher run uses an independent cache, so the native step starts at the same position.
        teacher = model(input_ids=first, past_key_values=copy.deepcopy(cache), use_cache=True).logits[:, -1, :].clone()
        try:
            print(f"Preparing {args.backend} runtime and original BF16 weights.", file=sys.stderr, flush=True)
            runtime = Runtime(library, args.xclbin if args.backend == "fpga" else None,
                              settings["fpga_device"])
            report["replaced_linears"] = replace_linears(model, model_path, runtime)
            print(f"Registered {report['replaced_linears']} linear layers; starting native decode.",
                  file=sys.stderr, flush=True)
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for step in range(args.max_new_tokens - 1):
                if generated[-1] in eos:
                    break
                output = model(input_ids=torch.tensor([[generated[-1]]]),
                               past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = output.logits[:, -1, :]
                if step == 0:
                    difference = (logits - teacher).double()
                    report["first_decode_max_abs_error"] = difference.abs().max().item()
                    report["first_decode_relative_l2_error"] = (difference.norm() / teacher.double().norm().clamp_min(1e-12)).item()
                    torch.testing.assert_close(logits, teacher, rtol=1e-3, atol=1e-3)
                generated.append(int(logits.argmax()))
                print(f"Native decode step {step + 1}/{args.max_new_tokens - 1} completed; "
                      f"linear calls: {runtime.calls}.", file=sys.stderr, flush=True)
            report["native_linear_calls"] = runtime.calls
            report["fpga_executed"] = args.backend == "fpga" and runtime.calls > 0
        finally:
            if runtime:
                runtime.close()
    report["generated_ids"] = generated
    report["answer"] = tokenizer.decode(generated, skip_special_tokens=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
