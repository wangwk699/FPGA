#!/usr/bin/env python3
"""Offline PyTorch reference for full-layer traces, logits and greedy output."""
import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="请用一句话介绍杭州。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=PROJECT / "reference/default")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Reference directory already exists: {args.output}")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    deployment = json.loads((PROJECT.parent / "deployment.json").read_text())
    model_path = deployment["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True).eval()
    text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt")
    if len(inputs["input_ids"][0]) + args.max_new_tokens > 2048 or args.max_new_tokens < 2:
        parser.error("Reference requires at least 2 new tokens and total context <= 2048")
    trace = [None] * 30
    def hook(index):
        def capture(_module, _inputs, output):
            trace[index] = output[0].detach().float().numpy().copy()
        return capture
    handles = [model.model.embed_tokens.register_forward_hook(hook(0))]
    handles += [layer.register_forward_hook(hook(i + 1)) for i, layer in enumerate(model.model.layers)]
    handles += [model.model.norm.register_forward_hook(hook(29))]
    arrays = {}
    with torch.inference_mode():
        output = model(**inputs, use_cache=True)
        cache = output.past_key_values
        all_trace = np.stack(trace, axis=1)
        arrays["prefill_trace_first"] = all_trace[0]
        arrays["prefill_trace_last"] = all_trace[-1]
        arrays["prefill_logits"] = output.logits[0, -1].numpy().copy()
        generated = [int(output.logits[0, -1].argmax())]
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for step in range(args.max_new_tokens - 1):
            if generated[-1] in eos:
                break
            output = model(input_ids=torch.tensor([[generated[-1]]]), past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if step == 0:
                arrays["first_decode_trace"] = np.stack(trace, axis=1)[0]
                arrays["first_decode_logits"] = output.logits[0, -1].numpy().copy()
            generated.append(int(output.logits[0, -1].argmax()))
    for handle in handles:
        handle.remove()
    if "first_decode_logits" not in arrays:
        raise RuntimeError("Reference prompt produced EOS before an incremental decode step")
    report = {"execution": "pytorch_cpu_fp32_reference", "quantized": False,
        "prompt": args.prompt, "input_ids": inputs["input_ids"][0].tolist(),
        "generated_ids": generated, "max_new_tokens": args.max_new_tokens,
        "answer": tokenizer.decode(generated, skip_special_tokens=True),
        "eos_token_ids": sorted(eos), "model": model_path, "revision": deployment["revision"],
        "trace_slots": ["embedding"] + [f"layer_{i}_output" for i in range(28)] + ["final_rmsnorm"]}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "vectors.npz", **arrays)
    (args.output / "reference.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
