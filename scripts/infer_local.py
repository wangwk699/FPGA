#!/usr/bin/env python3
"""Offline Qwen3 CPU inference. This entry point never claims FPGA execution."""
import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt", default="请用一句话介绍杭州。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    settings = json.loads((root / "deployment.json").read_text())
    model_path = args.model or Path(settings["model_snapshot"])
    if not (model_path / "model.safetensors").is_file():
        parser.error(f"Local model missing: {model_path}; no download fallback")
    if args.max_new_tokens < 1 or args.threads < 1:
        parser.error("Token count and thread count must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32,
        attn_implementation="eager", local_files_only=True).eval()
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt")
    if inputs["input_ids"].shape[1] + args.max_new_tokens > 2048:
        parser.error("Input plus requested output exceeds 2048 tokens")
    print("Execution device: CPU; weights: original BF16 expanded to FP32; quantization: none", flush=True)
    with torch.inference_mode():
        outputs = model.generate(**inputs, do_sample=False,
            max_new_tokens=args.max_new_tokens, use_cache=True)
    print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
