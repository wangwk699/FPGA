#!/usr/bin/env python3
"""Local files only. Generate FP32 reference from original floating-point weights."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reference"))
    parser.add_argument("--prompt", default="请用一句话介绍杭州。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if not (args.model / "config.json").is_file():
        parser.error("Missing local config.json; this program never downloads a model")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).eval()
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt")
    if inputs["input_ids"].shape[1] + args.max_new_tokens > 2048:
        parser.error("Input plus output exceeds the initial 2048-token budget")
    if args.output.exists():
        parser.error("Output already exists; use a new directory")
    args.output.mkdir(parents=True)
    traces = {}
    linear_vectors = {}
    handles = []
    def capture(name):
        def hook(_module, _args, output):
            if isinstance(output, tuple):
                output = output[0]
            traces[name] = output.detach().cpu().clone()
        return hook
    layer = model.model.layers[0]
    for name in ["input_layernorm", "self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj", "self_attn.q_norm", "self_attn.k_norm",
                 "self_attn.o_proj", "post_attention_layernorm", "mlp.gate_proj",
                 "mlp.up_proj", "mlp.down_proj"]:
        handles.append(layer.get_submodule(name).register_forward_hook(capture(name)))
    handles.append(layer.register_forward_hook(capture("layer_0_output")))
    def capture_linear(name):
        def hook(_module, inputs, output):
            linear_vectors[name] = {
                "input": inputs[0].detach().reshape(-1, inputs[0].shape[-1])[-1].cpu().clone(),
                "output": output.detach().reshape(-1, output.shape[-1])[-1].cpu().clone(),
            }
        return hook
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(capture_linear(name)))
    with torch.inference_mode():
        result = model(**inputs, use_cache=True)
        for handle in handles:
            handle.remove()
        torch.save({"input_ids": inputs["input_ids"], "traces": traces,
                    "linear_vectors": linear_vectors,
                    "last_logits": result.logits[:, -1, :].cpu()}, args.output / "prefill.pt")
        # Check cached single-token decoding against full-prefix recomputation.
        next_token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cached = model(input_ids=next_token, past_key_values=result.past_key_values, use_cache=True)
        full = model(input_ids=torch.cat([inputs["input_ids"], next_token], dim=1), use_cache=False)
        torch.testing.assert_close(cached.logits[:, -1, :], full.logits[:, -1, :], rtol=1e-3, atol=1e-3)
        output = model.generate(**inputs, do_sample=False,
            max_new_tokens=args.max_new_tokens, use_cache=True)
    answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    (args.output / "result.json").write_text(json.dumps({
        "prompt": args.prompt, "chat_text": text, "answer": answer,
        "input_ids": inputs["input_ids"].tolist(), "generated_ids": output.tolist(),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "compute_dtype": "float32", "quantized": False,
        "cache_vs_full_max_abs_error": (cached.logits[:, -1, :] - full.logits[:, -1, :]).abs().max().item(),
    }, ensure_ascii=False, indent=2) + "\n")
    print(answer)


if __name__ == "__main__":
    main()
