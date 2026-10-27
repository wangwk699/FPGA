"""Native linear backend. CPU C++ execution and FPGA execution are explicitly distinct."""
import ctypes
from pathlib import Path
import torch


class Runtime:
    def __init__(self, library, xclbin=None, device="0000:01:00.1"):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.qwen_runtime_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.qwen_runtime_open.restype = ctypes.c_void_p
        self.lib.qwen_runtime_close.argtypes = [ctypes.c_void_p]
        self.lib.qwen_runtime_close.restype = None
        self.lib.qwen_runtime_error.argtypes = []
        self.lib.qwen_runtime_error.restype = ctypes.c_char_p
        self.lib.qwen_runtime_add_weight.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.qwen_runtime_add_weight.restype = ctypes.c_int
        self.lib.qwen_runtime_linear.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.qwen_runtime_linear.restype = ctypes.c_int
        self.handle = self.lib.qwen_runtime_open(
            str(xclbin).encode() if xclbin else None, device.encode())
        self.calls = 0
        if not self.handle:
            raise RuntimeError(self.error())

    def error(self):
        return self.lib.qwen_runtime_error().decode()

    def add_weight(self, weight):
        if weight.dtype != torch.bfloat16 or weight.ndim != 2 or not weight.is_contiguous():
            raise ValueError("Expected contiguous original BF16 matrix")
        slot = self.lib.qwen_runtime_add_weight(self.handle, weight.data_ptr(), *weight.shape)
        if slot < 0:
            raise RuntimeError(self.error())
        return slot

    def linear(self, slot, x, output_size):
        if not self.handle:
            raise RuntimeError("Native runtime already closed")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("Native GEMV requires CPU FP32 activations")
        x = x.contiguous()
        result = torch.empty((*x.shape[:-1], output_size), dtype=torch.float32)
        for source, target in zip(x.reshape(-1, x.shape[-1]), result.reshape(-1, output_size)):
            status = self.lib.qwen_runtime_linear(self.handle, slot, source.data_ptr(), target.data_ptr())
            if status:
                raise RuntimeError(self.error())
            self.calls += 1
        return result

    def close(self):
        if self.handle:
            self.lib.qwen_runtime_close(self.handle)
            self.handle = None


class NativeLinear(torch.nn.Module):
    def __init__(self, runtime, original_bf16_weight):
        super().__init__()
        self.runtime = runtime
        self.out_features, self.in_features = original_bf16_weight.shape
        self.slot = runtime.add_weight(original_bf16_weight)

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError("Input dimension does not match the registered weight")
        return self.runtime.linear(self.slot, x, self.out_features)


def replace_linears(model, model_path, runtime):
    from safetensors import safe_open
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear)]
    if len(modules) != 197:
        raise ValueError(f"Expected 197 linear layers, got {len(modules)}")
    with safe_open(Path(model_path) / "model.safetensors", framework="pt", device="cpu") as source:
        for name, module in modules:
            if module.bias is not None:
                raise ValueError("Only bias-free Qwen3-0.6B linear layers are supported")
            key = "model.embed_tokens.weight" if name == "lm_head" else name + ".weight"
            weight = source.get_tensor(key)
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, NativeLinear(runtime, weight))
    return len(modules)
