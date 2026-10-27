"""Explicit CPU-C++ reference or FPGA backend for the complete model kernel."""
import ctypes
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT.parent/'scripts'))
from model_profiles import expected_shapes, geometry, weight_bytes
VOCAB = 151936
TRACE_SHAPE = (30, 1024)


class FullRuntime:
    def __init__(self, backend="fpga", weights=None, xclbin=None, device="0000:01:00.1",
                 project=None, model_name='Qwen3-0.6B'):
        if backend not in {"fpga", "cpu-cpp"}:
            raise ValueError("Explicit fpga or cpu-cpp backend required")
        project=Path(project) if project else PROJECT
        weights = Path(weights) if weights else project / "weights"
        xclbin = Path(xclbin) if xclbin else project / "build/hw/qwen_full_decode.xclbin"
        if backend == "fpga" and (not xclbin.is_file() or os.environ.get("XCL_EMULATION_MODE")):
            raise ValueError("Real FPGA xclbin required and XCL_EMULATION_MODE must be unset")
        manifest = json.loads((weights / "manifest.json").read_text())
        if manifest["format"] != "qwen3-tiled-bf16-v2" or manifest["quantized"] or len(manifest["tensors"]) != 310:
            raise ValueError("Invalid unquantized full-decoder weight manifest")
        expected_shapes(model_name,manifest['config'])
        g=geometry(model_name)
        self.vocab=g['vocab_size']
        self.trace_shape=(g['num_hidden_layers']+2,g['hidden_size'])
        if manifest['weight_bytes']!=weight_bytes(model_name) or (weights/'weights.bin').stat().st_size!=weight_bytes(model_name):
            raise ValueError('Selected model and packed weight image size differ')
        raw_offsets = (weights / "offsets.bin").read_bytes()
        raw_frequency = (weights / "inv_freq.bin").read_bytes()
        if len(raw_offsets) != 310 * 8 or len(raw_frequency) != 64 * 4:
            raise ValueError("Invalid model descriptor sizes")
        descriptors = (ctypes.c_uint64 * 310).from_buffer_copy(raw_offsets)
        frequencies = (ctypes.c_float * 64).from_buffer_copy(raw_frequency)
        library = project / f"build/libfull_{'fpga' if backend == 'fpga' else 'cpu'}.so"
        self.lib = ctypes.CDLL(str(library))
        if model_name=='Qwen3-1.7B':
            if not hasattr(self.lib,'qwen_full_model_signature'):
                raise ValueError('1.7B requires its own native runtime library')
            signature=self.lib.qwen_full_model_signature
            signature.argtypes=[];signature.restype=ctypes.c_uint64
            if signature()!=((g['hidden_size']<<32)|g['intermediate_size']):
                raise ValueError('Native runtime was compiled for a different model')
        self.lib.qwen_full_error.argtypes = []
        self.lib.qwen_full_error.restype = ctypes.c_char_p
        self.lib.qwen_full_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float), ctypes.c_char_p, ctypes.c_char_p]
        self.lib.qwen_full_open.restype = ctypes.c_void_p
        self.lib.qwen_full_close.argtypes = [ctypes.c_void_p]
        self.lib.qwen_full_close.restype = None
        self.lib.qwen_full_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.qwen_full_step.restype = ctypes.c_int
        for name in ["qwen_full_logits", "qwen_full_trace"]:
            function = getattr(self.lib, name)
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
            function.restype = ctypes.c_int
        self.handle = self.lib.qwen_full_open(str(weights / "weights.bin").encode(), descriptors,
            frequencies, str(xclbin).encode() if backend == "fpga" else None, device.encode())
        if not self.handle:
            raise RuntimeError(self.error())
        self.backend, self.library, self.xclbin = backend, library, xclbin
        self.calls = 0

    def error(self):
        return self.lib.qwen_full_error().decode()

    def step(self, token, emit_logits=True, capture_trace=False):
        if not self.handle:
            raise RuntimeError("Full decoder is closed")
        output = ctypes.c_int(-1)
        status = self.lib.qwen_full_step(self.handle, int(token), int(emit_logits), int(capture_trace), ctypes.byref(output))
        if status:
            raise RuntimeError(self.error())
        self.calls += 1
        return output.value

    def logits(self):
        import numpy as np
        output = (ctypes.c_float * self.vocab)()
        if self.lib.qwen_full_logits(self.handle, output):
            raise RuntimeError(self.error())
        return np.ctypeslib.as_array(output).copy()

    def trace(self):
        import numpy as np
        output = (ctypes.c_float * (self.trace_shape[0] * self.trace_shape[1]))()
        if self.lib.qwen_full_trace(self.handle, output):
            raise RuntimeError(self.error())
        return np.ctypeslib.as_array(output).reshape(self.trace_shape).copy()

    def close(self):
        if self.handle:
            self.lib.qwen_full_close(self.handle)
            self.handle = None
