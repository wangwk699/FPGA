import ctypes
import unittest
from pathlib import Path


class NativeRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "build/libqwen_runtime_cpu.so"
        cls.lib = ctypes.CDLL(str(path))
        cls.lib.qwen_runtime_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        cls.lib.qwen_runtime_open.restype = ctypes.c_void_p
        cls.lib.qwen_runtime_close.argtypes = [ctypes.c_void_p]
        cls.lib.qwen_runtime_add_weight.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        cls.lib.qwen_runtime_add_weight.restype = ctypes.c_int
        cls.lib.qwen_runtime_linear.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        cls.lib.qwen_runtime_linear.restype = ctypes.c_int

    def setUp(self):
        self.handle = self.lib.qwen_runtime_open(None, None)
        self.assertTrue(self.handle)

    def tearDown(self):
        self.lib.qwen_runtime_close(self.handle)

    def test_output_projection_tiling_and_weight_lifetime(self):
        rows, cols = 4099, 3
        # 1, -2, 0.5 in exact BF16, expected dot with [2, 3, 4] is -2.
        weight = (ctypes.c_uint16 * (rows * cols))(*([0x3f80, 0xc000, 0x3f00] * rows))
        slot = self.lib.qwen_runtime_add_weight(self.handle, weight, rows, cols)
        self.assertEqual(slot, 0)
        # Registration must own a copy: source can be reused/freed.
        ctypes.memset(weight, 0, ctypes.sizeof(weight))
        source = (ctypes.c_float * cols)(2, 3, 4)
        output = (ctypes.c_float * rows)()
        self.assertEqual(self.lib.qwen_runtime_linear(self.handle, slot, source, output), 0)
        self.assertTrue(all(value == -2 for value in output))

    def test_separate_weight_handles_and_repeated_calls(self):
        a = (ctypes.c_uint16 * 2)(0x3f80, 0x4000)
        b = (ctypes.c_uint16 * 2)(0xbf80, 0xc000)
        slots = [self.lib.qwen_runtime_add_weight(self.handle, weight, 1, 2) for weight in [a, b]]
        source, output = (ctypes.c_float * 2)(3, 4), (ctypes.c_float * 1)()
        for slot, expected in [(slots[0], 11), (slots[1], -11), (slots[0], 11)]:
            self.assertEqual(self.lib.qwen_runtime_linear(self.handle, slot, source, output), 0)
            self.assertEqual(output[0], expected)

    def test_invalid_handles_and_sizes_return_errors(self):
        weight = (ctypes.c_uint16 * 1)(0x3f80)
        data = (ctypes.c_float * 1)(1)
        self.assertEqual(self.lib.qwen_runtime_add_weight(self.handle, weight, 1, 3073), -1)
        self.assertEqual(self.lib.qwen_runtime_add_weight(None, weight, 1, 1), -1)
        self.assertEqual(self.lib.qwen_runtime_linear(self.handle, 100, data, data), -1)


if __name__ == "__main__":
    unittest.main()
