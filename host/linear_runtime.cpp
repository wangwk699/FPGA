// Stable C ABI for the same BF16/FP32 GEMV on CPU software or an XRT kernel.
#include "gemv.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef QWEN_XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#endif

namespace {
thread_local std::string last_error;
struct Weight {
  int rows, cols;
#ifdef QWEN_XRT
  xrt::bo data;
#else
  std::vector<std::uint16_t> data;
#endif
};
struct Context {
  std::vector<Weight> weights;
#ifdef QWEN_XRT
  xrt::device device;
  xrt::kernel kernel;
  xrt::bo input, output;
  Context(const char* xclbin, const char* bdf) : device(bdf) {
    const auto uuid = device.load_xclbin(xclbin);
    kernel = xrt::kernel(device, uuid, "qwen_gemv");
    input = xrt::bo(device, MAX_COLS * sizeof(float), kernel.group_id(1));
    output = xrt::bo(device, MAX_ROWS * sizeof(float), kernel.group_id(2));
  }
#else
  Context(const char*, const char*) {}
#endif
};
}

extern "C" {
const char* qwen_runtime_error() { return last_error.c_str(); }
void* qwen_runtime_open(const char* xclbin, const char* bdf) {
  try {
    last_error.clear();
#ifdef QWEN_XRT
    if (!xclbin || !bdf) throw std::runtime_error("xclbin and device BDF required");
#endif
    return new Context(xclbin, bdf);
  } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
void qwen_runtime_close(void* handle) { delete static_cast<Context*>(handle); }
int qwen_runtime_add_weight(void* handle, const std::uint16_t* source, int rows, int cols) {
  try {
    if (!handle || !source || rows < 1 || rows > 151936 || cols < 1 || cols > MAX_COLS)
      throw std::runtime_error("Invalid weight arguments");
    auto& ctx = *static_cast<Context*>(handle);
    const std::size_t count = std::size_t(rows) * cols;
#ifdef QWEN_XRT
    // Device-only weights avoid pinning the full 1.2GB model in host memory.
    xrt::bo data(ctx.device, count * sizeof(std::uint16_t),
                 xrt::bo::flags::device_only, ctx.kernel.group_id(0));
    const std::size_t bytes = count * sizeof(std::uint16_t);
    const std::size_t chunk_bytes = 1024 * 1024;
    const auto* source_bytes = reinterpret_cast<const char*>(source);
    for (std::size_t offset = 0; offset < bytes; offset += chunk_bytes)
      data.write(source_bytes + offset, std::min(chunk_bytes, bytes - offset), offset);
    ctx.weights.push_back(Weight{rows, cols, std::move(data)});
#else
    ctx.weights.push_back(Weight{rows, cols, std::vector<std::uint16_t>(source, source + count)});
#endif
    return static_cast<int>(ctx.weights.size() - 1);
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int qwen_runtime_linear(void* handle, int id, const float* input, float* output) {
  try {
    if (!handle || !input || !output) throw std::runtime_error("Invalid linear arguments");
    auto& ctx = *static_cast<Context*>(handle);
    if (id < 0 || std::size_t(id) >= ctx.weights.size()) throw std::runtime_error("Unknown weight id");
    const auto& weight = ctx.weights[id];
#ifdef QWEN_XRT
    ctx.input.write(input, weight.cols * sizeof(float), 0);
    ctx.input.sync(XCL_BO_SYNC_BO_TO_DEVICE, weight.cols * sizeof(float), 0);
#endif
    for (int start = 0; start < weight.rows; start += MAX_ROWS) {
      const int rows = std::min(MAX_ROWS, weight.rows - start);
#ifdef QWEN_XRT
      const std::size_t offset = std::size_t(start) * weight.cols * sizeof(std::uint16_t);
      const std::size_t tile_bytes = std::size_t(rows) * weight.cols * sizeof(std::uint16_t);
      if (offset + tile_bytes > weight.data.size()) throw std::runtime_error("Weight tile exceeds its buffer");
      // XRT sub-BO construction maps its parent, which device-only weights
      // cannot provide. Retain the parent BO binding, then set the tile's
      // 64-bit device address before starting this AP_CTRL_HS kernel.
      xrt::run run(ctx.kernel);
      run.set_arg(0, weight.data);
      run.set_arg(1, ctx.input);
      run.set_arg(2, ctx.output);
      run.set_arg(3, rows);
      run.set_arg(4, weight.cols);
      const std::uint64_t address = weight.data.address() + offset;
      run.set_arg(0, address);
      run.start();
      if (run.wait() != ERT_CMD_STATE_COMPLETED) throw std::runtime_error("GEMV kernel failed");
      ctx.output.sync(XCL_BO_SYNC_BO_FROM_DEVICE, rows * sizeof(float), 0);
      ctx.output.read(output + start, rows * sizeof(float), 0);
#else
      qwen_gemv(weight.data.data() + std::size_t(start) * weight.cols,
                input, output + start, rows, weight.cols);
#endif
    }
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
}
