#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: gemv_probe kernel.xclbin [device BDF]\n";
    return 2;
  }
  try {
    xrt::device device(argc > 2 ? argv[2] : "0000:01:00.1");
    auto uuid = device.load_xclbin(argv[1]);
    xrt::kernel kernel(device, uuid, "qwen_gemv");
    constexpr int rows = 2048, cols = 1024;
    std::vector<std::uint16_t> w(rows * cols);
    std::vector<float> x(cols), expected(rows);
    for (int c = 0; c < cols; ++c) x[c] = float(c % 17 - 8) / 16;
    for (int r = 0; r < rows; ++r) {
      double sum = 0;
      for (int c = 0; c < cols; ++c) {
        float value = float((r * 7 + c * 3) % 31 - 15) / 32;
        std::uint32_t bits;
        std::memcpy(&bits, &value, 4);
        w[r * cols + c] = std::uint16_t(bits >> 16);
        sum += double(value) * x[c];
      }
      expected[r] = float(sum);
    }
    xrt::bo wb(device, w.size() * 2, kernel.group_id(0));
    xrt::bo xb(device, x.size() * 4, kernel.group_id(1));
    xrt::bo yb(device, rows * 4, kernel.group_id(2));
    wb.write(w.data()); xb.write(x.data());
    wb.sync(XCL_BO_SYNC_BO_TO_DEVICE); xb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kernel(wb, xb, yb, rows, cols);
    run.wait();
    yb.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* y = yb.map<float*>();
    float max_error = 0;
    for (int r = 0; r < rows; ++r) {
      if (!std::isfinite(y[r])) throw std::runtime_error("nonfinite output");
      max_error = std::fmax(max_error, std::fabs(y[r] - expected[r]));
    }
    std::cout << "q_proj-shape synthetic GEMV: max_abs_error=" << max_error << '\n';
    if (max_error > 1e-4f) throw std::runtime_error("GEMV comparison failed");
    std::cout << "PASS (synthetic weights, not a model inference)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
