#include "gemv.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(int rows, int cols) {
    std::vector<std::uint16_t> w(rows * cols);
    std::vector<float> x(cols), y(rows);
    for (int c = 0; c < cols; ++c) x[c] = float(c % 17 - 8) / 16;
    for (int i = 0; i < rows * cols; ++i) {
        // Synthetic BF16 values exercise signs and a range of exponents/mantissas.
        w[i] = std::uint16_t(0x3b00 + i % 1800) | ((i % 3 == 0) ? 0x8000 : 0);
    }
    qwen_gemv(w.data(), x.data(), y.data(), rows, cols);
    double max_error = 0;
    for (int r = 0; r < rows; ++r) {
        double ref = 0, magnitude = 0;
        for (int c = 0; c < cols; ++c) {
            const auto b = w[r * cols + c];
            const int exponent = (b >> 7) & 255;
            const double sign = (b & 0x8000) ? -1.0 : 1.0;
            const double value = sign * std::ldexp(1.0 + (b & 127) / 128.0, exponent - 127);
            ref += value * x[c];
            magnitude += std::fabs(value * x[c]);
        }
        const double error = std::fabs(y[r] - ref);
        if (!std::isfinite(y[r]) || error > 2e-6 * magnitude + 1e-5)
            throw std::runtime_error("GEMV numeric error");
        max_error = std::fmax(max_error, error);
    }
    std::cout << "PASS " << rows << 'x' << cols << " max_abs_error=" << max_error << '\n';
}
int main() {
  try {
    check(7, 13);
    check(2048, 1024);  // q_proj
    check(1024, 2048);  // o_proj
    check(3072, 1024);  // gate/up
    check(1024, 3072);  // down
    float sentinel = 123;
    qwen_gemv(nullptr, nullptr, &sentinel, 1, MAX_COLS + 1);
    if (sentinel != 123) throw std::runtime_error("invalid size wrote output");
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
