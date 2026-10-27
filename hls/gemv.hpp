#pragma once
#include <cstdint>

// Row-major W[rows][cols], BF16 storage; x/y and arithmetic are FP32.
// Limits apply to one invocation; split the vocabulary projection into row tiles.
constexpr int MAX_COLS = 3072;
constexpr int MAX_ROWS = 4096;
extern "C" void qwen_gemv(const std::uint16_t* weights, const float* x,
                          float* y, int rows, int cols);
