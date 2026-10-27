#include "gemv.hpp"
#include <cstring>

static float expand_bf16(std::uint16_t value) {
    const std::uint32_t bits = std::uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

extern "C" void qwen_gemv(const std::uint16_t* weights, const float* x,
                          float* y, int rows, int cols) {
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem_w max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem_x
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem_y
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=rows bundle=control
#pragma HLS INTERFACE s_axilite port=cols bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if (rows <= 0 || rows > MAX_ROWS || cols <= 0 || cols > MAX_COLS) return;
    float input[MAX_COLS];
    for (int c = 0; c < cols; ++c) input[c] = x[c];
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            // Correctness-first sequential reduction. No throughput promise.
            sum += expand_bf16(weights[r * cols + c]) * input[c];
        }
        y[r] = sum;
    }
}
