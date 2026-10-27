#pragma once
#include <cstddef>
#include <cstdint>

namespace fullqwen {
constexpr int HIDDEN = 1024;
constexpr int INTERMEDIATE = 3072;
constexpr int LAYERS = 28;
constexpr int Q_HEADS = 16;
constexpr int KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_WIDTH = Q_HEADS * HEAD_DIM;
constexpr int KV_WIDTH = KV_HEADS * HEAD_DIM;
constexpr int VOCAB = 151936;
constexpr int CONTEXT = 2048;
constexpr int LAYER_TENSORS = 11;
constexpr int TENSORS = 2 + LAYERS * LAYER_TENSORS;
constexpr int TRACE_FLOATS = (LAYERS + 2) * HIDDEN;
constexpr std::size_t CACHE_FLOATS = std::size_t(LAYERS) * 2 * CONTEXT * KV_WIDTH;
constexpr float EPS = 1.0e-6f;
// Per-layer tensor order after embedding (0) and final RMSNorm (1).
enum Tensor { INPUT_NORM, Q_PROJ, K_PROJ, V_PROJ, Q_NORM, K_NORM,
              O_PROJ, POST_NORM, GATE_PROJ, UP_PROJ, DOWN_PROJ };
}

// A complete single-token forward pass. All model arithmetic is inside this
// kernel. KV cache persists in HBM across calls; only token IDs/control cross
// PCIe in ordinary inference. Optional logits/trace reads are validation only.
extern "C" void qwen_full_decode(const std::uint16_t* weights,
    const std::uint64_t* offsets, const float* inv_freq, float* kv_cache,
    float* logits, float* trace, int* result, int token, int position,
    int emit_logits, int emit_trace);
