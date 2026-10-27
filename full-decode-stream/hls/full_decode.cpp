#include "full_decode.hpp"
#include <cfloat>
#include <cmath>
#include <cstring>
#ifdef __SYNTHESIS__
#include <hls_math.h>
#endif

using namespace fullqwen;

namespace {
std::uint16_t lane_bits(const QwenWeightWord& word, int lane) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
    return word.range(lane * 16 + 15, lane * 16).to_uint();
#else
    return word.lanes[lane];
#endif
}
std::uint16_t weight_at(const QwenWeightWord* weights, std::uint64_t index) {
#pragma HLS INLINE
    const QwenWeightWord word = weights[index >> 5];
    return lane_bits(word, int(index & 31));
}
float bf16(std::uint16_t word) {
    std::uint32_t bits = std::uint32_t(word) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float exp32(float x) {
#ifdef __SYNTHESIS__
    return hls::expf(x);
#else
    return ::expf(x);
#endif
}
float sqrt32(float x) {
#ifdef __SYNTHESIS__
    return hls::sqrtf(x);
#else
    return ::sqrtf(x);
#endif
}
float sin32(float x) {
#ifdef __SYNTHESIS__
    return hls::sinf(x);
#else
    return ::sinf(x);
#endif
}
float cos32(float x) {
#ifdef __SYNTHESIS__
    return hls::cosf(x);
#else
    return ::cosf(x);
#endif
}
bool finite32(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

float sum16(const float* values) {
#pragma HLS INLINE
    float level8[8], level4[4], level2[2];
#pragma HLS ARRAY_PARTITION variable=level8 complete
#pragma HLS ARRAY_PARTITION variable=level4 complete
#pragma HLS ARRAY_PARTITION variable=level2 complete
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        level8[i] = values[2 * i] + values[2 * i + 1];
    }
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        level4[i] = level8[2 * i] + level8[2 * i + 1];
    }
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        level2[i] = level4[2 * i] + level4[2 * i + 1];
    }
    return level2[0] + level2[1];
}

// All supported matrices have 64-byte aligned rows and 32-divisible columns.
// Each beat contains 32 ORIGINAL BF16 values, expanded losslessly to FP32.
// Sixteen independent FP32 partial sums expose a true recurrence distance of
// sixteen iterations, instead of falsely removing the accumulation dependency.
void dot_tile(const QwenWeightWord* weights, std::uint64_t word_base,
              const float* input, float* output, int cols) {
#pragma HLS INLINE
    float partial[16] = {};
#pragma HLS ARRAY_PARTITION variable=partial complete
    // Row-interleaved 16x32 lossless tiles make every AXI request contiguous.
    // Each row accumulator is revisited after sixteen beats, not every cycle.
    for (int beat = 0; beat < (cols / 32) * 16; ++beat) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=partial type=inter direction=RAW distance=16 true
        const int row = beat & 15;
        const int block = beat >> 4;
        QwenWeightWord word = weights[word_base + beat];
        float pair[16];
#pragma HLS ARRAY_PARTITION variable=pair complete
        for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
            float first = bf16(lane_bits(word, 2 * i)) * input[block * 32 + 2 * i];
            float second = bf16(lane_bits(word, 2 * i + 1)) * input[block * 32 + 2 * i + 1];
            pair[i] = first + second;
        }
        float block_sum = sum16(pair);
        partial[row] = (block == 0 ? 0.0f : partial[row]) + block_sum;
    }
    for (int row = 0; row < 16; ++row) {
#pragma HLS UNROLL
        output[row] = partial[row];
    }
}

void matvec(const QwenWeightWord* weights, std::uint64_t base,
            const float* input, float* output, int rows, int cols) {
#pragma HLS INLINE off
    for (int r = 0; r < rows; r += 16) {
#pragma HLS PIPELINE off
        float tile[16];
#pragma HLS ARRAY_PARTITION variable=tile complete
        dot_tile(weights, (base >> 5) + std::uint64_t(r / 16) * (cols / 32) * 16,
                 input, tile, cols);
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE II=1
            output[r + i] = tile[i];
        }
    }
}

void rmsnorm(const QwenWeightWord* weights, std::uint64_t base,
             const float* input, float* output, int size) {
#pragma HLS INLINE off
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) sum += input[i] * input[i];
    const float inverse = 1.0f / sqrt32(sum / float(size) + EPS);
    for (int i = 0; i < size; ++i)
        output[i] = (input[i] * inverse) * bf16(weight_at(weights, base + i));
}

void norm_heads(const QwenWeightWord* weights, std::uint64_t base,
                float* values, int heads) {
#pragma HLS INLINE off
    for (int h = 0; h < heads; ++h) {
#pragma HLS PIPELINE off
        float sum = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            float value = values[h * HEAD_DIM + d];
            sum += value * value;
        }
        float inverse = 1.0f / sqrt32(sum / float(HEAD_DIM) + EPS);
        for (int d = 0; d < HEAD_DIM; ++d)
            values[h * HEAD_DIM + d] = (values[h * HEAD_DIM + d] * inverse) * bf16(weight_at(weights, base + d));
    }
}

void rope(float* values, int heads, const float* cosine, const float* sine) {
#pragma HLS INLINE off
    for (int h = 0; h < heads; ++h) {
#pragma HLS PIPELINE off
        for (int d = 0; d < HEAD_DIM / 2; ++d) {
            const int a = h * HEAD_DIM + d;
            const int b = a + HEAD_DIM / 2;
            const float first = values[a], second = values[b];
            values[a] = first * cosine[d] - second * sine[d];
            values[b] = second * cosine[d] + first * sine[d];
        }
    }
}

void attention(float* cache, int layer, int position,
               const float* query, const float* key, const float* value,
               float* output) {
#pragma HLS INLINE off
    float scores[2][CONTEXT];
#pragma HLS ARRAY_PARTITION variable=scores complete dim=1
#pragma HLS BIND_STORAGE variable=scores type=ram_2p impl=bram
    const std::size_t kbase = std::size_t(layer * 2) * CONTEXT * KV_WIDTH;
    const std::size_t vbase = kbase + std::size_t(CONTEXT) * KV_WIDTH;
    for (int d = 0; d < KV_WIDTH; ++d) {
        cache[kbase + std::size_t(position) * KV_WIDTH + d] = key[d];
        cache[vbase + std::size_t(position) * KV_WIDTH + d] = value[d];
    }
    // Both Q heads sharing a KV head consume each historical K/V row once.
    for (int kv = 0; kv < KV_HEADS; ++kv) {
#pragma HLS PIPELINE off
        float q[2][HEAD_DIM], accum[2][HEAD_DIM], cached[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=q complete dim=0
#pragma HLS ARRAY_PARTITION variable=accum complete dim=0
#pragma HLS ARRAY_PARTITION variable=cached complete
        for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE II=2
            q[0][d] = query[(kv * 2) * HEAD_DIM + d];
            q[1][d] = query[(kv * 2 + 1) * HEAD_DIM + d];
            accum[0][d] = 0.0f;
            accum[1][d] = 0.0f;
        }
        float maximum0 = -FLT_MAX, maximum1 = -FLT_MAX;
        for (int t = 0; t <= position; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
#pragma HLS PIPELINE off
            for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE II=1
                cached[d] = cache[kbase + std::size_t(t) * KV_WIDTH + kv * HEAD_DIM + d];
            }
            float total0 = 0.0f, total1 = 0.0f;
            for (int block = 0; block < HEAD_DIM / 32; ++block) {
#pragma HLS PIPELINE II=8
                float pair0[16], pair1[16];
#pragma HLS ARRAY_PARTITION variable=pair0 complete
#pragma HLS ARRAY_PARTITION variable=pair1 complete
                for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
                    int d = block * 32 + 2 * i;
                    pair0[i] = q[0][d] * cached[d] + q[0][d + 1] * cached[d + 1];
                    pair1[i] = q[1][d] * cached[d] + q[1][d + 1] * cached[d + 1];
                }
                total0 += sum16(pair0);
                total1 += sum16(pair1);
            }
            float score0 = total0 * 0.08838834764831845f;
            float score1 = total1 * 0.08838834764831845f;
            scores[0][t] = score0;
            scores[1][t] = score1;
            if (score0 > maximum0) maximum0 = score0;
            if (score1 > maximum1) maximum1 = score1;
        }
        float denominator0 = 0.0f, denominator1 = 0.0f;
        for (int t = 0; t <= position; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
#pragma HLS PIPELINE II=8
            float exp0 = exp32(scores[0][t] - maximum0);
            float exp1 = exp32(scores[1][t] - maximum1);
            scores[0][t] = exp0;
            scores[1][t] = exp1;
            denominator0 += exp0;
            denominator1 += exp1;
        }
        for (int t = 0; t <= position; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
#pragma HLS PIPELINE off
            float probability0 = scores[0][t] / denominator0;
            float probability1 = scores[1][t] / denominator1;
            for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE II=1
                cached[d] = cache[vbase + std::size_t(t) * KV_WIDTH + kv * HEAD_DIM + d];
            }
            for (int block = 0; block < HEAD_DIM / 32; ++block) {
#pragma HLS PIPELINE II=8
                for (int lane = 0; lane < 32; ++lane) {
#pragma HLS UNROLL
                    int d = block * 32 + lane;
                    accum[0][d] += probability0 * cached[d];
                    accum[1][d] += probability1 * cached[d];
                }
            }
        }
        for (int h = 0; h < 2; ++h)
            for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE II=1
                output[(kv * 2 + h) * HEAD_DIM + d] = accum[h][d];
            }
    }
}

int project_vocab(const QwenWeightWord* weights, std::uint64_t base,
                  const float* hidden, float* logits, int& numerical_error) {
#pragma HLS INLINE off
    float maximum = -FLT_MAX;
    int best = 0;
    numerical_error = 0;
    for (int r = 0; r < VOCAB; r += 16) {
#pragma HLS PIPELINE off
        float tile[16];
#pragma HLS ARRAY_PARTITION variable=tile complete
        dot_tile(weights, (base >> 5) + std::uint64_t(r / 16) * (HIDDEN / 32) * 16,
                 hidden, tile, HIDDEN);
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE II=3
            float sum = tile[i];
            logits[r + i] = sum;
            if (!finite32(sum)) numerical_error = 2;
            if (sum > maximum) { maximum = sum; best = r + i; }
        }
    }
    return best;
}
}

extern "C" void qwen_full_decode(const QwenWeightWord* weights,
    const std::uint64_t* offsets, const float* inv_freq, float* kv_cache,
    float* logits, float* trace, int* result, int token, int position,
    int emit_logits, int emit_trace) {
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem_w max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=offsets offset=slave bundle=gmem_meta
#pragma HLS INTERFACE m_axi port=inv_freq offset=slave bundle=gmem_meta
#pragma HLS INTERFACE m_axi port=kv_cache offset=slave bundle=gmem_kv max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=logits offset=slave bundle=gmem_out
#pragma HLS INTERFACE m_axi port=trace offset=slave bundle=gmem_out
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=offsets bundle=control
#pragma HLS INTERFACE s_axilite port=inv_freq bundle=control
#pragma HLS INTERFACE s_axilite port=kv_cache bundle=control
#pragma HLS INTERFACE s_axilite port=logits bundle=control
#pragma HLS INTERFACE s_axilite port=trace bundle=control
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=token bundle=control
#pragma HLS INTERFACE s_axilite port=position bundle=control
#pragma HLS INTERFACE s_axilite port=emit_logits bundle=control
#pragma HLS INTERFACE s_axilite port=emit_trace bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    result[0] = -1;
    result[1] = 1;
    result[2] = position;
    result[3] = 0;
    if (token < 0 || token >= VOCAB || position < 0 || position >= CONTEXT) return;
    float hidden[HIDDEN], normalized[HIDDEN], query[Q_WIDTH];
    float key[KV_WIDTH], value[KV_WIDTH], attended[Q_WIDTH], projected[HIDDEN];
    float gate[INTERMEDIATE], up[INTERMEDIATE];
    float cosine[HEAD_DIM / 2], sine[HEAD_DIM / 2];
#pragma HLS BIND_STORAGE variable=hidden type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=normalized type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=query type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=key type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=value type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=attended type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=projected type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=gate type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=up type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=normalized cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=attended cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=gate cyclic factor=32 dim=1
    for (int d = 0; d < HEAD_DIM / 2; ++d) {
        float angle = float(position) * inv_freq[d];
        cosine[d] = cos32(angle);
        sine[d] = sin32(angle);
    }
    for (int i = 0; i < HIDDEN; ++i) {
        const std::uint64_t word_index = (offsets[0] >> 5) + std::uint64_t(token / 16) * (HIDDEN / 32) * 16 + (i / 32) * 16 + token % 16;
        hidden[i] = bf16(lane_bits(weights[word_index], i % 32));
        if (emit_trace) trace[i] = hidden[i];
    }
    for (int layer = 0; layer < LAYERS; ++layer) {
#pragma HLS PIPELINE off
        const int base = 2 + layer * LAYER_TENSORS;
        rmsnorm(weights, offsets[base + INPUT_NORM], hidden, normalized, HIDDEN);
        matvec(weights, offsets[base + Q_PROJ], normalized, query, Q_WIDTH, HIDDEN);
        matvec(weights, offsets[base + K_PROJ], normalized, key, KV_WIDTH, HIDDEN);
        matvec(weights, offsets[base + V_PROJ], normalized, value, KV_WIDTH, HIDDEN);
        norm_heads(weights, offsets[base + Q_NORM], query, Q_HEADS);
        norm_heads(weights, offsets[base + K_NORM], key, KV_HEADS);
        rope(query, Q_HEADS, cosine, sine);
        rope(key, KV_HEADS, cosine, sine);
        attention(kv_cache, layer, position, query, key, value, attended);
        matvec(weights, offsets[base + O_PROJ], attended, projected, HIDDEN, Q_WIDTH);
        for (int i = 0; i < HIDDEN; ++i) hidden[i] += projected[i];
        rmsnorm(weights, offsets[base + POST_NORM], hidden, normalized, HIDDEN);
        matvec(weights, offsets[base + GATE_PROJ], normalized, gate, INTERMEDIATE, HIDDEN);
        matvec(weights, offsets[base + UP_PROJ], normalized, up, INTERMEDIATE, HIDDEN);
        for (int i = 0; i < INTERMEDIATE; ++i)
            gate[i] = (gate[i] / (1.0f + exp32(-gate[i]))) * up[i];
        matvec(weights, offsets[base + DOWN_PROJ], gate, projected, HIDDEN, INTERMEDIATE);
        for (int i = 0; i < HIDDEN; ++i) {
            hidden[i] += projected[i];
            if (emit_trace) trace[(layer + 1) * HIDDEN + i] = hidden[i];
        }
    }
    rmsnorm(weights, offsets[1], hidden, normalized, HIDDEN);
    if (emit_trace)
        for (int i = 0; i < HIDDEN; ++i) trace[(LAYERS + 1) * HIDDEN + i] = normalized[i];
    int numerical_error = 0;
    if (emit_logits) result[0] = project_vocab(weights, offsets[0], normalized, logits, numerical_error);
    result[1] = numerical_error;
    result[2] = position + 1;
    result[3] = LAYERS;
}
