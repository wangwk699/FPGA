#include "full_decode.hpp"
#include <array>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#ifdef QWEN_XRT
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#endif

using namespace fullqwen;
namespace {
thread_local std::string last_error;
constexpr std::size_t WEIGHT_BYTES = 1192099840;
struct MappedWeights {
    const std::uint16_t* data = nullptr;
    explicit MappedWeights(const char* path) {
        if (!path) throw std::runtime_error("Weight image required");
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open weight image");
        struct stat info {};
        if (::fstat(fd, &info) || std::size_t(info.st_size) != WEIGHT_BYTES) {
            ::close(fd);
            throw std::runtime_error("Unexpected Qwen3-0.6B weight image size");
        }
        void* address = ::mmap(nullptr, WEIGHT_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (address == MAP_FAILED) throw std::runtime_error("Cannot map weight image");
        data = static_cast<const std::uint16_t*>(address);
    }
    ~MappedWeights() { if (data) ::munmap(const_cast<std::uint16_t*>(data), WEIGHT_BYTES); }
    MappedWeights(const MappedWeights&) = delete;
    MappedWeights& operator=(const MappedWeights&) = delete;
};

void validate_offsets(const std::uint64_t* offsets) {
    const std::size_t counts[LAYER_TENSORS] = {
        HIDDEN, Q_WIDTH * HIDDEN, KV_WIDTH * HIDDEN, KV_WIDTH * HIDDEN,
        HEAD_DIM, HEAD_DIM, HIDDEN * Q_WIDTH, HIDDEN,
        INTERMEDIATE * HIDDEN, INTERMEDIATE * HIDDEN, HIDDEN * INTERMEDIATE};
    for (int i = 0; i < TENSORS; ++i) {
        std::size_t size = i == 0 ? std::size_t(VOCAB) * HIDDEN : i == 1 ? HIDDEN : counts[(i - 2) % LAYER_TENSORS];
        if (offsets[i] > WEIGHT_BYTES / 2 || size > WEIGHT_BYTES / 2 - offsets[i])
            throw std::runtime_error("Tensor offset outside weight image");
    }
}

struct Context {
    int position = 0;
    bool have_logits = false, have_trace = false;
    std::array<int, 4> status {};
#ifdef QWEN_XRT
    xrt::device device;
    xrt::kernel kernel;
    xrt::bo weights, offsets_bo, frequencies, cache, logits, trace, result;
    Context(const char* image, const std::uint64_t* offsets, const float* freq,
            const char* xclbin, const char* bdf) : device(bdf) {
        if (!xclbin || !bdf) throw std::runtime_error("FPGA xclbin and BDF required");
        validate_offsets(offsets);
        MappedWeights source(image);
        const auto uuid = device.load_xclbin(xclbin);
        kernel = xrt::kernel(device, uuid, "qwen_full_decode");
        weights = xrt::bo(device, WEIGHT_BYTES, xrt::bo::flags::device_only, kernel.group_id(0));
        offsets_bo = xrt::bo(device, TENSORS * sizeof(std::uint64_t), kernel.group_id(1));
        frequencies = xrt::bo(device, HEAD_DIM / 2 * sizeof(float), kernel.group_id(2));
        cache = xrt::bo(device, CACHE_FLOATS * sizeof(float), xrt::bo::flags::device_only, kernel.group_id(3));
        logits = xrt::bo(device, VOCAB * sizeof(float), kernel.group_id(4));
        trace = xrt::bo(device, TRACE_FLOATS * sizeof(float), kernel.group_id(5));
        result = xrt::bo(device, status.size() * sizeof(int), kernel.group_id(6));
        const char* bytes = reinterpret_cast<const char*>(source.data);
        constexpr std::size_t chunk = 1024 * 1024;
        for (std::size_t offset = 0; offset < WEIGHT_BYTES; offset += chunk)
            weights.write(bytes + offset, std::min(chunk, WEIGHT_BYTES - offset), offset);
        offsets_bo.write(offsets);
        offsets_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        frequencies.write(freq);
        frequencies.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    void step(int token, bool emit_logits, bool emit_trace) {
        auto run = kernel(weights, offsets_bo, frequencies, cache, logits, trace, result,
                          token, position, int(emit_logits), int(emit_trace));
        if (run.wait(std::chrono::milliseconds{180000}) != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("Full FPGA decoder did not complete");
        result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        result.read(status.data());
    }
    void read_logits(float* output) {
        logits.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        logits.read(output);
    }
    void read_trace(float* output) {
        trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        trace.read(output);
    }
#else
    MappedWeights weights;
    std::array<std::uint64_t, TENSORS> layout;
    std::array<float, HEAD_DIM / 2> frequencies;
    std::vector<float> cache, logits, trace;
    Context(const char* image, const std::uint64_t* offsets, const float* freq,
            const char*, const char*) : weights(image), cache(CACHE_FLOATS), logits(VOCAB), trace(TRACE_FLOATS) {
        validate_offsets(offsets);
        std::copy_n(offsets, TENSORS, layout.begin());
        std::copy_n(freq, HEAD_DIM / 2, frequencies.begin());
    }
    void step(int token, bool emit_logits, bool emit_trace) {
        qwen_full_decode(weights.data, layout.data(), frequencies.data(), cache.data(),
            logits.data(), trace.data(), status.data(), token, position, emit_logits, emit_trace);
    }
    void read_logits(float* output) { std::copy(logits.begin(), logits.end(), output); }
    void read_trace(float* output) { std::copy(trace.begin(), trace.end(), output); }
#endif
};
}

extern "C" {
const char* qwen_full_error() { return last_error.c_str(); }
void* qwen_full_open(const char* image, const std::uint64_t* offsets, const float* frequencies,
                     const char* xclbin, const char* bdf) {
    try {
        last_error.clear();
        if (!offsets || !frequencies) throw std::runtime_error("Model descriptors required");
#ifdef QWEN_XRT
        if (!xclbin || !bdf) throw std::runtime_error("FPGA xclbin and BDF required");
#endif
        return new Context(image, offsets, frequencies, xclbin, bdf);
    } catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
void qwen_full_close(void* handle) { delete static_cast<Context*>(handle); }
int qwen_full_step(void* handle, int token, int emit_logits, int emit_trace, int* next_token) {
    try {
        if (!handle || !next_token) throw std::runtime_error("Invalid decoder handle/output");
        auto& context = *static_cast<Context*>(handle);
        if (token < 0 || token >= VOCAB || context.position >= CONTEXT)
            throw std::runtime_error("Token ID or 2048-token context limit exceeded");
        context.step(token, emit_logits, emit_trace);
        if (context.status[1] || context.status[2] != context.position + 1 || context.status[3] != LAYERS)
            throw std::runtime_error("Full decoder status/layer-count validation failed: " + std::to_string(context.status[1]));
        context.position = context.status[2];
        context.have_logits = emit_logits;
        context.have_trace = emit_trace;
        *next_token = context.status[0];
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int qwen_full_logits(void* handle, float* output) {
    try {
        if (!handle || !output || !static_cast<Context*>(handle)->have_logits)
            throw std::runtime_error("No logits available");
        static_cast<Context*>(handle)->read_logits(output);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int qwen_full_trace(void* handle, float* output) {
    try {
        if (!handle || !output || !static_cast<Context*>(handle)->have_trace)
            throw std::runtime_error("No diagnostic trace available");
        static_cast<Context*>(handle)->read_trace(output);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
}
