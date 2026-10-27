#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    xrt::device device(argc > 1 ? argv[1] : "0000:01:00.1");
    if (argc > 3) device.load_xclbin(argv[3]);
    constexpr std::size_t bytes = 4 * 1024 * 1024;
    std::vector<std::uint32_t> expected(bytes / 4);
    const unsigned bank_count = argc > 2 ? std::stoul(argv[2]) : 1;
    if (bank_count == 0 || bank_count > 32) throw std::runtime_error("bank count must be 1..32");
    for (unsigned bank = 0; bank < bank_count; ++bank) {
      for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = std::uint32_t(i * 2654435761u) ^ (bank * 0x10203u);
      xrt::bo buffer(device, bytes, bank);
      auto* mapped = buffer.map<std::uint32_t*>();
      std::memcpy(mapped, expected.data(), bytes);
      auto start = std::chrono::steady_clock::now();
      buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      // Clear the host mapping: checking it unchanged would not prove a readback.
      std::memset(mapped, 0, bytes);
      buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto end = std::chrono::steady_clock::now();
      if (std::memcmp(mapped, expected.data(), bytes))
        throw std::runtime_error("readback mismatch on bank " + std::to_string(bank));
      double seconds = std::chrono::duration<double>(end - start).count();
      std::cout << "HBM[" << bank << "] PASS, 4 MiB roundtrip, "
                << (2.0 * bytes / 1e6 / seconds) << " MB/s combined DMA\n";
    }
    std::cout << "PASS: " << bank_count << " enabled HBM bank(s), sampled DMA roundtrip only; not a full-memory or compute test.\n";
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
