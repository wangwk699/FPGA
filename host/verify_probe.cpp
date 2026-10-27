#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: verify_probe official-verify.xclbin [device BDF]\n";
    return 2;
  }
  try {
    xrt::device device(argc > 2 ? argv[2] : "0000:01:00.1");
    auto uuid = device.load_xclbin(argv[1]);
    xrt::kernel kernel(device, uuid, "verify");
    xrt::bo buffer(device, 4096, kernel.group_id(0));
    auto* data = buffer.map<char*>();
    std::memset(data, 0, 4096);
    buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kernel(buffer);
    run.wait();
    buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (std::memcmp(data, "Hello World", 11) != 0)
      throw std::runtime_error("Unexpected official verify output");
    std::cout << "PASS: AMD verify kernel executed on U55C and returned Hello World\n";
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
