CXX ?= g++
PYTHON ?= python3
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-unknown-pragmas -ffp-contract=off
-include config/local.mk
XRT_ROOT ?= $(CURDIR)/.deps/xrt-dev/usr
XRT_INCLUDE ?= $(XRT_ROOT)/include
XRT_LIB ?= /usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2
PLATFORM ?= $(CURDIR)/platforms/u55c/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
TARGET ?= hw
VPP_LINK_FLAGS ?= --vivado.synth.jobs 1 --vivado.impl.jobs 1 --vivado.param general.maxThreads=2 --linkhook.do_first vpl.synth,$(CURDIR)/scripts/use_vendor_clock_adapter.tcl

.PHONY: test host kernel kernel-link-resume clean csim runtime
runtime: build/libqwen_runtime_cpu.so build/libqwen_runtime_fpga.so
build/libqwen_runtime_cpu.so: host/linear_runtime.cpp hls/gemv.cpp hls/gemv.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -fPIC -Ihls host/linear_runtime.cpp hls/gemv.cpp -o $@
build/libqwen_runtime_fpga.so: host/linear_runtime.cpp hls/gemv.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -fPIC -DQWEN_XRT -Ihls -I$(XRT_INCLUDE) host/linear_runtime.cpp $(XRT_LIB) -pthread -o $@
csim: build/libqwen_gemv_csim.so
build/libqwen_gemv_csim.so: hls/gemv.cpp hls/gemv.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -fPIC -Ihls hls/gemv.cpp -o $@
test: build/gemv_test build/libqwen_runtime_cpu.so
	./build/gemv_test
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

build/gemv_test: hls/gemv.cpp hls/gemv.hpp tests/gemv_test.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Ihls hls/gemv.cpp tests/gemv_test.cpp -o $@

host: build/dma_probe build/gemv_probe build/verify_probe
build/%: host/%.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -I$(XRT_INCLUDE) $< $(XRT_LIB) -pthread -o $@

kernel:
	@command -v v++ >/dev/null || { echo 'Missing Vitis v++ compiler'; exit 1; }
	@test -n "$(PLATFORM)" || { echo 'Set PLATFORM=/absolute/path/to/U55C.xpfm'; exit 1; }
	mkdir -p build/$(TARGET)
	v++ -c -t $(TARGET) --platform "$(PLATFORM)" -k qwen_gemv -Ihls hls/gemv.cpp -o build/$(TARGET)/qwen_gemv.xo
	v++ -l -t $(TARGET) --platform "$(PLATFORM)" $(VPP_LINK_FLAGS) --config hls/connectivity.cfg build/$(TARGET)/qwen_gemv.xo -o build/$(TARGET)/qwen_gemv.xclbin

kernel-link-resume:
	@test -s build/$(TARGET)/qwen_gemv.xo || { echo 'Missing compiled kernel .xo'; exit 1; }
	@test -s _x/link/vivado/vpl/prj/prj.xpr || { echo 'Missing existing Vivado project'; exit 1; }
	$(PYTHON) scripts/use_vendor_clock_adapter.py --apply
	v++ -l -t $(TARGET) --platform "$(PLATFORM)" $(VPP_LINK_FLAGS) --from_step vpl.synth --save-temps --config hls/connectivity.cfg build/$(TARGET)/qwen_gemv.xo -o build/$(TARGET)/qwen_gemv.xclbin

clean:
	rm -rf build
