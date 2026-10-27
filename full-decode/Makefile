CXX ?= g++
PYTHON ?= python3
ROOT := $(abspath ..)
-include $(ROOT)/config/local.mk
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-unknown-pragmas -ffp-contract=off
PLATFORM ?= $(ROOT)/platforms/u55c/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
XRT_INCLUDE ?= $(firstword $(foreach dir,/opt/xilinx/xrt/include /usr/include $(ROOT)/.deps/xrt-dev/usr/include,$(if $(wildcard $(dir)/xrt/xrt_device.h),$(dir))))
XRT_LIB ?= $(firstword $(wildcard /opt/xilinx/xrt/lib/libxrt_coreutil.so /usr/lib/x86_64-linux-gnu/libxrt_coreutil.so /usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2))
VPP_LINK_FLAGS ?= --vivado.synth.jobs 2 --vivado.impl.jobs 1 --vivado.param general.maxThreads=2 --remote_ip_cache $(ROOT)/.ipcache --linkhook.do_first vpl.synth,$(ROOT)/scripts/use_vendor_clock_adapter.tcl

.PHONY: runtime runtime-cpu runtime-fpga compile kernel link resume-link
runtime: build/libfull_cpu.so build/libfull_fpga.so
runtime-cpu: build/libfull_cpu.so
runtime-fpga: build/libfull_fpga.so
build/libfull_cpu.so: host/full_runtime.cpp hls/full_decode.cpp hls/full_decode.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -fPIC -Ihls host/full_runtime.cpp hls/full_decode.cpp -o $@
build/libfull_fpga.so: host/full_runtime.cpp hls/full_decode.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -fPIC -DQWEN_XRT -Ihls -I$(XRT_INCLUDE) host/full_runtime.cpp $(XRT_LIB) -pthread -o $@
compile:
	mkdir -p build/hw reports
	v++ -c -t hw --platform "$(PLATFORM)" -k qwen_full_decode -Ihls hls/full_decode.cpp -o build/hw/qwen_full_decode.xo
link:
	v++ -l -t hw --platform "$(PLATFORM)" $(VPP_LINK_FLAGS) --save-temps --config hls/connectivity.cfg build/hw/qwen_full_decode.xo -o build/hw/qwen_full_decode.xclbin
kernel: compile
	$(MAKE) link
resume-link:
	/usr/bin/python3 -I $(ROOT)/scripts/use_vendor_clock_adapter.py --build-root $(CURDIR)/_x/link/vivado/vpl --apply
	v++ -l -t hw --platform "$(PLATFORM)" $(VPP_LINK_FLAGS) --save-temps --from_step vpl.synth --config hls/connectivity.cfg build/hw/qwen_full_decode.xo -o build/hw/qwen_full_decode.xclbin
