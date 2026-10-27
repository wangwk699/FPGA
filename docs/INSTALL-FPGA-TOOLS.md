# U55C 编译工具安装准备（2026-10-26）

> 这是 2026-10-26 的安装与混合部署历史记录。2026-10-27 已完成全部 Decoder 上 FPGA；新机器请按 [从零部署与 GitHub 发布指南](FROM-ZERO-TO-GITHUB.md) 操作。

最终状态：2026-10-26 21:47，不量化的 CPU + U55C 混合推理已实机验证通过，全部输出 token 与 CPU 参考一致。
FPGA 执行解码阶段全部 197 个线性层，CPU 执行 prefill、Attention 等算子；完整 Decoder 全部上 FPGA 尚未实现。
直接运行 `/home/wangwk/qwen3-u55c/run_hybrid.sh --backend fpga --max-new-tokens 32`，无需再次启动安装器或编译器。
最终结果见 `../reports/FINAL-DEPLOYMENT-20261026.json`。

## device-only 权重分块修复

首次真实模型运行在 XRT 子 BO 构造时报错 `device only buffer has no host buffer`。
主机运行库已改为保留父 BO 绑定，启动每个分块前设置对应的 64 位设备地址，继续使用 device-only 权重和 1MiB 上传块。
无需重新编译电路。4103×1024 跨分块、17×3072 最大列数实机回归均通过，误差均为 0；之后真实模型推理通过。
已有电路需重新编译主机代码并验证时，可用 `scripts/auto_deploy.py --verify-existing`；这会重新运行硬件测试和模型推理，不重复下载或综合。

## 平台时钟适配器兼容性修复

深入检查 OOC 子任务日志发现，6 个综合任务因平台包中
`pxi_ii_infra/clk_metadata_adapter_v1_0` 的加密 Verilog 无法解析而失败。
总日志的任务计数会包含已经失败的任务，因此应同时检查 `deployment_status.py` 输出的失败子任务列表。

同一个 AMD 平台包在 `other_ipdefs/clk_metadata_adapter_v1_0` 提供了相同 VLNV、相同端口的明文实现，功能为 `assign clk_out=clk_in`。
独立综合中，加密版本复现失败，厂商提供的明文版本通过；记录见 `reports/CLOCK-METADATA-COMPATIBILITY-20261026.json`。
`scripts/use_vendor_clock_adapter.py` 先验证两份文件的 SHA256，再备份并替换派生构建目录中的对应文件，保留完整厂商版权声明。
它不会更改原始 XSA、平台包、静态 shell 或模型。

Makefile 已配置综合前钩子 `scripts/use_vendor_clock_adapter.tcl`，通过 Vitis 官方 `--linkhook.do_first vpl.synth,...` 运行，重置失败任务并复用其余检查点。
已有构建可通过 `scripts/auto_deploy.py --resume-link` 从 `vpl.synth` 继续。当前构建仍运行时不要启动第二个流程。
已有工程断点恢复可能保留旧的钩子配置，因此恢复目标还会在启动 v++ 前显式选择已生成目录中的适配器源码。
19:59 启动的恢复轮次实际复用了 IP 综合缓存；原先失败的 6 个任务全部通过，记录见 `reports/CLOCK-ADAPTER-RECOVERY-20261026.json`。之后整体电路构建和模型实机验证均已完成。

官方参数依据：

- [linkhook](https://docs.amd.com/r/2025.1-English/ug1702-vitis-accelerated-reference/linkhook-Options)
- [from_step](https://docs.amd.com/r/2025.1-English/ug1702-vitis-accelerated-reference/v-General-Options)

当前进展：两个下载文件已校验，U55C 开发平台已导入且接口匹配；AMD 认证已成功。
Vitis 与补充 HBM 器件组件均已安装，安装目标 `/home/wangwk/amd`；v++、Vivado 版本及平台检查通过，`.xo`、`.xclbin` 生成和实机验证均已完成。

## 平台链接恢复（2026-10-26 17:34）

`create_bd` 在创建 `clkwiz_hbm` 时报告速度文件缺少 MMCM/PLL 抖动与相位误差参数，并提示时钟区域文件不存在。
`scripts/check_u55c_clock.tcl` 用独立工程复现同一错误，说明不依赖模型或 GEMV 内核。
安装记录只选择了 Alveo/edge 组件，未选择 Virtex UltraScale+ HBM；本地 `xcu55c` 目录缺少速度文件。
已通过官方 `xsetup -b Add` 补充该 HBM 器件系列，新增下载约 7.77GB；之后独立时钟 IP 检查实际通过，`reports/auto-check_u55c_clock.log` 包含 `U55C_CLOCK_IP_CREATED`，原先错误已解除。
配置文件为 `config/vitis-2025.1-hbm.txt`，恢复入口为 `scripts/auto_deploy.py --add-hbm`。该恢复步骤及后续硬件构建已完成，无需重复启动或重新下载。
初次失败日志保存在 `reports/build-kernel-before-hbm-recovery-*.log`；后续实际 FPGA 验证已通过，见最终报告。

## 安装收尾恢复（2026-10-26 16:55）

安装器在生成器件列表时因 Vivado 找不到 `libtinfo.so.5` 停滞。Ubuntu 26 未匹配到工具自带的 Ubuntu 24 兼容库目录。
已在 Vivado 和 Vitis 各自的 `lib/lnx64.o/libtinfo.so.5` 创建到 `Ubuntu/24/libtinfo.so.5` 的本地链接；没有修改系统库。
随后实际重新生成器件列表，结束原先停滞的 Vivado 子进程，安装器完成收尾并生成 `settings64.sh`。
恢复记录见 `reports/INSTALL-POSTPROCESS-RECOVERY-20261026.json`；Vitis HLS 已开始对 `xcu55c-fsvh2892-2L-e` 执行 `csynth_design`。

如编译任务停止且需要重新启动，可使用 `scripts/auto_deploy.py --build-only`，直接使用已安装工具继续构建和验证，不再运行下载器。正在运行时不要重复启动。

## 后台流程与进度

自动流程直接启动并监管安装器，记录退出码及脱敏失败原因，本次流程已完成。
组件下载因 TLS/网络或校验错误失败时，最多尝试 3 次并复用缓存；认证失败、磁盘不足和主动取消不会反复重试。
安装成功后才自动依次执行：

1. 检查 v++、Vivado 和平台信息。
2. 编译 qwen_gemv.xclbin（降低并行任务数以控制内存占用）。
3. 编译主机程序与 XRT 调用库。
4. 在 U55C 上运行合成 GEMV 探针。
5. 运行真实模型的 CPU/FPGA 混合推理，并校验实际 FPGA 调用记录。

非网络错误或下载重试耗尽后会停止并记录错误；只有真实 FPGA 推理报告验证通过后才会标记部署完成。
这是 CPU prefill/Attention + FPGA decode 线性层的方案，不是整个 Decoder 都在 FPGA 执行。

查看当前进度：

```bash
cd /home/wangwk/qwen3-u55c
python3 scripts/deployment_status.py
```

- 当前状态：`reports/auto-deploy-status.json`
- 安装输出：`reports/vitis-install-resume.log`
- 后台流程日志：`reports/auto-deploy-worker.log`
- 阶段输出：`reports/auto-*.log`
- 成功后真实推理结果：`reports/hybrid-fpga-auto.json`

当前流程已完成，没有再次认证或重新启动安装器的需要。
本次没有修改系统 XRT 或刷新板卡持久固件；实机验证已加载新的用户计算电路。

已导入的平台路径：

```text
/home/wangwk/qwen3-u55c/platforms/u55c/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

Makefile 已默认采用此平台。

## 版本选择和当前机器条件

选用 AMD U55C 官方下载页列出的 Vitis 2025.1 工具流。
所需开发平台为 `xilinx_u55c_gen3x16_xdma_3_202210_1`，与当前 base_3 shell 对应。

当前主机检查结果：Ubuntu 26.04.1、约 32GB 物理内存、8GB swap、约 429GB 磁盘可用。
Vitis 2025.1 官方支持 Ubuntu 22.04/24.04，未列出 26.04；加速开发最低 64GB 内存、推荐 80GB，完整安装磁盘约 200GB（随所选组件变化）。
这意味着磁盘空间足够准备工具，但当前 OS 不在该版本支持列表，内存低于官方加速开发最低值。
建议使用 Ubuntu 24.04 且至少 64GB 内存的构建机，或者由管理员提供符合条件的构建环境。不要为安装编译器直接重装当前已经能使用 U55C 的主机。
构建机不必安装 U55C；生成 xclbin 后可以复制回本机进行实机验证。

## 需要用户通过 AMD 登录下载的两个文件

1. AMD Unified Installer 2025.1 的 Linux Self Extracting Web Installer，约 335.53MB：

   `FPGAs_AdaptiveSoCs_Unified_SDI_2025.1_0530_0145_Lin64.bin`

   下载页：https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2025-1.html

2. U55C Development Target Platform，约 165.05MB：

   `xilinx-u55c-gen3x16-xdma-3-202210-1-dev_1-3514517_all.deb`

   下载页：https://www.amd.com/en/support/downloads/alveo-downloads.html/accelerators/alveo/u55c.html

下载阶段两条链接均要求 AMD 登录。用户现已下载并提供这两个文件；以下下载信息保留作为记录。
可将文件保存在 `/home/wangwk/qwen3-u55c/installers/`；若使用另一台构建机，可放到该机器的用户目录。
Linux Web Installer 只是启动安装器；运行时还需下载选中的实际工具组件。

## 安装器中需要选择的组件

- 产品选择完整 **Vitis**（包括 Vivado 和 v++），不是 Vitis Embedded Development。
- 启用 **Install devices for Alveo and Edge acceleration platforms**。
- 同时启用 **Virtex UltraScale+ HBM FPGAs**。当前默认配置已修正为启用该系列；初次仅选择 Alveo 的配置保存在 `reports/vitis-2025.1-initial-alveo-only.txt`。
- 安装到有足够空间的目录，使用用户目录可避免主工具安装依赖 root；系统依赖若缺失仍可能需要管理员配置。
- 无需为此重新安装已经正常工作的主机 XRT 或刷新 U55C shell。

在满足条件的构建环境运行下载的 Linux 安装器：

```bash
chmod u+x FPGAs_AdaptiveSoCs_Unified_SDI_2025.1_0530_0145_Lin64.bin
./FPGAs_AdaptiveSoCs_Unified_SDI_2025.1_0530_0145_Lin64.bin
```

Web Installer 要求用户自己登录 AMD 并接受适用协议。

## 导入平台和构建

开发平台已按下面方法解包到用户目录；没有执行 deb 安装脚本，也没有刷新板卡（无需再次解包）：

```bash
cd /home/wangwk/qwen3-u55c
mkdir -p platforms/u55c
dpkg-deb -x installers/xilinx-u55c-gen3x16-xdma-3-202210-1-dev_1-3514517_all.deb platforms/u55c
find platforms/u55c -name '*.xpfm'
```

加载实际安装目录中的 Vitis `settings64.sh` 后检查：

```bash
v++ --version
vivado -version
```

用上面找到的真实 `.xpfm` 路径构建：

```bash
make kernel PLATFORM=/absolute/path/to/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

生成后先在装有 U55C 的机器上执行 GEMV 探针，再运行真实模型混合推理。下面这些步骤已实测通过，可用于复现：

```bash
./build/gemv_probe build/hw/qwen_gemv.xclbin 0000:01:00.1
./run_hybrid.sh --backend fpga --xclbin build/hw/qwen_gemv.xclbin \
  --max-new-tokens 32 --report reports/hybrid-fpga.json
```

## 官方依据

- U55C 下载页及开发平台：https://www.amd.com/en/support/downloads/alveo-downloads.html/accelerators/alveo/u55c.html
- Vitis 2025.1 系统要求：https://docs.amd.com/r/2025.1-English/ug1742-vitis-release-notes/Installation-Requirements
- 完整 Vitis 和 Alveo 设备选项：https://docs.amd.com/r/2025.1-English/ug1742-vitis-release-notes/Installing-the-Vitis-Software-Platform
