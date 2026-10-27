# FPGA

Qwen3-0.6B / Alveo U55C 浮点部署工程。

**同服务器、家目录完全隔离的用户请先按 [独立用户部署清单](docs/ISOLATED-USER-CHECKLIST.md) 操作。**

**完整从零复现说明：[从零部署与 GitHub 发布指南](docs/FROM-ZERO-TO-GITHUB.md)。** 包含全部下载项、环境安装、权重准备、软硬件验证、常见故障、源码清单及上传方式。当前主线为 `full-decode-stream/`；历史测量摘要见 [benchmark-summary.json](docs/evidence/benchmark-summary.json)。源码发布包不包含本机 `reports/`、模型和位流，文中这些历史产物链接需在原部署机器查看或自行重建。

**完整 FPGA Decode 已于 2026-10-27 实机验证通过**：28 层的 RMSNorm、QKV/QK Norm、RoPE、完整 Attention、KV cache、MLP、残差以及 embedding/输出投影/argmax 均在 U55C 执行。CPU 负责分词、加载和调度；模型不量化，层间激活与 KV 不回传 CPU 参与计算。22 个生成 token 与 PyTorch 参考完全一致，39 次完整内核调用覆盖 1,092 次层计算，五组层输出/logits 数值检查全部通过。

直接运行：

```bash
./full-decode-stream/run_full.sh --backend fpga \
  --prompt '请用一句话介绍杭州。' --max-new-tokens 32
```

**第二轮性能优化已实机验证通过**：采用 16 行交错矩阵乘、512 位读取及成对 Q head 的 KV 复用，增量 Decode 从上一版 `2.737` 提升到 **`8.898 token/s`，再加速 3.25 倍**，相对最初 `0.06268 token/s` 约快 142 倍；平均 token 延迟为 0.1124 秒。同一提示下 prefill 为 1.47 秒，同位流普通运行整次约 4.15 秒。输入和全部输出 token 完全一致，原始 BF16 权重、FP32 计算、完整 FPGA 执行均通过验证。KV Cache 一直启用，FP32 数据存放于 HBM，72-token 历史输入验证也已通过。

运行方式、限制和证据见 [当前完整 Decode 工程](full-decode-stream/README.md) 与 [性能报告](full-decode-stream/reports/PERFORMANCE.json)。上一优化版保留在 `full-decode-fast/`，原完整 FPGA 基线保留在 `full-decode/`；下文保留此前混合部署的运行方式和历史结果。

截至 2026-10-26，**不量化的 CPU + U55C 混合推理已实机验证通过**：CPU 执行 prefill、Attention 等算子，FPGA 执行解码阶段全部 197 个线性层。
完整回答的所有 token 与 CPU 参考一致，FPGA 原生线性层调用 4,137 次。该方案不是整个 Decoder 都在 FPGA 执行。
构建、运行及数值证据见 [最终报告](reports/FINAL-DEPLOYMENT-20261026.json) 和 [执行说明](reports/DEPLOYMENT-20261026.md)。

模型使用用户已经下载到 `/home/share` 的原始 BF16 权重。本次没有重新下载模型、没有量化、没有修改共享模型文件。
矩阵运算将 BF16 精确扩展为 FP32，使用 FP32 累加；CPU 上的激活和 KV cache 为 FP32。

## 已验证结果

- 模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`。
- 权重文件 SHA256 与共享缓存 blob 文件名一致；完整配置见 `deployment.json`。
- 310 个独立 BF16 张量，596,049,920 个参数，1,192,099,840 字节张量数据。
- 无损导出为 3 个分片，每片小于 480MiB；共享 embedding/lm_head 不重复存储，全部张量回读 SHA256 校验通过。
- CPU 推理成功；KV cache 单步解码与完整前缀重算最大绝对误差 `3.05176e-5`。
- 197 项真实线性层测试全部通过，包含所有 28 层的线性层与输出投影；最大绝对误差 `3.96729e-4`，最大相对 L2 误差 `1.27333e-6`。
- 完整解码程序替换全部 197 个线性层后，调用 C++ GEMV 4,137 次，生成 token 与 PyTorch 参考完全一致；首个解码步骤 logits 最大绝对误差 `4.95911e-5`。
- 23 项离线单元测试及 5 组 GEMV 数值测试通过；CPU 和 XRT 原生调用库均编译成功。
- Vitis 2025.1 实际生成 `build/hw/qwen_gemv.xclbin`；141 个 IP 综合任务、顶层综合、布局布线和 bitstream 生成全部通过。
- 布线后所有指定时序约束满足：WNS 0.003ns、WHS 0.009ns，setup/hold 失败端点均为 0；115 条总线偏斜约束全部通过。
- U55C 合成 GEMV 探针、device-only 权重跨 4096 行分块及最大列数 3072 的实机回归全部通过，最大绝对误差均为 0。
- 真实模型 FPGA 混合推理完成 21 个原生解码步骤、4,137 次线性层调用；22 个输出 token（含 EOS、含首枚 CPU prefill token）与 CPU 参考完全一致。
- 首个 FPGA 解码步骤 logits 最大绝对误差 `4.95911e-5`，相对 L2 误差 `1.69972e-6`；本次运行耗时约 335.74 秒。

## 运行 U55C 混合推理

```bash
./run_hybrid.sh --backend fpga \
  --prompt '请用一句话介绍杭州。' --max-new-tokens 32
```

无需手动激活 Conda 或加载 Vitis 环境。入口默认使用现有 `llm` 环境、已生成的 xclbin 和固定 revision 的本地模型，完全离线。
首枚 token 来自 CPU prefill；后续解码每步的 197 个线性层在 U55C 执行，Attention、归一化、RoPE、KV cache、残差和采样仍由 CPU 执行。
当前支持 batch=1，输入加输出最多 2048 token；关闭 thinking，使用 greedy 解码。
这是正确性优先的顺序浮点实现，本次完整回答约用 5 分 36 秒，尚未进行吞吐优化。

已验证输出：

> 杭州是中国浙江省的省会，位于中国东南部，以西湖闻名，是著名的旅游城市。

查看完成状态：`python3 ./scripts/deployment_status.py`。

## 离线 CPU 参考

```bash
cd "$HOME/FPGA"
./run_cpu.sh --prompt '请用一句话介绍杭州。' --max-new-tokens 64
```

该入口使用现有 `llm` Conda 环境，无需再执行 `conda activate`，默认加载已固定 revision 的共享模型；完全离线，不重复下载。
需要更换 Python 时可设置 `QWEN_PYTHON=/path/to/python`；模型路径可通过 `--model` 指定。
最大输入加输出为 2048 token，batch=1，关闭 thinking，greedy 解码。

## 软件验证混合推理调用链

```bash
make runtime
./run_hybrid.sh --backend csim --max-new-tokens 32 \
  --report reports/my-csim-run.json
```

这里的 `csim` 是用普通 g++ 执行相同 GEMV C++ 源码，并非 FPGA 执行，也不是 Vitis RTL 协同仿真。
架构为 CPU prefill，解码时所有线性层调用原生 GEMV；Attention、归一化、RoPE、KV cache、残差和采样仍由 CPU 执行。
第一枚输出 token 来自 CPU prefill；后续 token 的 197 个线性层均通过原生调用接口执行。
默认用 PyTorch 检查首个原生解码步骤的 logits；native 运行结果写明 `backend` 和 `fpga_executed`。

## 已部署的工具与重新构建

开发平台、Vitis 2025.1 与 HBM 器件组件均已配置；完整硬件构建和模型实机推理已通过。安装与兼容性修复过程见 [工具说明](docs/INSTALL-FPGA-TOOLS.md)。当前机器的 OS 和内存仍不在 Vitis 2025.1 官方加速开发要求内，本项目记录的是本机这次实际通过的结果。

已验证环境：

- 板卡：`0000:01:00.1`，XRT 2.21.75，Ready。
- shell：`xilinx_u55c_gen3x16_xdma_base_3`。
- 开发平台：`xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm`。
- 平台接口 UUID：`b7ac1abe1e3e1cb686d5a81232452676`。

已有电路可直接运行，无需重新编译。若修改 HLS 内核后需要重建：

```bash
cd "$HOME/FPGA"
source "$HOME/amd/2025.1/Vitis/settings64.sh"
make kernel
./build/gemv_probe build/hw/qwen_gemv.xclbin 0000:01:00.1
```

合成数据探针通过后，再验证真实模型的混合推理：

```bash
./run_hybrid.sh --backend fpga \
  --xclbin build/hw/qwen_gemv.xclbin \
  --max-new-tokens 32 --report reports/hybrid-fpga.json
```

上述构建和实机验证已完成，结果见 `reports/auto-gemv_hardware_probe.log`、`reports/fpga-runtime-probe-auto.json` 和 `reports/hybrid-fpga-auto.json`。
没有 xclbin 时 FPGA 入口会明确退出，不会自动切换到 CPU。
此处 `run_hybrid.sh` 的硬件方案是 CPU+FPGA 混合推理；整个 Decoder 的硬件实现及验证进展见独立的 `full-decode/` 工程。
当前 GEMV 优先保证正确性，未做并行度、流水线、时序及吞吐优化，不承诺加速收益。

XRT 权重使用 device-only BO，按 1MiB 块上传，输入/输出只用小型主机映射缓冲区，避免将整个模型锁页。
权重端口连接 HBM[0:31]，输出词表投影按最多 4096 行分块执行。
这套内存连接与分块方案已经实机验证。device-only 权重通过父 BO 绑定及 64 位设备地址偏移分块，不创建需要主机映射的子 BO。

## 主要文件

| 文件 | 用途 |
|---|---|
| `deployment.json` | 本地模型位置、原始文件哈希及当前部署状态 |
| `run_cpu.sh` | 用户可直接运行的离线 CPU 入口 |
| `run_hybrid.sh` | C++ 软件验证 / FPGA 混合推理入口 |
| `scripts/reference.py` | 保存首层输出及全部线性层真实输入/输出，检查 KV cache |
| `scripts/validate_real_gemv.py` | 对所有 197 个线性层做真实权重验证 |
| `scripts/export_weights.py` | 无损导出与 SHA256 回读校验 |
| `host/linear_runtime.cpp` | CPU/XRT 原生 GEMV C 接口 |
| `scripts/native_linear.py` | PyTorch 线性层与原生接口集成 |
| `hls/gemv.cpp` | BF16 权重、FP32 计算内核源码 |
| `build/hw/qwen_gemv.xclbin` | 已实机验证的 U55C 电路 |
| `scripts/validate_fpga_runtime.py` | device-only 上传与非零分块偏移的真实 FPGA 回归 |
| `reports/FINAL-DEPLOYMENT-20261026.json` | 最终结果、xclbin 哈希、数值误差与证据路径 |
| `weights/qwen3-0.6b/manifest.json` | 已导出张量与分片信息 |
| `reports/DEPLOYMENT-20261026.md` | 本次实际完成项、证据和阻塞项 |

## 板卡检查历史

2026-10-25 已运行 AMD 官方 verify，返回 Hello World；临时加载匹配的官方 bandwidth 电路后，32 个 HBM bank 的采样 DMA 均通过，并恢复原 verify 电路。
2026-10-26 已加载自定义 qwen_gemv 电路并完成真实模型混合推理；没有刷新板卡持久固件。板卡 HEALTHY，计算单元为 `qwen_gemv:qwen_gemv_1`。
旧验卡记录见 `reports/RESULT.md`；本次模型验证记录以 `reports/DEPLOYMENT-20261026.md` 为准。

## 官方来源

- [Qwen3-0.6B 模型配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [Qwen3 官方实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
- [AMD U55C 下载页](https://www.amd.com/en/support/downloads/alveo-downloads.html/accelerators/alveo/u55c.html)
- [U55C base_3 平台说明](https://docs.amd.com/r/en-US/ug1120-alveo-platforms/U55C-Gen3x16-XDMA-base_3-Platform)
