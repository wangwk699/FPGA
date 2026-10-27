# Qwen3-0.6B 完整 FPGA Decode 性能优化

本目录保留第一轮优化的 2.737 token/s 版本。当前已验证版本为 [full-decode-stream](../full-decode-stream/README.md)，实测 8.898 token/s，再加速 3.25 倍，并继续使用 FPGA HBM 中的 FP32 KV Cache。

已于 2026-10-27 在 U55C 实机验证：增量 Decode 从 **0.06268 提升到 2.737 token/s，加速 43.67 倍**。全部 22 个输出 token 与基线及 PyTorch FP32 参考一致，五组数值检查全部通过。

这是独立优化工程。`../full-decode/` 中的源代码、位流、原生库和报告作为基线保留；复用同一份原始 BF16 权重和离线参考，不下载模型、不量化，模型计算全部在 FPGA。

## 实测结果

| 指标 | 原完整 FPGA 基线 | 优化版 |
| --- | ---: | ---: |
| 增量 Decode | 0.06268 token/s | **2.737 token/s** |
| 平均每个增量 token | 15.954 秒 | **0.3654 秒** |
| 21 个增量 Decode 步 | 335.038 秒 | **7.673 秒** |
| 18 个输入 token 的 prefill | 215.963 秒 | **4.297 秒** |
| 含加载的整次运行 | 566.418 秒 | **27.960 秒** |

比较使用相同输入、相同 22 个输出 token（含 EOS）、同一板卡、相同 FP32 数值检查。Decode 加速 43.67 倍，prefill 约 50.26 倍，整次运行约 20.26 倍。首次验证的初始化与加载耗时约 15.96 秒，包含新位流加载，单独计入整次运行，不计入增量 Decode 速度。上述速度属于该提示下的实测，不代表所有上下文长度。

普通运行入口另行实测通过：相同 22 个输出 token 全部一致，诊断张量回读为 0；再次运行同一位流时，初始化与加载约 0.325 秒、prefill 4.286 秒、Decode 7.669 秒（2.738 token/s），总计 **12.310 秒**。模型权重仍由入口加载；这次运行不包含更换位流的开销。见 [普通入口验证](reports/production-validation.json)。两次运行后 XRT 观察到新位流完整 Decode CU 累计调用 78 次，状态正常。

证据：[性能比较](reports/PERFORMANCE.json)、[完整实机结果](reports/full-fpga.json)、[最终部署报告](reports/FINAL-DEPLOYMENT.json)。布线后 setup/hold/pulse-width 失败端点均为 0，WNS +0.003ns、WHS +0.009ns，115 条总线偏斜约束全部通过。

## 优化内容

原矩阵乘和词表投影的内层循环每 8 个时钟启动一次标量乘加。优化版本使用显式 512 位权重接口，每拍读取 32 个原始 BF16 权重，精确展开成 FP32 后并行相乘，经过平衡加法树和 16 路交错 FP32 累加器归约。输入激活按 32 个 bank 划分，支持并行读取。

交错累加保留真实的 16 次迭代依赖距离，没有将真实累加依赖标记为不存在。并行归约改变浮点加法顺序，因此必须重新逐层比较数值结果，不能假定与标量版本逐位相同。

全部 28 层、Attention、RoPE、KV cache、归一化、MLP、残差、embedding、最终投影和 argmax 仍在 FPGA。CPU 负责分词、一次性权重加载、提交 token ID 和显示。KV cache 仍为 HBM 中的 FP32，层间激活仍在 FPGA。

## 验证方法

软件阶段使用完整模型和既有 PyTorch FP32 参考，检查首/末 prefill token、首个增量 Decode token 的全部层输出，以及两组 logits；还要求全部 22 个输出 token ID 一致。普通 C++ 验证的耗时不作为 FPGA 加速证据。

硬件阶段重新构建位流，检查布线时序与总线偏斜，实际运行 U55C，再做相同数值和输出检查。速度对比使用同一组 18 个输入 token、22 个输出 token（含 EOS），比较 21 个增量 Decode 步的时间，加载和 prefill 单独统计。

基线来自 `../full-decode/reports/full-fpga.json`。实际硬件运行已完成 39 次完整内核调用、1,092 次层计算，首个增量 Decode 的 logits 最大绝对误差 `4.00543e-5`、相对 L2 误差 `1.69943e-6`。查看完成状态：

```bash
python3 ./scripts/deployment_status.py
```

查看旧完整 FPGA 基线状态可加 `--baseline`；旧混合方案状态可加 `--hybrid`。

## 运行与恢复

可直接运行，无需重新编译：

```bash
./full-decode-fast/run_full.sh --backend fpga \
  --prompt '请用一句话介绍杭州。' --max-new-tokens 32
```

batch=1，输入加输出最多 2048 token，greedy，关闭 thinking。缺少真实位流时明确退出，不回退 CPU。可用 `--verify ./full-decode-fast/reference/default` 读取诊断结果做离线参考比较，普通运行仅回读 token/status。

构建入口：`scripts/deploy_full.py`。可使用 `--from-stage link`、`--from-stage resume-link` 或 `--from-stage verify` 恢复到相应阶段。该入口只在通过实机验证且增量 Decode 至少加速两倍后更新默认部署配置；此前的可运行版本保留。
