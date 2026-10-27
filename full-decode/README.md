# Qwen3-0.6B 完整 FPGA Decode

本文描述初版完整 FPGA 基线。当前已验证版本将增量 Decode 从 0.06268 提升到 8.898 token/s（约 142 倍），新入口是 [full-decode-stream/run_full.sh](../full-decode-stream/run_full.sh)，详情见 [当前版本说明](../full-decode-stream/README.md)。第一轮 2.737 token/s 版本保留在 `../full-decode-fast/`，本目录保留最初基线程序、位流和原始结果供复现比较。

已于 2026-10-27 在 U55C 实机验证通过：28 层 Decoder 的完整单 token 前向计算均在 FPGA 执行，保留原始 BF16 权重并精确扩展为 FP32 运算，不量化。原有 CPU+FPGA 混合部署的程序和位流仍保留。

## 执行划分

| 运算 | 执行位置 |
| --- | --- |
| Token embedding | FPGA |
| 每层输入/后注意力 RMSNorm | FPGA |
| Q/K/V、O、gate/up/down 投影 | FPGA |
| Q/K head RMSNorm、RoPE | FPGA |
| KV cache 写入及读取 | FPGA，FP32 cache 在 HBM |
| GQA 的 QKᵀ、缩放、因果 softmax、AV | FPGA |
| SiLU、逐元素乘法、两次残差相加 | FPGA |
| 最终 RMSNorm、共享 lm_head、greedy argmax | FPGA |
| 分词、模型初始化、提交 token ID、显示输出 | CPU |

28 层复用同一个硬件引擎，按模型顺序执行；每个 token 只调用一次完整内核。层间激活留在片上，KV cache 留在 HBM，正常推理不逐层经过 PCIe，也不在 CPU 上执行 PyTorch 模型。输入提示也按 token 顺序在 FPGA 上 prefill。

当前接口为 batch=1、输入加输出最多 2048 token、greedy、关闭 thinking。这里对齐的是模型计算的设备划分；当前为顺序浮点实现，速度仍需并行化优化，尚未做 GPU 性能对标。

## 当前验证与状态

实际 FPGA 推理已通过五组数值检查：首/末 prefill token 和首个增量 decode token 各 30 个输出槽位，以及 prefill/首个 decode 的两组 logits。22 个生成 token（含 EOS）与 PyTorch FP32 参考完全一致。18 个输入 token 和 21 个增量 decode 步共调用完整内核 39 次，覆盖 1,092 次层计算。首个 decode 的 logits 最大绝对误差为 `6.00815e-5`，相对 L2 误差为 `2.43839e-6`。

布线后所有时序约束满足：WNS `+0.003ns`、WHS `+0.009ns`，setup/hold/pulse-width 失败端点均为 0；115 条总线偏斜约束全部通过，最小余量 `1.600ns`。XRT 实际观察到 `qwen_full_decode:qwen_full_decode_1` 的 39 次调用，状态为 `DONE|IDLE|READY`，加载 UUID 与生成的位流一致。

本次默认提示实测：加载 15.39 秒，FPGA prefill 215.96 秒，21 个增量 decode 步 335.04 秒，增量速度约 **0.0627 token/s**，总计 566.42 秒。这是该提示、该次运行的数据；prefill 当前按 token 顺序执行。

证据：[最终部署报告](reports/FINAL-DEPLOYMENT.json)、[FPGA 数值与输出报告](reports/full-fpga.json)、[时序与总线偏斜](reports/signoff.json)。此前的 [完整 C++ 软件验证](reports/full-cpp-validation.json) 单独保留。

正常运行入口也已单独实机验证：关闭层输出/logits 回读，使用另一组短提示完成 24 次完整内核调用，两个输出 token（含 EOS）与保存的 PyTorch 参考一致，诊断张量回读数为 0。见 [正常入口验证](reports/production-validation.json)。两次运行后 XRT 记录完整 Decode CU 累计调用 63 次，状态正常。

查看完成状态：

```bash
python3 ./scripts/deployment_status.py
```

原有混合部署状态可用同一命令加 `--hybrid` 查看。

## 运行

可直接运行，无需重新编译：

```bash
./full-decode/run_full.sh --backend fpga \
  --prompt '请用一句话介绍杭州。' --max-new-tokens 32
```

入口使用现有 `llm` 环境，只读固定 revision 的本地 tokenizer 和打包权重；无需下载或手动激活 Conda。缺少真实位流或设备时直接报错，不回退 CPU。

附加 `--verify ./full-decode/reference/default` 可读取设备诊断数据，与对应提示的离线参考比较。诊断只比对结果，不向 FPGA 提供激活或代替硬件计算。普通运行不读取层输出或 logits，只读 token/status。

显式执行软件版本：

```bash
cd ./full-decode
make runtime
./run_full.sh --backend cpu-cpp --verify reference/default \
  --max-new-tokens 32 --report reports/full-cpp-validation.json
```

## 数据布局

`weights/weights.bin` 为 310 个 BF16 张量的无损连续布局，共 1,192,099,840 字节。每个张量 SHA256 与此前验证的导出文件一致，embedding/lm_head 共用一份。`offsets.bin` 是张量偏移，`inv_freq.bin` 是模型 RoPE 固定频率参数；角度、正弦/余弦和旋转在 FPGA 计算。共享原模型未修改。

权重使用 HBM[0:3]，FP32 KV cache 使用 HBM[4]，容量 448 MiB；偏移/频率在 HBM[5]，结果和可选诊断缓冲在 HBM[6]。大缓冲使用 device-only BO，避免占用超过用户 memlock 上限的固定主机内存。

数值检查使用逐元素 `abs(error) <= 0.005 + 0.0001*abs(reference)`，并要求每层相对 L2 误差 <= 1e-4；原始误差全部保留在 JSON 中。大于 8000 的真实 residual 激活使用相对尺度比较，最终生成 token 还必须完全一致。

## 构建与恢复

```bash
python3 ./full-decode/scripts/deploy_full.py
```

`--from-stage link` 使用已成功编译的 XO；`--from-stage resume-link` 复用已生成的 Vivado 工程，从综合恢复；`--from-stage verify` 使用已绑定源码哈希的位流重新验证。脚本加互斥锁，日志写入 `reports/auto-*.log`，失败保留具体阶段和错误，不设置完成标志。不刷固件，不修改原始开发平台包。
