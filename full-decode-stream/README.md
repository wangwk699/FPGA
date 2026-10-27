# 多行交错矩阵乘与 KV Cache 复用

新机器首次部署请按 [从零部署指南](../docs/FROM-ZERO-TO-GITHUB.md) 操作。本版本可直接读取原始模型打包，不需要先编译历史版本；没有旧版性能报告也能完成独立部署与数值验收。

2026-10-27 已在 U55C 实机验证通过：增量 Decode 从上一版 **2.737 提升到 8.898 token/s，再加速 3.25 倍**，相对最初 0.06268 token/s 版本约快 142 倍。此前的可运行版本完整保留；部署配置已选择本版本。

| 同一默认提示的指标 | 上一版 | 本版 |
| --- | ---: | ---: |
| 增量 Decode | 2.737 token/s | **8.898 token/s** |
| 平均增量 token 延迟 | 0.3654 秒 | **0.1124 秒** |
| 21 个增量 Decode 步 | 7.673 秒 | **2.360 秒** |
| 18-token prefill | 4.297 秒 | **1.471 秒** |

输入及全部 22 个输出 token（含 EOS）完全一致，五组逐层输出/logits 数值检查全部通过。首个 Decode logits 最大绝对误差及相对 L2 误差见 [实机数值报告](reports/full-fpga.json)，完整对比见 [性能报告](reports/PERFORMANCE.json)。全部布线时序约束满足，WNS +0.003ns、WHS +0.009ns、失败端点为 0；116 条总线偏斜约束全部通过。

普通入口关闭诊断回读也已实测通过：8.929 token/s，沿用同一位流再次运行整次约 **4.15 秒**。首次验证包含新位流初始化与加载约 15.98 秒，总计约 19.85 秒，单独统计，不计入 Decode 速度。72-token 输入的数值、输出及 Cache 历史验证也通过；在位置 72 的一个增量步从 0.4465 秒降到 0.1364 秒。该较长输入只有一个增量步，不作为长生成平均速度。

证据：[最终部署报告](reports/FINAL-DEPLOYMENT.json)、[普通入口](reports/production-validation.json)、[KV Cache 历史验证](reports/kv-cache-validation.json)。上述速度属于对应提示和上下文长度的实测。

矩阵按 16 行、32 列进行无损布局重排，连续 512 位读取在 16 个输出行之间交错执行，减少逐行启动和排空流水线的开销。原始 BF16 位模式不变，310 个张量全部经过逆重排 SHA256 验证。计算和累加继续使用 FP32。

Attention 将共享一个 KV head 的两个 Q head 配对处理，每个历史 K/V 行从 HBM 读取一次，供两个 Q head 使用；点积和 AV 的特征维度并行计算。新旧版本都使用 KV Cache，优化的是 Cache 的访问和复用方式。

KV Cache 在 HBM 中以 FP32 存储，每层保存 RoPE 后的 K 和投影后的 V。batch=1 时，28 层、8 个 KV head、head_dim=128、2048 token 的固定容量为 448MiB，每个新 token 追加 224KiB 的有效 K/V。旧 token 的 K/V 投影及前馈层不重新执行；当前 Q 仍需与历史 K 做注意力计算，再组合历史 V。

软件与实际 FPGA 均已通过默认提示和 72-token 输入的参考比较。查看完成状态：

```bash
python3 scripts/deployment_status.py
```

`--previous` 查看 2.737 token/s 版本状态，`--baseline` 查看最初完整 FPGA 版本状态。

可直接运行，无需重新编译：

```bash
./full-decode-stream/run_full.sh --backend fpga \
  --prompt '请用一句话介绍杭州。' --max-new-tokens 32
```

全部 28 层、embedding、最终输出投影和 argmax 在 FPGA；CPU 负责分词、加载与控制。batch=1、总长度最多 2048 token、greedy、关闭 thinking。普通运行只回读 token/status，验证模式可回读诊断张量。无模型下载、无量化、无 CPU 模型计算回退。
