# 使用 --model 选择 Qwen3-0.6B 或 Qwen3-1.7B

入口仍然是 `full-decode-stream/run_full.sh`，默认模型为 Qwen3-0.6B。两种模型都使用原始 BF16 权重、FP32 运算、FP32 KV Cache，不量化。

```bash
./full-decode-stream/run_full.sh --backend fpga --model Qwen3-0.6B \
  --prompt '请用一句话介绍杭州。' --max-new-tokens 32

./full-decode-stream/run_full.sh --backend fpga --model Qwen3-1.7B \
  --prompt '请用一句话介绍杭州。' --max-new-tokens 32
```

也接受 `--model 0.6B`、`--model 1.7B` 和 `--model Qwen/Qwen3-1.7B`。原先省略 `--model` 的命令继续运行 0.6B。


已在 U55C 实机完成两种模型的数值验证。1.7B 默认提示输入 18 token、输出 20 token（含 EOS），37 次内核调用覆盖 1036 次层计算；全部 token 和五组数值检查与 PyTorch FP32 参考一致。增量 Decode 为 **3.457 token/s**，首次位流加载约 18.54 秒、Prefill 约 4.26 秒，增量速度不包含加载和 Prefill。干净源码副本首次自动准备共享模型后也已实机通过，约 3.450 token/s。0.6B 原已验证内核保持不变。

1.7B 布线签核：WNS +0.003ns、WHS +0.009ns，setup/hold/pulse-width 失败端点均为 0；118 条 bus-skew 约束全部通过。各次推理速度与上下文有关。完整摘要见 [验证记录](evidence/benchmark-summary.json)。

## 同服务器用户如何使用

1. 更新自己的源码并激活已有 `llm` 环境：

   ```bash
   cd "$HOME/FPGA"
   git pull --ff-only
   conda activate llm
   python scripts/check_python_environment.py
   ```

2. 尚未配置基础共享依赖时，先运行一次：

   ```bash
   python scripts/use_shared_resources.py
   make -C full-decode-stream runtime-cpu runtime-fpga
   ```

   已完成原清单前 7 步的用户不需要重复这一步。

3. 按上面的 `--model` 命令选择模型。若仓库带有已通过实机验收的 `config/models/Qwen3-1.7B.shared.example.json`，首次选择 1.7B 会自动验证共享文件、生成个人模型配置和构建目录、编译个人主机库，然后开始推理。不会下载模型、重建 Conda 或自动综合电路。

   也可提前完成首次准备：

   ```bash
   python scripts/prepare_model.py --model Qwen3-1.7B --shared
   ```

4. 同一张板卡仍要约定使用时段。切换模型会加载对应 `.xclbin`，加载时间与增量 Decode 时间分别统计。不同用户的代码、配置和日志独立，模型、工具和已签核位流共享。

缺少指定模型的配置/共享发布记录时会明确报错，不会静默改用另一个模型或在 CPU 上替代模型计算。

## 为什么不能只切换 tokenizer 或模型目录

| 项目 | Qwen3-0.6B | Qwen3-1.7B |
| --- | ---: | ---: |
| hidden_size | 1024 | 2048 |
| intermediate_size | 3072 | 6144 |
| Transformer 层数 | 28 | 28 |
| Q heads / KV heads / head_dim | 16 / 8 / 128 | 16 / 8 / 128 |
| 独立 BF16 权重数据字节数 | 1,192,099,840 | 3,441,149,952 |
| 权重 HBM bank | 0–3，合计 2GiB | 0–7，合计 4GiB |
| KV / 元数据 / 输出 HBM bank | 4 / 5 / 6 | 8 / 9 / 10 |
| 2048-token KV Cache | 448MiB | 448MiB |

形状依据：[0.6B 官方配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)、[1.7B 官方配置](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)。权重字节数由全部 310 个独立张量形状计算，去除重复的共享 lm_head，未进行量化。KV head 数与 head_dim 相同，因此两者的 Cache 容量相同。

原始 0.6B HLS 和 C++ 主机源码保留原样。1.7B 的构建从同一算法生成模型尺寸不同的头文件、主机代码与 HBM 连接配置，并使用独立的权重、库和位流。程序核对权重形状、1.7B 原生库签名、模型配置与位流 SHA256/源码绑定，防止混用两个模型的产物。

## 文件布局

```text
full-decode-stream/                        已验证的 0.6B 原始内核和公共 Python 入口
full-decode-stream/variants/Qwen3-1.7B/     个人生成的 1.7B 源码、构建、报告，不提交 Git
config/models/Qwen3-1.7B.local.json        个人路径和部署状态，不提交 Git
config/models/Qwen3-1.7B.shared.example.json 共享资源身份与固定哈希，提交 Git
scripts/model_profiles.py                 两种模型的形状、工作目录和位流绑定规则
scripts/prepare_model.py                  从已有本地 checkpoint 生成独立模型工程
```

1.7B 共享资源发布后使用：

```text
/home/share/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
/home/share/fpga/models/Qwen3-1.7B/tiled-bf16-v2
/home/share/fpga/reference/Qwen3-1.7B/default
/home/share/fpga/artifacts/qwen3-1.7b-stream/qwen_full_decode.xclbin
/home/share/fpga/resources-Qwen3-1.7B.json
```

## 维护者从原始模型独立构建 1.7B

共享缓存中已经有完整的 1.7B 两片 safetensors，本流程不下载文件。`prepare_model.py` 也不会下载缺失文件：

```bash
python scripts/prepare_model.py --model Qwen3-1.7B
python full-decode-stream/scripts/pack_weights.py --model-name Qwen3-1.7B
make -C full-decode-stream/variants/Qwen3-1.7B runtime
python full-decode-stream/scripts/make_reference.py --model Qwen3-1.7B
./full-decode-stream/run_full.sh --model Qwen3-1.7B --backend cpu-cpp \
  --verify full-decode-stream/variants/Qwen3-1.7B/reference/default \
  --report full-decode-stream/variants/Qwen3-1.7B/reports/full-cpp-validation.json
python full-decode-stream/scripts/deploy_full.py --model Qwen3-1.7B
```

需要其他本地 snapshot 时给 prepare 加 `--snapshot /实际路径`。主机代码、打包与 CPU 参考均按指定模型选择；1.7B 打包支持分片 checkpoint 并检查重复 lm_head 与 embedding 内容一致。

查看该模型进度：

```bash
python scripts/deployment_status.py --model Qwen3-1.7B
```

恢复链接用 `deploy_full.py --model Qwen3-1.7B --from-stage resume-link`；恢复硬件验收用 `--from-stage verify`。仍要求已有对应的完整阶段记录，不能跳过缺失的编译或数值验收。

只有布线时序、五组实机数值检查、输出 token、物理 CU 和 UUID 全部通过后，维护者才发布共享模型产物：

```bash
python scripts/publish_model_artifacts.py --model Qwen3-1.7B
```

该命令复制已验证产物到新的模型专属共享目录、保留个人原始构建、最后才写共享发布记录。不会覆盖已有 0.6B 模型或位流，也不会把模型文件上传 GitHub。新生成的共享 `.example.json` 和更新后的源码/文档才提交到仓库。
