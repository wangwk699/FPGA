# FPGA：同服务器用户复用 /home/share 的部署清单

仓库：<https://github.com/wangwk699/FPGA>。本清单针对当前这台服务器，当前实现是 `full-decode-stream/`：原始 BF16 权重、FP32 运算，不量化，28 层完整计算在 FPGA。

**每人已有 Python 3.11 的 `llm` 环境；模型、工具和平台使用服务器上已有的共享版本。不需要重新安装 Miniconda、创建环境、下载模型或安装 Vitis。** 每人的代码、本机配置、主机库、日志和自行编译结果仍保留在自己的家目录。

## 已有的共享资源

所有用户 `.bashrc` 已配置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/share
export HF_XET_HIGH_PERFORMANCE=1
```

不要再将 `HF_HOME` 改回个人缓存。`HF_XET_HIGH_PERFORMANCE` 对应 hf-xet。若另行设置过 `HF_HUB_CACHE` 或 `HF_XET_CACHE`，确保它们没有指向旧的个人目录。

| 内容 | 统一路径 |
| --- | --- |
| 原始 Qwen3-0.6B snapshot | `/home/share/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca` |
| Vitis/Vivado 2025.1 完整安装 | `/home/share/fpga/amd/2025.1` |
| Vitis 环境脚本 | `/home/share/fpga/amd/2025.1/Vitis/settings64.sh` |
| U55C 开发平台 | `/home/share/fpga/platforms/u55c/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm` |
| XRT 和 UUID 开发头文件 | `/home/share/fpga/xrt-dev/usr/include` |
| 原始安装包及开发 DEB | `/home/share/fpga/downloads` |
| 当前无损打包权重 | `/home/share/fpga/models/Qwen3-0.6B/tiled-bf16-v2` |
| 默认提示的参考数据 | `/home/share/fpga/reference/Qwen3-0.6B/default` |
| 已签核的完整 Decoder 位流 | `/home/share/fpga/artifacts/qwen3-0.6b-stream/qwen_full_decode.xclbin` |
| 路径、哈希及既有位流签核摘要 | `/home/share/fpga/resources.json` |

共享工具对其他用户只读可执行；模型、安装包和其他共享资料可编辑。工具更新由维护者统一进行。脚本将共享文件的 SHA256 与 GitHub 中的 `config/shared-server.example.json` 对照，检测文件被误改或版本混用。个人 AMD/GitHub 登录令牌、许可证和 Conda 环境没有搬到共享目录。

## 1. 获取或更新自己的源码

首次使用：

```bash
cd "$HOME"
git clone https://github.com/wangwk699/FPGA.git
cd "$HOME/FPGA"
```

已经克隆过且无未提交源码修改时：

```bash
cd "$HOME/FPGA"
git pull --ff-only
```

## 2. 激活已有 llm 环境，仅检查依赖

```bash
conda activate llm
export QWEN_REPO="$HOME/FPGA"
export QWEN_PYTHON="$(command -v python)"
cd "$QWEN_REPO"
python --version
python scripts/check_python_environment.py
```

不执行 `conda create`，也不重复安装已有常用依赖。检查脚本不会安装任何东西：全部匹配即直接继续；仅当缺少包或版本不同，按它输出的命令补齐对应包。特别是先前安装的常用依赖不一定包含 Torch，而 CPU 数值参考需要 Torch。CPU 版 Torch 即可。

## 3. 绑定共享资源，生成自己的配置

```bash
python scripts/use_shared_resources.py
```

脚本不下载、不安装、不重建环境，会：

1. 核对共享模型、平台、安装资料、权重、参考和位流的哈希及源码对应关系。
2. 创建自己的 `deployment.json`、`config/local.mk`、`config/vitis-2025.1.local.txt`，全部指向共享模型/工具/平台及自己当前的 Python。
3. 在自己的 `full-decode-stream/weights`、`reference/default` 建立指向共享数据的链接；不复制 1.2GB 权重。
4. 登记共享位流路径，不把它链接到个人编译输出位置，避免后续编译覆盖共享位流。

脚本不会覆盖已有本机配置或数据。已经配置过时先运行 `python scripts/use_shared_resources.py --check-only`；若要切换旧配置，应先备份自己的本机配置并处理旧数据目录，不能直接清除还要使用的部署记录。

共享数据已经准备好，不必运行 `snapshot_download`、`pack_weights.py`、`make_reference.py` 或 AMD 安装器。

## 4. 检查板卡和编译自己的主机库

```bash
xrt-smi examine -d 0000:01:00.1
make -C full-decode-stream runtime-cpu
make -C full-decode-stream runtime-fpga
```

系统 XRT 驱动及运行库沿用服务器版本 `2.21.75`，不重复安装。个人主机库会使用共享 XRT 开发头文件和系统运行库编译。新账号若无设备访问权限，由管理员补充设备权限。

## 5. 先通过 CPU 数值验收

```bash
./full-decode-stream/run_full.sh --backend cpu-cpp \
  --verify full-decode-stream/reference/default \
  --max-new-tokens 32 \
  --report full-decode-stream/reports/full-cpp-validation.json
```

应得到 `matches_reference_tokens: true` 和五项 `passed: true`。这里使用共享参考数据，无需重新下载或生成。

## 6. 使用共享位流完成 FPGA 实机验收

先约定独占使用 U55C 的时段，再执行：

```bash
unset XCL_EMULATION_MODE
./full-decode-stream/run_full.sh --backend fpga \
  --xclbin /home/share/fpga/artifacts/qwen3-0.6b-stream/qwen_full_decode.xclbin \
  --verify full-decode-stream/reference/default \
  --max-new-tokens 32 \
  --report full-decode-stream/reports/full-fpga.json
```

检查自己的报告：`fpga_executed`、`full_decoder_on_fpga`、`matches_reference_tokens` 为 true；五组数值检查通过。默认提示为 39 次内核调用、1092 次层计算，CPU 模型 forward 与层间 CPU 计算往返均为零。

此路径复用了已经签核的电路，没有重新综合布线；新用户自己的实机数值报告与共享资源哈希共同确认这次运行。无需伪造一份新编译的部署状态。

## 7. 以后直接运行

```bash
cd "$HOME/FPGA"
./full-decode-stream/run_full.sh --backend fpga \
  --prompt '请用一句话介绍杭州。' --max-new-tokens 32
```

入口优先使用个人已编译的位流；若没有本地位流，就使用第 3 步登记的共享位流并核对其 SHA256。要明确指定共享版本，使用第 6 步的 `--xclbin` 参数。

batch=1，输入加输出最多 2048 token，greedy，关闭 thinking。每次运行仍需错开板卡使用时段，家目录隔离不会隔离同一张物理 FPGA。

## 8. 只有要自行重编译电路时，才做以下可选步骤

原始模型、工具、平台和打包权重仍然共享，不再下载。自己的 `_x/`、IP cache、`.xo`、`.xclbin` 和时序报告留在自己目录：

```bash
source /home/share/fpga/amd/2025.1/Vitis/settings64.sh
v++ --version
vivado -version
unset XCL_EMULATION_MODE
"$QWEN_PYTHON" full-decode-stream/scripts/deploy_full.py
```

已有第 5 步的 CPU 数值报告后，可由脚本完成独立编译、布线签核和板卡验收。先约定上板时段；不要执行共享目录里的安装/补装操作。完整编译和恢复说明见 [完整指南](FROM-ZERO-TO-GITHUB.md)。

若确实遇到旧平台 `clk_metadata_adapter` 兼容问题，停止失败的个人构建后，仅修复自己的派生目录：

```bash
/usr/bin/python3 -I scripts/use_vendor_clock_adapter.py \
  --build-root "$QWEN_REPO/full-decode-stream/_x/link/vivado/vpl" --apply
"$QWEN_PYTHON" full-decode-stream/scripts/deploy_full.py --from-stage resume-link
```

修改 HLS 或权重布局时，不应继续宣称共享位流对应修改后的源码。此时自行编译，并在需要改变权重时使用新的个人 `--weights` 目录；不要直接覆盖全员共用的权重和参考数据。

## 使用另一种模型

本清单前 7 步完成基础共享部署后，可通过 `--model Qwen3-0.6B` 或 `--model Qwen3-1.7B` 选择模型。已发布 1.7B 共享资源时，第一次选择会自动配置个人运行目录并编译主机库；不重新下载模型或综合电路。详见 [多模型说明](MULTI-MODEL.md)。

## 维护记录

2026-10-27 已按所有者确认的方案，将现有 AMD 2025.1 安装同盘迁移到共享位置，修正 12 份生成的启动配置和 60 个内部绝对链接。原用户位置保留兼容链接。模型本来就在 `/home/share/hub`，未重新下载或改变；安装器认证、GitHub 凭据及个人 Conda 环境保持私有。
