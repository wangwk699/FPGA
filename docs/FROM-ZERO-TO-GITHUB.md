# FPGA：Qwen3-0.6B 在 Alveo U55C 上从零部署与 GitHub 发布指南

记录日期：2026-10-27。项目仓库：[wangwk699/FPGA](https://github.com/wangwk699/FPGA)。同服务器且家目录隔离的用户优先按 [独立用户部署清单](ISOLATED-USER-CHECKLIST.md) 操作。面向拿到本工程源码、原始模型和一张 U55C 的合作者。

本文以已经实机验证的 `full-decode-stream/` 为部署主线，同时记录这次从硬件检查、工具安装到完整 Decoder 和两轮优化的过程。新机器可以直接构建最新版，不必依次编译历史版本。

## 1. 最终要部署什么

目标模型是原始 **Qwen/Qwen3-0.6B**，固定 revision：

```text
c1899de289a04d12100db370d81485cdf75e47ca
```

不进行 INT8/INT4 等量化。模型原有 BF16 参数保持原始位模式，进入计算单元后精确扩展成 FP32；运算、累加、层间激活与 KV Cache 使用 FP32。权重的分块、转置和去除重复 lm_head 都是无损的数据布局操作。

| 模块 | 日常 FPGA 推理时的执行位置 |
| --- | --- |
| 聊天模板、分词、文本解码 | CPU / Python tokenizer |
| 加载权重、启动内核、读取生成 token | CPU / C++ XRT 主机程序 |
| Embedding | FPGA |
| 28 层 RMSNorm、Q/K/V 投影、QK Norm、RoPE | FPGA |
| GQA Attention、softmax、历史 K/V 读取、新 K/V 写入 | FPGA，Cache 在 HBM |
| Attention 输出投影、两次残差 | FPGA |
| MLP gate/up、SiLU、逐元素乘法、down | FPGA |
| 最终 RMSNorm、共享 lm_head、greedy argmax | FPGA |
| 生成 PyTorch 参考答案和诊断张量 | 仅验收准备阶段在 CPU 运行 |

Prefill 也在 FPGA，目前逐 token 执行。一次内核调用完成一个 token 的全部 28 层。正常推理不调用 CPU 模型 forward，不把层间激活或 KV 传回 CPU 计算。`--verify` 会额外回读诊断张量用于比较，比较结果不会参与下一 token 的生成。

当前支持 batch=1、输入与生成总长度不超过 2048 token、greedy 解码、关闭 thinking。代码针对这一模型形状固定设计，不是可加载任意 Qwen 模型的通用推理框架。

固定形状为 hidden_size=1024、intermediate_size=3072、28 层、16 个 Q head、8 个 KV head、head_dim=128、vocab_size=151936、rope_theta=1000000。注意 Q 投影输出宽度为 2048，不能用 hidden_size / Q-head 数推断 head_dim；准备脚本会显式检查这些配置。

KV Cache 一直启用：每层保存 RoPE 后的 K 和投影后的 V，新增 token 只追加自己的 K/V，历史 token 不重算其投影和 MLP。28 层、8 个 KV head、head_dim=128、2048 token 的 FP32 Cache 容量为 **448MiB**，每个有效 token 的 K/V 合计 **224KiB**。当前 Attention 把共用同一 KV head 的两个 Q head 配对，复用从 HBM 读出的 K/V。

## 2. 本次实际经历的阶段与可核验结果

| 阶段 | 做了什么 | 结果及保留的源码 |
| --- | --- | --- |
| 确认板卡与运行时 | 阅读 U55C 资料、检查 XRT/PCIe/HBM，运行官方测试内核并检查 32 个 HBM bank | 确认硬件可用；根目录 `host/`、`hls/`、`scripts/` 保留探针和早期 GEMV 工程 |
| 准备原始模型 | 读取用户已下载的 checkpoint，检查形状、BF16、共享权重，导出无损权重并生成 CPU 参考 | 310 个独立张量，596,049,920 个独立参数；未修改共享模型 |
| 补齐开发工具 | 下载 AMD 2025.1 网络安装器和 U55C 开发平台，认证、续传、安装 Vitis/Vivado | 解决中断下载、缺 HBM 器件支持、工具本地兼容库及平台时钟适配器问题 |
| CPU + FPGA 混合推理 | FPGA 执行 Decode 中 197 个线性层；CPU 执行 Prefill、Attention 等 | 全部输出 token 与 CPU 参考一致；`run_hybrid.sh` 保留这一阶段 |
| 完整 Decoder 上 FPGA | 把 Norm、RoPE、Attention、KV、MLP、输出选择一并放到 FPGA | `full-decode/`；增量 Decode **0.06268 token/s**，约 15.95 秒/token |
| 第一轮优化 | 512 位权重读取、32 路 BF16→FP32 乘法、16 路交错累加 | `full-decode-fast/`；**2.737 token/s** |
| 第二轮优化 | 16 行×32 列权重布局、多行交错 MAC、配对 Q head 复用 K/V、32 路 Attention | 当前 `full-decode-stream/`；**8.898 token/s**，约 0.1124 秒/token |

最后两次加速都重新完成了数值、真实板卡执行、位流身份和布线时序验证。历史摘要可提交到 GitHub，见 [evidence/benchmark-summary.json](evidence/benchmark-summary.json)。原机各版本 `reports/` 保存更完整的日志和报告，源码发布包不包含这些原始产物。

当前位流的原机布线签核为 WNS +0.003ns、WHS +0.009ns，setup/hold/pulse-width 失败端点均为 0，116 条 bus-skew 约束全部通过。新构建也必须检查自己的实际签核结果。

同一默认提示“请用一句话介绍杭州。”得到 18 个输入 token、22 个输出 token（含 EOS），其中首个输出来自 Prefill，随后 21 个增量 Decode 步耗时 2.360 秒。共有 39 次 token 内核调用，即 1092 次层计算；所有输出 token 与 CPU FP32 参考一致，五组诊断比较通过。参考答案为：

> 杭州是中国浙江省的省会，位于中国东南部，以西湖闻名，是著名的旅游城市。

这个速度只代表对应提示和上下文长度，不是任意长上下文的固定吞吐。首次验证包含位流初始化与加载约 15.98 秒，另行统计；同位流再次普通运行整次约 4.15 秒。72-token 输入的历史数值与 Cache 验证也通过，其中只有一个增量步，不能当成长生成的平均速度。

## 3. 合作者需要的机器

### 3.1 硬件和操作系统

运行机需要 Alveo U55C、正常供电散热、可用的 PCIe 插槽、XRT 驱动和适配板卡的基础 shell。U55C 有 16GB HBM、32 个 bank，每 bank 512MiB。本工程器件为 `xcu55c-fsvh2892-2L-e`。

编译机需要 x86-64 Linux、C++ 编译器和 Vitis/Vivado 2025.1。AMD 对加速开发列出的内存要求为最低 64GB、推荐 80GB，完整 Vitis 安装空间约 200GB，随组件选择变化。建议连下载缓存和构建文件共预留至少 250–300GB 空间。Ubuntu 24.04/24.04.1 及其列出的 GA 内核在支持列表内；操作系统、内核和 XRT 必须配套。详见 [AMD 2025.1 安装要求](https://docs.amd.com/r/2025.1-English/ug1742-vitis-release-notes/Installation-Requirements)。

本次实际是在 Ubuntu 26.04.1、7.0.0-38 内核、Ryzen 9700X、约 30GiB 可用物理内存加 8GiB swap 上完成的；这不是上述官方支持/内存要求下的推荐搭配。合作者优先使用官方支持环境，内存较小时可按本工程设置降低综合并行数。编译机与装有板卡的运行机可以分开，见第 12 节。

### 3.2 原机环境，供排错对照

| 项目 | 原机的实际值 |
| --- | --- |
| U55C user function BDF | `0000:01:00.1`，管理 function 为 `.0` |
| 基础 shell | `xilinx_u55c_gen3x16_xdma_base_3` |
| shell logic UUID | `97088961-feae-da91-52a2-1d9dfd63ccef` |
| 平台接口 UUID | `b7ac1abe1e3e1cb686d5a81232452676` |
| XRT | `2.21.75`，本机 Ubuntu/Debian 包版本 `1:2.21.75+dfsg-4` |
| AMD 工具 | Vitis/Vivado `2025.1`，目录 `/home/wangwk/amd/2025.1/` |
| Python | 3.11，原机 Conda 环境名 `llm` |
| memlock 上限 | 8MiB；主机代码用 device-only 大 BO 和 1MiB 上传块适配 |

合作者的 BDF、用户名和目录可能完全不同。应填写自己的本机配置，不要直接复制原机 `deployment.json`。

## 4. 下载清单

| 需要下载的内容 | 来源与用途 | 是否放进 GitHub |
| --- | --- | --- |
| 本工程源码 | 你的 GitHub 仓库；编译 HLS 内核、主机库、准备权重与验收 | 是 |
| Qwen3-0.6B 固定 revision | [模型官方仓库](https://huggingface.co/Qwen/Qwen3-0.6B)；原始 `model.safetensors`、配置、tokenizer | 否，按模型仓库获取 |
| Miniconda 或 Python 3.11 | 创建 Python 环境；已有 Python 3.11 可直接用 venv | 否 |
| Python 依赖 | 仓库中的 `requirements-reproduce.txt`；Torch、Transformers、NumPy 等 | 只提交依赖清单 |
| XRT 运行时、驱动、开发头文件 | [U55C 官方下载页](https://www.amd.com/en/support/downloads/alveo-downloads.html/accelerators/alveo/u55c.html) / 对应发行版的软件源 | 否 |
| AMD 2025.1 Linux 网络安装器 | [AMD 2025.1 下载页](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2025-1.html) | 否 |
| U55C XDMA 开发平台 | 同一 U55C 下载页，Development Target Platform | 否 |

这次下载的两个 AMD 文件是：

```text
FPGAs_AdaptiveSoCs_Unified_SDI_2025.1_0530_0145_Lin64.bin
xilinx-u55c-gen3x16-xdma-3-202210-1-dev_1-3514517_all.deb
```

前者约 336MiB，只是网络安装器；后续还要下载几十 GB 的工具组件。后者约 165MiB，包含开发平台 `.xpfm`、XSA/IP 等。原始 `model.safetensors` 约 **1.50GB**，含显式重复的共享 lm_head；无损去重后的设备权重为 **1,192,099,840 字节**。为模型缓存、打包结果和参考计算额外预留数 GB。

本方案使用自定义 HLS 内核和 XRT，不需要下载 Vitis AI/DPU、量化模型或 CUDA 工具包。AMD 工具下载需要 AMD 账号；公开的 Qwen3-0.6B 模型下载无需预先登录 Hugging Face。

## 5. 获取源码、检查板卡、创建 Python 环境

以下命令是给合作者执行的操作步骤。尖括号内容替换成实际仓库地址；所有命令默认在仓库根目录运行。仓库、AMD 工具及平台路径尽量不要有空格。

```bash
git clone https://github.com/wangwk699/FPGA.git
cd FPGA
export QWEN_REPO="$PWD"
mkdir -p installers platforms/u55c reports
```

Ubuntu 上准备普通构建工具：

```bash
sudo apt update
sudo apt install build-essential git pkg-config python3-venv uuid-dev pciutils acl
lspci -nn
uname -r
df -h .
free -h
```

先确认 U55C 的 BDF，记录自己的 user function：

```bash
export QWEN_DEVICE=0000:01:00.1
```

如果运行机已经装有可工作的 XRT，直接检查并沿用。新机器按 U55C 官方文档安装与系统、内核匹配的 XRT 运行时/驱动及开发头文件；AMD DEB 包可通过 `sudo apt install ./实际的XRT安装包.deb` 安装。驱动构建可能需要与当前内核精确对应的 headers。

本机发行版把 XRT 拆成 `libxrt-alveo2`、`libxrt2`、`libxrt-utils`、`libxrt-utils-alveo`、`xrt-xocl-dkms`，开发头文件来自 `libxrt-dev`。这是本机的软件包形式，不代表其他 Ubuntu 版本都提供这些同名包；不要把 Ubuntu 26.04 的 DEB 直接装到 Ubuntu 24.04。

如果使用 AMD 安装路径，先加载运行时环境：

```bash
source /opt/xilinx/xrt/setup.sh
```

发行版安装没有该文件时无需 source。检查工具与板卡：

```bash
xrt-smi --version
xrt-smi examine -d "$QWEN_DEVICE"
```

某些 XRT 版本使用 `xbutil` 名称，改为 `xbutil examine -d "$QWEN_DEVICE"`。确认板卡健康、HBM 可见、shell/interface 与开发平台匹配。板卡访问权限由管理员按 XRT 安装要求配置。本工程加载用户 `.xclbin`；若新卡还没有可用的基础 shell，应先由管理员按 U55C 官方部署文档完成这一独立步骤。

创建 Python 环境可选 Conda：

```bash
conda create -n qwen-u55c python=3.11 pip -y
conda activate qwen-u55c
python -m pip install -r requirements-reproduce.txt
export QWEN_PYTHON="$(command -v python)"
```

或者使用已安装的 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
export QWEN_PYTHON="$(command -v python)"
```

二选一即可。Conda 不是下载模型的前提，也不是 FPGA 必需组件。固定依赖版本取自本次成功环境；PyTorch 只用于 CPU 参考和准备工具，CPU 版满足需要。依赖安装可能下载较大的 Torch wheel，与 AMD 工具下载是两件事。

## 6. 准备原始模型

### 6.1 已下载模型时

原机模型已经存在，下列路径只供本机使用，**不需再次下载**：

```bash
export QWEN_MODEL=/home/share/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

合作者把 `QWEN_MODEL` 改成自己的完整模型目录。必须包含 `config.json`、`model.safetensors`、`tokenizer.json`、`tokenizer_config.json`、`generation_config.json`；使用完整 snapshot 可避免遗漏 tokenizer 附属文件。

### 6.2 尚未下载时

个人缓存可设置 `HF_HOME="$HOME/.cache/huggingface"`。若多人共用 `/home/.cache/huggingface`，使用已由管理员配置好的共享组、目录权限和默认 ACL：

```bash
export HF_HOME=/home/.cache/huggingface
export HF_XET_HIGH_PERFORMANCE=1
# 使用镜像时再设置；访问官方源可省略这一行
export HF_ENDPOINT=https://hf-mirror.com
```

仅设置变量不会授予目录写权限。管理员已配置共享组后，新会话需重新登录以取得新组；用 `id`、`getfacl /home/.cache/huggingface`、`test -w /home/.cache/huggingface` 检查。所有上级目录也要允许遍历。多人缓存内的已有文件及 `.locks` 也要可供组写，新建用户还需加入共享组；不必为下载给整个 `/home` 设置任意写权限。

`HF_XET_HIGH_PERFORMANCE` 对应 **hf-xet**，不是 `hf_transfer`。`HF_HOME` 下默认的模型仓库缓存是 `hub/`；配置应在 Python 导入 Hub 库前生效。见 [Hub 环境变量说明](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)。公开模型无需在共享目录保存个人登录令牌。

在刚创建的 Python 环境中运行一次：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
)
print("模型目录：", path)
PY
```

把打印的目录赋给 `QWEN_MODEL`。重复调用会复用相同 revision 的有效缓存；下载 API 说明见 [Hub 下载指南](https://huggingface.co/docs/huggingface_hub/guides/download)。下载完成后，本文的打包、参考生成和推理都只读本地模型，不会自动联网补文件。

核验固定 checkpoint：

```bash
sha256sum "$QWEN_MODEL/model.safetensors"
```

本次固定 revision 对应：

```text
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
```

若哈希不同，先确认 revision、文件是否完整及来源，不应直接沿用本次数值结论。

## 7. 导入开发平台，生成本机配置

将第 4 节的两个 AMD 下载文件放入仓库 `installers/`。先解包 U55C 开发平台：

```bash
dpkg-deb -x installers/xilinx-u55c-gen3x16-xdma-3-202210-1-dev_1-3514517_all.deb platforms/u55c
export QWEN_PLATFORM="$QWEN_REPO/platforms/u55c/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm"
test -f "$QWEN_PLATFORM"
```

这是解包到项目目录，不运行 DEB 安装脚本，也不替板卡刷基础固件。`.xpfm` 是编译平台描述；它与 XRT 和板卡基础 shell 的职责不同。平台依据见 [U55C base_3 官方说明](https://docs.amd.com/r/en-US/ug1120-alveo-platforms/U55C-Gen3x16-XDMA-base_3-Platform)。

创建只属于本机的配置：

```bash
python scripts/configure.py \
  --model "$QWEN_MODEL" \
  --python "$QWEN_PYTHON" \
  --device "$QWEN_DEVICE" \
  --platform "$QWEN_PLATFORM" \
  --vitis-root "$HOME/amd"
```

脚本会检查模型形状及 BF16 类型、计算原始模型 SHA256，并生成：

```text
deployment.json                     本机模型、Python、BDF、工具位置及后续部署状态
config/local.mk                     本机平台/XRT 编译路径
config/vitis-2025.1.local.txt         本机 AMD 安装器配置
```

这三项已被 `.gitignore` 排除，GitHub 提交相应 `.example` 模板即可。脚本默认拒绝覆盖已有配置；显式 `--force` 会重建配置并清除其中已有的部署完成记录，仅在需要重新配置时使用。

如果 XRT 头文件或库在非标准目录，首次配置时再加：

```text
--xrt-include /实际目录/include --xrt-lib /实际目录/lib/libxrt_coreutil.so
```

`--xrt-include` 指向包含 `xrt/xrt_device.h` 的目录；`--xrt-lib` 指向库文件。Makefile 可自动识别常见 `/opt/xilinx/xrt/`、`/usr/include` 和项目 `.deps/xrt-dev/usr/include`。已有 Vitis 的机器也可在配置时加 `--vitis-settings /实际路径/settings64.sh`，随后跳过安装。

## 8. 安装 Vitis/Vivado 2025.1

已有可用 2025.1 工具及 U55C/HBM 器件支持时，直接进入第 8.3 节。

### 8.1 解出网络安装器并认证

```bash
mkdir -p .deps
bash installers/FPGAs_AdaptiveSoCs_Unified_SDI_2025.1_0530_0145_Lin64.bin \
  --noexec --target "$QWEN_REPO/.deps/vitis-installer-2025.1"
bash scripts/setup_vitis.sh auth
```

最后一条在自己的交互终端按提示登录 AMD 账号并生成下载认证。不要把账号、令牌或含签名下载 URL 的安装日志提交到 GitHub。认证令牌过期时重新执行 `auth`。

### 8.2 安装正确的组件

检查 `config/vitis-2025.1.local.txt` 的 `Destination`，默认是 `$HOME/amd` 展开后的绝对路径。必须有：

1. Vitis Unified Software Platform，包含 `v++` 和 Vivado 工具链。
2. **Install devices for Alveo and edge acceleration platforms**。
3. **Virtex UltraScale+ HBM FPGAs**。

本工程配置模板已选中后二项。只选 Alveo 而漏选 HBM 会出现 U55C 时钟器件数据缺失；这正是本次遇到过的问题。安装方式也可参考 [AMD 官方安装说明](https://docs.amd.com/r/2025.1-English/ug1742-vitis-release-notes/Installing-the-Vitis-Software-Platform)。

```bash
bash scripts/setup_vitis.sh install > reports/vitis-install-local.log 2>&1
```

上述命令会下载并安装组件。下载期间可以从另一个终端查看：

```bash
tail -n 30 reports/vitis-install-local.log
```

重启或网络故障后，先确认没有旧安装器继续运行，再用同一目录、同一配置重新执行 `install`，安装器可复用已下载的有效文件；不要删除其下载缓存。安装日志出现 `running` 或 `installing_tools` 不能单独证明完成。

已经安装但漏了 HBM 组件时才执行：

```bash
bash scripts/setup_vitis.sh add-hbm > reports/vitis-add-hbm-local.log 2>&1
```

本次补装 HBM 下载约 7.77GB；首次组件下载显示过约 45GB，这两个值是历史记录，不能作为新机器必定相同的下载大小。

### 8.3 确认安装真正完成

本次安装布局如下；若安装器采用其他布局，改为实际路径：

```bash
export QWEN_VITIS_SETTINGS="$HOME/amd/2025.1/Vitis/settings64.sh"
test -f "$QWEN_VITIS_SETTINGS"
source "$QWEN_VITIS_SETTINGS"
v++ --version
vivado -version
platforminfo -p "$QWEN_PLATFORM"
```

应看到 Vitis/Vivado 2025.1、U55C 器件与正确平台信息。工具自带 `/Vitis/2025.1/` 的另一种布局也可使用。自动部署脚本依次接受 `QWEN_VITIS_SETTINGS`、本机配置指定路径或配置的安装根目录；不要求用户名与原机相同。

可再独立确认 HBM 时钟 IP：

```bash
vivado -nolog -nojournal -mode batch -source scripts/check_u55c_clock.tcl
```

日志中的 `U55C_CLOCK_IP_CREATED` 是检查成功标志。若报 `libtinfo.so.5` 或时钟参数缺失，按第 14 节排查后再编译模型电路。

## 9. 为当前内核打包权重，并生成独立 CPU 参考

确认处于正确 Python 环境。`run_full.sh` 优先使用 `QWEN_PYTHON`，其次使用 `deployment.json` 中的 Python 路径，最后才使用 PATH 中的 `python3`。

### 9.1 原始 checkpoint → 当前无损权重布局

```bash
python full-decode-stream/scripts/pack_weights.py --model "$QWEN_MODEL"
```

该命令直接读取原始模型，不需要根目录旧版分片或 `full-decode/weights/`。输出：

```text
full-decode-stream/weights/weights.bin      BF16 原位模式，16 行×32 列分块布局
full-decode-stream/weights/offsets.bin      310 个张量的元素偏移
full-decode-stream/weights/inv_freq.bin     FP32 RoPE 逆频率
full-decode-stream/weights/manifest.json    形状、源文件与逐张量/文件哈希
```

checkpoint 有 311 个参数记录，其中 lm_head 与 embedding 完全相同；脚本先检查二者形状、类型和内容，再保留 310 个独立张量。每个打包张量都会逆重排并与原始张量 SHA256 比较。已有输出目录时只校验，不覆盖；要切换模型或布局，应使用新的 `--output` 目录，并在运行时通过 `--weights` 指定。

固定模型的当前布局应得到：

| 文件 | SHA256 |
| --- | --- |
| weights.bin | `8ccd3d1534df6559143c08dbdf5f72f8fae97b59fe4167af0294c8743fe1bdfc` |
| offsets.bin | `d62d70eaa4ff2f198e558c1167c15b70cbd22e2451758677ad8e73abc688f5e6` |
| inv_freq.bin | `138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e` |

### 9.2 编译 C++ 软件参考，生成 PyTorch 参考

```bash
make -C full-decode-stream runtime-cpu
python full-decode-stream/scripts/make_reference.py
```

第二条读取本地模型，在 CPU FP32 上生成默认提示的答案、逐层输出和 logits，保存为 `full-decode-stream/reference/default/`。已有参考目录时拒绝覆盖。这里临时使用 CPU 模型是为了建立验收基准；之后实际 FPGA 推理不依赖这个模型 forward。

### 9.3 先做软件数值验收

```bash
./full-decode-stream/run_full.sh --backend cpu-cpp \
  --verify full-decode-stream/reference/default \
  --max-new-tokens 32 \
  --report full-decode-stream/reports/full-cpp-validation.json
```

应得到 `matches_reference_tokens: true` 和五项 `passed: true`：首个/最后一个 Prefill token 的层输出、Prefill logits、首个增量 Decode 的层输出及 logits。层输出比较包含 embedding、28 层与最终 Norm，既检查元素误差，也检查逐层相对 L2 误差。

这一步是 CPU 软件执行，还不证明 FPGA 部署成功。它可以先发现权重顺序、RoPE、Attention 或数值实现错误，避免浪费电路编译时间。根目录的低层回归还可运行：

```bash
make test
```

本次为 23 项单元测试与 5 组 GEMV 数值测试全部通过。

## 10. 构建 FPGA 电路，并完成实机验收

先确保第 9 节成功、板卡空闲、`QWEN_DEVICE` 配置正确。若是在新的终端，重新激活 Python 环境；脚本也能从本机配置找到工具路径。

```bash
unset XCL_EMULATION_MODE
python full-decode-stream/scripts/deploy_full.py
```

这条才是当前版本的完整部署入口。根目录 `scripts/auto_deploy.py` 是早期 GEMV/混合部署入口，不能用它替代当前完整 Decoder 的部署步骤。

后台执行可用：

```bash
mkdir -p full-decode-stream/reports
nohup "$QWEN_PYTHON" full-decode-stream/scripts/deploy_full.py \
  > full-decode-stream/reports/deploy-worker-local.log 2>&1 < /dev/null &
```

前台、后台二选一，不要启动两份。脚本有进程锁，但不应并行操作同一张卡或同一构建目录。编译可能持续数小时；本次某些完整构建约 1–2 小时，时间随资源、缓存与环境变化。

自动流程会依次：

1. 检查已通过的 C++ 数值报告与源码/原生库绑定。
2. `v++ -c -t hw`：将 `hls/full_decode.cpp` 编译为 `.xo`，记录源码与 `.xo` 哈希。
3. `v++ -l -t hw`：结合开发平台和 `connectivity.cfg` 综合、布局布线、生成 `.xclbin`，绑定位流哈希。
4. 编译 XRT 主机库 `libfull_fpga.so`。
5. 检查布线后 setup、hold、pulse-width 的失败端点数为零，并检查所有 bus-skew 约束。
6. 用真实 FPGA 运行默认模型提示，检查全部 token 与五组数值比较。
7. 从 XRT 查询 CU 使用计数、健康状态和已加载 UUID，与新编译位流对应。
8. 保存完成报告，并更新本机 `deployment.json` 的完整 FPGA 部署状态。

当前 HBM 绑定：权重 `HBM[0:3]`，KV Cache `HBM[4]`，偏移与逆频率 `HBM[5]`，输出/logits/诊断 `HBM[6]`。权重和 KV 不每步经 PCIe 回传；主机只提交当前 token 和控制参数，正常生成只读回 token/status。

默认最多 2 个综合任务、1 个实现任务、Vivado 线程上限 2，用来控制内存。初次平台 IP 综合可能产生大量子任务；HLS 的估算性能或“生成了 xclbin”都不足以替代最终板卡与时序验收。

### 10.1 查看进度

```bash
python full-decode-stream/scripts/status.py
tail -n 40 full-decode-stream/reports/deploy-worker-local.log
```

关注 `Status`、`Stage`、日志更新时间及失败子任务，而不是只看 `running`。服务器重启后，旧状态文件可能还写着 running，状态工具会检查进程身份并提示陈旧状态。

### 10.2 主要输出与成功条件

| 文件 | 内容 |
| --- | --- |
| `full-decode-stream/build/hw/qwen_full_decode.xo` | HLS 编译内核 |
| `full-decode-stream/build/hw/qwen_full_decode.xclbin` | 可加载用户位流 |
| `full-decode-stream/build/libfull_cpu.so`、`libfull_fpga.so` | 软件参考库、XRT 主机库 |
| `full-decode-stream/reports/compiled-source.json`、`linked-source.json` | 源码、内核和位流绑定 |
| `full-decode-stream/reports/signoff.json` | 布线时序及 bus-skew 验收 |
| `full-decode-stream/reports/full-fpga.json` | 实机输出、数值比较与性能 |
| `full-decode-stream/reports/FINAL-DEPLOYMENT.json` | 完整部署结果 |
| `full-decode-stream/reports/status.json` | 阶段和完成状态 |
| `full-decode-stream/reports/PERFORMANCE.json` | 当前测量，可选历史版本比较 |

最终应看到 `full_decoder_on_fpga: true`、`fpga_executed: true`、`matches_reference_tokens: true`，全部五项数值检查通过、28 层全部覆盖、CPU model forward/层间 CPU 往返为零，时序和板卡证据同时通过。

新仓库没有旧版性能报告也可以独立验收。不需要把原机的 `baseline.json` 复制过去；只有要比较速度时，才提供相同输入、相同全部输出 token 的真实 FPGA 报告：

```bash
python full-decode-stream/scripts/deploy_full.py --from-stage verify \
  --baseline-report /实际路径/旧版同提示的FPGA报告.json
```

这会重新运行验收，要求本机已有自动流程生成的链接记录和完整实现目录。`--min-speedup` 是可选的性能门槛，独立部署无需设置。新机器的布线结果、UUID 和位流 SHA256 可以与原机不同，应与该次本机生成的证据一致，而不是强求二进制复刻。

## 11. 完成部署后的日常运行

在仓库根目录执行：

```bash
./full-decode-stream/run_full.sh --backend fpga \
  --prompt '请用一句话介绍杭州。' \
  --max-new-tokens 32 \
  --report full-decode-stream/reports/my-run.json
```

不需要每次重新安装、打包或编译。推理需要 Python tokenizer、原始模型的配置/tokenizer 文件、已打包的设备权重、与平台兼容的 `.xclbin`、本机编译的 XRT 主机库和运行时。入口会强制使用本地 tokenizer；缺文件时失败，不会自动下载或回退到 CPU 模型。

速度按 **增量生成步数 / decode_seconds** 计算，不把加载和 Prefill 混进去。默认示例有 22 个输出 token，其中 21 个是增量步骤。报告同时给出 `load_seconds`、`prefill_seconds`、`decode_seconds` 和 `elapsed_seconds`。

更换提示后验收，先生成对应参考，避免拿新提示与旧参考比较：

```bash
python full-decode-stream/scripts/make_reference.py \
  --prompt '1+1等于几？请只回答数字。' \
  --max-new-tokens 8 \
  --output full-decode-stream/reference/arithmetic
./full-decode-stream/run_full.sh --backend fpga \
  --prompt '1+1等于几？请只回答数字。' \
  --max-new-tokens 8 \
  --verify full-decode-stream/reference/arithmetic \
  --report full-decode-stream/reports/arithmetic-fpga.json
```

参考生成要求至少两个输出 token 的额度，并且能实际进入一个增量步。总长度检查包含聊天模板 token；不要只按可见中文字符数判断是否超过 2048。

## 12. 编译机与板卡运行机分开时

完整复现可以分两台主机：编译机不要求安装 U55C，但需要 Vitis/Vivado、平台包和足够资源；运行机需要 U55C、驱动、XRT 与模型/运行数据，不需要 Vitis 编译器。

编译机在前述配置、权重和软件数值验收后，可只执行：

```bash
source "$QWEN_VITIS_SETTINGS"
make -C full-decode-stream compile
make -C full-decode-stream link
```

得到 `.xclbin` 后，将其和相应源码、模型 revision、已打包权重及必要的构建/时序报告交给运行机。运行机生成自己的 `deployment.json` 并 `make -C full-decode-stream runtime-fpga`，再执行带 `--verify` 的 FPGA 命令，可用 `--xclbin /实际路径/位流.xclbin` 指定位流。原生 `.so` 应在运行机编译，避免发行版、glibc 和 XRT ABI 不一致。

手动执行 Makefile 不会自动生成部署脚本的全部 `compiled-source.json` / `linked-source.json` / 完成状态。不要仅复制位流后就运行 `--from-stage verify` 并假设这些记录存在；需要保留并迁移自动流程的完整构建证据，或按手工流程重新核验时序、哈希、XRT 身份与数值结果。若要自动流程最终一步完成所有记录，优先在构建和验收都可访问板卡的环境执行第 10 节。

## 13. 中断后从哪里继续

先查明停止原因并确认没有旧工作进程，再决定恢复阶段。不要删除 `_x/`、`.xo`、IP cache 或下载缓存后期待断点续传。

| 已完成的阶段 | 恢复方式 |
| --- | --- |
| 只有 AMD 下载/安装中断 | 按第 8 节使用相同配置重跑安装器，认证过期则先 auth |
| 原始模型已下载 | 指定已有 `QWEN_MODEL`，不要重新下载 |
| 当前权重已打包 | 重跑 pack 只校验现有目录 |
| CPU 参考和软件验收完成，尚未编译内核 | 默认执行当前 `deploy_full.py` |
| 自动流程完成 compile，link 未开始或需重做 | `python full-decode-stream/scripts/deploy_full.py --from-stage link` |
| 自动流程已生成 VPL 工程，综合阶段中断/适配器修复后继续 | `python full-decode-stream/scripts/deploy_full.py --from-stage resume-link` |
| 自动流程 link 成功、板卡验收尚未完成 | `python full-decode-stream/scripts/deploy_full.py --from-stage verify` |
| 验收报告、签核及位流均保留，仅汇总失败 | `--from-stage finalize`，会复核已有证据并查询当前物理身份 |

恢复参数并不是任意跳步开关。各阶段都要求对应源码哈希、构建产物、参考报告和状态仍存在；缺少记录应回到能重新生成它的阶段。

## 14. 这次遇到的故障，以及新机器如何处理

### 14.1 模型缓存报 PermissionError

先看报错指向 `hub/`、`.locks/`、`xet/` 还是上级目录，再用 `id` 和 `getfacl` 检查。`.bashrc` 指定缓存位置不等于文件系统授权。共享组成员变更后重新登录；已有不可写文件需管理员补组写权限/ACL。原机管理员已执行共享目录与默认 ACL 配置时，不必无理由重复创建目录。

### 14.2 显示 installing_tools，却没有 settings64.sh

这表示安装尚未成功，可能仍在下载、认证过期、取消、重启、校验失败或收尾失败。检查安装器退出状态与最新日志，再验证 `settings64.sh`、`v++ --version` 和 `vivado -version`，不要只靠旧 JSON 状态推断完成。

### 14.3 Vivado 缺 libtinfo.so.5

本次 Ubuntu 26 不在工具支持列表，工具没有自动选到其自带的 Ubuntu 24 兼容库。处理时在 **Vivado/Vitis 自己的** `lib/lnx64.o/` 中，把 `libtinfo.so.5` 链接到同一工具目录已有的 `Ubuntu/24/libtinfo.so.5`，没有更改系统库。仅当对应文件确实存在、报错也相同时才适用；不要把系统 `libtinfo.so.6` 冒充成 `.5`。推荐环境优先按 AMD 的依赖安装说明处理。

### 14.4 U55C 时钟 IP 报 MMCM/PLL 抖动参数或器件文件缺失

检查是否漏装 **Virtex UltraScale+ HBM FPGAs**。补装后再运行 `check_u55c_clock.tcl`；该独立检查不依赖 Qwen 权重。原机补装后成功，再继续 link。

### 14.5 2022 年平台内 clk_metadata_adapter 加密副本无法综合

本次开发平台与 2025.1 工具组合，在派生平台 IP 的加密适配器副本上失败。同一个 AMD 平台包的 `other_ipdefs` 中已有同 VLNV/端口的未加密厂商实现。工程的 `use_vendor_clock_adapter.py` 验证旧/新文件 SHA256，完整复制厂商实现并保留版权，只改 `_x/` 派生目录，备份到该版本 `reports/`；不解密，也不改原始 XSA、平台压缩包或静态 shell。

Makefile 配置了 [Vitis 综合前 linkhook](https://docs.amd.com/r/2025.1-English/ug1702-vitis-accelerated-reference/linkhook-Options)，恢复 link 前也会应用修复。部分初次 IP 任务可能在顶层综合钩子之前运行；如果这些任务先失败，应停止/等待现有流程结束，确认 VPL 已解出平台源码，然后执行：

```bash
/usr/bin/python3 -I scripts/use_vendor_clock_adapter.py \
  --build-root "$QWEN_REPO/full-decode-stream/_x/link/vivado/vpl" --apply
python full-decode-stream/scripts/deploy_full.py --from-stage resume-link
```

它只接受工程预定义的派生构建路径，遇到未知文件哈希会停止，不能用于其他平台任意替换。`-I` 用于隔离 Vivado 可能设置的 Python 环境，避免系统 Python 误用工具自带的库。

### 14.6 XRT 报 device only buffer has no host buffer / 大 BO 分配失败

本次 memlock 为 8MiB，整模型权重与 KV 使用 device-only BO，通过小块上传。早期混合实现用子 BO 时失败，已改为父 BO 地址加偏移；完整 Decoder 主机库沿用适配的大内存分配方式。合作者应编译当前主机源码，不要复用旧 `.so` 或把所有大缓冲直接改成 pinned host BO。

### 14.7 位流载入失败或输出不匹配

依次检查 BDF、XRT 驱动与用户权限、shell/interface UUID、平台及 HBM 绑定、模型 revision、权重布局和当前 `.xclbin` 是否属于同一版本。`full-decode-fast` 的权重/位流不能与 stream 版混用。真实硬件运行前取消 `XCL_EMULATION_MODE`；先通过软件比较，再定位 FPGA 问题。

### 14.8 工具命令找不到、内存耗尽或已有锁

检查 `QWEN_VITIS_SETTINGS`、本机配置及 XRT PATH。最新版验收会查找 PATH 或 `/opt/xilinx/xrt/bin/` 中的 `xrt-smi`/`xbutil`、`xclbinutil`。内存不足时降低并行数、换资源充足的编译机，并检查实际子任务日志。遇到部署锁先查旧进程，不要同时启动第二个流程或删除活跃构建目录。

## 15. 哪些内容应上传 GitHub

合作者从零复现需要“源码 + 依赖清单 + 配置模板 + 操作说明”，大型厂商工具、原机状态和模型数据通过各自来源取得。

| 路径 | 是否提交 | 原因 |
| --- | --- | --- |
| `full-decode-stream/hls/` | 必须 | 当前完整 Decoder 电路源码、头文件、HBM 连接配置 |
| `full-decode-stream/host/` | 必须 | CPU 软件验证与 XRT 主机运行库 |
| `full-decode-stream/scripts/`、`Makefile`、`run_full.sh`、`README.md` | 必须 | 打包、参考、部署、运行和状态工具 |
| 根目录 `scripts/` | 必须保留相关公共脚本；建议全部源码提交 | 最新版依赖其中 safetensors 解析、部署监督器、时钟兼容脚本、Python 入口与本机配置工具 |
| `config/deployment.example.json`、`vitis-2025.1.example.txt` | 必须 | 合作者生成本机配置和安装器配置的模板 |
| `requirements-reproduce.txt`、`requirements-reference.txt` | 必须 | 固定成功环境及参考依赖说明 |
| `docs/`、README、`.gitignore` | 必须 | 操作步骤、故障说明、发布边界 |
| 根目录 `hls/`、`host/`、`tests/`、`Makefile` | 建议 | 早期 GEMV/混合工程、探针及 23 项单元测试的依赖 |
| `full-decode/`、`full-decode-fast/` 的源码部分 | 建议 | 完整上板与两轮优化的可读历史，便于研究/比较；不需要预先构建它们 |
| `docs/evidence/benchmark-summary.json` | 建议 | 去除本机路径/安装下载信息的历史结果摘要 |
| `deployment.json`、`config/local.mk`、`*.local.txt` | 不提交 | 本机路径、BDF、完成状态由合作者自己生成 |
| 原始 `weights/`、模型 snapshot、`reference/`、`.safetensors`、`.bin` | 不提交 | 大型数据，按固定 revision 下载并由源码重新生成 |
| `build/`、`_x/`、`.ipcache/`、`.Xil/`、`.srcs/` | 不提交 | 平台/IP/编译中间产物，可重建 |
| `installers/`、`platforms/`、`.deps/`、AMD 安装目录 | 不提交 | 第三方安装包、平台包、工具与本机依赖 |
| 原始 `reports/`、日志、认证文件、`.env` | 不直接提交 | 本机运行信息、绝对路径，安装日志还可能包含带认证信息的下载地址 |

不要只上传 `full-decode-stream/` 而漏掉根目录公共脚本和模板。此次建议上传三个完整 Decoder 版本及早期工程的**全部源码**：总体很小，可以保留完整技术过程，同时用 README 清楚标出 stream 是主线。

若只希望合作者“直接运行”，可以另外考虑分发已编译 `.xclbin`，并附模型 revision、源码 commit、工具/平台版本、位流 SHA256、接口 UUID 及实机数值报告。它应是独立的可选 Release 产物，需确认所含厂商组件的分发条件；主分支无需依赖它才能从源码构建。大型模型不要为了这个目的塞入 Git 历史，也不需要为了提交本工程源码启用 Git LFS。

## 16. 当前布局是否需要改

**不需要先整体搬目录才能上传。** 当前按技术演进阶段分目录，适合保留已验证版本；问题主要在本机配置和生成物混在工作目录，以及曾经存在对旧权重/旧报告的依赖。本次已补齐这些影响复现的部分，保留当前源码位置，避免改变经过验证的 HLS/主机代码和路径规则。

建议仓库保持如下结构；标记“本机”的内容不进入 Git：

```text
FPGA/
├── README.md                         入口，指向本指南和当前版本
├── requirements-reproduce.txt        成功环境版本记录
├── requirements-reference.txt        原参考依赖清单
├── .gitignore
├── config/
│   ├── deployment.example.json
│   ├── vitis-2025.1.example.txt
│   ├── local.mk                      本机
│   └── vitis-2025.1.local.txt         本机
├── scripts/                          公共准备、监督、兼容及发布工具
├── docs/
│   ├── FROM-ZERO-TO-GITHUB.md         本指南
│   ├── INSTALL-FPGA-TOOLS.md          2026-10-26 历史安装记录
│   └── evidence/benchmark-summary.json
├── full-decode-stream/               当前主线
│   ├── hls/                          电路 C++ / HBM 绑定
│   ├── host/                         C++ 主机库
│   ├── scripts/                      打包、运行、验收
│   ├── Makefile / run_full.sh / README.md
│   └── build/ weights/ reference/ reports/ _x/   本机生成
├── full-decode-fast/                 上一优化版，只提交源码
├── full-decode/                      最初完整 FPGA 版，只提交源码
├── hls/ host/ tests/ Makefile         早期 GEMV/混合工程及回归
├── deployment.json                   本机
└── installers/ platforms/ .deps/ .ipcache/ dist/   本机
```

为发布所做的实际改动：

1. 本机绝对路径转入配置，新增 `configure.py`、`.example` 模板和 `config/local.mk` 入口。
2. 运行脚本按配置/环境选择 Python，不再把原机 Conda 路径当作默认值。
3. 当前权重打包直接读取原始 safetensors，无需先运行历史版本；保留逐张量无损校验。
4. 当前验收允许没有历史 baseline，仍必须完成独立数值、板卡和时序检查。
5. 主机编译拆分 `runtime-cpu`、`runtime-fpga`，XRT 常见路径自动识别。
6. 新增固定依赖、生成物忽略规则和按源码白名单导出的 GitHub 发布工具。

HLS 算法与 C++ 主机计算代码保持不变，本次未为文档重新综合电路。以后若要统一成 `src/hls`、`src/host`、`python`、`examples/legacy` 也可以，但应另作一次重构：同步修改所有 Makefile、脚本定位、派生构建路径白名单、文档和测试，并重新确认构建与数值结果。当前没有必须如此搬迁的理由。

## 17. 用干净源码目录创建 GitHub 仓库

原工作目录约 11GB，大部分是下载、模型、IP 缓存和构建产物。为避免把它们误加进首次提交，已经提供白名单导出脚本：

```bash
python scripts/export_github.py
```

输出：

```text
dist/FPGA-github/                准备提交的干净源码树
dist/FPGA-github.tar.gz           同一源码树的压缩包
```

导出目录带有 `SOURCE-MANIFEST.json`，列出每个源文件的大小和 SHA256。脚本只复制源码/模板/文档及指定的结果摘要，不复制模型、位流、本机配置、原始日志和符号链接。输出已经存在时会拒绝覆盖；要重新导出，使用新的 `--output /新的目录/名称`，避免覆盖你在旧导出目录中的 Git 工作。

先在 GitHub 创建空仓库，再在**导出目录**执行以下命令，替换真实仓库地址：

```bash
cd "$QWEN_REPO/dist/FPGA-github"
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Add reproducible Qwen3-0.6B deployment for Alveo U55C"
git branch -M main
git remote add origin https://github.com/wangwk699/FPGA.git
git push -u origin main
```

执行提交前检查暂存列表，应该只有源码、文档、配置模板和摘要，没有 `.deb`、`.xclbin`、模型或 `deployment.json`。本次准备工作没有替你选择源码开源许可证；公开发布前应由代码权利人确定合适的 `LICENSE`，并保留依赖和厂商代码的原始声明。模型、AMD 软件的许可证与自己的源码许可证分别适用。

仓库地址已确定为 `wangwk699/FPGA`；本节命令用于以后自行提交更新。发布仅包含白名单源码，原部署工作目录、模型与已验证位流继续保留。

## 18. 复现闭环与本次发布准备的验证边界

合作者成功复现应能从“干净源码 + 自己下载的原始模型/AMD 工具/平台/XRT”完成：本机配置 → 无损权重 → CPU 参考 → C++ 数值比较 → HLS 编译与 link → 时序签核 → 真实 FPGA 数值与 CU/UUID 验收 → 正常生成。这个链条不依赖原机旧版 `weights/`、旧参考、旧报告或完成状态文件。

为检查发布材料，本次已经从白名单导出的干净副本实际执行：生成新配置、直接读取本地原始 checkpoint 打包、比较三份权重文件哈希、生成新的 PyTorch 参考、编译新的 CPU/XRT 主机库、完成 C++ 与真实 U55C 两种后端验证。二者均得到 39 次调用、1092 次层计算、全部输出 token 与五组数值检查一致；该副本 FPGA 实测约 **8.907 token/s**。

这次副本验证复用原来已经签核的 `.xclbin`，没有重新下载模型、重装工具或重新跑数小时的综合布线；另外用该次副本数值报告与既有时序证据，实际通过了没有历史 baseline 的独立验收汇总。原始硬件构建的实际成功记录来自此前三版完整部署，摘录在 `benchmark-summary.json`。因此，“源码准备与现有位流上板验证已通过”和“另一台机器重新安装及重新布线仍需执行本文步骤”应分别理解。
