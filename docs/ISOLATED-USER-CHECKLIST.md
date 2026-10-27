# FPGA：同服务器、家目录隔离用户从零部署清单

仓库：<https://github.com/wangwk699/FPGA>。当前主线为 `full-decode-stream/`，原始 BF16 权重、FP32 运算，不量化，28 层完整计算在 FPGA。

下面全部以**另一个用户自己的账号**执行。`$HOME` 始终代表该用户自己的家目录，不要把它手动改成别人的目录。代码、模型缓存、下载文件、Vitis 和平台均在自己的目录，不依赖 `/home/wangwk`、`/home/share` 或 `/home/.cache`。

系统安装的 XRT 驱动和运行库可以沿用；PCIe U55C 是同一个物理设备，家目录隔离不等于设备隔离。软件准备和编译可以各自进行，加载位流/上板推理前应约定独占使用时段，避免互相替换正在运行的电路。

以下从第 1 步依次执行。需要重新开终端时，重新激活环境，并恢复 `QWEN_REPO`、`QWEN_PYTHON`、`QWEN_MODEL` 等变量。

## 1. 在自己的家目录克隆仓库

```bash
cd "$HOME"
git clone https://github.com/wangwk699/FPGA.git
cd "$HOME/FPGA"
export QWEN_REPO="$PWD"
mkdir -p installers platforms/u55c reports .deps
df -h "$HOME"
free -h
```

预留约 250–300GB 空间用于工具、下载缓存和编译中间文件；实际随组件/缓存变化。官方加速开发要求内存至少 64GB、推荐 80GB，本机约 30GiB 加 swap 曾完成编译，但不是推荐资源配置。本机 Ubuntu 26.04 不在 Vitis 2025.1 官方支持列表，需要留意第 8、14 步的实际兼容问题。

## 2. 确认系统工具和 U55C 权限

```bash
command -v git
command -v g++
command -v make
command -v dpkg-deb
xrt-smi --version
xrt-smi examine -d 0000:01:00.1
```

本服务器已装 XRT `2.21.75`，U55C user BDF 是 `0000:01:00.1`，shell 为 `xilinx_u55c_gen3x16_xdma_base_3`。若实际设备改变，以查询结果为准。有些 XRT 使用 `xbutil` 命令名。

新账号若报设备访问权限错误，请管理员按既有 XRT 配置授予该账号设备访问权限；无需为了新用户重装全局驱动或刷新板卡基础固件。缺 `g++`/`make`/UUID 依赖时由管理员补齐系统包，或按第 6 步解出本地开发头文件。

## 3. 准备自己能使用的 Python 3.11

已有自己的 Conda 时直接创建环境；不能调用其他用户家目录里的 Conda。

没有 Conda 时，可先在自己家目录安装 Miniconda：

```bash
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o "$QWEN_REPO/installers/Miniconda3-Linux-x86_64.sh"
bash "$QWEN_REPO/installers/Miniconda3-Linux-x86_64.sh" -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

然后：

```bash
conda create -n fpga-qwen --override-channels -c conda-forge python=3.11 pip -y
conda activate fpga-qwen
cd "$QWEN_REPO"
python -m pip install -r requirements-reproduce.txt
export QWEN_PYTHON="$(command -v python)"
```

如果已装自己的 Python 3.11，也可用 `.venv` 替代 Conda。CPU 版 Torch 满足准备/数值参考用途，不需要 GPU 或 CUDA。

## 4. 将原始模型下载到自己的缓存

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_XET_HIGH_PERFORMANCE=1
# 需要镜像时设置；官方源可省略
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
)
PY

export QWEN_MODEL="$HF_HOME/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
sha256sum "$QWEN_MODEL/model.safetensors"
```

`model.safetensors` 约 1.50GB，预期 SHA256：

```text
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
```

如果曾在 `.bashrc` 设置过 `HF_HUB_CACHE`，它会覆盖 `HF_HOME/hub` 的默认位置；本清单已把它显式设为 `"$HF_HOME/hub"`。不要让旧配置指向其他用户目录。高性能后端是 hf-xet，不是 hf_transfer。

## 5. 下载自己的一份 AMD 安装器和平台

从以下官方页面下载，保存到 `$HOME/FPGA/installers/`：

- [AMD 2025.1 下载](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2025-1.html)：`FPGAs_AdaptiveSoCs_Unified_SDI_2025.1_0530_0145_Lin64.bin`，约 336MiB，后续还要联网下载几十 GB。
- [U55C 下载](https://www.amd.com/en/support/downloads/alveo-downloads.html/accelerators/alveo/u55c.html)：Development Target Platform，`xilinx-u55c-gen3x16-xdma-3-202210-1-dev_1-3514517_all.deb`，约 165MiB。

需要使用该用户自己的 AMD 下载认证。源码仓库不包含厂商安装包、平台、模型或位流。

## 6. 在自己的目录解出平台和匹配的 XRT 开发头文件

```bash
cd "$QWEN_REPO"
dpkg-deb -x installers/xilinx-u55c-gen3x16-xdma-3-202210-1-dev_1-3514517_all.deb platforms/u55c
export QWEN_PLATFORM="$QWEN_REPO/platforms/u55c/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm"
```

本机是发行版拆包的 XRT，运行库已在 `/usr/lib/x86_64-linux-gnu/`。新用户不能使用其他用户目录里的 `.deps`，可不经 sudo 下载并解出**匹配现有运行库版本**的开发包：

```bash
cd "$QWEN_REPO/installers"
apt download "libxrt-dev=$(dpkg-query -W -f='${Version}' libxrt2)"
apt download uuid-dev
for qwen_deb in libxrt-dev_*.deb uuid-dev_*.deb; do
  dpkg-deb -x "$qwen_deb" "$QWEN_REPO/.deps/xrt-dev"
done
cd "$QWEN_REPO"
test -f .deps/xrt-dev/usr/include/xrt/xrt_device.h
test -f .deps/xrt-dev/usr/include/uuid/uuid.h
test -f /usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2
```

如果软件源已没有安装版本，应向管理员索取同版本开发包或对应软件源；不能任意用不匹配头文件替代。以上包名和库路径针对本服务器，迁移到其他发行版按其 XRT 安装布局调整。

## 7. 生成自己的本机配置

```bash
"$QWEN_PYTHON" scripts/configure.py \
  --model "$QWEN_MODEL" \
  --python "$QWEN_PYTHON" \
  --device 0000:01:00.1 \
  --platform "$QWEN_PLATFORM" \
  --vitis-root "$HOME/amd" \
  --xrt-include "$QWEN_REPO/.deps/xrt-dev/usr/include" \
  --xrt-lib /usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2
```

生成自己的 `deployment.json`、`config/local.mk`、`config/vitis-2025.1.local.txt`。不要复制原用户的部署完成配置；这一步不下载模型、不访问板卡，也不覆盖已有配置。

## 8. 在自己的家目录安装 Vitis/Vivado 2025.1

```bash
bash installers/FPGAs_AdaptiveSoCs_Unified_SDI_2025.1_0530_0145_Lin64.bin \
  --noexec --target "$QWEN_REPO/.deps/vitis-installer-2025.1"
bash scripts/setup_vitis.sh auth
```

在交互终端按提示用自己的 AMD 账号认证，然后安装：

```bash
bash scripts/setup_vitis.sh install > reports/vitis-install-local.log 2>&1
```

模板已经包含 Alveo 与 **Virtex UltraScale+ HBM FPGAs**，不得漏选 HBM。安装目标是自己的 `$HOME/amd`。从第二个终端查看安装日志，下载中断后保留缓存、确认旧安装器已退出，再用同一配置恢复。

本服务器 Ubuntu 26.04 的原安装曾卡在 `libtinfo.so.5`。若日志出现同样错误，等对应组件已解出后，在自己的工具目录执行以下条件修复；只使用 AMD 已附带的 Ubuntu 24 兼容库：

```bash
for qwen_component in Vivado Vitis; do
  qwen_lib="$HOME/amd/2025.1/$qwen_component/lib/lnx64.o"
  if [ -f "$qwen_lib/Ubuntu/24/libtinfo.so.5" ] && [ ! -e "$qwen_lib/libtinfo.so.5" ]; then
    ln -s Ubuntu/24/libtinfo.so.5 "$qwen_lib/libtinfo.so.5"
  fi
done
```

若旧收尾子进程已经失败或停止，结束自己的失败安装任务后重跑相同安装命令。不要修改系统库，也不要结束其他用户的进程。更多说明见 [完整指南的故障章节](FROM-ZERO-TO-GITHUB.md#14-这次遇到的故障以及新机器如何处理)。

## 9. 验证编译工具与 HBM 支持完整

```bash
export QWEN_VITIS_SETTINGS="$HOME/amd/2025.1/Vitis/settings64.sh"
source "$QWEN_VITIS_SETTINGS"
v++ --version
vivado -version
platforminfo -p "$QWEN_PLATFORM"
vivado -nolog -nojournal -mode batch -source scripts/check_u55c_clock.tcl
```

必须得到 2025.1 工具版本及 `U55C_CLOCK_IP_CREATED`。缺 HBM 组件时先 `bash scripts/setup_vitis.sh add-hbm`，修复后再继续。

## 10. 在自己的目录生成无损权重

```bash
"$QWEN_PYTHON" full-decode-stream/scripts/pack_weights.py --model "$QWEN_MODEL"
```

输出在自己的 `full-decode-stream/weights/`，310 个独立张量逐一逆重排检查。它直接读原始 checkpoint，不需要其他用户或旧版本的权重。

## 11. 生成参考并先通过 CPU 软件验收

```bash
make -C full-decode-stream runtime-cpu
"$QWEN_PYTHON" full-decode-stream/scripts/make_reference.py
./full-decode-stream/run_full.sh --backend cpu-cpp \
  --verify full-decode-stream/reference/default \
  --max-new-tokens 32 \
  --report full-decode-stream/reports/full-cpp-validation.json
make test PYTHON="$QWEN_PYTHON"
```

预期全部生成 token 相同、五组数值检查通过，单元测试通过。这一步没有在 FPGA 运行。

## 12. 自己编译位流并在约定时段完成板卡验收

该命令把编译、签核和真实板卡测试串联起来。确保最后上板阶段不会与另一用户使用同一张 U55C 冲突：

```bash
unset XCL_EMULATION_MODE
"$QWEN_PYTHON" full-decode-stream/scripts/deploy_full.py
```

新终端查看进度：

```bash
cd "$HOME/FPGA"
python3 full-decode-stream/scripts/status.py
```

流程会生成自己的一份 `.xo`、`.xclbin`、主机库、布线时序报告和实机数值报告。没有历史 baseline 也可部署，不用先构建 `full-decode/` 或 `full-decode-fast/`。

## 13. 若首次 link 因平台时钟适配器失败，修复后恢复

只有日志明确报 `clk_metadata_adapter` 对应问题且旧构建已停止时，执行：

```bash
/usr/bin/python3 -I scripts/use_vendor_clock_adapter.py \
  --build-root "$QWEN_REPO/full-decode-stream/_x/link/vivado/vpl" --apply
"$QWEN_PYTHON" full-decode-stream/scripts/deploy_full.py --from-stage resume-link
```

这只修改自己的派生构建目录，不修改平台原包或基础 shell。其他故障按照完整指南第 13、14 节选择恢复阶段，不要无条件删掉缓存重来。

## 14. 确认从零部署成功

检查自己的 `full-decode-stream/reports/FINAL-DEPLOYMENT.json`、`full-fpga.json` 和 `signoff.json`：

- 状态 complete，`full_decoder_on_fpga` 和 `fpga_executed` 为 true。
- `matches_reference_tokens` 为 true，五项数值检查全部通过。
- 默认提示 39 次内核调用、1092 次层计算，CPU model forward 和层间 CPU 计算往返为零。
- 布线时序及 bus-skew 均通过，XRT 查询到的 CU 使用计数和 UUID 与自己位流一致。

原机同提示约 8.9 token/s 是参考测量；另一用户应以自己的实测和签核为准。batch=1，输入加输出最多 2048 token，greedy，关闭 thinking。

## 15. 以后直接推理，无需再下载或编译

```bash
cd "$HOME/FPGA"
./full-decode-stream/run_full.sh --backend fpga \
  --prompt '请用一句话介绍杭州。' \
  --max-new-tokens 32 \
  --report full-decode-stream/reports/my-run.json
```

仍需约定板卡使用时段。所有模型、配置和产物都来自该用户自己的目录；系统 XRT 和物理板卡由服务器提供。
