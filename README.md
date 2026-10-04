# Vista4D × FlowLong：基于 4D 点云的视频重拍与长视频稳定化

## 推荐入口：每个视频一份 YAML

视频路径、384p/720p、分割关键词、相机平滑、prompt 和推理参数可统一写入 [configs/example.yaml](configs/example.yaml)。完整字段、逐步命令和产物说明见 [视频配置指南](docs/video_configuration.md)。Conda 环境和 GPU仍在命令行指定。

```bash
# 已安装依赖并激活主环境后，只读预览，不启动推理。
python -m scripts.test_video.run_video_experiment --config configs/1776148878076.yaml
```

单个 49-frame 片段设置 `video.mode: single_clip`、`single_clip.start_frame` 和可选 `single_clip.name`，然后通过 `--stages split recon smooth render inference --execute` 分阶段或顺序执行。`CLIP_NAME/EXAMPLE` 默认自动生成。本文后续的直接 Bash 命令保留为底层/旧流程说明；YAML 流程输出到 `results/configured/<视频名>/<分辨率>/<run_name>/`，不会自动接管旧实验目录。

本仓库基于 [Vista4D: Video Reshooting with 4D Point Clouds](https://eyeline-labs.github.io/Vista4D)（CVPR 2026 Highlight，[论文](https://arxiv.org/abs/2604.21915) / [模型](https://huggingface.co/Eyeline-Labs/Vista4D)）。输入一段单目视频和一条目标相机轨迹，Vista4D 先把视频重建为 4D 点云，再在目标相机下渲染点云，最后用 Wan2.1-T2V-14B 微调的视频扩散模型生成“沿新轨迹重新拍摄”的视频。

在原版基础上，本仓库主要做了以下扩展：

1. **单片段流程可以直接用于自己的视频**：新增片段准备、DA3 默认重建、相机轨迹平滑，以及 384p / 720p 分辨率校验和低显存预设；并在 NVIDIA GB10（aarch64，CUDA 13）上跑通。
2. **支持超过 49 帧的长视频**：切窗与尾部 padding、全局 reconstruction 对齐、一次性 shared-static 渲染，以及 FlowLong 联合去噪，减少窗口接缝。
3. **提供实验管理和轨迹工具**：YAML 配置、分阶段执行、运行记录、轨迹可视化和相机设计 UI。

本文是入口文档：说明如何安装环境、下载模型，并对**单个 49 帧片段**完成前处理和 Vista4D 推理。其他主题见下表：

| 文档 | 内容 |
| --- | --- |
| [README.md](README.md)（本文） | 环境、模型、单片段前处理与推理、原版功能入口 |
| [READMEv2.md](READMEv2.md) | 长视频：切窗、全局重建、shared-static 渲染、FlowLong 联合去噪 |
| [docs/FlowLong_Plan.md](docs/FlowLong_Plan.md) | FlowLong 实现设计 |
| [docs/report_usage.md](docs/report_usage.md) | 各阶段耗时与显存实测 |
| [docs/report_low_vram.md](docs/report_low_vram.md) | 低显存预设验证 |

---

## 目录

- [1. 相对原版 Vista4D 的改动](#1-相对原版-vista4d-的改动)
- [2. 环境安装](#2-环境安装)
- [3. 下载模型权重](#3-下载模型权重)
- [4. 快速验证：官方示例片段](#4-快速验证官方示例片段)
- [5. 自定义视频：单片段完整流程](#5-自定义视频单片段完整流程)
- [6. 推理参数与显存预设](#6-推理参数与显存预设)
- [7. 长视频：FlowLong](#7-长视频flowlong)
- [8. 相机轨迹可视化](#8-相机轨迹可视化)
- [9. 原版 Vista4D 功能](#9-原版-vista4d-功能)
- [10. 目录结构](#10-目录结构)
- [11. 测试](#11-测试)
- [12. 常见问题](#12-常见问题)

---

## 1. 相对原版 Vista4D 的改动

| 方面 | 改动 | 主要文件 |
| --- | --- | --- |
| 4D 重建 | 默认使用 Depth Anything 3（DA3），Pi3X 保留为 `RECON_METHOD=pi3`；所有模型从本地 `checkpoints/` 读取 | `scripts/preprocess/recon_and_seg_single.py` |
| 动态分割 | SAM3 官方实现，读取本地 `checkpoints/sam3/sam3.pt` | `utils/recon_and_seg/seg_sam3_official.py` |
| 自定义视频 | 从任意视频的指定起始帧截取 49 帧，中心裁剪并缩放到目标分辨率 | `scripts/preprocess/prepare_custom_single_video.py` |
| 单片段流水线 | 前处理、轨迹平滑、渲染、推理四个脚本共享一份配置，结果目录按片段名自动派生 | `scripts/test_video/{config,recon_and_seg,smooth,render,inference}.sh` |
| 相机轨迹 | 对重建相机轨迹做高斯平滑，作为“稳定化”目标轨迹 | `scripts/preprocess/smooth_camera_trajectory.py` |
| 分辨率 | 384p = 672×384、720p = 1280×720 两个 profile，尺寸与 checkpoint 不匹配时直接报错 | `utils/resolution.py` |
| 显存 | `--vram_preset full / balanced / low_vram`（block-swap offload、可选 FP8） | `utils/vram_presets.py` |
| 性能分析 | 前处理可输出逐阶段耗时和 CUDA 峰值显存 JSON（`--profile_json`） | `scripts/preprocess/recon_and_seg_single.py` |
| 硬件 | 适配 NVIDIA GB10（aarch64、CUDA capability 12.1、CUDA 13、统一内存） | 见 [2.3 节](#23-方案-bnvidia-gb10aarch64cuda-13) |
| 长视频 | 规则切窗 + manifest、Sim(3) 全局对齐、shared-static 渲染、FlowLong 联合去噪、baseline center-cut | 见 [READMEv2.md](READMEv2.md) |
| 实验管理 | 统一入口（只打印计划 / `--execute` 执行）、GPU 占用保护、排队执行 | `scripts/test_video/run_video_experiment.py`、`queue_video_experiment.py` |

原版的 4D 场景重组（点云编辑）、动态场景扩展（DSE）和相机设计 UI 仍然保留，见[第 9 节](#9-原版-vista4d-功能)。

---

## 2. 环境安装

### 2.1 基本要求

- Linux，NVIDIA GPU。
- Conda（Miniconda / Miniforge 均可）。
- 系统工具：`ffmpeg`、`ffprobe`。长视频流程另外用到 `jq`、`tmux`；相机 UI 需要 Node.js。
- 显存：384p 单片段在默认 `full` 预设下，CUDA 峰值约 45 GiB allocated / 49 GiB reserved（GB10 实测，见 [5.7 节](#57-参考耗时与显存)）。显存不足时见[第 6 节](#6-推理参数与显存预设)。
- 所有命令都在**仓库根目录**执行。

以下两种安装方案二选一。

### 2.2 方案 A：x86_64 + CUDA 12.8（原版方案）

```bash
conda create -n vista4d python=3.12 -y
conda activate vista4d

# 系统 CUDA 不是 12.8（或不确定）时，在环境内安装独立的 CUDA toolkit 和编译器
conda install -c nvidia cuda-toolkit=12.8
conda install -c conda-forge gxx_linux-64
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

可选的加速组件：

```bash
# 在无 GPU 的登录节点编译，或需要跨 GPU 架构运行时，先指定目标架构，例如：
# 8.0 = A100，8.6 = A40 / RTX A6000，8.9 = L40 / RTX 4090，9.0 = H100 / H200
# export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"

pip install flash-attn==2.8.3 --no-build-isolation   # 从源码编译时耗时较长
pip install "xfuser[flash-attn]==0.4.5"              # 仅多 GPU USP 推理需要
```

没有安装 flash-attn 时，DiT attention 会自动回退到 PyTorch SDPA，结果正确但速度较慢。

### 2.3 方案 B：NVIDIA GB10（aarch64，CUDA 13）

本仓库的实验机器使用这个环境，环境名为 `vista4d-pgx`。GB10 的 CUDA capability 是 12.1，需要 cu130 版本的 PyTorch。该环境没有安装 `xformers`、`decord`、`pycolmap`（aarch64 + CUDA 13 下缺少可直接安装的预编译包，且本仓库的运行路径不依赖它们），因此安装时从 requirements 中去掉这三项，并用 `decord2` 提供 `decord` 模块。

```bash
conda create -n vista4d-pgx python=3.12 pip -y
conda activate vista4d-pgx

python -m pip install --upgrade pip setuptools wheel packaging psutil ninja

# CUDA 13.0 toolkit 安装到环境内，不使用系统 CUDA 12.8
conda install -y --override-channels \
  -c nvidia/label/cuda-13.0.3 -c conda-forge cuda-toolkit=13.0.3

python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130

# requirements.txt 去掉 aarch64 上不可用的三项后安装
grep -v -x -E 'xformers|decord|pycolmap' requirements.txt > /tmp/vista4d-requirements-pgx.txt
python -m pip install -r /tmp/vista4d-requirements-pgx.txt
python -m pip install decord2==3.4.0
python -m pip install -U "huggingface_hub[cli]"
```

**编译 CUDA 扩展前**，需要让编译器使用环境内的 CUDA 13 并指定 GB10 的目标架构。可以把以下内容保存为仓库根目录下的 `env_pgx.sh`（该文件已在 `.gitignore` 中，不随仓库分发），激活环境后执行 `source env_pgx.sh`：

```bash
#!/usr/bin/env bash
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Activate the vista4d-pgx Conda environment first."
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

# 不加载系统 CUDA 12.8 的库
unset LD_LIBRARY_PATH
CUDA_TARGET_LIB="$(find "$CONDA_PREFIX/targets" -maxdepth 2 -type d -name lib -print -quit 2>/dev/null)"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
if [ -n "$CUDA_TARGET_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDA_TARGET_LIB:$LD_LIBRARY_PATH"
fi

# GB10 是 SM121；通用 PyTorch 扩展按 12.0 + PTX 编译
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export FLASH_ATTN_CUDA_ARCHS="120"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
```

日常推理不强制执行 `env_pgx.sh`；如果遇到 CUDA 库版本冲突，再执行它。

GB10 上的已知限制：

| 现象 | 说明 |
| --- | --- |
| PyTorch 启动时警告 `cuda capability 12.1 ... supported (8.0) - (12.0)` | 预编译 kernel 的上限是 12.0，可以忽略，推理正常 |
| flash-attn、xformers、xfuser 均未安装 | DiT 使用 PyTorch SDPA；不能使用多 GPU USP（本机也只有一块 GPU） |
| `low_vram` 预设报 `CUBLAS_STATUS_NOT_SUPPORTED` | FP8 计算（`torch._scaled_mm`）在本机不可用，必须加 `--no-fp8_compute` |
| `nvidia-smi` 显示 `Memory-Usage: Not Supported` | 统一内存架构。请以报告 JSON 中的 `torch.cuda.max_memory_allocated/reserved` 为准 |
| `--vram_limit` 基本不起作用 | 统一内存下空闲显存的语义不同，offload 阈值很少触发，详见 [docs/report_low_vram.md](docs/report_low_vram.md) |

### 2.4 检查环境

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import cv2, OpenEXR, decord, transformers
import sam3, depth_anything_3                   # 仓库内置的 SAM3 与 DA3 代码
from diffsynth.pipelines.wan_video_vista4d import Vista4DPipeline
try:
    import flash_attn; print("attention: flash-attn 2")
except ImportError:
    print("attention: PyTorch SDPA")
print("ok")
PY

command -v ffmpeg ffprobe
```

`transformers` 必须是 4.x（requirements 固定为 4.57.6），5.x 会导致 DA3 无法运行。

---

## 3. 下载模型权重

### 3.1 需要的模型

| 模型 | Hugging Face 仓库 | 本地目录 | 用途 |
| --- | --- | --- | --- |
| Vista4D | `Eyeline-Labs/Vista4D` | `checkpoints/vista4d/` | 扩散模型微调权重：`384p49_step=30000/`、`720p49_step=3000/`，每个目录含 `dit.pth` 和 `config.yaml` |
| Wan2.1-T2V-14B | `Wan-AI/Wan2.1-T2V-14B` | `checkpoints/wan/Wan2.1-T2V-14B/` | 基础 DiT、T5 文本编码器、VAE、tokenizer |
| DA3 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | `checkpoints/DA3NESTED-GIANT-LARGE-1.1/` | 4D 重建（默认） |
| SAM3 | `facebook/sam3`（**需要申请访问权限**） | `checkpoints/sam3/` | 动态区域分割，读取其中的 `sam3.pt` |
| Pi3X（可选） | `yyfz233/Pi3X` | `checkpoints/Pi3X/` | 仅 `RECON_METHOD=pi3` 时需要 |

两个 Vista4D checkpoint：

| checkpoint | 分辨率 | 帧数 | 训练步数 | 说明 |
| --- | --- | --- | --- | --- |
| `384p49_step=30000` | 672×384 | 49 | 30000 | 从 Wan2.1-T2V-14B 微调 |
| `720p49_step=3000` | 1280×720 | 49 | 3000 | 从 `384p49_step=30000` 继续微调 |

720p 必须使用 720p checkpoint，不能只把 384p 的宽高改大。

### 3.2 下载

先在 [facebook/sam3](https://huggingface.co/facebook/sam3) 页面申请访问权限，然后登录：

```bash
hf auth login
```

#### 方式一：逐个下载到脚本期望的位置（推荐）

```bash
hf download Eyeline-Labs/Vista4D --local-dir checkpoints/vista4d
hf download Wan-AI/Wan2.1-T2V-14B --local-dir checkpoints/wan/Wan2.1-T2V-14B
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir checkpoints/DA3NESTED-GIANT-LARGE-1.1
hf download facebook/sam3 --local-dir checkpoints/sam3

# 可选：只用某个分辨率时，Vista4D 可以只下载对应子目录
# hf download Eyeline-Labs/Vista4D --local-dir checkpoints/vista4d --include "384p49_step=30000/*"

# 可选：使用 Pi3X 重建时
# hf download yyfz233/Pi3X --local-dir checkpoints/Pi3X
```

#### 方式二：断点续传脚本

`scripts/download_hf_checkpoints.sh` 依次下载上面四个必需仓库，中断后自动重试，每个仓库完成后写入 `.hf_download_complete` 标记，适合放在 tmux 中长时间运行。该脚本按**仓库名**建目录，下载完成后需要手动移动两个目录：

```bash
bash scripts/download_hf_checkpoints.sh

mv checkpoints/Vista4D checkpoints/vista4d
mkdir -p checkpoints/wan && mv checkpoints/Wan2.1-T2V-14B checkpoints/wan/
```

### 3.3 检查目录

以下检查针对原始 `.pth` 布局。现在也支持 FP32 无损 safetensors 分片；转换、加载和哈希兼容说明见 [checkpoint 分片指南](docs/checkpoint_sharding.md)。默认路径不变，分片版使用独立目录并显式选择。

```bash
for f in checkpoints/vista4d/384p49_step=30000/dit.pth \
         checkpoints/vista4d/384p49_step=30000/config.yaml \
         checkpoints/vista4d/720p49_step=3000/dit.pth \
         checkpoints/vista4d/720p49_step=3000/config.yaml \
         checkpoints/wan/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
         checkpoints/wan/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
         checkpoints/DA3NESTED-GIANT-LARGE-1.1/model.safetensors \
         checkpoints/sam3/sam3.pt; do
  test -f "$f" && echo "ok      $f" || echo "MISSING $f"
done
ls checkpoints/wan/Wan2.1-T2V-14B/diffusion_pytorch_model*.safetensors | wc -l
```

---

## 4. 快速验证：官方示例片段

`media/single/` 提供 8 段 720p 示例视频和作者设计的目标相机：`couple-newspaper`、`couple-walk`、`elderly-tennis`、`mountain-hike`、`park-selfie`、`parkour`、`snowboard`、`soapbox`。三条命令即可跑通完整流程：

```bash
# 1) 4D 重建 + 动态分割（720p、49 帧，DA3_PROCESS_RES=896）
EXAMPLE=couple-newspaper RECON_METHOD=da3 \
bash scripts/preprocess/example_recon_and_seg_single.sh

# 2) 在示例目标相机下渲染点云
EXAMPLE=couple-newspaper RESOLUTION=384p \
bash scripts/preprocess/example_render_single.sh

# 3) Vista4D 推理（示例 prompt 已内置）
EXAMPLE=couple-newspaper RESOLUTION=384p \
bash scripts/inference/example_inference_single.sh
```

结果分别位于 `results/single/couple-newspaper/` 下的 `recon_and_seg/`、`render_384p/`、`vista4d_384p/`。

> **注意**：示例目标相机是作者基于 **Pi3X** 重建设计的。DA3 重建的场景尺度不同，相机可能不在预期位置。要复现原版效果，可以用 `RECON_METHOD=pi3`（需要 Pi3X 权重）；或者用 `recon_and_seg_single.py --scene_scale` 调整尺度；也可以用相机 UI（[10.3 节](#104-相机设计-ui)）重新设计相机。

---

## 5. 自定义视频：单片段完整流程

### 5.1 流程概览

Vista4D 的原生输入是**一个 49 帧片段**。对自己的视频，单片段流程是：

```text
原始视频
  → [5.2] 截取 49 帧，中心裁剪并缩放到 672×384 或 1280×720
  → [5.3] DA3 深度 / 相机 + SAM3 动态与天空 mask             → recon_and_seg/
  → [5.4] 生成目标相机轨迹：平滑轨迹 / 原轨迹 / 自定义轨迹
  → [5.5] 点云 unprojection，在目标相机下渲染                 → render_<res>_smooth/
  → [5.6] Vista4D 扩散推理                                    → vista4d_<res>_smooth/
```

后四步使用 `scripts/test_video/` 下的四个脚本，它们都读取 `scripts/test_video/config.sh`，由同一组环境变量决定输入和输出路径：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `SOURCE_VIDEO` | 仓库内置的测试片段 | 49 帧输入片段 |
| `RESOLUTION` | `384p` | `384p` 或 `720p`，同时决定宽高和 checkpoint |
| `NUM_FRAMES` | `49` | 片段帧数，保持 49 |
| `EXAMPLE` | 由文件名派生 | 结果目录名：`results/single/$EXAMPLE/` |
| `USE_SMOOTHED_CAMERA` | `true` | 渲染时使用平滑后的轨迹 |
| `CAM_PATH` | 由上一项派生 | 显式指定目标相机 `.npz`，优先级最高 |

`EXAMPLE` 的派生规则：文件名以 `_384p49` 或 `_720p49` 结尾时，把后缀替换为当前 `RESOLUTION`，例如 `walk_384p49.mp4` 在 `RESOLUTION=720p` 下对应 `walk_720p49`；否则直接使用文件名。**建议片段按 `<名称>_<分辨率>49.mp4` 命名**，这样 384p 和 720p 的结果不会写进同一个目录。

先在终端设置本节复用的变量：

```bash
conda activate vista4d-pgx            # 或 vista4d
export RESOLUTION=384p                # 或 720p
export CLIP_NAME=my_video_frames000000_000048_${RESOLUTION}49
export SOURCE_VIDEO=./media/single/${CLIP_NAME}.mp4
```

### 5.2 截取 49 帧片段

```bash
python -m scripts.preprocess.prepare_custom_single_video \
  --input ./data/my_video.mp4 \
  --output_dir ./media/single \
  --output_name "$CLIP_NAME" \
  --start_frame 0 \
  --num_frames 49 \
  --width 672 --height 384            # 720p 用 --width 1280 --height 720
```

- 从 `--start_frame`（从 0 开始）连续读取 `--num_frames` 帧；超出视频长度会报错。
- 先按目标宽高比做**中心裁剪**，再用 Lanczos 缩放到目标尺寸，保持原 FPS，不保留音频。
- 输出为 `media/single/<output_name>.mp4`。

`recon_and_seg_single.py` 遇到超过 49 帧的输入时，会自动截取**中间**的 49 帧。为了明确控制时间范围，请先用本步骤截取片段。

超过 49 帧、需要完整生成的视频，请使用长视频流程（[第 7 节](#7-长视频flowlong)）。

### 5.3 4D 重建与动态分割（DA3 + SAM3）

```bash
SOURCE_VIDEO="$SOURCE_VIDEO" RESOLUTION="$RESOLUTION" \
RECON_METHOD=da3 \
DA3_PROCESS_RES=672 \
SEG_KEYWORDS="person man woman hand phone bag backpack car" \
SAVE_VIS=true \
bash scripts/test_video/recon_and_seg.sh
```

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RECON_METHOD` | `da3` | `da3` 或 `pi3` |
| `DA3_PROCESS_RES` | `672` | DA3 的处理分辨率（长边）。384p 用 672；720p 建议 1280（与长视频统一入口一致），显存不足时降到 896 或更低 |
| `DA3_MODEL_ID` | `./checkpoints/DA3NESTED-GIANT-LARGE-1.1` | DA3 权重目录 |
| `PI3_MODEL_ID` | `./checkpoints/Pi3X` | Pi3X 权重目录 |
| `SEG_KEYWORDS` | `person man woman hand phone bag backpack car` | SAM3 文本提示词，以空格分隔。应覆盖画面中**所有会动的物体** |
| `SAVE_VIS` | `true` | 额外保存深度 / 天空 / 分割 / 动态 mask 四宫格可视化 `vis.mp4` |
| `OUTPUT_FOLDER` | `results/single/$EXAMPLE/recon_and_seg` | 输出目录 |

输出：

```text
results/single/<EXAMPLE>/recon_and_seg/
├── video.mp4          裁剪缩放后的 49 帧源视频
├── cameras.npz        每帧相机外参（C2W）与内参
├── depths/            每帧深度，00000.exr ...（float16）
├── dynamic_mask/      每帧动态区域 mask，00000.png ...
├── sky_mask/          每帧天空 mask
└── vis.mp4            SAVE_VIS=true 时的可视化
```

使用建议：

- **分割提示词决定哪些点被当作动态点**。漏掉的运动物体会被当作静态点在时间上持久化，渲染时出现拖影。建议打开 `vis.mp4` 检查动态 mask。
- DA3 自带天空分割；Pi3X 没有，会额外用 SAM3 的 `sky` 提示词分割天空。
- 需要逐阶段耗时和显存时，直接调用 Python 入口并加 `--profile_json <path>`：

  ```bash
  python -m scripts.preprocess.recon_and_seg_single \
    --video_path "$SOURCE_VIDEO" --output_folder ./results/single/${CLIP_NAME}/recon_and_seg \
    --seg_keywords person car --recon_method da3 \
    --da3_model_id ./checkpoints/DA3NESTED-GIANT-LARGE-1.1 --da3_process_res 672 \
    --height 384 --width 672 --num_frames 49 \
    --profile_json ./results/single/${CLIP_NAME}/recon_and_seg/profile.json
  ```

### 5.4 生成目标相机轨迹

根据目的选择以下一种：

#### A. 稳定化：平滑原始轨迹（默认）

```bash
SOURCE_VIDEO="$SOURCE_VIDEO" RESOLUTION="$RESOLUTION" \
TRANSLATION_SIGMA=4 ROTATION_SIGMA=4 \
bash scripts/test_video/smooth.sh
```

输出 `recon_and_seg/cameras_gaussian_smooth.npz`。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TRANSLATION_SIGMA` | `4.0` | 平移的高斯平滑尺度，单位为帧；越大越平稳，但偏离原轨迹越多 |
| `ROTATION_SIGMA` | `4.0` | 旋转的高斯平滑尺度，单位为帧 |
| `SMOOTH_MODE` | `nearest` | 边界处理，传给 `scipy.ndimage.gaussian_filter1d` |
| `ANCHOR_FIRST` | `true` | 平滑后把第一帧重新对齐到原始第一帧位姿 |

脚本会打印平滑前后的轨迹统计。需要画图时见 [9.2 节](#92-相机轨迹可视化)。

#### B. 沿原始轨迹重拍

不需要执行平滑。后续渲染和推理都加 `USE_SMOOTHED_CAMERA=false`，结果目录变为 `render_<res>/` 和 `vista4d_<res>/`。

#### C. 自定义轨迹

用相机 UI（[10.3 节](#104-相机设计-ui)）加载 `recon_and_seg/` 并导出 `.npz`。渲染时指定 `CAM_PATH`，并用 `RENDER_FOLDER` 起一个独立的目录名：

```bash
CAM_PATH=cam_ui/exported_cameras/output_cameras.npz RENDER_FOLDER=render_384p_custom
```

推理时传入相同的 `RENDER_FOLDER`，或直接指定 `INPUT_FOLDER` / `OUTPUT_FOLDER`。

### 5.5 点云渲染

```bash
SOURCE_VIDEO="$SOURCE_VIDEO" RESOLUTION="$RESOLUTION" \
SAVE_VIS=true RENDER_ONLY_NECESSARY=true \
bash scripts/test_video/render.sh
```

脚本把重建结果反投影为 4D 点云：静态点在整个片段内持久化，动态点只保留在所属帧。然后在目标相机下逐帧渲染。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `USE_SMOOTHED_CAMERA` | `true` | `true` 使用 `cameras_gaussian_smooth.npz`，`false` 使用 `cameras.npz` |
| `CAM_PATH` | 自动 | 显式目标相机 |
| `RENDER_FOLDER` | `render_<res>_smooth` 或 `render_<res>` | 输出子目录名 |
| `RENDER_ONLY_NECESSARY` | `true` | 只渲染推理所需的输出。设为 `false` 时额外输出非持久化渲染、二次重投影等，用于对比实验 |
| `SAVE_VIS` | `true` | 保存可视化 |

输出（推理的输入）：

```text
results/single/<EXAMPLE>/render_384p_smooth/
├── video_src.mp4        源视角视频
├── video_pc.mp4         目标相机下的点云渲染
├── cameras_src.npz      源相机
├── cameras_tgt.npz      目标相机
├── depths_src/  depths_pc/
├── alpha_mask_src/  alpha_mask_pc/      有效像素（点云覆盖）mask
├── dynamic_mask_src/  dynamic_mask_pc/
├── static_mask_src/  static_mask_pc/
└── sky_mask_src/
```

推理前建议先看一遍 `video_pc.mp4`：空洞和拉伸是正常的，会由扩散模型补全；但如果人物被“冻结”在背景里，或者背景随人物一起移动，说明 `SEG_KEYWORDS` 需要调整。

### 5.6 Vista4D 推理

```bash
SOURCE_VIDEO="$SOURCE_VIDEO" RESOLUTION="$RESOLUTION" \
PROMPT="A realistic handheld smartphone video of a person walking along a city street, natural lighting, detailed surroundings." \
SEEDS=10027 \
bash scripts/test_video/inference.sh
```

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PROMPT` | 示例片段有内置 prompt，其他片段使用通用手持拍摄描述 | 文本条件。建议描述主体、动作、场景和光照 |
| `SEEDS` | `10027` | 一个或多个 seed，以空格分隔；多个 seed 在一次运行中按 batch 生成 |
| `NUM_INFERENCE_STEPS` | `50` | 去噪步数 |
| `CFG_SCALE` | `5.0` | classifier-free guidance 强度 |
| `SIGMA_SHIFT` | `5.0` | flow-matching scheduler 的时间偏移 |
| `VISTA4D_FOLDER` | 按分辨率选择 | checkpoint 目录 |
| `LOCAL_WAN_FOLDER` / `WAN_NAME` | `./checkpoints/wan` / `Wan2.1-T2V-14B` | 基础模型位置 |
| `INPUT_FOLDER` | `render_<res>_smooth` | 渲染条件目录 |
| `OUTPUT_FOLDER` | `vista4d_<res>_smooth` | 输出目录 |
| `EXTRA_ARGS` | 空 | 追加到 `inference.py` 的参数，例如 `--tile_vae --vram_preset balanced`（见[第 6 节](#6-推理参数与显存预设)） |
| `TRIAL_INFERENCE_TIME` | `false` | `true` 时只计时 3 次，**不保存视频** |
| `USE_USP` / `NUM_GPUS` | `false` / `8` | 多 GPU 序列并行，需要 xfuser |

输出：

```text
results/single/<EXAMPLE>/vista4d_384p_smooth/
├── video_seed=10027.mp4    生成结果
├── source.mp4              输入源视频（按模型分辨率裁剪）
├── point_cloud.mp4         输入点云渲染
├── point_cloud_masks.mp4   R = alpha，G = 动态，B = 两者并集
└── gifs/                   上述三个输入的 GIF 预览
```

多 GPU 推理（需要安装 xfuser，每张卡分担序列）：

```bash
USE_USP=true NUM_GPUS=4 SOURCE_VIDEO="$SOURCE_VIDEO" RESOLUTION="$RESOLUTION" \
bash scripts/test_video/inference.sh
```

### 5.7 参考耗时与显存

以下是本仓库在单块 NVIDIA GB10（统一内存）上的实测值，只用于估算排期，其他硬件会不同。

| 阶段 | 384p | 720p | 来源 |
| --- | --- | --- | --- |
| DA3 + SAM3（每个 49 帧片段，9 个提示词） | 约 195 s | 约 281 s | [report_usage.md](docs/report_usage.md) |
| DA3 峰值显存（allocated / reserved） | 18.0 / 24.7 GiB | 42.4 / 69.1 GiB | 同上。720p 使用 `DA3_PROCESS_RES=1280` |
| SAM3 峰值显存 | 6.2 / 6.5 GiB | 6.3 / 6.6 GiB | 同上 |
| Vista4D 推理（50 步，CFG 5，`full`） | 约 1 h / 片段，45.3 / 48.5 GiB | 去噪约 7 h（508 s/步） | 384p：baseline 报告；720p：`low_vram --no-fp8_compute` 计时运行，未保存视频 |

SAM3 耗时主要随提示词数量增加，与分辨率关系不大；DA3 对分辨率非常敏感。

### 5.8 完整命令汇总

```bash
conda activate vista4d-pgx
export RESOLUTION=384p
export CLIP_NAME=my_video_frames000000_000048_${RESOLUTION}49
export SOURCE_VIDEO=./media/single/${CLIP_NAME}.mp4

python -m scripts.preprocess.prepare_custom_single_video \
  --input ./data/my_video.mp4 --output_dir ./media/single --output_name "$CLIP_NAME" \
  --start_frame 0 --num_frames 49 --width 672 --height 384

SEG_KEYWORDS="person car" bash scripts/test_video/recon_and_seg.sh
bash scripts/test_video/smooth.sh
bash scripts/test_video/render.sh
PROMPT="..." bash scripts/test_video/inference.sh

ls results/single/${CLIP_NAME}/vista4d_${RESOLUTION}_smooth/
```

四个脚本都会先打印实际使用的变量（`Script kwargs:`），运行前请确认 `EXAMPLE`、输入和输出目录符合预期。

---

## 6. 推理参数与显存预设

`scripts/inference/inference.py` 的参数（通过 `EXTRA_ARGS` 追加，或直接调用）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--height` / `--width` / `--num_frames` | 384 / 672 / 49 | 必须与 checkpoint 一致 |
| `--num_inference_steps` | 50 | 去噪步数 |
| `--cfg_scale` | 5.0 | CFG 强度 |
| `--sigma_shift` | 5.0 | scheduler 时间偏移 |
| `--seed` | 10027 | 可传多个 |
| `--negative_prompt` | 内置 | 负向提示词 |
| `--tile_vae` | 关闭 | VAE 空间分块编解码，降低解码峰值显存，720p 建议开启 |
| `--cfg_merge` | 关闭 | 条件与无条件分支合并为一个 batch |
| `--use_usp` | 关闭 | 序列并行，需要 `torchrun` 和 xfuser |
| `--num_inference_time_trials N` | 无 | 只计时 N 次，不保存结果 |
| `--vram_preset` | `full` | 显存预设，见下表 |
| `--vram_limit` | 按预设 | 常驻 GPU 的模型块预算（GB），覆盖预设值 |
| `--fp8_compute` / `--no-fp8_compute` | 按预设 | 是否让 DiT 以 FP8 计算 |

显存预设（同样适用于 `inference_flowlong.py`）：

| 预设 | offload | DiT 计算精度 | 默认 `vram_limit` | 适用 |
| --- | --- | --- | --- | --- |
| `full` | 关闭，模型常驻 GPU | bf16 | 不限 | 显存充足，速度最快 |
| `balanced` | CPU↔GPU block-swap | bf16 | 16 GB | 独立显存较小的 GPU |
| `low_vram` | CPU↔GPU block-swap | FP8（仅 DiT；T5、VAE 保持 bf16） | 10 GB | 需要硬件支持 FP8 |

示例：

```bash
# 720p 片段（已按第 5 节在 RESOLUTION=720p 下完成前处理和渲染），分块 VAE + offload
RESOLUTION=720p SOURCE_VIDEO=./media/single/<片段名>_720p49.mp4 \
EXTRA_ARGS="--tile_vae --vram_preset balanced --vram_limit 16" \
bash scripts/test_video/inference.sh

# GB10 上使用 low_vram 时必须关闭 FP8
EXTRA_ARGS="--vram_preset low_vram --no-fp8_compute" bash scripts/test_video/inference.sh
```

预设只改变模型的驻留与计算精度，不改变 Vista4D 的条件或生成语义。`vram_limit` 是决定驻留多少模型块的预算，不是总显存硬上限，latent、条件和 VAE 解码仍会额外占用显存。

---

## 7. 长视频：FlowLong

Vista4D 一次只能生成 49 帧。长视频直接切成独立片段分别推理，会在窗口接缝处出现几何、条件和生成结果三个层面的不一致。本仓库依次解决：

```text
长视频
  → 两套规则窗口：baseline 49/5/44（分段重建）与 FlowLong 49/25/24（联合去噪）
  → 每个 baseline 窗口执行 DA3 + SAM3
  → Sim(3) 对齐到同一世界坐标 → 完整轨迹只做一次全局平滑
  → 对完整轨迹做一次 shared-static 渲染 → 无损切回两套窗口，并严格校验 overlap 逐像素一致
  → FlowLong：每个去噪时间步内预测所有窗口，在 latent overlap 上做 Tweedie matching，得到唯一的全局 x̂₀
  → 全局 latent 只解码一次，按 manifest 裁掉尾部 padding
```

统一入口（不加 `--execute` 时只打印计划，不执行）：

```bash
export VIDEO=./data/my_video.mp4
export RESOLUTION=384p
export SEED=10027
export SEG_KEYWORDS="person man woman hand phone bag backpack car stroller"
export PROMPT="A realistic handheld smartphone video ..."

# 1) 查看窗口数、padding 和将要执行的命令
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seed "$SEED" \
  --stages split recon stitch render inference --phases matching_only

# 2) 前处理：切窗、DA3/SAM3、拼接与平滑、shared-static 渲染
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seg-keywords "$SEG_KEYWORDS" \
  --stages split recon stitch render --execute

# 3) 50 步 FlowLong 推理（matching-only；另一个可选方案是 --phases t0.6）
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seed "$SEED" --prompt "$PROMPT" \
  --stages inference --phases matching_only --execute
```

结果：`results/flowlong_eval/<视频名>_<分辨率>_seed=<seed>/matching_only/video_seed=<seed>.mp4`。

`--phases` 可选 `baseline`（各窗口独立推理后 center-cut 合并）、`matching_only`、`t0.5`、`t0.6`、`t0.7`。**省略 `--phases` 会运行 baseline 和全部四个 FlowLong 方案**，长视频耗时很长，请显式指定。

分阶段命令、1 步 smoke 检查、参数、输出目录、tmux 后台运行、中断恢复与排错，见 **[READMEv2.md](READMEv2.md)**。

> `scripts/test_video/` 下的 `run_splits_*.sh`、`merge_splits.sh` 属于早期逐窗口流程。当前推荐流程由上述统一入口和 READMEv2 中的脚本组成，不再需要它们。

---

## 8. 相机轨迹可视化

```bash
R=results/full/<视频名>_stitched_384p/recon_and_seg
# 平滑前后轨迹；--up_from_recon 用深度图拟合地面，把视图旋转到重力方向
python scripts/postprocess/visualize_camera_trajectory.py \
  --smooth $R/cameras_gaussian_smooth.npz --up_from_recon $R \
  --manifest media/flowlong_splits/<视频名>_splits_manifest.json \
  --output results/diagnostics/camera_traj_384p_levelled.png
```

单片段可以直接传 `--smooth results/single/<EXAMPLE>/recon_and_seg/cameras_gaussian_smooth.npz --raw results/single/<EXAMPLE>/recon_and_seg/cameras.npz`。

## 9. 原版 Vista4D 功能

以下功能沿用原版实现，示例脚本的重建方法默认改为 DA3（`RECON_METHOD=pi3` 可切回）。更多原理说明见[上游项目主页](https://eyeline-labs.github.io/Vista4D)。

### 9.1 4D 场景重组（点云编辑）

对 4D 点云中的主体做平移、旋转、缩放、删除、复制，或插入其他场景的主体，再用编辑后的点云渲染条件推理。示例：`couple-hug_duplicate-car`、`couple-hug_couple-newspaper`、`funeral-procession_remove-priest`、`funeral-procession_rhino`、`hike_enlarge-backpack`、`hike_cow`、`swing_shrink-person`、`swing_couple-walk`。

```bash
EXAMPLE=hike RECON_METHOD=da3 bash scripts/preprocess/example_recon_and_seg_edit.sh   # 主场景与插入场景
EXAMPLE=hike_cow RESOLUTION=720p bash scripts/preprocess/example_render_edit.sh       # 应用 media/edit/hike_cow.json
EXAMPLE=hike_cow RESOLUTION=720p bash scripts/inference/example_inference_edit.sh     # → results/edit/hike_cow/vista4d_720p/
```

编辑 JSON 由相机 UI 导出。每条编辑包含：

- `target.kind`：`existing` / `duplicate` / `insert`；
- `ops`：`translate` / `rotate` / `scale` / `remove`；
- `scope`：`global` 或 `frame`；
- 可选的 `mask_expansion` 和 `centroid_threshold`。

编辑的解析与应用在 `utils/point_cloud/edit.py`。

### 9.2 动态场景扩展（DSE）

把源视频与另外拍摄的场景画面联合重建为同一个 4D 点云，减少扩散模型需要凭空补全的区域。示例：`conference-punch`、`conference-study`、`hall-cartwheel`、`lounge-cup`、`lounge-drink`、`plaza-point`、`room-lift`、`room-walk`。

```bash
EXAMPLE=lounge-cup RECON_METHOD=da3 bash scripts/preprocess/example_recon_and_seg_dse.sh
EXAMPLE=lounge-cup RESOLUTION=720p bash scripts/preprocess/example_render_dse.sh      # DSE_FRAME_INTERVAL 默认 4
EXAMPLE=lounge-cup RESOLUTION=720p bash scripts/inference/example_inference_dse.sh    # → results/dse/lounge-cup/vista4d_720p/
```

### 9.3 相机设计 UI

基于 [Viser](https://viser.studio)，用于加载 `recon_and_seg/` 结果、浏览 4D 点云、设置关键帧相机并导出插值轨迹 `.npz`，也支持点云编辑。需要 Node.js：

```bash
conda install conda-forge::nodejs    # 或使用系统 / nvm 安装的 Node.js
bash cam_ui/startup.sh               # Viser 9997，FastAPI 9998，React 前端 9999
```

远程服务器上用 `ssh -L 9999:localhost:9999 -L 9997:localhost:9997 <user>@<host>` 转发端口后，在本地打开 `http://localhost:9999`。

操作步骤：

1. 在 *Folder path* 中填入 `results/single/<EXAMPLE>/recon_and_seg/`，点击 **Load**。
2. 用 WASD + Q/E 移动视角，鼠标拖动旋转（不支持 roll）。
3. 选定帧后点击 **Capture current view** 记录关键帧，并设置 zoom。
4. 用 **Auto-follow camera** 预览轨迹。
5. 点击 **Export cameras**，默认写到 `cam_ui/exported_cameras/output_cameras.npz`。

导出的文件用作 [5.4 节](#54-生成目标相机轨迹) 方案 C 的 `CAM_PATH`。

---

## 10. 目录结构

```text
vista4d/
├── diffsynth/                   DiffSynth-Studio：Wan / Vista4D pipeline
│   └── pipelines/flowlong.py    FlowLong 联合去噪与 overlap matching
├── depth_anything_3/  pi3/  sam3/   内置的重建与分割代码
├── scripts/
│   ├── preprocess/              片段准备、重建分割、平滑、渲染、切窗、拼接
│   ├── inference/               inference.py（单片段）、inference_flowlong.py、inference_split_baseline.py
│   ├── postprocess/             视频拼接与相机轨迹可视化
│   └── test_video/              自定义视频流水线脚本与统一入口 run_video_experiment.py
├── utils/                       分辨率 profile、manifest、显存预设、媒体读写、点云
├── cam_ui/                      相机设计 UI
├── tests/                       CPU / mock 单元测试
├── docs/                        设计文档与实测报告
├── media/
│   ├── single/  edit/  dse/     原版示例视频与相机；自定义 49 帧片段也放在 single/
│   ├── splits/                  baseline 窗口（384p；720p 在 splits/720p/）
│   └── flowlong_splits/         FlowLong 窗口（384p；720p 在 flowlong_splits/720p/）
├── checkpoints/                 模型权重（不入库）
└── results/                     所有输出（不入库）
    ├── single/<片段名>/          recon_and_seg/、render_*/、vista4d_*/
    ├── full/<视频名>_stitched_<res>/     长视频全局重建与完整渲染
    ├── shared_static_single/  flowlong_single/   长视频两套窗口的切片条件
    └── flowlong_eval/           长视频生成结果（历史目录名）
```

---

## 11. 测试

单元测试覆盖窗口几何、padding、manifest、FlowLong matching、配置与 checkpoint 等，在 CPU 上运行，不加载模型：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v

# shell 脚本语法检查
for f in scripts/test_video/*.sh; do bash -n "$f" || echo "syntax error: $f"; done
```

测试不能替代真实模型 smoke 运行。

---

## 12. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `Cannot detect the model type. File: []` | 模型路径下没有匹配的文件。按 [3.3 节](#33-检查目录) 检查 `checkpoints/wan/Wan2.1-T2V-14B/` 和 `checkpoints/vista4d/<分辨率目录>/` |
| `resolution=384p requires width x height 672x384` | 输入尺寸、`RESOLUTION` 与 checkpoint 不一致。所有步骤使用同一个 `RESOLUTION` |
| `Source video must have at least args.num_frames=49 frames` | 输入片段不足 49 帧。用 [5.2 节](#52-截取-49-帧片段) 重新截取 |
| 找不到 `./checkpoints/sam3/sam3.pt` 或下载 SAM3 时 401 / 403 | 未申请 SAM3 访问权限或未 `hf auth login` |
| DA3 在 720p 下 OOM | 降低 `DA3_PROCESS_RES`（如 1280 → 896） |
| Vista4D 推理 OOM | 加 `--tile_vae`，或使用 `--vram_preset balanced`（[第 6 节](#6-推理参数与显存预设)） |
| `CUBLAS_STATUS_NOT_SUPPORTED` | 当前 GPU 不支持 FP8 计算（如 GB10）。加 `--no-fp8_compute` |
| 推理很慢 | 确认 flash-attn 是否可用（[2.4 节](#24-检查环境)）；没有时使用 SDPA，属于预期 |
| `--use_usp` 报 `No module named 'xfuser'` | USP 需要安装 xfuser；单卡请保持 `USE_USP=false` |
| 示例相机位置明显不对 | 示例相机基于 Pi3X 重建设计，见[第 4 节](#4-快速验证官方示例片段)的说明 |
| 运动的人或物被“冻结”在背景中 | `SEG_KEYWORDS` 漏掉了该物体，补充提示词后重新运行 5.3–5.6 |
| 384p 与 720p 结果互相覆盖 | 片段文件名没有 `_<res>49` 后缀，导致 `EXAMPLE` 相同。按 [5.1 节](#51-流程概览) 的命名约定重命名，或显式设置 `EXAMPLE` |
| `nvidia-smi` 显示 `Not Supported` | 统一内存设备的正常现象，以报告 JSON 中的 PyTorch 峰值统计为准 |


---

## 致谢与引用

本仓库基于 Eyeline Labs 的 [Vista4D](https://github.com/Eyeline-Labs/Vista4D) 开发，并使用了 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)、[Wan 2.1](https://github.com/Wan-Video/Wan2.1)、[Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)、[Pi3(X)](https://github.com/yyfz/Pi3)、[Segment Anything 3](https://github.com/facebookresearch/sam3)、[Viser](https://viser.studio) 等项目。长视频联合去噪参考了 FlowLong 的 timestep 同步思路（见 [docs/FlowLong_Plan.md](docs/FlowLong_Plan.md)）。

使用 Vista4D 时请引用原论文：

```bibtex
@InProceedings{lin2026vista4d,
    author    = {Lin, {Kuan Heng} and Liu, Zhizheng and Salamanca, Pablo and Kant, Yash and Burgert, Ryan and Xu, Yuancheng and Namekata, Koichi and Zhao, Yiwei and Zhou, Bolei and Goldblum, Micah and Debevec, Paul and Yu, Ning},
    title     = {{Vista4D}: Video Reshooting with 4D Point Clouds},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {32671--32682}
}
```
