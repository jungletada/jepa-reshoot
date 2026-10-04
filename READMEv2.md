# Vista4D × FlowLong：长视频重新运镜使用指南

## 推荐：使用 YAML 执行

参数集中在 [configs/example.yaml](configs/example.yaml)，每个视频维护一份。设置 `video.mode: full`，使用 `flowlong.baseline`、`matching_only` 和 `stochastic_thresholds` 选择变体；例如 `[0.5, 0.6]` 不会额外执行 0.7。完整规则见 [配置指南](docs/video_configuration.md)。

```bash
# 环境激活后先预览；未加 --execute 不会启动任何阶段。
python -m scripts.test_video.run_video_experiment --config configs/1776148878076.yaml

# 按顺序完成两套切片、DA3/SAM3、全局重建/平滑及共享渲染。
python -m scripts.test_video.run_video_experiment --config configs/1776148878076.yaml \
  --gpu 0 --stages split recon stitch render --execute

# 运行配置选中的生成变体。
python -m scripts.test_video.run_video_experiment --config configs/1776148878076.yaml \
  --gpu 0 --stages inference --execute
```

将 GPU 0 换成分配给自己的卡。YAML 模式允许配置推理步数/CFG/sigma shift，默认仍为 50/5/5；窗口结构仍限定 49/25/24。YAML 产物位于 `results/configured/<TARGET>/<RESOLUTION>/<run_name>/`；下文旧 Bash 示例及旧路径仍可独立使用，但不要与 YAML 产物混用。改变已执行配置时使用新 `run_name`，不会默认覆盖旧结果。

本文说明如何在本仓库中，用固定 49-frame 的 Vista4D 模型处理长视频：构建统一几何与渲染条件，在每个去噪时间步同步所有窗口，最后生成完整视频。以下原理和限制以本仓库实现为准。

- 模型安装与原始 Vista4D 功能：见 [README.md](README.md)。
- 实现设计：见 [docs/FlowLong_Plan.md](docs/FlowLong_Plan.md)。

## 1. FlowLong 原理

### 1.1 独立切片与联合去噪的区别

独立切片推理让每个窗口分别完成去噪。即使输入有 overlap，两个窗口仍可能对同一时刻生成不同的颜色、纹理或人物细节；事后 center-cut 拼接不能约束生成过程。

FlowLong 不改变 Vista4D 的 49-frame 模型输入，也不需要重新训练模型，而是在采样过程中维护共享的全局 latent 时间轴：

```text
输入视频
  → baseline / FlowLong 两套规则窗口
  → baseline 窗口分别执行 DA3 + SAM3
  → reconstruction 全局对齐、拼接与相机轨迹平滑
  → 一次生成完整 shared-static render
  → 按两套 manifest 切出共享条件
  → 每个 timestep 内预测所有窗口，再匹配 overlap
  → 解码全局 latent，裁掉 padding，输出完整视频
```

“联合”指时间步同步，不要求所有窗口同时进入一次 forward。`microbatch_size=1` 时可以逐窗口计算，但必须收集同一 timestep 的所有预测、完成匹配后，才能推进下一步。

baseline 视频是可选的对照输出。不过本仓库前处理仍使用 baseline 窗口做分段 reconstruction，因此只生成 FlowLong 视频也需要准备两套窗口。

### 1.2 窗口与尾部 padding

当前完整流程固定使用以下布局，索引从 0 开始：

| 用途 | 每窗帧数 | overlap | stride | 起点 |
|---|---:|---:|---:|---|
| baseline reconstruction / 独立推理 | 49 | 5 | 44 | 0, 44, 88, … |
| FlowLong 联合推理 | 49 | 25 | 24 | 0, 24, 48, … |

对 `N` 帧视频和步长 `D`：

```text
窗口数 M       = 1 + ceil(max(N - 49, 0) / D)
窗口起点       = 0, D, 2D, ..., (M - 1)D
补齐后的总长度 = (M - 1)D + 49
尾部 padding  = 补齐后的总长度 - N
```

尾窗保持在规则网格上，不向左移动来凑足 49 帧。不足部分重复最后一个真实帧；RGB、depth、mask 和 camera 同步补齐，输出时依据 manifest 裁掉 padding。两种 stride 都是 4 的倍数，满足时间压缩对齐要求。

Wan/Vista4D 的时间压缩因子为 4，因此 FlowLong 的 latent 布局为：

```text
latent window  F = (49 - 1) / 4 + 1 = 13
latent stride  S = 24 / 4 = 6
latent overlap O = F - S = 7
全局 latent 长度 = 13 + (M - 1) × 6
```

不要直接把 overlap 改成任意值：本实现校验固定的 49/25/24 布局，并要求 latent overlap ≥ latent stride。完整 FlowLong 入口要求至少两个窗口；单窗口视频应使用主 README 中的 Vista4D 入口。

### 1.3 每个 timestep 内的匹配

用 `t` 表示当前噪声水平、`s` 表示下一步噪声水平，满足 `0 ≤ s ≤ t ≤ 1`：

1. 取得各窗口状态 `x_t`，用 Vista4D 和各自条件预测 CFG velocity `v_t`。
2. 在 float32 中计算预测干净状态：`x̂_0 = x_t - t × v_t`。
3. 对相邻窗口的 7 个 latent overlap 位置做线性 Tweedie matching，构成唯一的全局 `x̂_0`。
4. 更新全局状态到 `x_s`，再切回窗口，进入下一 timestep。
5. 去噪结束后解码全局 latent，而不是逐窗解码后拼接 MP4。

匹配权重为：

```text
λ[i]     = i / 6，i = 0..6
blend[i] = (1 - λ[i]) × left_x̂_0[i] + λ[i] × right_x̂_0[i]
```

代码按相邻窗口对从左向右写入；三窗共同覆盖的位置使用最右窗口对的结果，不是三窗平均。随后所有窗口从同一个全局状态切片，因此匹配后的 overlap 应完全一致。

相关代码：[diffsynth/pipelines/flowlong.py](diffsynth/pipelines/flowlong.py)。

### 1.4 matching-only 与阈值 `t*`

本实现提供两种状态更新：

```text
随机更新：x_s = (1 - s) × x̂_0 + s × ε，ε 为新采样的全局高斯噪声

确定性更新：x̂_1 = [x_t - (1 - t) × x̂_0] / t
            x_s = (1 - s) × x̂_0 + s × x̂_1
```

最后一步 `s=0` 直接返回匹配后的 `x̂_0`。

- **matching-only**：每一步仍做 overlap matching，但关闭随机更新。
- **stochastic FlowLong**：在 `t ≥ t*` 时随机更新，低噪声阶段确定性更新。

`t*` 是 scheduler 的噪声阈值，不是步数比例。阈值越低，通常随机更新阶段越长；实际随机步数也受 steps 和 `sigma_shift` 影响，应查看输出 JSON 的逐步记录。

matching 约束共享状态，但不保证消除所有视觉问题。错误的深度、mask、相机对齐或遮挡条件仍会导致伪影；随机更新也不保证对所有视频都优于 matching-only。

### 1.5 为什么要共享 reconstruction 和 render

两个窗口中的同一源帧，必须对应同一全局坐标、目标相机和渲染结果。分别重建、分别平滑或独立建立静态点云，会使相同 overlap 帧的条件发生变化。

因此流程先拼接 reconstruction，再平滑完整轨迹，并生成一次 full shared-static render。两套窗口条件均从它切片得到；推理加载时检查窗口尺寸、padding 及来源信息。

默认目标轨迹是全局平滑后的相机轨迹。自定义运镜需要准备匹配完整时间轴的目标轨迹并重新生成共享渲染条件，不能只修改 prompt，也不能只替换某个窗口的 camera。

## 2. 环境、模型与输入

先按 [README.md](README.md) 安装 Vista4D、DA3、SAM3 和依赖，确保 PyTorch/CUDA 与 GPU 兼容。以下命令均在仓库根目录的 Bash 中执行：

```bash
# 激活安装了本仓库依赖的环境；已有环境也可能名为 vista4d-pgx
conda activate vista4d

command -v python
command -v ffmpeg
command -v ffprobe
nvidia-smi
df -h .
```

SAM3 需要可用的模型权重及相应访问权限，准备方法见主 README 的 preprocessing 部分。下文用 `jq` 读取 JSON，用 `tmux` 挂后台，两者不是 sampler 本身的依赖。

### 2.1 分辨率与模型路径

| `RESOLUTION` | 输出尺寸 | Vista4D checkpoint 目录 |
|---|---:|---|
| `384p` | 672×384 | `checkpoints/vista4d/384p49_step=30000/` |
| `720p` | 1280×720 | `checkpoints/vista4d/720p49_step=3000/` |

原始 Vista4D 目录需要 `dit.pth` 和 `config.yaml`；也支持无损 safetensors 分片目录（索引、全部分片及 `config.yaml`），见 [分片指南](docs/checkpoint_sharding.md)。统一入口用 `--vista4d-folder` 选择，底层脚本用 `VISTA4D_FOLDER`。共享基础权重默认位于
`checkpoints/wan/Wan2.1-T2V-14B/`，DA3 默认位于
`checkpoints/DA3NESTED-GIANT-LARGE-1.1/`。下载后的目录名应与脚本配置一致。

720p 必须使用对应 checkpoint，不能只放大 384p 的 `HEIGHT/WIDTH`。

### 2.2 设置输入变量

将文件名改为自己的视频，同一终端后续命令复用这些变量：

```bash
export VIDEO=./data/my_video.mp4
export TARGET="$(basename "${VIDEO%.*}")"
export RESOLUTION=384p              # 或 720p
export SEED=10027
export SEG_KEYWORDS="person man woman hand phone bag backpack car stroller"
export PROMPT="A realistic handheld smartphone video of people in an everyday scene, with natural body motion, realistic lighting, stable camera motion, and detailed surroundings."

test -f "$VIDEO"
```

`TARGET` 是文件名去掉扩展名。为兼容统一入口，建议只使用字母、数字、下划线和连字符。不同视频不要使用相同 stem；原地替换同名视频后，也不能直接复用旧条件。

视频长度由输入和 manifest 决定，不固定为 310 帧。流程按帧索引和平均 FPS 建立均匀时间轴，不保留逐帧原始 PTS，也不传播音频；需要保留变帧率时间戳或音画同步时，应另行处理。

## 3. 快速开始：统一入口

统一入口 [run_video_experiment.py](scripts/test_video/run_video_experiment.py) 管理视频身份、分辨率、阶段顺序和路径，无需为每个视频复制 launcher。

### 3.1 只查看计划

```bash
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seed "$SEED" \
  --stages split recon stitch render inference \
  --phases matching_only
```

不加 `--execute` 时，只读取视频并输出 JSON 计划，不创建实验目录、加载模型或启动任务。计划包含帧数、窗口数、padding、目标路径和将调用的命令。

### 3.2 执行前处理

```bash
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" \
  --seg-keywords "$SEG_KEYWORDS" \
  --stages split recon stitch render --execute
```

依次完成两套切片、DA3/SAM3、reconstruction 拼接、全局轨迹平滑、shared-static rendering。之后可按第 4 节运行 smoke。

### 3.3 生成完整视频

前处理和 smoke 完成后，选择一个 50-step 方案：

```bash
# 仅生成 matching-only
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seed "$SEED" \
  --prompt "$PROMPT" \
  --stages inference --phases matching_only --execute

# 或生成启用随机更新的 t*=0.6 方案
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seed "$SEED" \
  --prompt "$PROMPT" \
  --stages inference --phases t0.6 --execute
```

正式参数为 50 steps、CFG=5、sigma_shift=5。两条命令是不同输出方案，不要求都运行。结果位置见第 6 节。

### 3.4 统一入口参数

| 选项 | 含义 |
|---|---|
| `--video` | 输入视频路径，必填 |
| `--resolution` | `384p` 或 `720p` |
| `--seed` | 正式推理 seed，默认 10027 |
| `--vista4d-folder` | 可选的 checkpoint 目录，支持原 `.pth` 布局或 safetensors 分片；分辨率必须匹配 |
| `--seg-keywords` | SAM3 分割关键词，按视频内容调整 |
| `--prompt` | 正式推理文本条件 |
| `--stages` | 按顺序选择 `split recon stitch render inference` |
| `--phases` | `baseline matching_only t0.5 t0.6 t0.7` |
| `--execute` | 真正执行；省略时只输出计划 |

省略 `--stages` 会选择完整前处理和 inference；省略 `--phases` 会选择 baseline 和全部四个 FlowLong variant，**不是只运行一个候选**。不需要完整对照时应显式指定 phase。

统一入口清除历史 shell 脚本配置覆盖值，保留 CUDA、conda、缓存等运行环境。使用它时通过 CLI 设置 prompt、seed 等；要改底层参数，使用第 4、5 节的 Bash 接口。

当前 GPU 占用拒绝逻辑已禁用，`--gpu` 仍负责选择设备。已有切片或 reconstruction 会触发保护；检查后省略相应阶段以复用。入口不会自动启动 tmux，也不支持整个流水线任意位置无条件续跑。

## 4. 分阶段操作与校验

本节用于控制参数、定位错误或复用上游产物。它与第 3 节是两种操作方式，**已通过统一入口完成的阶段不要重复执行**。

### 4.1 Stage 1：两套规则窗口

复用第 2.2 节变量，设置分辨率相关路径。切换分辨率时重新执行此配置块：

```bash
case "$RESOLUTION" in
  384p)
    export WIDTH=672 HEIGHT=384
    export BASELINE_SPLITS_DIR=./media/splits
    export FLOWLONG_SPLITS_DIR=./media/flowlong_splits
    export VISTA4D_FOLDER=./checkpoints/vista4d/384p49_step=30000
    ;;
  720p)
    export WIDTH=1280 HEIGHT=720
    export BASELINE_SPLITS_DIR=./media/splits/720p
    export FLOWLONG_SPLITS_DIR=./media/flowlong_splits/720p
    export VISTA4D_FOLDER=./checkpoints/vista4d/720p49_step=3000
    ;;
  *) echo "Unsupported RESOLUTION=$RESOLUTION" >&2; exit 2 ;;
esac

export FULL_SEQUENCE_ROOT=./results/full/${TARGET}_stitched_${RESOLUTION}
export BASELINE_CONDITION_ROOT=./results/shared_static_single
export FLOWLONG_RESULT_ROOT=./results/flowlong_single

python -m utils.vista4d_checkpoint resolve "$VISTA4D_FOLDER"
test -f "$VISTA4D_FOLDER/config.yaml"
test -d ./checkpoints/wan/Wan2.1-T2V-14B
```

仅在尚未生成切片时运行：

```bash
OUTPUT_DIR="$BASELINE_SPLITS_DIR" CLIP_FRAMES=49 OVERLAP=5 \
START_FRAME=0 MAX_FRAMES= TEMPORAL_ALIGNMENT=4 INCLUDE_TAIL=true \
bash scripts/test_video/split_video.sh "$VIDEO"

OUTPUT_DIR="$FLOWLONG_SPLITS_DIR" CLIP_FRAMES=49 OVERLAP=25 \
START_FRAME=0 MAX_FRAMES= TEMPORAL_ALIGNMENT=4 INCLUDE_TAIL=true \
bash scripts/test_video/split_video.sh "$VIDEO"
```

`MAX_FRAMES=` 表示不截短源视频。每套目录包含 `<TARGET>_splits_manifest.json`、CSV 和实际编码为 49 帧的 MP4。后续以 JSON 的有效区间及 padding 元数据为准，不通过文件名猜测长度。

检查窗口布局和两套源范围：

```bash
python - <<'PY'
import json, os
from pathlib import Path
from utils.split_manifest import validate_flowlong_window_manifest
from utils.resolution import validate_manifest_resolution

manifests = []
for key, overlap in [('BASELINE_SPLITS_DIR', 5), ('FLOWLONG_SPLITS_DIR', 25)]:
    path = Path(os.environ[key]) / (os.environ['TARGET'] + '_splits_manifest.json')
    m = json.loads(path.read_text())
    clips = validate_flowlong_window_manifest(m, overlap=overlap)
    validate_manifest_resolution(m, resolution=os.environ['RESOLUTION'],
                                width=int(os.environ['WIDTH']), height=int(os.environ['HEIGHT']))
    print(key, 'starts=', [c['start_frame'] for c in clips], 'tail_padding=', clips[-1]['pad_right'])
    manifests.append(m)
for key in ('input_path', 'start_frame', 'end_exclusive', 'fps'):
    assert manifests[0][key] == manifests[1][key], key
print('Window geometry: PASS')
PY
```

### 4.2 Stage 2：DA3 / SAM3、拼接与共享渲染

先对 baseline 窗口执行 DA3 + SAM3。统一入口按 JSON 遍历窗口并逐个校验：

```bash
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" \
  --seg-keywords "$SEG_KEYWORDS" --stages recon --execute
```

输出位于 `results/single/<split-stem>/recon_and_seg/`，包含源视频、depth、dynamic/sky mask 和 camera。不要把很长的视频直接送入一次 DA3 调用。

若要调整 DA3 处理分辨率，可对明确选定的窗口使用底层脚本。先将示例路径换成实际窗口；此接口没有统一入口的已有目录保护：

```bash
SOURCE_VIDEO=./media/splits/your_split.mp4 \
RECON_METHOD=da3 DA3_PROCESS_RES=672 SAVE_VIS=false \
bash scripts/test_video/recon_and_seg.sh
```

统一入口的 `DA3_PROCESS_RES` 默认取目标宽度，即 672 或 1280。它影响重建成本和质量，同一组对比应使用一致设置。

拼接 reconstruction，平滑完整轨迹，再重新切片：

```bash
SPLITS_DIR="$BASELINE_SPLITS_DIR" \
INPUT_RESULT_ROOT=./results/single \
OUTPUT_RESULT_ROOT=./results/stitched_single \
FULL_RESULT_BASE=./results/full \
TRANSLATION_SIGMA=8 ROTATION_SIGMA=10 \
FORCE_STITCH=false OVERWRITE_SPLITS=false \
bash scripts/test_video/stitch_splits_smooth_and_slice.sh "$TARGET"
```

`TRANSLATION_SIGMA` / `ROTATION_SIGMA` 控制相机轨迹的高斯平滑尺度（以帧序列为单位），与扩散采样的 `SIGMA_SHIFT` 无关。检查
`$FULL_SEQUENCE_ROOT/recon_and_seg/stitch_report.json` 的尺度及对齐误差，并查看轨迹；不要将明显错误的几何条件送入采样。

生成一次 full shared-static render，并切出两套条件：

```bash
STATIC_FRAME_STRIDE=4 RENDER_CHUNK_SIZE=4 \
OVERWRITE_FULL_RENDER=false \
OVERWRITE_BASELINE_SPLITS=false OVERWRITE_FLOWLONG_SPLITS=false \
SAVE_VISUALS=false \
bash scripts/test_video/prepare_flowlong_ab_conditions.sh "$TARGET"
```

`STATIC_FRAME_STRIDE` 控制建立静态点云时的时间采样，不改变 FlowLong 的窗口 stride；`RENDER_CHUNK_SIZE` 控制渲染批量，不改变输出帧数。

完整条件位于 `$FULL_SEQUENCE_ROOT/render_${RESOLUTION}_smooth_shared_static/`，包括
`video_src.mp4`、`video_pc.mp4`、depth、mask、`cameras_src.npz`、`cameras_tgt.npz` 和 `shared_static_render.json`。

逐窗口条件位于：

```text
results/shared_static_single/<baseline-split>/render_<resolution>_smooth/
results/flowlong_single/<flowlong-split>/render_<resolution>_smooth/
```

### 4.3 Stage 3：1-step smoke

smoke 检查模型读取、完整输出长度和 overlap matching，不用于判断画质。先设置第 4.1 节的路径变量，再运行：

```bash
export SMOKE=./results/flowlong_smoke/${TARGET}_${RESOLUTION}_1step_cfg1

OUTPUT_FOLDER="$SMOKE" \
NUM_INFERENCE_STEPS=1 CFG_SCALE=1.0 SIGMA_SHIFT=5.0 \
FLOWLONG_STOCHASTIC_THRESHOLD=0.6 FLOWLONG_DISABLE_STOCHASTIC=false \
FLOWLONG_MICROBATCH_SIZE=1 SEEDS="$SEED" \
TILE_VAE=true USE_USP=false CFG_MERGE=false FORCE=false \
bash scripts/test_video/run_flowlong_inference.sh "$TARGET"
```

检查视频与报告：

```bash
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames \
  -of default=noprint_wrappers=1 "$SMOKE/video_seed=${SEED}.mp4"

jq '{resolution, width, height, output_frames, geometry, wall_seconds,
     decoded: .pipeline.decoded_frames_before_trim,
     trimmed: .pipeline.trimmed_frames,
     peak_allocated_gib: .pipeline.cuda_peak_allocated_gib,
     peak_reserved_gib: .pipeline.cuda_peak_reserved_gib,
     steps: .pipeline.steps}' \
  "$SMOKE/flowlong_report_seed=${SEED}.json"
```

应满足：输出尺寸匹配 profile；实际帧数等于 manifest 的
`end_exclusive - start_frame`；解码长度减去 trim 等于有效输出长度；
`.pipeline.steps[].overlap_after_max_abs` 全为 `0.0`，误差和耗时为有限数值。

CFG=1 的 smoke 与 CFG=5 正式推理计算量不同。估计正式单步时间时，使用独立目录重新做 CFG=5 的 smoke；不要直接将 CFG=1 总耗时乘以 50。模型加载、条件编码和最终 VAE 解码也是额外开销。

### 4.4 Stage 4：50-step 正式推理

可使用第 3.3 节统一入口，或直接调用 Bash wrapper：

```bash
# 仅运行 matching-only
PHASES=matching_only SEED="$SEED" PROMPT="$PROMPT" FORCE=false \
bash scripts/test_video/run_flowlong_stage4.sh "$TARGET"

# 或仅运行 t*=0.6
PHASES=t0.6 SEED="$SEED" PROMPT="$PROMPT" FORCE=false \
bash scripts/test_video/run_flowlong_stage4.sh "$TARGET"
```

Stage 4 固定 50 steps、CFG=5、sigma_shift=5。`baseline` 独立推理各窗口后 center-cut 合并；其他 phase 使用联合 sampler。

## 5. 参数与自定义运行

### 5.1 底层推理参数

以下环境变量适用于 `run_flowlong_inference.sh`，不是统一 CLI 的任意环境覆盖接口：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `NUM_INFERENCE_STEPS` | 50 | 去噪步数 |
| `CFG_SCALE` | 5.0 | 文本条件引导强度 |
| `SIGMA_SHIFT` | 5.0 | 调整 scheduler 噪声时间分布，不是相机平滑参数 |
| `FLOWLONG_STOCHASTIC_THRESHOLD` | 0.6 | 随机更新阈值 `t*` |
| `FLOWLONG_DISABLE_STOCHASTIC` | false | true 表示 matching-only |
| `FLOWLONG_MICROBATCH_SIZE` | 1 | 一次预测的窗口数；增大可能增加显存，不保证更快 |
| `SEEDS` | 10027 | 一个或多个 seed，以空格分隔 |
| `TILE_VAE` | true | 启用 VAE 空间分块 |
| `USE_USP` / `CFG_MERGE` | false / false | 当前 FlowLong 接口要求保持 false |
| `OUTPUT_FOLDER` | 自动派生 | 自定义结果目录 |
| `FORCE` | false | true 允许覆盖，使用前确认目标 |

例如，使用独立路径运行 20 steps、`t*=0.7`：

```bash
OUTPUT_FOLDER="./results/flowlong/${TARGET}_${RESOLUTION}_steps20_t0p7" \
NUM_INFERENCE_STEPS=20 CFG_SCALE=5 SIGMA_SHIFT=5 \
FLOWLONG_STOCHASTIC_THRESHOLD=0.7 FLOWLONG_DISABLE_STOCHASTIC=false \
FLOWLONG_MICROBATCH_SIZE=1 SEEDS="$SEED" FORCE=false \
bash scripts/test_video/run_flowlong_inference.sh "$TARGET"
```

更换参数应使用新目录；底层 wrapper 发现同 seed 的 MP4 和 JSON 都存在就跳过，并不据此确认它们符合新参数。

### 5.2 显存选项

优先保持 `FLOWLONG_MICROBATCH_SIZE=1` 和 `TILE_VAE=true`。需要模型 offload 时，用 `EXTRA_ARGS` 传递 preset：

```bash
OUTPUT_FOLDER="./results/flowlong/${TARGET}_${RESOLUTION}_balanced" \
SEEDS="$SEED" \
EXTRA_ARGS="--vram_preset balanced --vram_limit 16" \
bash scripts/test_video/run_flowlong_inference.sh "$TARGET"
```

`full` 保持模型常驻 GPU；`balanced` 使用 bf16 CPU/GPU block-swap；`low_vram` 额外启用 DiT FP8。FP8 需要硬件和软件栈支持，相关算子不兼容时可用 `--no-fp8_compute` 禁用。

预算用于决定驻留模型块，不是总显存硬上限；全局 latent、条件及解码也占用内存。长视频和高分辨率应先做 smoke。

## 6. 输出位置、后台运行与恢复

### 6.1 目录约定

`<target>` 是视频 stem，`<resolution>` 为 `384p` 或 `720p`：

```text
results/
├── single/<split-stem>/recon_and_seg/              # 分段 DA3/SAM3
├── full/<target>_stitched_<resolution>/            # 全局 reconstruction 和 render
├── shared_static_single/<baseline-split>/         # baseline 窗口条件
├── flowlong_single/<flowlong-split>/               # FlowLong 窗口条件
├── flowlong_smoke/<自定义 smoke 目录>/
└── flowlong_eval/<target>_<resolution>_seed=<seed>/
    ├── experiment_spec.json
    ├── baseline/
    │   ├── video_seed=<seed>_center_cut.mp4
    │   ├── baseline_report_seed=<seed>.json
    │   └── clips/
    ├── matching_only/
    ├── flowlong_t0p5/
    ├── flowlong_t0p6/
    ├── flowlong_t0p7/
    └── metrics/
```

各 FlowLong variant 的最终视频为 `video_seed=<seed>.mp4`，报告为
`flowlong_report_seed=<seed>.json`。只有被执行的组才会产生结果。

Stage 4 日志位于 `logs/flowlong_stage4/<target>_<resolution>_seed=<seed>_<run-id>/`。
直接调用 `run_flowlong_inference.sh` 则写到 `OUTPUT_FOLDER`；未指定时使用
`results/flowlong/<target>_vista4d_<resolution>_smooth/`，不要与 Stage 4 目录混淆。

### 6.2 使用 tmux

创建会话后，在里面激活环境、进入仓库并重新设置第 2.2 节变量。tmux 不保证继承客户端刚设置的所有 shell 变量。

```bash
tmux new -s flowlong-run
```

在新会话中执行所需命令。例如，前处理和 smoke 校验通过后：

```bash
# 在此终端完成环境激活与第 2.2 节变量配置后运行
set -o pipefail
mkdir -p logs/manual
python -u -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seed "$SEED" \
  --prompt "$PROMPT" --stages inference --phases matching_only --execute \
  2>&1 | tee "logs/manual/${TARGET}_${RESOLUTION}_matching_$(date +%Y%m%d_%H%M%S).log"
```

按 `Ctrl-b` 后按 `d` 分离；用 `tmux attach -t flowlong-run` 返回。tmux 保持进程运行，但不提供断电恢复或自动重试。

### 6.3 跳过、恢复与覆盖

- **baseline**：Stage 4 按窗口校验 sidecar，可复用合格结果。
- **FlowLong**：完成的 variant 可跳过；中途终止的去噪没有 timestep checkpoint，需从该 variant 开头重算。
- **不完整输出**：MP4、JSON 或元数据不完整时可能拒绝复用；先确认目标，再选择独立目录或明确覆盖。
- **split / recon**：统一入口拒绝已有结果。部分窗口失败时，应检查已有窗口，再用底层脚本处理缺失窗口，不要盲目重跑全部前处理。
- **stitch / render**：默认不强制重建或覆盖切片；stitch 仍会重新计算平滑轨迹。上游输入或参数变化后，需要同步更新下游条件和推理。

仅在明确要重做指定 variant 时使用覆盖：

```bash
PHASES=t0.6 SEED="$SEED" PROMPT="$PROMPT" FORCE=true \
bash scripts/test_video/run_flowlong_stage4.sh "$TARGET"
```

此命令允许覆盖同 seed 的 `t0.6` 结果，不应作为日常恢复命令。重做完整渲染需显式设置 `OVERWRITE_FULL_RENDER=true`；不要为了跳过校验随意打开覆盖开关。

## 7. 排错与开发入口

| 现象 | 排查方向 |
|---|---|
| `Cannot detect the model type. File: []` | 检查 checkpoint 路径和实际文件，确认权重不是放在不同名称的目录 |
| 尾段跳变或长度不对 | 检查规则 stride、`valid_num_frames`、`pad_right`，不要移动尾窗或保留生成 padding |
| OOM 或推理很慢 | 检查内存、attention 后端、microbatch、VAE 和 offload；用相同 CFG 的 smoke 定位 |
| 改参数后立即跳过 | 换独立 `OUTPUT_FOLDER`，或确认目标后显式覆盖；文件存在不等于参数符合 |
| 颜色、人物仍不稳定 | matching 只同步共享状态，仍需检查重建、渲染条件和采样设置 |

报告中的 `pipeline.cuda_peak_allocated_gib` / `pipeline.cuda_peak_reserved_gib`
是 PyTorch 分配器统计，不是整个系统或进程的全部内存。统一内存设备上，也不能据 `nvidia-smi` 缺失数值推断没有占用。

主要代码入口：

| 模块 | 文件 |
|---|---|
| 多视频流水线与路径 | `scripts/test_video/run_video_experiment.py`、`utils/video_experiment.py` |
| 窗口与 padding | `scripts/preprocess/split_video_into_clips.py`、`utils/split_manifest.py` |
| 共享条件生成 | `scripts/test_video/prepare_flowlong_ab_conditions.sh` |
| 联合采样与匹配 | `diffsynth/pipelines/flowlong.py` |
| Vista4D 接入 | `diffsynth/pipelines/wan_video_vista4d.py`、`scripts/inference/inference_flowlong.py` |

CPU/mock 回归与 shell 语法检查：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v

bash -n scripts/test_video/prepare_flowlong_ab_conditions.sh
bash -n scripts/test_video/run_flowlong_inference.sh
bash -n scripts/test_video/run_flowlong_stage4.sh
```

