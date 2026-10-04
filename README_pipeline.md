# Vista4D / FlowLong 完整帧率生成流程

## YAML 统一入口（推荐）

每个视频使用一份 [YAML 配置](docs/video_configuration.md)。环境、模型与单片段说明见 [README.md](README.md)，长视频原理见 [READMEv2.md](READMEv2.md)。

```bash
# 只读预览。
python -m scripts.test_video.run_video_experiment --config configs/1776148878076.yaml

# 执行完整视频流程。
python -m scripts.test_video.run_video_experiment \
  --config configs/1776148878076.yaml --gpu 0 \
  --stages split recon stitch render inference --execute
```

完整视频的数据流：切出 baseline / FlowLong 窗口 → baseline 窗口重建与分割 → 全局对齐及相机平滑 → 共享点云渲染 → 两套窗口条件 → baseline 独立生成或 FlowLong 联合去噪。

重建数据在 `recon_and_seg/`，渲染条件在 `render_*/`，最终生成视频为 `video_seed=*.mp4`；baseline 成品带 `_center_cut` 后缀。点云渲染视频是模型输入。

YAML 产物在 `results/configured/<TARGET>/<RESOLUTION>/<run_name>/`。以下直接 Bash 示例沿用历史结果目录，不能假定与 YAML 流程自动复用。

## 0. 环境与模型


```bash
# 已有环境直接激活；首次安装按主 README 第 2 节选择 x86_64/cu128 或 GB10/cu130。
# 不要在实验运行期间重装依赖；下文使用本机已有的 vista4d-pgx 环境名。
conda activate vista4d-pgx
command -v python ffmpeg ffprobe jq tmux
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

```bash
# 首次下载基础权重；SAM3 须先在 Hugging Face 申请访问权限。已有完整权重可跳过。
hf auth login
hf download Eyeline-Labs/Vista4D --local-dir checkpoints/vista4d
hf download Wan-AI/Wan2.1-T2V-14B --local-dir checkpoints/wan/Wan2.1-T2V-14B
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir checkpoints/DA3NESTED-GIANT-LARGE-1.1
hf download facebook/sam3 --local-dir checkpoints/sam3
```

权重预期位置与用途：

| 路径 | 用途 / 应检查的内容 |
|---|---|
| `checkpoints/vista4d/384p49_step=30000/` | 384p Vista4D 条件生成权重与 `config.yaml` |
| `checkpoints/vista4d/720p49_step=3000/` | 720p 对应权重与 `config.yaml`，不能与 384p 混用 |
| `checkpoints/wan/Wan2.1-T2V-14B/` | Wan 模型文件、文本编码器、tokenizer、`Wan2.1_VAE.pth` |
| `checkpoints/DA3NESTED-GIANT-LARGE-1.1/` | 默认 DA3 重建模型目录 |
| `checkpoints/sam3/` | SAM3 下载位置；访问授权、实际模型加载与缓存配置仍按主 README 检查 |

环境检查命令应能找到 `ffmpeg`、`ffprobe`、`jq`、`tmux`，且主环境能访问所选 GPU；这些检查没有生成结果视频。模型下载位置不等于全部模型均已成功加载，最终以 smoke 为准。

已提供可选的 [FP32 checkpoint 分片工具](docs/checkpoint_sharding.md)，默认每片 4 GB；原 `.pth` 不受影响。统一入口使用 `--vista4d-folder <分片目录>`，底层 Bash 使用 `VISTA4D_FOLDER=<分片目录>`。它们是权重存储选项，不需要重复前处理。

## 1. 公共配置与后台会话

统一 Python 入口在共享多卡机器上可用 `--gpu` 选择 `nvidia-smi` 中的物理卡号。卡号会转换成 UUID 并传递给所有阶段，避免 CUDA 逻辑编号混淆；此参数不自动启用多卡并行。

```bash
# 仅使用物理 GPU 1；先预览计划，确认后在同一命令末尾添加 --execute。
python -m scripts.test_video.run_video_experiment \
  --video ./data/my_video.mp4 --resolution 384p --gpu 1
```

`--gpu` 优先于已有的 `CUDA_VISIBLE_DEVICES`，在调度系统中应只选择分配给自己的设备。不传 `--gpu` 时支持继承 UUID 形式的 `CUDA_VISIBLE_DEVICES`；继承数字形式须显式设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，否则拒绝猜测设备映射。

**当前工作树的安全边界**：`run_video_experiment.py` 的 `gpu_environment()` 仍查询计算进程，但 `if busy: raise RuntimeError(...)` 已被注释，因此目前不会因为 GPU 忙而拒绝启动。它不锁卡，也不自动排队；执行前必须自行确认设备分配与占用。选卡只作用于这一次统一入口调用，不会修改父 shell，也不会自动影响随后直接运行的 Bash 命令。后续统一入口命令也应传同一个 `--gpu`，或在当前 shell 显式配置分配给自己的 GPU UUID。

```bash
# 可选：先创建 tmux 会话，再在会话内部进入仓库、激活环境并执行后续配置。
tmux new -s vista4d-pipeline
```

按 `Ctrl-b` 后按 `d` 分离；返回用 `tmux attach -t vista4d-pipeline`。tmux 不提供断电恢复，也不自动继承客户端刚设置的全部变量。

```bash
# 修改视频、分辨率、分割关键词和 prompt；不要用同名新视频复用旧条件。
conda activate vista4d-pgx
export VIDEO=./data/my_video.mp4
export TARGET="$(basename "${VIDEO%.*}")"
export RESOLUTION=384p                       # 384p 或 720p
export SEED=10027
export SEG_KEYWORDS="person man woman hand phone bag backpack car stroller"
export PROMPT="A realistic handheld smartphone video of people in an everyday scene, with natural body motion, realistic lighting, stable camera motion, and detailed surroundings."
export FLOWLONG_PYTHON="$(python -c 'import sys; print(sys.executable)')"

case "$RESOLUTION" in
  384p)
    export WIDTH=672 HEIGHT=384
    export BASELINE_SPLITS_DIR=./media/splits
    export FLOWLONG_SPLITS_DIR=./media/flowlong_splits
    export VISTA4D_FOLDER=./checkpoints/vista4d/384p49_step=30000 ;;
  720p)
    export WIDTH=1280 HEIGHT=720
    export BASELINE_SPLITS_DIR=./media/splits/720p
    export FLOWLONG_SPLITS_DIR=./media/flowlong_splits/720p
    export VISTA4D_FOLDER=./checkpoints/vista4d/720p49_step=3000 ;;
  *) echo 'RESOLUTION 必须为 384p 或 720p' >&2; exit 2 ;;
esac
export FULL_SEQUENCE_ROOT=./results/full/${TARGET}_stitched_${RESOLUTION}
export EVAL_ROOT=./results/flowlong_eval/${TARGET}_${RESOLUTION}_seed=${SEED}
export USE_USP=false CFG_MERGE=false FLOWLONG_DISABLE_STOCHASTIC=false
test -f "$VIDEO" && test -f "$VISTA4D_FOLDER/config.yaml"
python -m utils.vista4d_checkpoint resolve "$VISTA4D_FOLDER"
```

### 1.1 路径和命名规则

以下路径均相对于仓库根目录；`$变量` 对应上面或第 4.1 节导出的 shell 变量，`<说明>` 是占位符，不应原样复制到命令中。

| 名称 | 含义 / 默认值 |
|---|---|
| `TARGET` | 输入视频不带扩展名的文件名，例如 `my_video`。使用字母、数字、下划线或连字符；不要用空格或复杂标点 |
| `RESOLUTION` | `384p` 对应 `672×384`；`720p` 对应 `1280×720` |
| `BASELINE_SPLITS_DIR` | 384p：`media/splits/`；720p：`media/splits/720p/` |
| `FLOWLONG_SPLITS_DIR` | 384p：`media/flowlong_splits/`；720p：`media/flowlong_splits/720p/` |
| `FULL_SEQUENCE_ROOT` | `results/full/${TARGET}_stitched_${RESOLUTION}`，整条视频的全局重建与共享渲染 |
| `EVAL_ROOT` | `results/flowlong_eval/${TARGET}_${RESOLUTION}_seed=${SEED}`，完整 baseline/FlowLong 生成（历史目录名） |

长视频的窗口名不是单段 `CLIP_NAME`，而是切片 MP4 的 stem：

```text
<TARGET>_split<三位窗口号>_frames<六位起始帧>_<六位有效末帧>_<RESOLUTION>49
例如：my_video_split001_frames000024_000072_384p49
```

起止帧是从 0 开始的索引，文件名中的末帧是**有效末帧**，不包含为凑满 49 帧而复制的 padding。baseline 和 FlowLong 的窗口号、起止帧不同；即使第一窗名称相同，其条件也分别放在不同根目录，不能串用。应从各自 JSON manifest 的 `clips[].output_path` 取 stem，不要人工猜测。

前处理目录通常不带 seed，多个 seed 可以共享同一套未改变的条件。不同参数的生成结果应放入各自输出目录。

### 1.2 统一 Python 入口与底层 Bash 的配置不同

- 统一入口：[run_video_experiment.py](scripts/test_video/run_video_experiment.py)。默认只打印计划；加 `--execute` 才执行。默认阶段为 `split recon stitch render inference`；默认推理变体包含 `baseline matching_only t0.5 t0.6 t0.7`。本文命令显式选择阶段/变体，以便逐步验收。
- 统一入口会清理继承的流程配置变量，以避免上一条视频的 `OUTPUT_FOLDER`、`SOURCE_VIDEO` 等污染下一次运行。应通过 `--video`、`--resolution`、`--seed`、`--prompt`、`--seg-keywords`、`--factors`、`--vista4d-folder` 指定配置；不要以为 `export NUM_INFERENCE_STEPS=1` 会改变正式入口的 50-step 设置。
- 上面导出的变量用于本文直接运行的 Bash 命令。特别是自定义分片权重：Bash 读 `VISTA4D_FOLDER`，统一入口需在每次相关调用中显式加 `--vista4d-folder "$VISTA4D_FOLDER"`。
- 直接调用 Bash 时会继承已导出的变量，且若不指定参数，一些历史脚本有示例视频默认值。本文始终传 `"$TARGET"` 或 `"$VIDEO"`；换视频后不要继续使用旧的 `INPUT_VIDEO`、`OUTPUT_FOLDER`。
- 统一入口没有隐含 tmux、日志总文件或任务调度；日志保存和中断处理见第 5 节。

## 2. 单片段 Vista4D（README.md）

适合指定的连续 49 帧；完整长视频直接跳到第 3 节，无需先运行本节。

### 2.1 截取并规范化输入

读取 `$VIDEO` 的第 `0..48` 帧，按目标宽高比中心裁剪后缩放，保留源 FPS，不带音频。只生成一个 49 帧片段；不会生成长视频的 split manifest，也不会做 DA3 或生成稳定化结果。

```bash
# 2.1 截取连续 49 帧并缩放到目标尺寸；输入不足 49 帧会报错。
export CLIP_NAME=${TARGET}_frames000000_000048_${RESOLUTION}49
export SOURCE_VIDEO=./media/single/${CLIP_NAME}.mp4
python -m scripts.preprocess.prepare_custom_single_video \
  --input "$VIDEO" --output_dir ./media/single --output_name "$CLIP_NAME" \
  --start_frame 0 --num_frames 49 --width "$WIDTH" --height "$HEIGHT"
```

产物：`media/single/${CLIP_NAME}.mp4`。下一阶段由导出的 `SOURCE_VIDEO` 找到它。应先确认裁剪后目标人物没有出画；若修改 `--start_frame`，同时修改 `CLIP_NAME` 中的帧范围，避免名称与内容不符。

### 2.2 重建源场景并分割动态区域

入口：[recon_and_seg.sh](scripts/test_video/recon_and_seg.sh) → [recon_and_seg_single.py](scripts/preprocess/recon_and_seg_single.py)。DA3 估计每帧深度、相机位姿和内参；SAM3 依据 `SEG_KEYWORDS` 分割动态物体，并准备天空 mask。此时相机还是源视频的相机，不是平滑后的目标相机。

```bash
# 2.2 DA3 重建深度和相机，SAM3 分割动态区域；SAVE_VIS 保存人工检查视频。
RECON_METHOD=da3 DA3_PROCESS_RES="$WIDTH" SAVE_VIS=true \
bash scripts/test_video/recon_and_seg.sh
```

产物根目录：`results/single/${CLIP_NAME}/recon_and_seg/`。

| 相对该目录的路径 | 内容 / 下游用途 |
|---|---|
| `video.mp4` | 与重建数据一一对应的源画面，49 帧 |
| `depths/00000.exr` … `00048.exr` | 每帧深度，后续反投影构建点云；不是可直接播放的视频 |
| `dynamic_mask/00000.png` … | 动态区域二值 mask，用于区分动态/静态点 |
| `sky_mask/00000.png` … | 天空区域二值 mask |
| `cameras.npz` | `cam_c2w`：49 个 4×4 camera-to-world 位姿；`intrinsics`：对应每帧内参 |
| `vis.mp4` | 仅 `SAVE_VIS=true` 时保存的检查视频，用于看深度和分割是否合理 |

普通单片段重建不保证产生 `clips.json`；该文件在 DSE 或后面的全局拼接/重切流程中另有用途。逐帧深度和 mask 都从本目录的局部索引 `00000` 开始命名，不是原视频的全局帧号。

验收：源视频、深度、动态 mask、天空 mask 各有 49 帧/项，相机长度一致；人工查看 `vis.mp4`。错误的人物分割或深度不会被后续平滑自动修复。

### 2.3 生成稳定化目标相机

```bash
# 2.3 平滑原始相机轨迹，作为稳定化后的目标运镜。
TRANSLATION_SIGMA=4 ROTATION_SIGMA=4 bash scripts/test_video/smooth.sh
```

读取 `results/single/${CLIP_NAME}/recon_and_seg/cameras.npz`，分别平滑平移与旋转，默认锚定第一帧位姿。sigma 按帧时间轴计算，不是秒。

输出：同目录的 `cameras_gaussian_smooth.npz`，包含平滑后的 `cam_c2w`、保留的 `intrinsics` 和原始 `raw_cam_c2w`；不会改写源 `cameras.npz`，也不生成视频。下一阶段 `USE_SMOOTHED_CAMERA=true` 时读取这个文件。

### 2.4 在目标相机下渲染模型条件

```bash
# 2.4 在目标相机下渲染点云；video_pc.mp4 是输入条件，不是最终生成视频。
USE_SMOOTHED_CAMERA=true SAVE_VIS=true RENDER_ONLY_NECESSARY=true \
bash scripts/test_video/render.sh
```

读取 2.2 的 RGB、depth、mask、源相机以及 2.3 的目标相机，构建并投影点云。产物目录：`results/single/${CLIP_NAME}/render_${RESOLUTION}_smooth/`。

以下是本流程共用的**渲染条件文件约定**；后面的全局 shared-static 渲染和窗口条件也使用这些名称：

| 相对渲染目录的路径 | 含义 |
|---|---|
| `video_src.mp4` | 源视角 RGB 条件 |
| `video_pc.mp4` | 目标视角点云渲染 RGB；有空洞并不等于推理失败 |
| `depths_src/*.exr`、`depths_pc/*.exr` | 源视角/目标渲染视角的逐帧深度 |
| `alpha_mask_src/*.png`、`alpha_mask_pc/*.png` | 各视角的有效覆盖区域 |
| `dynamic_mask_src/*.png`、`dynamic_mask_pc/*.png` | 动态区域 |
| `static_mask_src/*.png`、`static_mask_pc/*.png` | 静态区域 |
| `sky_mask_src/*.png` | 源视角天空区域；不要假设一定有 `sky_mask_pc/` |
| `cameras_src.npz`、`cameras_tgt.npz` | 源相机与目标相机，各含 `cam_c2w` 和 `intrinsics` |

深度使用 EXR，mask 使用 0/255 PNG；逐帧文件在所属目录内从 `00000` 开始编号。`SAVE_VIS=true` 还会产生 `vis.mp4`。本文 `RENDER_ONLY_NECESSARY=true` 不要求额外的 `video_pc_ntp.mp4`、`video_pc2.mp4` 等可选视图。

验收：至少观看 `video_src.mp4`、`video_pc.mp4` 和 `vis.mp4`，确认目标运镜、人物位置与静态背景合理。此目录是 2.5 的 `INPUT_FOLDER`，不要把 `recon_and_seg/` 直接传给推理。

### 2.5 Vista4D 生成稳定化视频

```bash
# 2.5 正式 50-step Vista4D；结果保存在该片段的 vista4d_<分辨率>_smooth/。
USE_SMOOTHED_CAMERA=true SEEDS="$SEED" NUM_INFERENCE_STEPS=50 CFG_SCALE=5 SIGMA_SHIFT=5 \
EXTRA_ARGS="--tile_vae" bash scripts/test_video/inference.sh
ls "results/single/$CLIP_NAME/vista4d_${RESOLUTION}_smooth/video_seed=${SEED}.mp4"
```

入口：[inference.sh](scripts/test_video/inference.sh) → [inference.py](scripts/inference/inference.py)。加载对应分辨率的 Vista4D/Wan 权重，在渲染条件和 prompt 引导下执行 50 次去噪。

输出目录：`results/single/${CLIP_NAME}/vista4d_${RESOLUTION}_smooth/`。

- `video_seed=${SEED}.mp4`：最终 49 帧视频，FPS 与输入条件一致。
- `source.mp4`、`point_cloud.mp4`、`point_cloud_masks.mp4`：模型输入的检查副本，不是另外三种生成方案。
- `gifs/`：wrapper 的 `--save_gif` 用于保存输入检查 GIF；不要据此要求每个生成视频都有 GIF。

这个单片段入口没有长视频的 `flowlong_report_seed=*.json`。验收应检查最终视频的帧数、尺寸并人工观看；若换 prompt、seed 或轨迹，使用明确的新输出位置保存对照。

## 3. 完整视频 baseline / FlowLong（READMEv2.md）

正式推理固定 50 steps、seed=`$SEED`、CFG=5、sigma_shift=5。完整 FlowLong 至少需要两个窗口。

3.2～3.5 是共用前处理：即使只准备运行 FlowLong，当前统一流程仍先用 baseline 的 49/5/44 窗口做重建，再将统一的全局条件切给 FlowLong。**这不要求先生成 baseline 视频**；不需要对 FlowLong 的全部重叠窗口再次运行 DA3/SAM3。

### 3.1 查看计划

```bash
# 只打印计划，不生成文件、不加载模型；确认视频长度、两套窗口与结果路径。
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seed "$SEED" \
  --seg-keywords "$SEG_KEYWORDS" --prompt "$PROMPT" \
  --stages split recon stitch render inference --phases baseline matching_only t0.5 t0.6
```


检查重点：输入是否为期望的视频；是否从第 0 帧覆盖到末帧；分辨率、窗口数和尾部 padding 是否合理；`inference_root` 是否会与旧实验重名。需要保存计划时自行重定向到一个新文件，见第 4.1 节。计划成功不代表权重、条件或显存已经通过运行验收。

### 3.2 视频切片

```bash
# baseline：49/5/44；FlowLong：49/25/24。规则 stride，尾部重复 padding。
# 已完成切片时跳过此步；统一入口不会无条件覆盖已有 manifest。
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --stages split --execute
```

执行两次 [split_video.sh](scripts/test_video/split_video.sh) → [split_video_into_clips.py](scripts/preprocess/split_video_into_clips.py)：中心裁剪/缩放原视频，按固定步长抽取连续 49 帧窗口；最后一窗不足 49 帧时重复末帧补齐。保留有效范围，后续最终输出必须去掉 padding。

| 窗口体系 | 长度 / 重叠 / 步长 | MP4 与 manifest 的保存位置 |
|---|---|---|
| baseline | 49 / 5 / 44 | `$BASELINE_SPLITS_DIR/<窗口名>.mp4`、`${TARGET}_splits_manifest.json`、`${TARGET}_splits_manifest.csv` |
| FlowLong | 49 / 25 / 24 | `$FLOWLONG_SPLITS_DIR/<窗口名>.mp4`、`${TARGET}_splits_manifest.json`、`${TARGET}_splits_manifest.csv` |

表中 manifest 文件也位于对应行的目录中，不是仓库根目录。JSON 是后续阶段的数据依据；CSV 便于人工查看，不包含全部顶层元信息，不能替代 JSON。

重要字段：

- 顶层 `input_path`、`fps`、`start_frame`、`end_exclusive` 标识源文件、帧率和有效范围；`end_exclusive` 不包含末端帧。
- `clips[].start_frame` / `end_frame` 是该窗有效范围，`end_frame` 包含末帧。
- `num_frames=49` 是编码帧数，`valid_num_frames` 是真实帧数，`pad_right` 是复制补齐数量。
- `output_path` 是窗口 MP4；`padded_end_frame` 是组窗逻辑位置，可能超出实际源视频末帧。

可用以下只读检查核对两套布局：

```bash
for folder in "$BASELINE_SPLITS_DIR" "$FLOWLONG_SPLITS_DIR"; do
  jq '{input_path, fps, start_frame, end_exclusive, num_clips,
       first: .clips[0], last: .clips[-1]}' \
    "$folder/${TARGET}_splits_manifest.json"
done
```

下一步只读取 baseline manifest。已有 manifest 时，统一入口会拒绝再次执行 `split`，而非自动覆盖；复用旧切片前须核对源文件内容未变，单凭同名、同路径和相同帧数不足以证明相同输入。

### 3.3 DA3 + SAM3

```bash
# 对 baseline 窗口逐一重建和分割；DA3_PROCESS_RES 自动取 672 或 1280。
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seg-keywords "$SEG_KEYWORDS" \
  --stages recon --execute
```

统一入口遍历 baseline JSON 中的窗口，逐一调用 `recon_and_seg.sh`，不处理 FlowLong manifest 中的另一套窗口。

每窗产物在：

```text
results/single/<baseline窗口名>/recon_and_seg/
  video.mp4
  depths/00000.exr … 00048.exr
  dynamic_mask/00000.png … 00048.png
  sky_mask/00000.png … 00048.png
  cameras.npz
```

文件语义同 2.2。当前统一入口设置 `SAVE_VIS=false`，因此本步默认**没有** `vis.mp4`，也不会自动写 profiling JSON。这里的相机/深度仍属于各自窗口的坐标与尺度，不能直接把相机数组首尾拼起来当成全局轨迹。

每窗结束后，统一入口检查 `video.mp4` 的尺寸和 49 帧长度、深度/mask 的文件数量、相机数组长度及有限值，成功打印 `RECON_VALIDATED <目录>`。全部完成后才进入 3.4。

如果其中一窗已经存在 `recon_and_seg/`，统一入口会报错，并不会自动跳过它。部分完成时不能直接重跑整条 `recon` 命令；应核验已完成窗口，用底层命令仅补缺失窗口，或在新命名空间重做，见第 5.2 节。

### 3.4 全局 reconstruction 拼接、平滑与重切片

```bash
# 对齐到统一坐标系，再平滑完整相机轨迹；平移 sigma=8、旋转 sigma=10。
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --stages stitch --execute
```

入口：[stitch_splits_smooth_and_slice.sh](scripts/test_video/stitch_splits_smooth_and_slice.sh)。这个阶段按顺序做三件事：

1. **全局对齐与拼接**：从 `results/single/<baseline窗口名>/recon_and_seg/` 读取局部重建，使用重叠区域估计尺度、旋转和平移的 Sim(3) 变换。统一相机坐标和深度尺度；重叠 RGB/depth/mask 按中心权重选择所属窗口，相机位姿则做加权融合。
2. **全局平滑**：对拼接后的完整相机序列统一平滑，使用平移 sigma=8、旋转 sigma=10。不是每段各自平滑后再拼接，从而避免边界轨迹不连续。
3. **重切片**：按 baseline manifest 切回 49 帧重建条件，并同步处理尾部 padding。此时切片都来自同一条全局时间轴。

具体产物：

| 位置 | 内容 / 用途 |
|---|---|
| `$FULL_SEQUENCE_ROOT/recon_and_seg/video.mp4` | 去掉重叠冗余和尾部 padding 后的完整有效源序列 |
| `$FULL_SEQUENCE_ROOT/recon_and_seg/{depths,dynamic_mask,sky_mask}/` | 同一全局时间轴上的深度和 mask，逐帧数量等于有效源帧数 `N` |
| `$FULL_SEQUENCE_ROOT/recon_and_seg/cameras.npz` | 对齐融合后的完整源相机，不再是各窗独立坐标 |
| `$FULL_SEQUENCE_ROOT/recon_and_seg/clips.json` | 完整源片段范围，`src` 对应有效全序列 |
| `$FULL_SEQUENCE_ROOT/recon_and_seg/stitch_report.json` | 每窗对齐尺度、旋转/平移、误差、全局帧归属等诊断 |
| `results/stitched_single/<baseline窗口名>/recon_and_seg/` | 重新切出的重建文件，同时包含源相机、平滑相机和 `full_sequence_slice.json` |
| `results/stitched_single/${TARGET}_splits_manifest_full_sequence_conditions.json` | 重建重切片的总索引/报告 |

`results/single/` 的原始局部重建保留不动；`results/stitched_single/` 不是最终生成视频目录。本流程也不要求落盘 `.ply` 点云：后续渲染根据 RGB、深度、mask 和相机构建点云。

验收：查看 `stitch_report.json` 是否覆盖全部窗口，关注异常尺度和接缝对齐误差；统一入口还会验证全局重建长度为 `N`、尺寸正确、相机有限。目标相机平滑只输出 NPZ，本步不会生成稳定化视频或相机轨迹可视化视频。

**复用边界**：已有 `cameras.npz` 时 wrapper 可以跳过重建拼接，但仍会重新写平滑相机；带切片 metadata 的旧重切片又可能被跳过。因此，修改平滑参数后不能只重跑此命令并假设所有下游已更新，必须有计划地重建受影响的切片、渲染与生成结果。

### 3.5 共享渲染

```bash
RESOLUTION="$RESOLUTION" STATIC_FRAME_STRIDE=4 RENDER_CHUNK_SIZE=4 \
  bash scripts/test_video/prepare_flowlong_ab_conditions.sh "$TARGET"
```

入口：[prepare_flowlong_ab_conditions.sh](scripts/test_video/prepare_flowlong_ab_conditions.sh)。读取两份 manifest、3.4 的全局重建和全局平滑相机，先校验两套 manifest 的源文件、有效范围、FPS 和分辨率一致，再执行：

1. 构建全局共享的静态点集；`STATIC_FRAME_STRIDE=4` 是构建静态点集时的帧采样步长，**不是把输出视频抽到 1/4 帧率**。
2. 对完整有效时间轴渲染源/目标条件；`RENDER_CHUNK_SIZE=4` 控制渲染分块，不改变输出帧数。
3. 从这一份全局结果切出 baseline 条件，再切出 FlowLong 条件；第二套复用全局渲染，不再独立重建或渲染点云。

保存位置：

| 产物 | 具体路径 |
|---|---|
| 全局渲染条件 | `$FULL_SEQUENCE_ROOT/render_${RESOLUTION}_smooth_shared_static/`，含 2.4 表中的 RGB、depth、mask、相机，长度为 `N` |
| 全局渲染 metadata | 上述目录中的 `shared_static_render.json`，记录来源、配置、帧数与共享渲染信息 |
| baseline 窗口条件 | `results/shared_static_single/<baseline窗口名>/render_${RESOLUTION}_smooth/` |
| FlowLong 窗口条件 | `results/flowlong_single/<FlowLong窗口名>/render_${RESOLUTION}_smooth/` |
| 每窗来源证明 | 各窗口渲染目录中的 `full_shared_static_slice.json`，记录父渲染目录、帧范围、padding 等 |
| baseline 切片总报告 | `results/shared_static_single/${TARGET}_splits_manifest_shared_static_render.json` |
| FlowLong 切片总报告 | `results/flowlong_single/${TARGET}_splits_manifest_shared_static_render.json` |


共享 RGB 及其窗口切片采用无损 RGB 编码，以便严格比较相同帧。不要把普通切片 MP4、单独渲染的各窗条件或 `results/stitched_single/` 的重建目录替代这里的输入。


### 3.6 1-step smoke 与验收

```bash
# CFG=5 与正式配置一致；这里只验证加载、输出长度和 matching，不评价画质。
export SMOKE=./results/flowlong_smoke/${TARGET}_${RESOLUTION}_1step_cfg5
OUTPUT_FOLDER="$SMOKE" NUM_INFERENCE_STEPS=1 CFG_SCALE=5 SIGMA_SHIFT=5 \
FLOWLONG_STOCHASTIC_THRESHOLD=0.6 FLOWLONG_DISABLE_STOCHASTIC=false \
FLOWLONG_MICROBATCH_SIZE=1 SEEDS="$SEED" TILE_VAE=true FORCE=false \
bash scripts/test_video/run_flowlong_inference.sh "$TARGET"

# 输出应等于完整有效帧数，所有 overlap_after_max_abs 应为 0；实际帧数用 ffprobe 检查。
EXPECTED_FRAMES=$(jq '.end_exclusive - .start_frame' "$FLOWLONG_SPLITS_DIR/${TARGET}_splits_manifest.json")
jq -e --argjson n "$EXPECTED_FRAMES" \
  '.output_frames == $n and all(.pipeline.steps[]; .overlap_after_max_abs == 0)' \
  "$SMOKE/flowlong_report_seed=${SEED}.json"
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames -of default=noprint_wrappers=1 \
  "$SMOKE/video_seed=${SEED}.mp4"
```

读取 3.2 的 FlowLong manifest 和 3.5 的 `results/flowlong_single/` 条件。入口 [run_flowlong_inference.sh](scripts/test_video/run_flowlong_inference.sh) 加载模型、执行 1-step 全窗口采样、全局解码并裁掉 padding。

输出位置：

- `$SMOKE/video_seed=${SEED}.mp4`：完整有效帧数的连通性测试视频。
- `$SMOKE/flowlong_report_seed=${SEED}.json`：窗口几何、条件来源、权重/manifest/输出 hash、实验参数、时间及每步 matching 统计。

本命令不经 stage4 日志 wrapper，默认日志只在终端。JSON 在完成视频保存之后才写出，不是运行中持续刷新的进度文件。`overlap_after_max_abs == 0` 只说明匹配后的共享 latent 一致；1-step 不用于画质结论，正式实验也不能用 smoke 视频代替。

### 3.7 50-step 正式推理

```bash
# 顺序运行 baseline、matching-only、t*=0.5、t*=0.6；只需一种时删去其余 phase。
# matching-only 关闭随机更新但仍匹配 overlap；t* 是噪声阈值，不是步数比例。
python -u -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seed "$SEED" --prompt "$PROMPT" \
  --stages inference --phases baseline matching_only t0.5 t0.6 --execute
```

入口：[run_flowlong_stage4.sh](scripts/test_video/run_flowlong_stage4.sh)。只执行指定 phase，按 `baseline → matching_only → t0.5 → t0.6 → t0.7` 的固定顺序运行，不是并行启动。

| phase | 执行内容 | 最终视频 | 运行报告 |
|---|---|---|---|
| `baseline` | 读取 `shared_static_single`，独立生成各 49 帧窗口，再对重叠区 center-cut 拼接并裁尾 | `$EVAL_ROOT/baseline/video_seed=${SEED}_center_cut.mp4` | 同目录 `baseline_report_seed=${SEED}.json` |
| `matching_only` | 读取 `flowlong_single`，每步匹配窗口 clean prediction，关闭随机更新 | `$EVAL_ROOT/matching_only/video_seed=${SEED}.mp4` | 同目录 `flowlong_report_seed=${SEED}.json` |
| `t0.5` | FlowLong，随机阶段阈值 0.5 | `$EVAL_ROOT/flowlong_t0p5/video_seed=${SEED}.mp4` | 同目录 `flowlong_report_seed=${SEED}.json` |
| `t0.6` | FlowLong，随机阶段阈值 0.6 | `$EVAL_ROOT/flowlong_t0p6/video_seed=${SEED}.mp4` | 同目录 `flowlong_report_seed=${SEED}.json` |
| `t0.7` | FlowLong，随机阶段阈值 0.7，需显式请求或使用入口默认 phase 集合 | `$EVAL_ROOT/flowlong_t0p7/video_seed=${SEED}.mp4` | 同目录 `flowlong_report_seed=${SEED}.json` |

baseline 的中间产物还包括：

```text
$EVAL_ROOT/baseline/clips/<baseline窗口名>/
  video_seed=<SEED>.mp4
  clip_report_seed=<SEED>.json
```

FlowLong 则维护全局 latent 并在结束后解码，不要求输出每个窗口的独立生成 MP4。所有完整变体最终均应等于有效源帧数 `N`，而不是 `窗口数×49`。

每次 stage4 调用的日志保存在：

```text
logs/flowlong_stage4/${TARGET}_${RESOLUTION}_seed=${SEED}_<YYYYmmdd_HHMMSS>/
  baseline.log                 # 选择 baseline 时
  matching_only.log             # 选择该变体时
  flowlong_t0p5.log
  flowlong_t0p6.log
  flowlong_t0p7.log
```

只产生所选 phase 对应的日志。

验收先看 MP4 帧数、尺寸和 FPS，再核对报告里的 `experiment`、`output_frames`、模型/manifest hash。FlowLong 的 `pipeline.steps` 数应为 50，逐步 `overlap_after_max_abs` 应为 0；仍需人工查看接缝闪烁、身份保持和动态物体质量。


## 4. 日志、复用与故障排查

### 4.1 到哪里看计划、进度和日志

| 阶段 / 入口 | 默认进度与日志位置 |
|---|---|
| 统一 Python 计划、split/recon/stitch/render | 标准输出；`--execute` 时打印 `STAGE_START` / `STAGE_COMPLETE`，没有自动总日志文件 |
| 3.6 直接运行完整 FlowLong smoke | 标准输出；完成后在 `$SMOKE` 写视频和 JSON |
| 3.7 stage4 | `logs/flowlong_stage4/${TARGET}_${RESOLUTION}_seed=${SEED}_<时间戳>/` |

需要总日志时，在已经完成变量配置的 shell 中显式保存，例如：

```bash
# 只保存一份完整推理计划，不启动实验。
export LOG_DIR="./logs/manual/${TARGET}_${RESOLUTION}_seed=${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
python -m scripts.test_video.run_video_experiment \
  --video "$VIDEO" --resolution "$RESOLUTION" --seed "$SEED" \
  --seg-keywords "$SEG_KEYWORDS" --prompt "$PROMPT" \
  --stages inference --phases t0.6 > "$LOG_DIR/plan.json"
```

正式执行时，对选定的原命令加 `--execute` 并另存为 `pipeline.log`。如果通过 `2>&1 | tee ...` 保存，先执行 `set -o pipefail`，防止 `tee` 成功掩盖上游失败。不要把这个执行日志命名为 `plan.json`，因为它混有进度与子进程输出，不再是单个 JSON。

报告通常在对应任务结束时才写入。运行中未出现 `flowlong_report_seed=*.json` 不一定表示失败；反过来，只出现视频却没有对应报告也不能当作完成。进度看当前日志，完成后核对报告与视频。

### 4.2 已有结果如何复用，中断后从哪继续

以下描述本文默认 `FORCE=false`、不传 overwrite 时的行为；这些入口并不是统一的“自动断点续跑”系统。

| 阶段 | 遇到已有结果的行为 | 继续方式 |
|---|---|---|
| `split` | 统一入口发现已有 manifest 就拒绝再次切片 | 核实两套 manifest 与源内容后，省略 `split` |
| `recon` | 统一入口发现任一窗口重建目录已存在就拒绝执行到该窗 | 全部完成则省略 `recon`；部分完成则按 manifest 用底层命令只补缺失窗口，不能盲目重跑整阶段 |
| `stitch` | 可跳过已存在的全局相机，但会重新平滑；部分重切片按 metadata 跳过 | 先确认参数和所有切片一致，修改轨迹后重建所有受影响下游 |
| `render` | 可复用带 metadata 的全局渲染；渲染切片目录已存在通常会报错 | 条件完整且参数一致时省略 `render`；部分结果需逐项核验，不能认为会自动补齐 |
| baseline | 对完整结果及逐窗结果检查 contract/hash，可复用完成窗口 | 使用相同配置补未完成窗口；残缺或参数不符的文件会报错 |
| 完整 FlowLong | Bash 对已有视频+报告做有限完成判断；无 timestep 断点 | 已完成且参数正确的变体省略；未完成变体从头采样，保存好残留文件后用新目录重跑 |

目录名并未编码所有实验变量。下面这些改动必须考虑使哪些产物失效：

- 更换源视频内容、裁剪范围或分辨率：从切片/重建开始重新准备，不能复用同 stem 的旧文件。
- 改 DA3/SAM3、分割关键词或重建处理：重新做相关重建、全局拼接、渲染及所有生成。
- 改 prompt、seed、模型、去噪配置：几何条件可在确认未变时复用，但生成及其后处理应放入匹配的新目录。

不要为绕过报错直接设置 `FORCE=true`：有些底层 overwrite 会删除并重建整个目标子目录。优先用新输出位置或先备份已核实的旧产物，再显式重建。

### 4.3 常见问题定位

| 现象 | 优先检查 |
|---|---|
| 找不到 manifest | `TARGET` 是否为视频 stem，分辨率对应目录是否正确，是否执行 3.2 |
| 找到的是旧视频/旧 seed 结果 | 当前 shell 的旧变量、输出目录、JSON 中的 prompt/seed/hash；不要只看文件名 |
| `Reconstruction already exists` | 是否把已完成的 `recon` 加进统一入口；按 4.2 处理部分完成，别覆盖已有窗口 |
| JSON 技术通过但画面仍闪烁或变形 | matching 数值一致不代表视觉质量；查看最终视频、人物和接缝 |

### 4.4 运行边界

- 统一入口不加 `--execute` 只打印计划；当前工作树的 GPU busy 拒绝逻辑已注释，统一入口与底层 Bash 都不能当作 GPU 锁或排队器，运行前确认所选 GPU 可用。
- 已有 `split/recon` 阶段不要重复执行；更换输入或轨迹后不能复用旧条件。本文不使用 `FORCE=true` 自动覆盖。
- FlowLong 中断后没有 timestep 断点恢复，需从该变体开头重算；完成结果能否复用取决于具体入口的校验。
- 新 tmux 会话要重新激活环境并配置变量；不要同时在一张 GPU 上启动多组推理。流程使用均匀帧时间轴，不保留音频。

## 5. 一个展开后的实际路径示例

下面使用当前仓库中的视频 `1776148878076`，仅帮助理解命名；新实验不要直接覆盖这些既有结果。其完整有效长度为 462 帧，示例分辨率 `384p`、seed=`10027`。

| 要找的内容 | 不带变量的具体路径 |
|---|---|
| 原始输入 | `data/1776148878076.mp4` |
| baseline manifest（11 窗） | `media/splits/1776148878076_splits_manifest.json` |
| FlowLong manifest（19 窗） | `media/flowlong_splits/1776148878076_splits_manifest.json` |
| baseline 第二窗原始重建 | `results/single/1776148878076_split001_frames000044_000092_384p49/recon_and_seg/` |
| 全局重建 | `results/full/1776148878076_stitched_384p/recon_and_seg/` |
| 全局目标轨迹 | `results/full/1776148878076_stitched_384p/recon_and_seg/cameras_gaussian_smooth.npz` |
| 全局目标视角点云条件 | `results/full/1776148878076_stitched_384p/render_384p_smooth_shared_static/video_pc.mp4` |
| baseline 第二窗共享渲染条件 | `results/shared_static_single/1776148878076_split001_frames000044_000092_384p49/render_384p_smooth/` |
| FlowLong 第二窗共享渲染条件 | `results/flowlong_single/1776148878076_split001_frames000024_000072_384p49/render_384p_smooth/` |
| 完整 t0.6 最终视频 | `results/flowlong_eval/1776148878076_384p_seed=10027/flowlong_t0p6/video_seed=10027.mp4` |

对于新视频，把 `TARGET` 换成实际 stem；窗口起止帧从新 manifest 获取。不能只全局替换视频名而沿用旧的窗口数或帧范围。

## 6. 面向开发者的代码入口与交付清单

### 6.1 修改某阶段时应读哪些文件

| 职责 | 实现入口 |
|---|---|
| 阶段编排、默认参数、环境隔离、选卡 | [run_video_experiment.py](scripts/test_video/run_video_experiment.py) |
| 分辨率、有效帧数探测、路径与窗口数量 | [video_experiment.py](utils/video_experiment.py) |
| 切片命名、manifest、尾部 padding | [split_video_into_clips.py](scripts/preprocess/split_video_into_clips.py)、[split_manifest.py](utils/split_manifest.py) |
| DA3/SAM3 重建与保存 | [recon_and_seg_single.py](scripts/preprocess/recon_and_seg_single.py) |
| 全局对齐、相机平滑、重建重切片 | [stitch_split_recon_by_manifest.py](scripts/preprocess/stitch_split_recon_by_manifest.py)、[smooth_camera_trajectory.py](scripts/preprocess/smooth_camera_trajectory.py)、[slice_full_recon_by_manifest.py](scripts/preprocess/slice_full_recon_by_manifest.py) |
| 共享渲染、渲染重切片 | [render_full_shared_static.py](scripts/preprocess/render_full_shared_static.py)、[slice_full_render_by_manifest.py](scripts/preprocess/slice_full_render_by_manifest.py) |
| baseline 生成、复用与 center-cut | [inference_split_baseline.py](scripts/inference/inference_split_baseline.py) |
| FlowLong 条件读取、生成与报告 | [inference_flowlong.py](scripts/inference/inference_flowlong.py)、[wan_video_vista4d.py](diffsynth/pipelines/wan_video_vista4d.py)、[flowlong.py](diffsynth/pipelines/flowlong.py) |

### 6.2 给其他开发者或展示端交付什么

- 交付最终视频及对应运行报告；需要复现时同时保留配置、manifest 和执行日志。
- 要继续生成其他变体/seed：还需保留实际窗口条件、全局重建/渲染和相机文件；仅有报告不够。模型与环境另行准备。
- 搬到另一台机器继续运行：JSON 里部分来源路径是绝对路径。单纯拷贝 `results/` 适合展示，不保证推理或严格来源验证立即可用；需要检查输入、权重、manifest 及父条件路径。
