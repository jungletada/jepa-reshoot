# Vista4D × FlowLong 实现计划

## 1. 文档状态与结论

- 目标仓库：Vista4D 当前工作树。
- 参考论文：[FlowLong: Inference-time Long Video Generation via Manifold-constrained Tweedie Matching](https://arxiv.org/html/2605.20910v1)，arXiv v1，2026-05-20。
- 官方仓库：[jhq1234/flowlong](https://github.com/jhq1234/flowlong)。截至本计划编写时，仓库仅包含 README 和演示资源，没有公开采样实现，因此本项目需要按论文公式自行实现。
- 总体结论：Vista4D 使用 Wan 的 rectified-flow sampler，能够在不训练或微调模型的前提下接入 FlowLong。实现位置必须在逐 timestep 去噪循环内部，而不是最终 MP4 合并阶段。

本计划采用以下已经锁定的设计：

1. 保留当前“独立切片 inference + merge”流程作为基线，不改变其默认行为。
2. 新增独立的 FlowLong 联合推理入口；一次加载模型，让所有窗口在相同 timestep 下交替执行模型前向。
3. 49 帧窗口改用 25 帧 overlap、24 帧规则 stride。它对应 13 个 latent frame、6 个 latent stride 和 7 个 latent overlap，满足论文要求的 `O >= S`。
4. 每步在 predicted-clean latent `x0_pred` 上执行论文的线性 Tweedie matching。
5. 高噪声阶段采用论文的 binary stochastic renoising，默认阈值 `t*=0.6`；低噪声阶段恢复确定性 ODE。
6. GB10 上默认 `microbatch_size=1`，避免把所有窗口同时送入 14B DiT。
7. 只在采样结束后对全局 latent 解码一次，然后裁掉尾部 padding；FlowLong 输出不再经过 center-cut。
8. 第一版继续使用当前逐窗口 VAE 条件编码，以保持 Vista4D 训练时的窗口输入分布；“整段条件一次编码再切 latent”不纳入第一版。

## 2. 目标、成功标准与非目标

### 2.1 目标

为长视频重新运镜新增一条训练无关的联合推理路径，使相邻 49 帧窗口不再从彼此独立的扩散轨迹生成，而是在每个去噪步共享一致的 overlap latent，降低窗口接缝处的主体变化、纹理跳变、亮度跳变和运动突变。

### 2.2 成功标准

功能正确性：

- 输入仍由现有完整重建、全局平滑相机和 shared-static render 产生。
- 对当前 310 帧视频，FlowLong 窗口起点严格为 `[0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264]`。
- 每个窗口仍为 49 个像素帧；最后一窗包含 46 个真实帧和 3 个 edge-padding 帧。
- 每个窗口在 latent 空间包含 13 帧，stride 为 6，overlap 为 7。
- 每次 Tweedie aggregation 后，相邻窗口对应的 overlap latent 必须数值一致，最大误差不超过 `1e-6`（以 float32 聚合结果计）。
- 最终全局 latent 解码为 313 帧，按 manifest 的真实范围裁成恰好 310 帧。
- 固定配置和 seed 时结果可复现。

### 2.3 第一版非目标

- 不重新训练或微调 Wan/Vista4D。
- 不修改 DiT attention、RoPE 或 checkpoint 格式。
- 不支持每个窗口使用不同 prompt；第一版所有窗口共享同一 prompt 和 negative prompt。
- 不支持 `cfg_merge=true`；仍使用当前 positive/negative 两次模型前向。
- 不支持 USP/多机联合 FlowLong；第一版明确要求 `USE_USP=false`。
- 不删除当前独立切片 inference、merge mode 或已有结果格式。
- 不在第一版实现全序列 source/point-cloud condition 的一次性 VAE 编码。

## 3. 当前链路与问题定位

当前长视频路径如下：

```text
长视频
  -> 49 帧窗口切片
  -> 全序列重建、相机平滑和 shared-static render
  -> 将连续条件重新切回各窗口
  -> 每个窗口单独启动 Vista4D inference
  -> 每个窗口独立初始化 latent noise 并完成全部去噪
  -> 解码成多个 MP4
  -> center_cut / trim / blend 合并
```

规则 stride 和尾部 padding 已经解决 VAE 时间相位错位；完整重建与 shared-static render 也使输入条件在真实 overlap 中保持连续。但是当前 `run_splits_pipeline.sh` 仍逐窗口启动独立 inference，而 `merge_split_videos.py` 只在像素输出已经确定之后选择或混合 overlap 帧。后处理无法让两个窗口重新回到同一条生成轨迹，因此接缝仍可能出现：

- 同一人物的纹理、姿态或局部结构不一致；
- 相邻帧的运动方向或速度突变；
- 亮度、色调、背景细节突然变化；
- 简单 average 导致双影，center-cut 则把差异压缩成一次硬切。

FlowLong 的价值在于把一致性约束前移到 `x0_pred`，在整个去噪过程中反复同步窗口，而不是修补最终 RGB。

## 4. 时间窗口与 latent 几何

### 4.1 固定参数

第一版固定/默认参数如下：

| 参数 | 符号 | 默认值 | 说明 |
|---|---:|---:|---|
| 像素窗口长度 | `W` | 49 | Vista4D 原生输入长度 |
| Wan VAE 时间压缩 | `r` | 4 | 仓库当前硬编码值 |
| latent 窗口长度 | `F` | 13 | `(W - 1) / r + 1` |
| 像素 stride |  | 24 | 必须为 4 的倍数 |
| 像素 overlap |  | 25 | `49 - 24` |
| latent stride | `S` | 6 | `24 / 4` |
| latent overlap | `O` | 7 | `F - S` |
| matching 权重 | `lambda` | 线性 | `[0, 1/6, ..., 1]` |
| inference steps |  | 50 | 与现有 Vista4D 默认一致 |
| sigma shift |  | 5.0 | 与现有 Wan scheduler 默认一致 |
| stochastic threshold | `t*` | 0.6 | 论文 v1 未公布数值，作为项目默认值 |
| DiT microbatch |  | 1 | GB10 安全默认值 |

必须验证：

```text
F = (W - 1) // r + 1
S = pixel_stride // r
O = F - S
O >= S
pixel_stride % r == 0
(W - 1) % r == 0
```

若使用当前 `overlap=5`：`F=13, S=11, O=2`，不满足 `O >= S`。FlowLong 入口必须直接报错并提示至少使用 `overlap=25`，不能静默退化成两 latent 帧的 matching。

### 4.2 当前 310 帧样例

使用 `stride=24` 后：

| 窗口 | 像素起点 | 真实范围 | 有效帧 | padding | latent 起点 | latent 范围 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0–48 | 49 | 0 | 0 | 0–12 |
| 1 | 24 | 24–72 | 49 | 0 | 6 | 6–18 |
| 2 | 48 | 48–96 | 49 | 0 | 12 | 12–24 |
| 3 | 72 | 72–120 | 49 | 0 | 18 | 18–30 |
| 4 | 96 | 96–144 | 49 | 0 | 24 | 24–36 |
| 5 | 120 | 120–168 | 49 | 0 | 30 | 30–42 |
| 6 | 144 | 144–192 | 49 | 0 | 36 | 36–48 |
| 7 | 168 | 168–216 | 49 | 0 | 42 | 42–54 |
| 8 | 192 | 192–240 | 49 | 0 | 48 | 48–60 |
| 9 | 216 | 216–264 | 49 | 0 | 54 | 54–66 |
| 10 | 240 | 240–288 | 49 | 0 | 60 | 60–72 |
| 11 | 264 | 264–309 | 46 | 3 | 66 | 66–78 |

全局 latent 长度为：

```text
N_latent = F + (K - 1) * S = 13 + 11 * 6 = 79
N_padded_pixel = 1 + r * (N_latent - 1) = 313
N_valid_pixel = 310
trim_right = 3
```

尾部 padding 只用于满足窗口和 VAE 长度约束。最终输出严格以 manifest 的真实 `end_exclusive` 裁剪，不依赖检测重复帧。

### 4.3 Manifest 约束

FlowLong 只接受 JSON manifest，并要求：

- `manifest_version >= 2`；
- 所有窗口 `num_frames=49`；
- 所有非尾窗 `valid_num_frames=49` 且 `pad_right=0`；
- 只有最后一窗允许 `pad_right>0`；
- 起点等差且 stride 固定为 24；
- 所有起点能被 4 整除；
- 相邻真实范围没有 gap；
- manifest 的 `total_frames`/处理范围能唯一决定最终裁剪长度。

第一版不接受非规则 stride、跳过中间窗口或只运行窗口子集，因为这会破坏全局 latent buffer 的几何关系。

## 5. FlowLong 采样算法

### 5.1 符号映射

Vista4D 当前 `FlowMatchScheduler("Wan")` 使用从噪声到数据的 rectified-flow：

```text
x_t = (1 - t) * x_0 + t * x_1
v_theta = x_1 - x_0
```

对第 `k` 个窗口和当前 shifted sigma `t`：

```text
x0_pred[k] = x_t[k] - t * v[k]
```

这里的 `v[k]` 必须是完成 CFG 后的 velocity：

```text
v[k] = v_negative[k] + cfg_scale * (v_positive[k] - v_negative[k])
```

matching 在 CFG 之后、scheduler step 之前执行。

### 5.2 Predicted-clean aggregation

每对相邻窗口在 7 个 latent overlap 上使用论文线性权重：

```text
lambda[i] = i / (O - 1),  i = 0..O-1
blend[i] = (1 - lambda[i]) * left_x0[i] + lambda[i] * right_x0[i]
```

聚合器创建单一 `x0_global`：

1. 第一个 blending zone 之前的 prefix 复制自窗口 0。
2. 每个相邻窗口的 blending zone 写入上述线性组合。
3. 最后一个 blending zone 之后的 suffix 复制自最后窗口。
4. 本配置 `S=6 < O=7`，相邻 blending zone 会重叠 1 个 latent frame；严格按论文 Appendix A.4 使用 rightmost-pair / last-writer-wins。
5. 聚合使用 float32 计算，写回后再转换为 pipeline dtype。

完成 aggregation 后，不再让两个窗口分别保存 overlap；全局索引只存一份。下一步模型前向前，根据 latent 起点从全局 state 重新切出每个 13 帧窗口，因此 overlap 从此完全共享。

### 5.3 Stochastic early-phase sampling

对 scheduler 当前 sigma `t` 和下一 sigma `s`：

```text
eta_t = 1 if stochastic_enabled and t >= 0.6 else 0
```

高噪声阶段：

```text
epsilon_global ~ N(0, I)
x_s_global = (1 - s) * x0_global + s * epsilon_global
```

低噪声阶段：

```text
x_t_global = 已存在的全局 state
x1_pred_global = (x_t_global - (1 - t) * x0_global) / t
x_s_global = (1 - s) * x0_global + s * x1_pred_global
```

最后一步 `s=0`：

```text
x_final_global = x0_global
```

正式 FlowLong 配置下，第一次 timestep 的窗口噪声彼此独立，且 `t=1 >= t*`，因此该步进入 stochastic 分支并生成第一份一致的全局 state。

为了支持 `matching-only / eta=0` 消融，第一步还要定义一个只用于 deterministic 更新的 `x_t_global`：使用与 clean aggregation 完全相同的几何和线性权重聚合各窗口的初始 `x_t`。这等价于先对每个窗口执行线性的 deterministic 更新，再用同一线性算子聚合 next state。正式 stochastic 路径可以计算该值用于诊断，但不会使用它。启用 stochastic 时必须验证 `0 < t* <= 1`；完全关闭 stochastic 由独立布尔开关控制，不能用非法 threshold 表示。

### 5.4 随机数与复现

对每个用户提供的 `base_seed`：

- 窗口初始噪声和 augmentation noise 使用 `window_seed[k] = base_seed + k`；
- stochastic global noise 使用独立 generator，seed 为 `base_seed + 1_000_003`；
- global generator 按 timestep 顺序消费随机数；
- 多个 base seed 顺序执行，每次重建 generator，不把多个 seed 合成显存 batch；
- report 中记录 base seed、所有 window seed 和 stochastic seed。

默认继续沿用 pipeline 的 `rand_device=cpu`，随后把噪声移动到目标 device，以便与当前 seed 行为一致。

### 5.5 伪代码

```python
geometry = build_flowlong_geometry(manifest, vae_temporal_factor=4)
state = prepare_all_window_conditions_and_independent_noise(...)
global_xt = None

for step_id, timestep in enumerate(scheduler.timesteps):
    t = scheduler.sigmas[step_id]
    s = scheduler.sigmas[step_id + 1] if not last_step else 0.0

    if step_id == 0:
        xt_windows = state.independent_window_noise
        xt_global_for_step = aggregate_window_values(xt_windows, geometry, dtype=float32)
    else:
        xt_windows = slice_global(global_xt, geometry.latent_starts, F=13)
        xt_global_for_step = global_xt

    velocity_windows = []
    for microbatch in windows(batch_size=1):
        v_pos = model(condition=microbatch, latents=xt_windows[microbatch])
        v_neg = model(negative_condition=microbatch, latents=xt_windows[microbatch])
        velocity_windows.append(v_neg + cfg_scale * (v_pos - v_neg))

    x0_windows = xt_windows - t * velocity_windows
    x0_global = aggregate_predicted_clean(x0_windows, geometry, dtype=float32)

    if stochastic_enabled and t >= stochastic_threshold:
        epsilon = randn_like_global(global_generator)
        global_xt = (1 - s) * x0_global + s * epsilon
    else:
        x1_global = (xt_global_for_step - (1 - t) * x0_global) / t
        global_xt = (1 - s) * x0_global + s * x1_global

final_latent = global_xt
decoded = vae.decode(final_latent)
decoded = decoded[:, :, :manifest_valid_frames, :, :]
output = vae_output_to_video(decoded)
```

## 6. 代码架构与接口设计

### 6.1 核心数学模块

新增 `diffsynth/pipelines/flowlong.py`，保持模型无关，不读取文件系统。包含：

```python
@dataclass(frozen=True)
class FlowLongGeometry:
    pixel_window: int
    pixel_stride: int
    pixel_overlap: int
    temporal_factor: int
    latent_window: int
    latent_stride: int
    latent_overlap: int
    pixel_starts: tuple[int, ...]
    latent_starts: tuple[int, ...]
    global_latent_frames: int
    padded_pixel_frames: int
    valid_pixel_frames: int

@dataclass(frozen=True)
class FlowLongSamplingConfig:
    stochastic_threshold: float = 0.6
    stochastic_enabled: bool = True
    microbatch_size: int = 1
    matching_dtype: torch.dtype = torch.float32
```

公开纯函数：

- `build_geometry_from_manifest(manifest, temporal_factor) -> FlowLongGeometry`
- `linear_blend_weights(overlap, device, dtype) -> Tensor`
- `aggregate_window_values(window_values, geometry) -> Tensor`
- `aggregate_predicted_clean(window_x0, geometry) -> Tensor`
- `slice_global_latents(global_latents, geometry) -> Tensor`
- `flowlong_next_state(global_xt, global_x0, t, s, stochastic, generator) -> Tensor`

这些函数不依赖 Vista4D checkpoint，所有几何和公式测试都能在 CPU 完成。

### 6.2 Vista4D pipeline 重构

修改 `diffsynth/pipelines/wan_video_vista4d.py`，先把现有 `__call__` 中的共用步骤抽成私有方法：

- `_prepare_inference_inputs(...)`：设置 scheduler、运行 units，返回 `inputs_shared`、positive/negative inputs 和 batch size；
- `_select_iteration_model(timestep, models, switch_boundary)`：保留 Wan 2.2 双 DiT 切换逻辑；
- `_predict_cfg_velocity(models, shared, positive, negative, timestep, cfg_scale)`：封装当前两次模型前向；
- `_decode_latents(latents, ...)`：统一 VAE decode 和 quantize。

原 `__call__` 必须改为调用这些私有方法，但采样公式、seed、输出和默认参数不能改变。先为原路径补回归测试，再新增：

```python
Vista4DPipeline.generate_flowlong(
    *,
    flowlong_geometry: FlowLongGeometry,
    flowlong_config: FlowLongSamplingConfig,
    base_seed: int,
    ...existing Vista4D conditions...
) -> tuple[list[Image.Image], dict]
```

`generate_flowlong` 的条件 batch 维度等于窗口数 `K`。模型前向按 `microbatch_size` 切 batch；所有 microbatch 必须完成同一个 timestep 后才能进行 aggregation，不能让某个窗口提前进入下一 timestep。

第一版入口验证：

- `cfg_merge` 必须为 `False`；
- `use_usp` 必须为 `False`；
- 所有窗口 prompt 完全相同；
- batch size 必须等于 manifest 窗口数；
- 每个窗口 latent 时间长度必须为 13。

### 6.3 Microbatch 切分规则

新增内部函数 `_slice_batch_value(value, start, end, batch_size)`：

- `Tensor` 且首维为 `batch_size`：切首维；
- `np.ndarray` 且首维为 `batch_size`：切首维；
- list/tuple 且长度为 `batch_size`：切片并保留类型；
- 标量、模型对象、无 batch tensor：原样传递；
- 遇到首维既不是 1、也不是 batch size，但字段被声明为 batch-sensitive 时立即报错。

不要根据所有 kwargs 盲目猜测。为 `model_fn_vista4d` 建立显式的 batch-sensitive 字段集合，至少覆盖：

```text
latents, context, cam_emb, clip_feature, y, y_empty,
source_video_latents, point_cloud_video_latents,
source_mask_latents, point_cloud_mask_latents
```

microbatch 结果按原窗口顺序 concat，确保 geometry 索引稳定。

### 6.4 FlowLong inference 入口

新增 `scripts/inference/inference_flowlong.py`。职责：

1. 读取 JSON manifest，建立并验证 geometry。
2. 根据每条 manifest `output_path` 推导 example name。
3. 从 `--result_root/<example>/<render_folder>` 加载每个窗口的：
   - `video_src.mp4`
   - `video_pc.mp4`
   - source/point-cloud alpha mask
   - source/point-cloud motion mask
   - `cameras_tgt.npz`
4. 验证所有窗口都为 49 个 encoded frames，且最后一窗 padding 元数据与文件一致。
5. 构造 `K` 窗口条件 batch，加载一次 pipeline。
6. 对每个 base seed 顺序调用 `generate_flowlong`。
7. 保存最终视频和 JSON report。

CLI 固定如下：

```text
--manifest PATH                         必填，JSON v2
--result_root PATH                      默认 ./results/flowlong_single
--render_folder NAME                    默认 render_384p_smooth
--output_folder PATH                    必填
--prompt TEXT                           必填
--negative_prompt TEXT                  沿用现有默认值
--height INT                            默认 384
--width INT                             默认 672
--num_frames INT                        必须为 49
--seed INT [INT ...]                    默认 10027；多个 seed 顺序执行
--num_inference_steps INT               默认 50
--sigma_shift FLOAT                     默认 5.0
--cfg_scale FLOAT                       默认 5.0
--flowlong_stochastic_threshold FLOAT   默认 0.6
--flowlong_disable_stochastic           默认关闭；仅用于 matching-only 消融
--flowlong_microbatch_size INT          默认 1
--overwrite                             默认拒绝覆盖
```

模型、tokenizer、Vista4D checkpoint 和 VAE tiling 参数与现有 `inference.py` 保持同名接口，便于 shell wrapper 复用。

### 6.5 Shell wrapper

新增 `scripts/test_video/run_flowlong_inference.sh <source_stem|manifest.json>`：

- 解析 `FLOWLONG_SPLITS_DIR`，默认 `./media/flowlong_splits`；
- 解析 `FLOWLONG_RESULT_ROOT`，默认 `./results/flowlong_single`；
- 解析 `FLOWLONG_OUTPUT_ROOT`，默认 `./results/flowlong`；
- 复用现有 Wan checkpoint、tokenizer、Vista4D config、prompt 和分辨率选择逻辑；
- 显式设置/打印 `USE_USP=false`、threshold、microbatch 和 seed；
- 输出目录固定为：

```text
results/flowlong/<SOURCE_STEM>_vista4d_<RESOLUTION>_smooth/
```

每个 seed 保存：

```text
video_seed=<SEED>.mp4
flowlong_report_seed=<SEED>.json
```

如果输出已存在且没有 `FORCE=true`，跳过对应 seed；`FORCE=true` 映射为 Python 的 `--overwrite`。

### 6.6 Report 内容

`flowlong_report_seed=<SEED>.json` 至少记录：

- 论文版本和实现版本；
- 当前 git commit 与 dirty 状态；
- manifest 绝对路径及 SHA-256；
- 所有像素/latent 起点、窗口长度、stride、overlap、padding；
- 模型、checkpoint、prompt hash；
- base/window/stochastic seeds；
- steps、sigma shift、CFG、threshold、microbatch；
- 每步 matching 前的 overlap MAE/max error；
- aggregation 后的 overlap max error；
- 每步耗时、总耗时、CUDA peak allocated/reserved；
- 解码帧数、有效帧数和裁剪帧数；
- 输出 MP4 路径。

默认只保存标量，不保存每步 latent。启用 `--save_step_diagnostics` 时，可保存少量指定 timestep 的低分辨率 decoded preview，但禁止默认写出全部 latent，以免占满磁盘。

## 7. 条件数据准备与运行目录

### 7.1 与当前结果隔离

FlowLong 使用单独目录，避免覆盖 `overlap=5` 基线：

```text
media/flowlong_splits/
results/flowlong_single/
results/flowlong/
logs/flowlong/
```

完整重建和完整 shared-static render 继续复用：

```text
results/full/1778135019043_stitched_384p/
```

因此不需要重新运行 DA3 或 SAM3。

### 7.2 当前样例准备命令

生成独立的 25-overlap manifest 和 5 个输入切片：

```bash
VIDEO=data/1778135019043.mp4
TARGET="$(basename "${VIDEO%.*}")"

OUTPUT_DIR=./media/flowlong_splits \
RESOLUTION=384p \
CLIP_FRAMES=49 \
OVERLAP=25 \
TEMPORAL_ALIGNMENT=4 \
bash scripts/test_video/split_video.sh "$VIDEO"
```

复用完整 shared-static render，只按新 manifest 切出 5 组 FlowLong 条件：

```bash
SPLITS_DIR=./media/flowlong_splits \
OUTPUT_RESULT_ROOT=./results/flowlong_single \
OVERWRITE_SPLITS=true \
bash scripts/test_video/render_shared_static_and_slice.sh "$TARGET"
```

实现完成后的联合推理命令：

```bash
LOCAL_WAN_FOLDER=./checkpoints \
FLOWLONG_SPLITS_DIR=./media/flowlong_splits \
FLOWLONG_RESULT_ROOT=./results/flowlong_single \
FLOWLONG_OUTPUT_ROOT=./results/flowlong \
FLOWLONG_STOCHASTIC_THRESHOLD=0.6 \
FLOWLONG_MICROBATCH_SIZE=1 \
SEEDS=10027 \
bash scripts/test_video/run_flowlong_inference.sh "$TARGET"
```

FlowLong 最终输出已经是单一全局解码视频，不再运行 `merge_splits.sh`。

## 8. 测试计划

### 8.1 CPU 单元测试

新增 `tests/test_flowlong_geometry.py`：

1. 310 帧、49 窗口、25 overlap 得到像素起点 `[0,24,48,...,264]`，共 12 个窗口。
2. 得到 latent geometry `F=13,S=6,O=7,N=37`。
3. 尾窗为 34 valid + 15 padding，最终 trim 为 15。
4. 当前 5 overlap 被拒绝，并明确报告 `F=13,S=11,O=2,O<S`。
5. 非 4 对齐起点、非规则 stride、中间 padding 和缺失窗口均被拒绝。

新增 `tests/test_flowlong_aggregation.py`：

1. 两窗口标量 tensor 能产生权重 `[0,1/6,...,1]`。
2. blend 左端严格等于左窗口，右端严格等于右窗口。
3. `S<O` 的重叠 blending zone 使用 rightmost-pair last-writer-wins。
4. 聚合再切片后，相邻 overlap 完全相等。
5. 输入 bf16 时聚合在 float32 完成，输出 dtype 恢复正确。
6. padding 只影响全局尾部，不覆盖真实帧位置。

新增 `tests/test_flowlong_scheduler.py`：

1. matching 关闭且 `eta=0` 时，新公式与当前 Euler scheduler step 数值等价。
2. `t>=t*` 时使用 stochastic 分支；`t<t*` 时使用 deterministic 分支。
3. 同 seed 完全可复现，不同 seed 产生不同 global noise。
4. 最后 `s=0` 时输出严格等于 matched `x0_global`。
5. 第一 deterministic step 使用聚合后的 `x_t_global`，结果与“逐窗更新后再聚合”等价。
6. 启用 stochastic 时非法 threshold 被拒绝；关闭 stochastic 时不消费 global stochastic generator。

新增 `tests/test_flowlong_microbatch.py`：

1. 用 mock model 验证 microbatch 1、2、K 的 velocity 顺序和结果一致。
2. positive/negative CFG 与原 pipeline 公式一致。
3. batch-sensitive 字段完整切分；未知不合法 batch shape 立即报错。

### 8.2 原 pipeline 回归测试

在重构 `Vista4DPipeline.__call__` 前，用 mock DiT/VAE 固化以下行为：

- 相同 seed 和输入得到相同 noise；
- scheduler timestep 数量和顺序不变；
- CFG 两次调用顺序不变；
- 原 `__call__` 最终 latent 与重构后逐元素一致；
- quantized/floatpoint 输出接口不变。

FlowLong 代码合入的前提是这些回归测试全部通过。

### 8.3 GPU smoke test

按以下顺序降低调试成本：

1. 两窗口、2 inference steps、384p、microbatch 1，验证完整模型前向和全局 decode。
2. 五窗口、2 steps，验证显存不会随窗口数线性增长到 OOM。
3. 五窗口、50 steps，生成正式候选。

GPU smoke test 必须记录：

- 单窗口模型前向 peak memory；
- 五窗口联合推理 peak memory；
- 每 timestep 的 K 次 forward 总耗时；
- VAE 全局 79-latent decode 的 peak memory。

如果 VAE 一次解码 79 latent OOM，第一降级方案是给 VAE decoder 增加时间 chunk/cache 解码，而不是恢复逐窗口解码；逐窗口 decode 会重新引入 VAE 边界。

## 9. 实施阶段与交付物

### 阶段 0：冻结基线

- 使用当前代码重新生成 `overlap=5` 的三窗口 inference 和 center-cut 输出。
- 保存命令、manifest、seed、prompt、checkpoint hash。
- 不再以已删除或参数不完整的旧 merged MP4 作为对照。

交付物：`results/flowlong_eval/baseline/` 和 baseline report。

### 阶段 1：核心数学和几何

- 实现 `FlowLongGeometry`、manifest validation、线性 aggregation、global slice 和 stochastic/deterministic next-state。
- 完成全部 CPU 单元测试。
- 此阶段不加载 Vista4D 模型。

交付物：核心模块和 CPU tests。

### 阶段 2：pipeline 安全重构

- 为现有 `Vista4DPipeline.__call__` 建立 mock 回归测试。
- 抽取 prepare、CFG prediction、DiT switch 和 decode 私有方法。
- 确认原单窗口 inference 行为不变。

交付物：重构后的 pipeline 和 baseline regression tests。

### 阶段 3：FlowLong 联合 sampler

- 实现 `generate_flowlong`。
- 实现显式 batch 字段切分和 microbatch。
- 加入每步 aggregation、binary stochastic phase、全局 RNG 和全局 decode。
- 加入 report 指标采集。

交付物：可由 tensor 条件调用的联合 sampler。

### 阶段 4：仓库工作流接入

- 实现 `inference_flowlong.py` 和 shell wrapper。
- 创建隔离的 flowlong splits/conditions/output/logs 路径。
- 更新 README 长视频章节，明确 FlowLong 不运行 merge。

交付物：从 manifest 到最终 MP4 的用户命令。

## 10. 风险与明确处理策略

### 10.1 论文没有公开实现与完整超参数

风险：官方仓库目前没有 sampler 代码，论文 v1 只给出 binary schedule，没有给出 `t*` 的实验取值。

处理：严格实现论文公式；项目默认 `0.6`，并把 `{0.5,0.6,0.7}` sweep 设为验收必做项。report 必须记录阈值，避免结果不可追溯。

### 10.2 Vista4D 比论文 T2V 多出强时序条件

风险：每个窗口的 source/point-cloud latent 由 causal VAE 独立编码，即使 RGB overlap 一致，窗口边缘的 condition latent 也可能因上下文不同而略有差异。

处理：第一版优先保持训练时的逐窗口编码分布，通过 x0 matching 克服输出分歧，并在 diagnostics 中记录 encoded condition overlap MAE。若它成为主要残差，第二版再增加“全局条件编码后按 latent 切片”的显式实验分支；不得在第一版默认开启这一分布外改动。

### 10.3 计算时间增加

风险：窗口从 3 个增加到 5 个，DiT forward 数量约为当前的 1.67 倍。

处理：模型和 prompt/condition embedding 只加载/计算一次；每 timestep 内窗口 microbatch 顺序执行。report 记录分阶段耗时。第一版不以牺牲 overlap 几何为代价缩短时间。

### 10.4 显存

风险：14B DiT 若直接以 `K=5` batch 前向会超过 GB10 显存。

处理：默认且推荐 `microbatch_size=1`；只在用户显式配置时允许更大值。所有窗口条件 latent 很小，可以驻留；大 activation 每次只保留一个 microbatch。每次 forward 后及时释放临时引用，但不在 timestep 内反复卸载 DiT。

### 10.5 Stochastic phase 降低条件遵循或细节

风险：高噪声阶段反复注入新噪声可能让相机条件或人物细节变弱。

处理：只在 `t>=t*` 使用 stochastic，后期保持 ODE；必须做 matching-only 消融和阈值 sweep。若 full FlowLong seam 更好但质量不达标，不直接关闭 matching，而是提高 `t*`、缩短 stochastic 阶段后重测。

### 10.6 全局 VAE decode

风险：当前常用路径只解码 13 latent frame，FlowLong 要解码 79 latent frame。

处理：优先使用现有 spatial tiled decode；先做 2-step smoke 测 peak memory。如果仍 OOM，实现保持 causal cache 的 temporal chunk decode。禁止退回逐窗口 decode + MP4 merge作为正式 FlowLong 输出。

### 10.7 尾部 padding 污染

风险：DiT 是双向时序注意力，最后 15 个重复条件帧可能影响尾部真实帧。

处理：tail padding 仍是不可避免的 native-window 输入；report 记录有效帧数与裁掉的 padding 数量，不允许 padding 帧写入最终视频。后续可实验把最后窗口向左多覆盖，但不能破坏 4 对齐和规则 stride。

## 11. 完成定义

以下条件全部满足后，FlowLong 实现才算完成：

- 核心几何、aggregation、scheduler、microbatch 和原 pipeline 回归测试全部通过。
- FlowLong 对不合法 5-overlap manifest 明确失败，对 25-overlap manifest 正确运行。
- 14B Vista4D 在 GB10 上以 microbatch 1 完成 5 窗口、50-step 联合采样。
- 全局 VAE 一次解码成功，最终视频恰好 310 帧，无 merge 步骤。
- report 证明每步 aggregation 后 overlap latent 一致。
- README 包含准备、运行、恢复和 baseline 生成命令。
- 现有单窗口及独立 split inference 默认路径保持兼容，用户可以随时切回基线。
