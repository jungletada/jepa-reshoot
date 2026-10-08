# Vista4D Paper Notes: 4D Reconstruction and Diffusion Inference

本文整理两个问题：

1. Vista4D 如何通过 segmentation mask 和 depth 构建 4D point cloud。
2. Vista4D 如何基于 Wan2.1 做 flow matching finetuning 与 inference。

本文同时记录当前长视频实验采用的统一流程：完整序列先运行 DA3/SAM3 和相机平滑，再按 manifest 切分所有 49 帧 conditions。旧的“每个 49 帧切片分别 reconstruction、segmentation 和 smoothing”流程不再使用。

对应代码位于：

- `/home/peng/code/Vista4D/scripts/preprocess/recon_and_seg_single.py`
- `/home/peng/code/Vista4D/scripts/preprocess/render_single.py`
- `/home/peng/code/Vista4D/utils/point_cloud/`
- `/home/peng/code/Vista4D/diffsynth/pipelines/wan_video_vista4d.py`
- `/home/peng/code/Vista4D/diffsynth/models/latent_encoder.py`
- `/home/peng/code/Vista4D/diffsynth/models/wan_video_dit.py`

## 0. 当前长视频统一流程：Full DA3/SAM3 First

### 0.1 为什么不再逐片运行 reconstruction

旧流程是：

```text
完整视频
-> 切成多个 49 帧视频
-> 每段分别运行 Pi3 + SAM3
-> 每段分别平滑 cameras.npz
-> render / diffusion / merge
```

这种做法会让 overlap 中同一个全局帧得到两套独立 conditions：

- Pi3 每次都重新估计 depth scale、focal、shift 和 camera coordinate system。
- 每段 `cam_c2w` 都被独立对齐到该段第一帧。
- SAM3 在每段开头重新初始化，动态 mask 的边界和跟踪结果可能变化。
- Gaussian smoothing 在每个 49 帧窗口分别执行，overlap 中的 target camera 也不相同。

即使输入 RGB 帧相同，这些 condition 差异也会放大分段 Vista4D 生成结果的不连续。

当前统一流程改为：

```text
原始长视频
-> 生成 split manifest 和 49 帧输入切片
-> 对 manifest 覆盖的完整时间范围统一 crop/resize
-> 完整序列一次运行 DA3 + SAM3
-> 对完整 cam_c2w 一次 Gaussian smoothing
-> 按 manifest 同步切分 RGB/depth/mask/raw camera/smoothed camera
-> 每段分别 render 和 Vista4D diffusion
-> center hard cut 合并
```

默认 reconstruction 现在是：

```bash
RECON_METHOD=da3
DA3_MODEL_ID=./checkpoints/DA3NESTED-GIANT-LARGE-1.1
DA3_PROCESS_RES=448
```

### 0.2 完整序列输出

统一入口会先生成一个经过相同 center crop/resize 的完整工作视频：

```text
media/full/<source>_frames<start>_<end>_384p<N>.mp4
```

随后完整运行 `recon_and_seg_single.py`，保存到：

```text
results/full/<source>_frames<start>_<end>_384p<N>_<recon_method>/recon_and_seg/
  video.mp4
  depths/
  dynamic_mask/
  sky_mask/
  cameras.npz
  cameras_gaussian_smooth.npz
```

完整 reconstruction 结果按方法隔离，例如目录后缀为 `_da3` 或 `_pi3`。裁剪后的完整 RGB 视频仍由两种方法共享，避免切换 `RECON_METHOD` 时误复用另一种方法的相机和深度缓存。

这里的 `cameras.npz` 和 `cameras_gaussian_smooth.npz` 都处于同一个完整序列 world coordinate system。相机 smoothing 必须在切片前完成。

### 0.3 Condition 切片

代码：

```text
scripts/preprocess/slice_full_recon_by_manifest.py
```

它读取完整 reconstruction 和 JSON manifest，然后对每个 `start_frame..end_frame` 同步切出：

```text
results/single/<split>_384p49/recon_and_seg/
  video.mp4
  depths/                  # 49 个 EXR
  dynamic_mask/            # 49 个 PNG
  sky_mask/                # 49 个 PNG
  cameras.npz              # 完整 raw trajectory 的对应 49 帧
  cameras_gaussian_smooth.npz
  clips.json
  full_sequence_slice.json # provenance 和全局帧区间
```

切片时不会对每段 camera 单独 re-anchor，也不会重新 smoothing。因此 overlap 中相同全局帧的 depth、mask、raw camera 和 smoothed camera 数值完全一致。

每段 `video.mp4` 默认使用 H.264 `yuv444p + CRF 0` 保存。原因是普通有损 H.264 会让两个切片中的同一 overlap RGB 帧因 GOP 上下文不同而产生细小像素差；无损保存保证两个切片解码后的 overlap RGB 也完全一致。

### 0.4 使用命令

统一 Shell 入口是：

```text
scripts/test_video/prepare_full_recon_and_slice.sh
```

先生成 split videos 和 manifest：

```bash
cd /home/peng/code/Vista4D
conda activate vista4d

bash scripts/test_video/split_video.sh ./data/1778135019043.mp4
```

然后只运行一次完整 DA3/SAM3、完整相机 smoothing，并切分 conditions：

```bash
bash scripts/test_video/run_splits_recon.sh 1778135019043
```

如果相同名称下已经存在旧的逐片 reconstruction，需要明确允许替换各切片的 `recon_and_seg` 文件夹：

```bash
OVERWRITE_SPLITS=true \
bash scripts/test_video/run_splits_recon.sh 1778135019043
```

如果换到另一台服务器后 manifest 里的 LHR 原始视频绝对路径不可用，可以显式覆盖：

```bash
SOURCE_VIDEO=/local/path/1778135019043.mp4 \
bash scripts/test_video/run_splits_recon.sh 1778135019043
```

完整预处理后，逐片 render：

```bash
bash scripts/test_video/run_splits_render.sh 1778135019043
```

在具有足够显存的推理服务器上逐片运行 diffusion：

```bash
USE_USP=true NUM_GPUS=8 \
bash scripts/test_video/run_splits_inference.sh 1778135019043
```

默认的 `run_splits_pipeline.sh` 现在只负责 `render inference`。它不再接受 `recon` 或 `smooth`，以防误回到逐片预处理。

批处理的跳过逻辑会检查依赖文件时间：render 必须比对应 camera condition 新，diffusion 必须比 render condition 新。因此替换完整序列 conditions 后，已有但过期的 render/inference 不会被误用。

如果只调整了 smoothing 参数，可以复用完整 DA3/SAM3 结果，重新平滑完整轨迹并覆盖 condition slices：

```bash
TRANSLATION_SIGMA=8 ROTATION_SIGMA=10 \
bash scripts/test_video/run_splits_smooth.sh 1778135019043
```

### 0.5 关键约束和 Pi3X 回退

1. JSON manifest 中的 `[start_frame, end_exclusive)` 必须与完整工作视频严格对应。
2. DA3、SAM3 和 camera smoothing 都只在完整序列上运行一次。
3. 每段的 source camera 和 target smoothed camera 必须从相同完整坐标系切出。
4. 不要再对切片后的 `cameras.npz` 单独运行 `smooth.sh`。
5. 需要复现实验中的 Pi3X 路径时，显式设置 `RECON_METHOD=pi3`。

DA3NESTED-GIANT-LARGE-1.1 会把完整序列一次送入 AnyView ViT-Giant 的跨帧 global attention，再运行 Metric ViT-Large。官方的 DPT head 虽然默认按 8 帧分块，但 backbone 不分块，因此显存随帧数和空间 token 数显著增长。在 48 GB RTX A6000 上，304 帧实测 `DA3_PROCESS_RES=504` 峰值为 47,715 MiB，而 `448` 峰值为 40,241 MiB。完整长视频入口因此默认使用 448，最终 depth 和 sky mask 仍会缩放回 672x384。

Pi3X 回退路径的 point/conf convolution heads 默认使用 `PI3_HEAD_CHUNK_SIZE=16` 沿帧维分块解码，用来规避长序列在卷积 head 中触发的 32-bit tensor index 上限。这个分块不拆分 temporal transformer，也不会改变完整序列的全局时序建模。

### 0.6 显存不足时：先分段 reconstruction，再统一对齐和平滑

如果完整 DA3/Pi3 reconstruction 无法装入显存，可以先对 manifest 中的每个 49 帧切片独立运行 reconstruction 和 segmentation，然后执行：

```bash
bash scripts/test_video/stitch_splits_smooth_and_slice.sh 1778135019043
```

`stitch_split_recon_by_manifest.py` 不会直接拼接各段的局部 C2W。它利用相邻切片的重叠帧估计 Sim(3)：

1. 从重叠帧的静态、非天空有效 depth 中估计尺度。
2. 对重叠 C2W 的相对 rotation 求 SO(3) 均值。
3. 用缩放和旋转后的相机中心估计 world translation。
4. 递归地把所有切片映射到 split000 的 world coordinate system。
5. RGB、depth 和 mask 在 overlap 使用 center owner hard cut；相机观测按切片中心权重求均值。
6. 在合并后的完整 raw camera trajectory 上只运行一次 Gaussian smoothing。
7. 最后按原 JSON manifest 切回 49 帧 conditions。

默认输出写到 `results/stitched_single/`，不会覆盖 `results/single/` 中独立 reconstruction 的原件。确认结果后，可写回标准 render 路径：

```bash
FORCE_STITCH=true OVERWRITE_SPLITS=true \
OUTPUT_RESULT_ROOT=./results/single \
bash scripts/test_video/stitch_splits_smooth_and_slice.sh 1778135019043
```

实现只依赖 manifest 中的切片列表，不假设固定为 3 段；实际 7 段会按相同方式顺序对齐。

## 1. Segmentation Mask + Depth 如何构建 4D Point Cloud

Vista4D 里的 4D reconstruction 不是优化一个 NeRF 或 4D Gaussian scene，而是更直接：

```text
RGB video
+ per-frame depth
+ per-frame source camera pose
+ intrinsics
+ dynamic/static/sky masks
-> world-space colored point cloud
-> per-point temporal visibility
```

这里的「4D」主要来自每个 3D point 的时间可见性。动态点只在来源帧可见，静态点在整个 clip 的目标时间轴上持续可见。

### 1.1 Reconstruction 产出 depth / camera / intrinsics

入口是：

```text
scripts/preprocess/recon_and_seg_single.py
```

这个脚本先加载 source video，裁成指定帧数并 resize：

```python
video_src, fps = load_video(args.video_path)
video_src = slice_center_frames(video_src, args.num_frames)
video_src = crop_and_resize_video(video_src, args.height, args.width)
```

然后用 Pi3X 或 DA3 做单目/视频重建。

### Pi3X 路径

代码：

```text
utils/recon_and_seg/recon_pi3.py
```

核心逻辑：

```python
results = model(frames[None])
local_points = results["local_points"][0]
focal, shift = recover_focal_shift(local_points)

depths = local_points[..., 2] + shift[..., None, None]
cam_c2w = results["camera_poses"][0]
intrinsics = focal_to_intrinsics(focal, height=height_new, width=width_new)

# Align camera poses so first frame is identity transformation
cam_c2w = np.linalg.inv(cam_c2w[0])[None] @ cam_c2w
```

Pi3X 输出每帧局部点云和相机位姿，Vista4D 从 `local_points[..., 2]` 加上 recovered shift 得到 depth，并把所有 camera pose 归一到第一帧坐标系。

### DA3 路径

代码：

```text
utils/recon_and_seg/recon_da3.py
```

核心逻辑：

```python
prediction = model.inference([...], process_res=process_res, export_format="npz")

depths = prediction.depth
mask_sky = prediction.sky
cam_w2c = prediction.extrinsics
intrinsics = K_to_intrinsics(prediction.intrinsics)

cam_w2c_44 = np.zeros((num_frames, 4, 4), dtype=np.float32)
cam_w2c_44[:, :3, :4] = cam_w2c
cam_w2c_44[:, 3, 3] = 1.0
cam_c2w = np.linalg.inv(cam_w2c_44)

# Align camera poses so first frame is identity transformation
cam_c2w = np.linalg.inv(cam_c2w[0])[None] @ cam_c2w
```

DA3 直接给 depth、sky mask、world-to-camera extrinsics 和 intrinsics。Vista4D 把 W2C 转成 C2W，再同样归一到第一帧。

最后 `recon_and_seg_single.py` 保存：

```text
recon_and_seg/
  video.mp4
  cameras.npz       # cam_c2w, intrinsics
  depths/
  dynamic_mask/
  sky_mask/
  clips.json        # optional, DSE 场景会用
```

## 1.2 Segmentation mask 的作用

动态 mask 由 SAM3 根据关键词生成：

```python
dynamic_mask, seg_frames = run_sam3_video(video, sam3_video_predictor, args.seg_keywords)
```

语义是：

```text
dynamic_mask: 会动的主体，例如 person / dog / ball / car
static_mask:  静态背景，默认等于 ~dynamic_mask
sky_mask:     天空，特殊处理为极远静态背景
```

加载 `recon_and_seg` 文件夹时，如果没有显式 `static_mask/`，代码会默认：

```python
static_mask = ~dynamic_mask
```

位置：

```text
utils/media.py::load_recon_and_seg
```

也就是说，segmentation mask 不是直接估计几何，而是决定每个 RGB-D 像素 lift 成 point 之后的时间行为。

## 1.3 Preprocess: 收缩 mask、过滤 depth outlier、处理 sky

代码：

```text
utils/point_cloud/preprocess.py
```

关键函数：

```python
preprocess_scene(...)
```

它做几件事：

1. 按目标帧窗口切片。
2. resize video/depth/mask/intrinsics 到目标分辨率。
3. 对 static mask 做 erosion/contract，避免动态物体边缘被误当静态背景。
4. 对 dynamic/static 区域分别做 depth outlier filtering。
5. 从 dynamic mask 中移除 sky。
6. 把 sky 合并进 static。
7. 把 sky depth 设成 `SKY_DEPTH = 1e3`。

核心代码：

```python
if contract_masks:
    static_mask = contract_mask(static_mask, radius=6, iterations=3)

dynamic_mask = dynamic_mask & ~get_depths_outliers(depths, dynamic_mask)
static_mask = static_mask & ~get_depths_outliers(depths, static_mask)

dynamic_mask = dynamic_mask & ~sky_mask
static_mask = static_mask | sky_mask
depths[sky_mask] = SKY_DEPTH
```

这样处理的目的：

- 动态物体边缘不要污染静态背景点云。
- depth 异常点不要成为错误 floating points。
- sky 没有可靠 depth，所以放到很远处，并作为静态背景参与渲染。

## 1.4 Unprojection: RGB-D 像素变成 world-space colored points

代码：

```text
utils/point_cloud/point_cloud.py::unproject
```

对每一帧每个像素，先构造像素中心：

```python
pixel_coords = [u + 0.5, v + 0.5, 1]
```

再用内参和 depth 反投影到相机坐标：

```python
points_cam = (pixel_coords @ inverse(K).mT) * depths[..., None]
```

再用 source camera C2W 变到世界坐标：

```python
R = cam_c2w[:, :3, :3]
T = cam_c2w[:, None, :3, 3]
points_world = (points_cam @ R.mT) + T
```

每个保留下来的像素成为：

```text
color:    source RGB pixel
position: world-space xyz
origin:   [frame, y, x]
```

保留条件是：

```python
mask_flat = (dynamic_mask | static_mask).view(-1)
```

## 1.5 关键：4D temporal visibility

Vista4D 的 4D 表达关键在 `visible`：

```python
visible = torch.zeros(num_points, num_frames)
visible.scatter_(1, origin_frame_index, True)
visible = (visible & dynamic_mask[..., None]) | static_mask[..., None]
```

展开成语义：

```text
dynamic point:
  只在它来源的那一帧 visible=True

static point:
  在所有帧 visible=True
```

这就是论文里 “temporally-persistent static points” 在代码中的落点。

如果一个人被正确分到 dynamic mask，他不会作为静态点残留在所有时间步里；如果背景被分到 static，它会跨整个 clip 被复用，形成更完整的新视角几何上下文。

## 1.6 Rendering: 用 target camera 渲染 point cloud condition

代码入口：

```text
scripts/preprocess/render_single.py
```

读取目标相机：

```python
cam_c2w_tgt, intrinsics_tgt = load_cameras(args.cam_path)
```

调用：

```python
render_video(
    video=video_src,
    depths=depths_src,
    cam_c2w_src=cam_c2w_src,
    cam_c2w_tgt=cam_c2w_tgt,
    intrinsics_src=intrinsics_src,
    intrinsics_tgt=intrinsics_tgt,
    dynamic_mask=dynamic_mask_src,
    static_mask=static_mask_src,
)
```

渲染时每个目标帧只取当前可见的点：

```python
visible_i = visible[:, i]
points_pos_i = points_pos[visible_i]
points_color_i = points_color[visible_i]
```

然后投影到目标 camera，并用 z-buffer + bilinear splatting 合成：

```text
video_pc.mp4
depths_pc/
alpha_mask_pc/
dynamic_mask_pc/
```

这些文件就是后续 diffusion inference 的强条件输入。

## 1.7 小结

Vista4D 的 4D reconstruction 可以概括为：

```text
per-frame RGB-D + source camera
+ dynamic/static/sky segmentation
-> world-space colored points
+ temporal visibility
-> target-camera point-cloud render
```

segmentation mask 的核心作用是决定时间行为：

```text
static/background/sky points: temporally persistent
dynamic/foreground points:   frame-local
```

这比只给 diffusion 一个 target camera embedding 强很多，因为模型不只知道“相机怎么走”，还看到“这条相机下几何上大概应该看到什么”。

---

## 2. Vista4D Flow Matching Finetuning 与 Inference

Vista4D diffusion 部分是在 `Wan2.1-T2V-14B` 上 finetune。它把 Wan 的 DiT 改造成一个可以同时看：

```text
noisy target latent
point-cloud render latent
source video latent
source/point-cloud masks
target camera Plucker embedding
text prompt
```

的 conditional video denoiser。

## 2.1 Finetuning 的目标

训练目标仍然是 flow matching。通用 loss 在：

```text
diffsynth/diffusion/loss.py::FlowMatchSFTLoss
```

核心逻辑：

```python
timestep_id = torch.randint(...)
timestep = pipe.scheduler.timesteps[timestep_id]

noise = torch.randn_like(inputs["input_latents"])
inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

noise_pred = pipe.model_fn(..., timestep=timestep)
loss = mse_loss(noise_pred, training_target)
loss = loss * pipe.scheduler.training_weight(timestep)
```

也就是说：

```text
真实 target video -> VAE latent x0
随机噪声 epsilon
随机 timestep t
加噪得到 x_t
模型预测 flow / velocity target
MSE 监督
```

训练时的 condition 是：

```text
source video
point-cloud render under target camera
alpha/motion masks
target camera
prompt
```

目标是还原 target video。

论文里强调，训练不是只用干净 synthetic render，而是使用 noisy reconstructed multiview data。也就是说 point-cloud render 本身会有深度错误、洞、错位、动态物体 artifact，模型被训练成既利用几何条件，又学会修这些 artifact。

## 2.2 对 Wan DiT 的架构改造

入口：

```text
diffsynth/pipelines/wan_video_vista4d.py::Vista4DPipeline.from_pretrained
```

代码会先加载 Wan base model：

```python
pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
pipe.dit = model_pool.fetch_model("wan_video_dit", index=2)
pipe.vae = model_pool.fetch_model("wan_video_vae")
pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
```

然后应用 Vista4D config：

```yaml
dit:
  positional_embedding_offset: 31
  latent_encoder:
    source_init_mode: wan_patch_embed
    point_cloud_init_mode: wan_patch_embed
    mask_init_mode: zero_init
    use_source_masks: True
    use_point_cloud_masks: True
  augmentation:
    source_noise_level: 0.0
    point_cloud_noise_level: 0.0
    image_noise_level: 0.0
```

checkpoint:

```text
checkpoints/vista4d/720p49_step=3000/dit.pth
```

### 2.2.1 LatentEncoder

代码：

```text
diffsynth/models/latent_encoder.py
```

`LatentEncoder` 维护三套 patch embedding：

```text
output_patch_embedding       # noisy target latent
source_patch_embedding       # source video latent + source masks
point_cloud_patch_embedding  # point-cloud video latent + point-cloud masks
```

初始化方式：

```text
output:      frozen Wan patch embedding
source RGB:  initialized from Wan patch embedding
point RGB:   initialized from Wan patch embedding
masks:       zero-init patch embedding
```

zero-init mask branch 的意义是：刚开始 finetune 时 mask 不会破坏 Wan 原来的行为，模型可以逐步学会使用 mask。

### 2.2.2 Camera encoder

每个 DiT block 加：

```python
block.cam_encoder = torch.nn.Linear(6, dim)
block.projector = torch.nn.Linear(dim, dim)
```

初始化：

```python
cam_encoder.weight = 0
cam_encoder.bias = 0
projector.weight = I
projector.bias = 0
```

代码在：

```text
diffsynth/pipelines/wan_video_vista4d.py::from_pretrained
```

这样一开始模型几乎等价于原 Wan，camera conditioning 是渐进学进去的。

## 2.3 条件如何进入模型

### 2.3.1 Source / Point-cloud video 走 VAE latent

代码：

```text
WanVideoUnit_Vista4DVideoInput.encode_videos
```

逻辑：

```python
source_video_latents = pipe.vae.encode(source_video)
point_cloud_video_latents = pipe.vae.encode(point_cloud_video)
```

也就是说，source video 和 point-cloud render 都先进入 Wan 的 VAE latent 空间。

### 2.3.2 Masks 直接打包成 latent patch 格式

mask 不走 VAE，而是用 `shuffle_mask` 变成和 latent patch 对齐的 channel-packed tensor。

输入：

```text
source_alpha_mask
source_motion_mask
point_cloud_alpha_mask
point_cloud_motion_mask
```

打包后形状近似为：

```text
b (2 * 4 * 8 * 8) f h w
```

其中：

```text
2: alpha mask + motion mask
4: Wan VAE temporal compression
8x8: Wan VAE spatial compression
```

代码：

```python
masks = rearrange(
    masks,
    "b c (f sf) (h sh) (w sw) -> b (c sf sh sw) f h w",
    sf=4, sh=8, sw=8,
)
```

### 2.3.3 Target camera 转成 Plucker embedding

代码：

```text
diffsynth/utils/vista4d/camera.py::get_plucker_embedding
```

输入：

```text
intrinsics: b f 4, [fx, fy, cx, cy]
cam_c2w:    b f 4 4
```

对每个 latent spatial patch 位置计算 camera ray：

```python
directions = normalize([(i - cx) / fx, (j - cy) / fy, 1])
rays_d = directions @ R_c2w.T
rays_o = cam_c2w[..., :3, 3]
rays_dxo = cross(rays_o, rays_d)
plucker = concat([rays_dxo, rays_d])
```

输出：

```text
cam_emb: b f h w 6
```

然后按 Wan 时间压缩 stride 4 下采样：

```python
cam_emb = cam_emb[:, ::pipe.time_division_factor]
```

对于 49 frames：

```text
49 frames -> 13 latent time steps
```

## 2.4 In-context conditioning: 三段 token 拼接

最关键的模型函数：

```text
diffsynth/pipelines/wan_video_vista4d.py::model_fn_vista4d
```

先 patchify 三组 latent：

```python
x, source_latents, point_cloud_latents, (f, h, w) = dit.latent_encoder(
    dit.patch_embedding,
    x,
    source_video_latents,
    source_mask_latents,
    point_cloud_video_latents,
    point_cloud_mask_latents,
)
```

然后拼成一条 self-attention 序列：

```python
x = torch.cat((x, point_cloud_latents, source_latents), dim=1)
```

语义：

```text
[ noisy target tokens | point-cloud condition tokens | source video tokens ]
```

这就是 Vista4D 的 in-context conditioning。它不是简单把 point cloud 当 ControlNet，也不是只 cross-attend source，而是让 noisy target / point cloud / source 在 DiT self-attention 中直接交互。

camera embedding 也复制三份，对齐三段 token：

```python
cam_emb = rearrange(cam_emb, "b f h w d -> b (f h w) d")
cam_emb = cam_emb.repeat(1, 3, 1)
```

## 2.5 Camera conditioning 在 DiT block 里怎么用

代码：

```text
diffsynth/models/wan_video_dit.py::DiTBlock.forward
```

核心：

```python
input_x = modulate(norm1(x), shift_msa, scale_msa)

if cam_emb is not None and hasattr(self, "cam_encoder"):
    cam_emb = self.cam_encoder(cam_emb)
    input_x = input_x + cam_emb

x_self_attn = self.self_attn(input_x, freqs)

if hasattr(self, "projector"):
    x_self_attn = self.projector(x_self_attn)

x = self.gate(x, gate_msa, x_self_attn)
x = x + self.cross_attn(norm3(x), context)
x = self.gate(x, gate_mlp, ffn(...))
```

所以 camera 是加到 self-attention 输入上，而 text prompt 仍通过 cross-attention context 进入。

## 2.6 Inference 流程

入口：

```text
scripts/inference/inference.py
```

inference 不直接吃裸 source video，而是吃 point-cloud render 后的 input folder：

```text
render_720p/
  video_src.mp4
  video_pc.mp4
  cameras_tgt.npz
  alpha_mask_src/
  dynamic_mask_src/
  alpha_mask_pc/
  dynamic_mask_pc/
```

加载逻辑：

```python
video_src, fps = load_video("video_src.mp4")
video_pc, _ = load_video("video_pc.mp4")
cam_c2w_tgt, intrinsics_tgt = load_cameras("cameras_tgt.npz")
```

然后构造输入：

```python
inputs = {
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "source_video": video_src,
    "point_cloud_video": video_pc,
    "source_alpha_mask": alpha_mask_src,
    "source_motion_mask": dynamic_mask_src,
    "point_cloud_alpha_mask": alpha_mask_pc,
    "point_cloud_motion_mask": dynamic_mask_pc,
    "target_cam_c2w": cam_c2w_tgt,
    "target_intrinsics": intrinsics_tgt,
}
```

pipeline inference：

```python
videos = pipe(**inputs, seed=args.seed, cfg_merge=args.cfg_merge, tiled=args.tile_vae)
```

## 2.7 Denoising loop

在：

```text
Vista4DPipeline.__call__
```

先生成初始 noise latent：

```python
inputs_shared["latents"] = noise
```

然后 50-step denoise：

```python
for timestep in self.scheduler.timesteps:
    noise_pred_posi = self.model_fn(..., prompt condition ...)

    if cfg_scale != 1.0:
        noise_pred_nega = self.model_fn(..., negative prompt ...)
        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
    else:
        noise_pred = noise_pred_posi

    latents = self.scheduler.step(noise_pred, timestep, latents)
```

最后 VAE decode：

```python
video = self.vae.decode(latents)
```

## 2.8 训练与推理的一句话对照

训练：

```text
target video latent x0
-> add flow-matching noise
-> DiT predicts training target
conditioned on source video + point-cloud render + masks + camera + prompt
```

推理：

```text
random noise latent
-> DiT denoises for 50 steps
conditioned on source video + point-cloud render + masks + camera + prompt
-> VAE decode to output video
```

## 2.9 和 ReCamMaster 的关键区别

ReCamMaster 更接近：

```text
source video latent + target camera embedding
```

Vista4D 是：

```text
source video latent
+ target-camera point-cloud render latent
+ alpha/motion masks
+ target camera Plucker rays
+ prompt
```

ReCamMaster 只告诉模型“相机应该怎么动”；Vista4D 先用 4D point cloud 渲染出“目标相机下几何上大概应该看到什么”，再让 diffusion model 补洞、修 artifact、生成未观测区域。

这也是它 camera control 更强的核心原因。
