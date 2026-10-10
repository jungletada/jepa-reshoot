# Manifold4D baseline

本实现对应 [reshoot-vjepa2.1-plan2.md](reshoot-vjepa2.1-plan2.md) 中先建立 Manifold4D baseline 的步骤：复用当前项目的重建、分割和目标视角点云渲染，替换生成器的起始分布和视觉条件结构。已有 Vista4D、JEPA adapter 和 FlowLong 入口独立保留。

参考 [Manifold4D 原文](https://arxiv.org/abs/2608.28174)、[官方代码](https://github.com/ManifoldTechLtd/Manifold4D/tree/14e1989e0e854a1c05eae0cf09533a7558adf20b)（固定 commit `14e1989e0e854a1c05eae0cf09533a7558adf20b`）和 [本地解读](readpaper/manifold4d.md)。网络适配参考官方 Apache-2.0 实现，[许可副本](third_party/Manifold4D-LICENSE.txt)随仓库保留。

## 算法与实现对应

| 内容 | 实现 |
| --- | --- |
| Eq. 4：`x1 = alpha * (z_render + sigma * epsilon) + (1-alpha) * epsilon`，两个分支共享同一份噪声，默认 `sigma=0.3` | `diffsynth/pipelines/manifold4d.py::manifold_prior` |
| Eq. 5：`xt=(1-t)*z_target+t*x1`，速度目标 `x1-z_target`，从高时间向低时间采样 | `flow_path`、`manifold_training_loss` |
| 两条 token 流 `[output, source]`；没有持续输入的渲染 RGB token 流 | `diffsynth/models/wan_video_manifold4d.py::Manifold4DModel` |
| 四个 patch embedding：输出 RGB、输出覆盖/运动掩码、源 RGB、源覆盖/运动掩码 | `output_rgb_patch_embed` 等 |
| 原生 Wan2.1 VAE：16 通道、空间压缩 8 倍、时间压缩 4 倍；首帧独立池化 | `pool_video_mask`、`prepare_inputs` |
| 官方 avgpool 掩码：在 VAE 网格池化为 2 通道，再通过掩码卷积 patchify | `render_mask`、`source_mask` |
| 目标和源相机各自的 Plücker 射线；相机帧选择 `0,4,8,...` | `camera_tokens`、`prepare_inputs` |
| RoPE 源流偏移 31；源流默认使用真实条件噪声时间，推理为 0 | `two_stream_freqs`、`cond_stream_t=honest` |
| 每块零初始化相机线性层和恒等初始化 attention projector | `Manifold4DModel` |
| 文本 CFG：正负分支共享源视频、掩码、相机和当前状态 | `sample_manifold` |
| 默认 flow UniPC；显式设置 Wan 官方时间网格；另提供 Euler 消融 | `make_scheduler` |
| Appendix B：覆盖/空洞区域分别归一化，动态区域双倍损失权重，独立条件 dropout | `balanced_flow_loss`、`manifold_training_loss` |

官方发布模型在去噪时仍读取覆盖/运动掩码，渲染 **RGB** 只参与起始状态。训练必须使用这个起始状态对应的 flow 目标；仅修改 Vista4D 的输入噪声不构成已训练的 Manifold4D baseline。

默认推理参数见 [configs/manifold4d.yaml](../configs/manifold4d.yaml)：384×672、49 帧、50 步、CFG 5、shift 5、UniPC。

## 权重和依赖

在现有 `vista4d` 环境安装本次新增的 scheduler 依赖：

```bash
conda activate vista4d
pip install diffusers==0.41.0
```

准备本地 [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) 基座和 [Manifold4D 增量权重](https://huggingface.co/manifoldtech/Manifold4D)：

```bash
hf download Wan-AI/Wan2.1-T2V-14B --local-dir checkpoints/Wan-AI/Wan2.1-T2V-14B
hf download manifoldtech/Manifold4D --local-dir checkpoints/manifold4d \
  --include conditioning_modules.pt camera_encoder.pt self_attn_full.pt
```

新增代码会校验三份权重的必需模块、键名、尺寸和 schema；缺文件或误用 Vista4D 权重时直接报错。代码没有自动启动权重下载。

```text
checkpoints/Wan-AI/Wan2.1-T2V-14B/
  diffusion_pytorch_model*.safetensors
  models_t5_umt5-xxl-enc-bf16.pth
  Wan2.1_VAE.pth
  google/umt5-xxl/
checkpoints/manifold4d/
  conditioning_modules.pt
  camera_encoder.pt
  self_attn_full.pt
```

`checkpoints/`、`datasets/`、`data/` 和 `results/` 已被 Git 忽略。训练缓存建议放到 `data/manifold4d/`，避免提交大文件。

## 输入和推理

先通过现有 `scripts/preprocess/render_single.py` 或 shared-static 渲染流程准备输入目录。无需切换几何前端。必须提供真实覆盖掩码和源相机，不能通过渲染黑色像素猜测空洞，也不能用目标相机替代源相机。

```text
results/.../render/
  video_src.mp4
  video_pc.mp4
  cameras_src.npz            # cam_c2w [T,4,4]；intrinsics [T,4]，像素单位 fx/fy/cx/cy
  cameras_tgt.npz
  alpha_mask_pc/*.png        # 必需，渲染覆盖
  alpha_mask_src/*.png       # 可选，默认全覆盖
  dynamic_mask_src/*.png     # 可选，默认无运动掩码
  dynamic_mask_pc/*.png      # 可选，默认无运动掩码
```

视频、相机和掩码必须共用时间轴，源视频和渲染视频 FPS 必须一致。输入较长时所有输入统一取中心窗口，输入不足时明确报错。空间中心裁剪同步更新相机内参；掩码保持平均后的部分覆盖值。

只检查输入，不加载模型：

```bash
python -m scripts.inference.inference_manifold4d \
  --input_folder results/your_clip/render --validate_only
```

生成 baseline：

```bash
python -m scripts.inference.inference_manifold4d \
  --wan_checkpoint checkpoints/Wan-AI/Wan2.1-T2V-14B \
  --manifold4d_checkpoint checkpoints/manifold4d \
  --input_folder results/your_clip/render \
  --output_folder results/your_clip/manifold4d \
  --prompt "Describe the source scene and its motion." \
  --vram_preset balanced --vram_limit 10 --tile_vae \
  --seed 42
```

输出 `video_seed=42.mp4`、生成网格对应的源/目标相机，以及 `run_metadata.json`（模型路径、输入路径、参数、seed、前端说明和参考 commit）。多 seed 可以传 `--seed 42 43`。`--cfg_merge` 合并 CFG 的两次前向，增加峰值显存；`--solver euler` 为采样器消融；`--prior_sigma 0` 为初始化噪声强度消融。

`balanced` 使用已有 CPU offload，并让新增的大型 projector 采用相同策略；`full` 保持模型驻 GPU。该入口使用 bf16，不提供 FP8 或 USP。完整 14B 在原生分辨率下仍需大量模型和 attention 显存；小模型验证不能保证单张 16 GB GPU 可运行完整 49 帧推理。

## 监督训练入口

提供先缓存冻结 VAE/T5、再训练生成器的最小入口。训练目标视频仅进入 VAE teacher 和损失，不进入源条件或渲染条件。需要自行准备同步的源视角/目标视角监督对及基于源数据的渲染。Vista4D-Eval-Data 没有真实目标视角 RGB，不能直接作为这些监督对；DyCheck 的相机与视频也须先转换到上述格式。

配对 manifest 的路径相对于 manifest 所在目录，例如 `data/manifold4d/pairs.jsonl`：

```json
{"input_folder":"pairs/scene001/render","target_video":"pairs/scene001/target.mp4","target_motion_mask":"pairs/scene001/target_dynamic","prompt":"A person walks through the scene."}
```

`target_motion_mask` 可选；提供后启用目标坐标下的动态区域双倍损失权重。缓存仅加载基座的 VAE/T5，不加载 14B DiT：

```bash
python -m scripts.manifold4d.prepare_training \
  --wan_checkpoint checkpoints/Wan-AI/Wan2.1-T2V-14B \
  --manifest data/manifold4d/pairs.jsonl \
  --output data/manifold4d/cache --tile_vae

python -m scripts.manifold4d.train \
  --wan_checkpoint checkpoints/Wan-AI/Wan2.1-T2V-14B \
  --manifest data/manifold4d/cache/manifest.jsonl \
  --output checkpoints/manifold4d_custom \
  --vram_preset full --lr 1e-5 --epochs 1
```

训练仅开放 self-attention Q/K/V/O 及其 RMSNorm、四个 patch embedding、相机编码器和 projector；其余基座冻结。默认均匀采样时间 `t∈[0,1]`；`--time_shift` 可显式调整时间分布。独立 dropout 默认 0.1，全无条件样本默认 0.05；丢弃 render 时覆盖设为 0，恢复 Gaussian endpoint。导出与官方相同的三份增量权重和 `config.yaml`。

传 `--manifold4d_checkpoint` 可以从已训练模块开启新的微调；当前入口不恢复 optimizer 状态。这个入口实现训练目标和单进程训练循环，没有复现论文的五源混合数据准备、点云腐蚀、double-reprojection、时间反转和 8 卡 FSDP 训练系统。大规模 14B 训练仍需配置相应的分布式训练和显存策略。

## 验证与比较边界

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_manifold4d.py -v
```

测试覆盖共享噪声、0/1/部分覆盖、首帧时序、速度符号、区域损失、两条 token 流、条件时间、训练梯度范围、官方格式权重往返、完整推理编排、多 seed 和 offload wrapper。

开发验证还在固定官方 commit 上比较了带非零相机/掩码权重的小型网络：统一 CPU SDPA、fp32 权重和输入时，最大输出误差约 `6.6e-7`；显式 sigma 网格下，diffusers UniPC 与 Wan 官方 FlowUniPC 的测试轨迹一致。CUDA bf16 小模型前向和带 gradient checkpointing 的反向检查通过。

当前默认几何来自本项目已有 DA3/Pi3 与渲染流程；原文使用 VGGT-Omega。应把结果记为 **Manifold4D generator + 当前项目几何前端**，在同一 render、分辨率、帧数、步数和 seed 下比较 Vista4D 与 Manifold4D。完整官方权重的 14B 视频推理和论文指标尚未验证，不能据小模型测试宣称完整论文复现。
