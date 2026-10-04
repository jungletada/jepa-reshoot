# 方案 1：实现分析与实验入口

本实现对应 [方案 1](reshoot-jepa-plan1.md#方案-1jepa-结构条件--vista4dmanifold4d-生成先验oracle预测生成)。这是可训练的实验链路，尚无真实数据上的效果结论。文中“能减轻遮挡误差”等描述应视为待验证假设。

## 设计判断

1. **先做 Oracle，但必须训练新增分支。** 现有 Vista4D checkpoint 不认识 JEPA 特征，直接挂随机 cross-attention 不能检验特征是否有效。冻结原生成器，先在多视角训练集上用目标视频特征训练适配器，再在独立场景上比较 Oracle 与基线。零初始化保证起点保持基线，不能把初始化后的相同输出当成实验收益。
2. **教师的预训练 predictor 不等于新视角预测器。** 本实现仅使用冻结教师的密集编码结果，另行训练接受源特征、源深度和相机的预测器。官方 V-JEPA 2/2.1 hub 返回 encoder 与预训练 predictor；这里只加载 encoder 的本地权重。接口依据 [官方 hub](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py) 和 [2.1 encoder](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py)。
3. **世界状态先采用可实现的近似。** 源特征网格通过源深度反投影，附加坐标与时间，再由固定数量的 learned memory tokens 聚合。它不是完整、去重的动态 4D 点云世界模型。目标查询包含相机射线、时间、从源点云渲染得到的可见性。没有目标 RGB/深度输入。
4. **Oracle→预测存在条件分布偏差。** 预测器训练后，适配器需要继续使用预测特征训练，或混合 Oracle/预测缓存训练。训练文件支持每个样本列出多种条件并均匀抽取。最终可部署的结论必须来自 `predicted` 模式。
5. **优先复用 Vista4D。** 当前仓库没有 Manifold4D。生成器的源视频、点云视频、相机条件和 FlowLong 联合去噪保留；新增分支只作用于生成视频 token。

## 代码路径和数据契约

| 部件 | 代码 | 行为 |
| --- | --- | --- |
| 冻结教师 | `jepa/teacher.py` | 使用本地官方代码和 checkpoint，`eval` + stop-gradient，保留全部帧；奇数长度只复制末帧补齐 tubelet |
| 缓存 | `jepa/features.py` | `B,C,T,H,W` 特征、原帧坐标、视频尺寸、教师权重/预处理标识和来源角色 |
| 预测器 | `jepa/predictor.py` | 源 RGB-D 特征 → 全时段 memory → 目标相机查询 → 密集特征；通道归一化 L2 监督 |
| 生成器适配器 | `jepa/adapter.py` | 选定 DiT 层的逐时间片 cross-attention、空间位置编码、零初始化输出投影 |
| 生成器训练 | `jepa/training.py`、`scripts/jepa/train_adapter.py` | 冻结 Vista4D，仅训练适配器，使用原 flow-matching schedule 与加权速度损失 |
| 推理 | `scripts/inference/inference.py`、`inference_flowlong.py` | 显式选择无条件/源特征/投影/Oracle/预测模式 |

教师特征按 tubelet 中心记录时间，例如 49 帧、tubelet=2 对应 `[0.5, 2.5, …, 46.5, 48]`。缓存加载时按真实帧坐标插值到 Wan 的 `[0,4,…,48]`，边界使用最近值。相机和深度在教师时间位置取最近帧。所有视频必须已经按生成器尺寸准备好，避免随后中心裁剪导致特征/相机错位。

适配器默认在 40 层 DiT 的第 `9,19,29,39` 层后注入，内部宽度 256，8 个头；每个时间片将教师空间网格平均池化到至多 `8×8`，再供生成 token 查询。保留二维位置，不进行全部视频 token 的全局交叉注意力，以控制显存。预测器跨时间聚合源信息，适配器本身逐时间片注入。

几何使用 OpenCV 约定（x 向右、y 向下、z 向前），`c2w`，像素单位内参 `(fx,fy,cx,cy)`；源和目标相机必须在同一重建世界系。坐标变换到第一帧源相机系，以源深度中位数归一化。`target_visibility` 必须来自**源重建的点云渲染**，不能从目标真值构造。缓存角色能防止常见误用，不能证明数据制作者没有混入目标信息。

## 准备特征和几何

使用现有 Vista4D 环境和模型。另在仓库外准备 [官方 V-JEPA 代码](https://github.com/facebookresearch/vjepa2) 和对应 checkpoint，安装其依赖；本仓库不自动克隆代码或下载权重。下面以 V-JEPA 2.1 ViT-L 为例，其 checkpoint 使用 `ema_encoder`；其它架构按官方 checkpoint 指定 `--architecture`、`--checkpoint-key`，不自动猜测或忽略缺失权重。

假设一个已准备的 49 帧、384×672 样本位于 `results/jepa/clip01/`：`render/` 是现有渲染产物（源 RGB、源深度、源/目标相机、点云 RGB、alpha mask）；`target.mp4` 是同步多视角目标真值。真实训练需多个独立场景；示例清单中的单行只说明格式。

```bash
python -m scripts.jepa.extract_features \
  --video results/jepa/clip01/render/video_src.mp4 --role source \
  --teacher-repo /path/to/vjepa2 --checkpoint /path/to/vjepa2_1_vitl.pt \
  --output results/jepa/clip01/source.pt

python -m scripts.jepa.extract_features \
  --video results/jepa/clip01/target.mp4 --role oracle \
  --teacher-repo /path/to/vjepa2 --checkpoint /path/to/vjepa2_1_vitl.pt \
  --output results/jepa/clip01/oracle.pt

python -m scripts.jepa.prepare_geometry \
  --render-folder results/jepa/clip01/render \
  --source results/jepa/clip01/source.pt \
  --output results/jepa/clip01/geometry.pt
```

教师预处理为 RGB `[0,1]`、保持整个视野的双线性 resize、ImageNet 均值/标准差归一化；默认 `--size 224 384`，可调整，但所有对照和训练必须统一。这里的完整视野矩形 resize 是本实验的选择，不能假定与某项官方 benchmark 的 crop 策略等价。架构、权重 SHA256、checkpoint key、resize 和 patch/tubelet 参数均写入 `encoder_id`。

`prepare_geometry` 仅打包渲染目录内的 `depths_src/`、`cameras_src.npz`、`cameras_tgt.npz`、`alpha_mask_pc/`。自定义数据也可保存六个同名概念的 tensor 字段：`source_depth[B,T,H,W]`、`source_c2w[B,T,4,4]`、`source_intrinsics[B,T,4]`、`target_c2w`、`target_intrinsics`、`target_visibility[B,T,H,W]`。具体字段名见 `prepare_geometry()`。源深度是相机 z 深度，不能替换成沿射线的欧氏距离。

## 第一阶段：Oracle 适配器

训练清单见 [oracle_train.example.jsonl](../configs/jepa/oracle_train.example.jsonl)，其中路径相对于清单所在目录。目标视频仅用于 VAE 编码后构造 flow-matching 训练目标，不传给生成器的源条件。下面公共参数可同时供适配器训练和推理使用（Bash）：

```bash
MODEL_ARGS=(
  --model_id_with_origin_paths 'Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan2.1-T2V-14B:Wan2.1_VAE.pth'
  --tokenizer_id_with_origin_path 'Wan2.1-T2V-14B:google/*'
  --local_model_folder ./checkpoints/wan
  --vista4d_checkpoint ./checkpoints/vista4d/384p49_step=30000
  --vista4d_config_path ./checkpoints/vista4d/384p49_step=30000/config.yaml
)

python -m scripts.jepa.train_adapter "${MODEL_ARGS[@]}" \
  --manifest configs/jepa/oracle_train.example.jsonl \
  --output results/jepa/oracle_adapter.pt --epochs 10 --tile_vae
```

训练采用单样本更新、FP32 适配器参数，基础模型 BF16，梯度 checkpointing。当前要求单 DiT、`--vram_preset full`，不支持训练时 USP、FP8 或 block offload。每个 epoch 重新编码输入视频，优先保证训练语义清晰；尚未增加缓存、分布式训练、优化器恢复或学习率调度。`--init-adapter` 加载权重用于下一阶段，不恢复优化器状态。

## 第二阶段：源条件预测与适配器继续训练

训练清单见 [predictor_train.example.jsonl](../configs/jepa/predictor_train.example.jsonl)，每行包含 `source`、`target`（Oracle 缓存，仅作监督）、`geometry`。教师、时间坐标、尺寸必须一致。不同相机样本需同步；训练/验证按场景划分，避免相邻片段泄露。

```bash
python -m scripts.jepa.train_predictor \
  --manifest configs/jepa/predictor_train.example.jsonl \
  --output results/jepa/predictor.pt --epochs 20

python -m scripts.jepa.predict_features \
  --checkpoint results/jepa/predictor.pt \
  --source results/jepa/clip01/source.pt --geometry results/jepa/clip01/geometry.pt \
  --output results/jepa/clip01/predicted.pt

python -m scripts.jepa.train_adapter "${MODEL_ARGS[@]}" \
  --manifest configs/jepa/mixed_train.example.jsonl \
  --init-adapter results/jepa/oracle_adapter.pt \
  --output results/jepa/mixed_adapter.pt --epochs 10 --tile_vae
```

预测器训练在源点云渲染的孔洞处增加损失权重：`1 + occlusion_weight * (1 - visibility)`，默认额外权重 1。这不是遮挡真值指标。方案中的 GEOM/TEXTURE 没有给出可计算定义、对应标签或损失权重，本次只实现明确的特征回归；没有用一个任意损失冒充这两项。

## 推理和消融

```bash
python -m scripts.inference.inference "${MODEL_ARGS[@]}" \
  --input_folder results/jepa/clip01/render \
  --output_folder results/jepa/clip01/generated_predicted \
  --prompt 'A realistic video of the scene.' --seed 10027 --tile_vae \
  --jepa_mode predicted --jepa_features results/jepa/clip01/predicted.pt \
  --jepa_adapter results/jepa/mixed_adapter.pt
```

| 模式 | 条件来源 | 实验含义 |
| --- | --- | --- |
| `none`（默认） | 无 JEPA | 原有基线 |
| `oracle` | 真实目标视频的教师特征 | 条件信息上界实验，不能作为源视频部署结果 |
| `source` | 源视频特征，未投影 | 无几何映射的特征对照 |
| `projected` | 对源点云渲染的 `video_pc.mp4` 执行 `extract_features --role projected` | 先几何投影、再编码的对照 |
| `predicted` | 新预测器输出 | 仅源信息推理 |

`source`/`projected` 对照应训练对应适配器或把相应缓存加入条件混合清单；直接替换 Oracle 适配器的输入会混入分布差异因素。比较时固定数据划分、prompt、seed、训练预算和原始 Vista4D 条件。`--jepa_scale 0` 可检查分支关闭行为；正负 CFG 分支共享同一结构条件，CFG 仍主要引导文本。

FlowLong 的 `scripts.inference.inference_flowlong` 同样接受这四个 `--jepa_*` 参数，`--jepa_features` 需按 manifest 顺序提供**每窗一个缓存**，缓存包含完整 49 帧（尾窗使用与视频相同的末帧 padding）。新增特征参与 microbatch 切分；原有窗口耦合/解码不变。教师对不同窗的编码可能在重叠区不同，当前没有强制跨窗特征一致性。统一视频 YAML 和旧 Bash 包装器暂未增加 JEPA 字段，实验请使用上述 Python 入口。

## 验证范围与下一步

```bash
python -m unittest tests.test_jepa tests.test_jepa_pipeline -v
```

测试覆盖奇数帧教师编码、时间坐标插值、相机反投影、Oracle 来源误用拒绝、stop-gradient、预测器小样本优化、checkpoint 保存/加载、真实小型 Wan DiT 上的零初始化等价与适配器梯度、多个 seed 的 CFG 合并等价、FlowLong microbatch、CPU offload wrapper。

没有在本机运行真实 V-JEPA/Vista4D 权重和 GPU 多视角训练，尚不能说明 FID/LPIPS、遮挡重建或相机控制得到改善。下一步是在独立验证场景完成 Oracle 适配器实验，再决定是否增加预测器训练规模与明确的几何监督。本次没有新增抽帧、插帧、独立条件校验或质量评估脚本。
