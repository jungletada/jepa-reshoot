# 官方 Manifold4D baseline

本项目的主实现已切换至 [官方 Manifold4D](https://github.com/ManifoldTechLtd/Manifold4D)，固定源码提交为 `cadeac5a6a7341de1ba76f770b3d07a020611fff`。引入清单见 [upstream/manifold4d.json](upstream/manifold4d.json)，官方使用说明见 [README.upstream.md](../README.upstream.md)。

## 实现来源

| 部分 | 主实现 |
| --- | --- |
| Wan2.1 双流 DiT | [manifold4d_14b.py](../manifold4d/model14b/manifold4d_14b.py) |
| 相机条件和 Plücker 射线 | [camera_encoder.py](../manifold4d/model/camera_encoder.py)、[helpers_14b.py](../manifold4d/model14b/helpers_14b.py) |
| 模型加载、Manifold 初始化、CFG 和 UniPC | [infer.py](../manifold4d/infer.py) |
| 完整场景生成 | [generate.py](../manifold4d/generate.py) |
| 预渲染条件示例 | [demo.py](../manifold4d/demo.py) |
| 模型与路径配置 | [manifold4d.yaml](../configs/manifold4d.yaml) |

官方实际发布路径使用 Wan2.1 的 16 通道 VAE latent；覆盖率与运动掩码池化为两通道。双流联合处理目标输出和源视频，相机条件逐层注入。目标点云渲染的 RGB 用于构造采样起点，渲染掩码仍作为去噪条件。

## 输入与资源

官方 `generate` 入口读取源帧、VGGT-Omega 深度与相机、动态掩码、T5 embedding 和目标轨迹；官方 `demo` 入口读取已渲染的源 / 目标条件视频、掩码、相机和文本 embedding，省去几何预处理。

模型权重路径为 `checkpoints/manifold4d/` 和 `checkpoints/wan/Wan2.1-T2V-14B/`。Wan2.1 源码路径为 `external/Wan2.1/`，用 `python3 scripts/setup_external.py wan` 检出。完整预处理需要另外准备 VGGT-Omega、SAM3 及相应模型资源，参见官方说明。

配置沿用官方 `paths`、`model`、`inference` 结构。官方 forward 接收源 / 目标相机和共享内参 `K`；后续接入不同内参的训练数据时，需要显式设计适配。帧采样、尺寸变换、相机坐标、掩码池化和负提示词都应按官方入口记录。

## 训练与研究扩展

官方代码不包含训练入口；本次迁移未将旧 DiffSynth 训练器接入官方模型。后续开发需要补充：

1. 同步源 / 目标监督对、按场景划分的数据清单，以及仅基于源观测生成的几何条件。
2. 面向官方模型的缓存和批处理接口，包括原始相机、内参及文本长度。
3. 与起始分布匹配的 flow 目标、条件 dropout、区域损失和数据增强。
4. 分布式训练、完整断点恢复和固定协议验证。
5. 训练和推理共用的初始化构造接口，供后续 JEPA prior 使用。

旧最小训练实现保存于 [archive/vista4d](../archive/README.md)，用于参考和迁移。当前研究计划见 [reshoot-vjepa2.1-plan2.md](reshoot-vjepa2.1-plan2.md)，论文解读见 [readpaper/manifold4d.md](readpaper/manifold4d.md)。

## 验证范围

本次迁移在开发机器上检查源码与上游的一致性、归档完整性、Python / Bash 语法、包发现和开发脚本行为。完整 14B 模型训练、视频推理及论文指标需要后续在运行机器上验证。
