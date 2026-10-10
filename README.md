# JEPA Reshoot：基于官方 Manifold4D 的研究开发

本仓库现在以 [ManifoldTechLtd/Manifold4D](https://github.com/ManifoldTechLtd/Manifold4D) 的官方实现作为 baseline 和后续开发基础。模型、预处理、点云渲染和推理源码采用官方目录结构；原 Vista4D / FlowLong / DiffSynth 实现已完整归档到 [archive/vista4d](archive/README.md)。

研究方向见 [reshoot-vjepa2.1-plan2.md](docs/reshoot-vjepa2.1-plan2.md)：通过目标视角对齐的 JEPA 预测构造生成起始分布。当前阶段先建立官方 baseline，后续再接入预测分支与对应训练。

## 源码与目录

引入版本为官方提交 `cadeac5a6a7341de1ba76f770b3d07a020611fff`，日期 2026-10-09。官方文件保持原内容，来源和 SHA-256 记录在 [docs/upstream/manifold4d.json](docs/upstream/manifold4d.json)。官方 README 保存为 [README.upstream.md](README.upstream.md)。

| 路径 | 用途 |
| --- | --- |
| `manifold4d/model14b/` | 官方 Wan2.1-T2V-14B 双流模型 |
| `manifold4d/infer.py` | 官方权重加载、初始化和扩散采样 |
| `manifold4d/generate.py` | 从预处理场景生成新视角视频 |
| `manifold4d/demo.py` | 使用官方预渲染示例输入 |
| `manifold4d/rendering/` | 官方点云处理与渲染 |
| `scripts/preprocess/` | VGGT-Omega、caption、T5、SAM3 和轨迹准备 |
| `configs/manifold4d.yaml` | 官方模型参数与资源路径 |
| `assets/demo/` | 官方三个示例场景与小型条件数据 |
| `external/` | 固定版本的外部源码依赖 |
| `checkpoints/`、`datasets/` | 本地权重与数据集，Git 忽略 |
| `docs/` | 当前研究计划、论文笔记和 baseline 说明 |
| `archive/vista4d/` | 原项目源码、文档、配置与测试快照 |

## 开发准备

本机用于开发。源码准备脚本仅检出依赖源码，不安装软件或下载权重：

```bash
# 查看固定版本与将要执行的源码准备步骤。
python3 scripts/setup_external.py wan --dry-run

# 官方生成器所需的 Wan2.1 源码。
python3 scripts/setup_external.py wan

# 完整预处理链需要时，另外准备 VGGT-Omega 和 SAM3。
python3 scripts/setup_external.py vggt sam3
```

依赖版本记录在 [external/sources.json](external/sources.json)。准备脚本保留已有检出；已有目录版本不同或存在本地修改时会停止，不覆盖代码。

训练 / 推理机器的环境安装参考 [官方安装说明](README.upstream.md#installation)。本项目的主 `requirements.txt` 和 `setup.py` 已采用官方文件。主模型安装只发现 `manifold4d` 包，归档源码不参与默认安装；默认 pytest 搜索也排除归档目录。

## 权重与推理入口

已下载权重继续使用原路径：

```text
checkpoints/manifold4d/
  conditioning_modules.pt
  camera_encoder.pt
  self_attn_full.pt
checkpoints/wan/Wan2.1-T2V-14B/
```

换到运行机器后，可使用官方入口：

```bash
python -m manifold4d.demo \
  --demo_dir assets/demo/woman-phone \
  --checkpoint checkpoints/manifold4d \
  --output_dir output/demo_woman_phone
```

完整场景输入、预处理及生成命令见 [官方 README](README.upstream.md) 和 [baseline 说明](docs/manifold4d_baseline.md)。资源路径沿用官方配置，支持 CLI 和环境变量覆盖。

## 训练开发边界

官方发布内容包括推理代码和权重，尚不包含训练入口。本次迁移建立官方源码开发基础；后续训练需要围绕官方模型接入监督样本、flow 目标、分布式训练、断点恢复与验证。

原有最小训练损失与缓存逻辑保存在 [归档训练代码](archive/vista4d/diffsynth/pipelines/manifold4d_training.py)，可作为后续迁移参考。当前缓存仅存预计算相机射线，官方模型接收相机位姿与内参，接入训练时需要明确调整该接口。JEPA prior 应由训练和推理共用的初始化接口实现。

## 许可

官方代码采用 [Apache-2.0](LICENSE)。官方权重和外部模型各自的使用条件见 [官方许可说明](README.upstream.md#license)。官方示例条件数据随源码保留；完整数据集、模型权重、外部检出和运行产物均不纳入 Git。
