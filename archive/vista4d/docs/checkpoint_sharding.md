# Vista4D 无损 checkpoint 分片

支持原始单个 `.pth`、单个 `.safetensors`、标准 `*.safetensors.index.json`，或包含唯一 checkpoint 入口的目录。多片文件必须通过索引加载，不支持随意 glob 后合并；目录内同时有 `.pth` 和索引时会报歧义，请传明确路径或使用独立目录。

## 1. 转换（按需手动执行）

在已安装仓库依赖的环境、仓库根目录执行。以下命令会读取真实权重并创建新目录，**不是 dry-run**；本次代码开发和 CPU 测试没有执行它们。

```bash
conda activate vista4d-pgx

# 保留 FP32，每片 tensor 数据最多 4 GB，输出到独立目录，不修改原文件。
CUDA_VISIBLE_DEVICES='' python -m scripts.convert_vista4d_checkpoint \
  --input checkpoints/vista4d/384p49_step=30000/dit.pth \
  --output checkpoints/vista4d/384p49_step=30000_sharded \
  --max-shard-size 4GB

CUDA_VISIBLE_DEVICES='' python -m scripts.convert_vista4d_checkpoint \
  --input checkpoints/vista4d/720p49_step=3000/dit.pth \
  --output checkpoints/vista4d/720p49_step=3000_sharded \
  --max-shard-size 4GB
```

`4GB` 是 4,000,000,000 字节；`4GiB` 是 4×1024³ 字节。限制指 tensor payload，文件头会增加少量字节。单个 tensor 超过上限时报错，不拆 tensor。默认上限为 `4GB`。

转换工具：

- 不改变任何 dtype；FP32 输入仍为 FP32，特殊浮点值、非连续张量和重复引用的键也会保留。
- 自动复制相邻的 `config.yaml`，可用 `--config` 指定其他配置文件。
- 先写临时目录，逐 tensor 检查名称、shape、dtype 和原始字节完全一致，通过后发布结果；失败只清理工具自己的临时目录。
- 输出目录已存在时拒绝操作，不覆盖、不自动删除原始 checkpoint。
- 原 `.pth` 优先 mmap 读取；每次转换一片，校验也逐片读取。仍需要 CPU 内存、磁盘缓存和足够空间，不保证总内存峰值只有 4 GB。

```text
384p49_step=30000_sharded/
├── config.yaml
├── conversion_report.json
├── model.safetensors.index.json
├── model-00001-of-000NN.safetensors
└── ...
```

总大小不会明显减少。保留原件时，需额外约一份 checkpoint 的磁盘空间；分片也不会直接降低模型推理显存或去噪时间。

## 2. 接入现有推理

以下命令会启动正式推理，须在对应前处理完成、GPU 空闲后另行执行。

```bash
# 统一多视频入口：通过 CLI 选择目录，环境变量覆盖会被该入口清理。
python -m scripts.test_video.run_video_experiment \
  --video ./data/my_video.mp4 --resolution 384p --seed 10027 \
  --vista4d-folder ./checkpoints/vista4d/384p49_step=30000_sharded \
  --stages inference --phases matching_only --execute

# 底层 wrapper：VISTA4D_FOLDER 同时提供权重和 config.yaml。
VISTA4D_FOLDER=./checkpoints/vista4d/384p49_step=30000_sharded \
RESOLUTION=384p SOURCE_VIDEO=./media/single/my_clip_384p49.mp4 \
bash scripts/test_video/inference.sh
```

FlowLong、Stage-4、官方示例脚本也支持 `VISTA4D_FOLDER`。底层 wrapper 可另用 `VISTA4D_CHECKPOINT` 明确指定 `.pth`、单个 `.safetensors` 或索引文件；配置仍从 `VISTA4D_FOLDER/config.yaml` 读取。直接调用 Python 推理入口时，`--vista4d_checkpoint` 接受上述文件或目录，`--vista4d_config_path` 仍需单独指定。

720p 必须使用 720p 权重和配置，不能只改输出宽高。未配置新目录时，原始 `.pth` 默认路径继续有效。

## 3. 加载、指纹与已有结果

加载器先检查全部分片 header 与索引的键集合，再检查模型参数形状和包装层映射，最后逐片复制权重。Vista4D 是对 Wan 基座的**部分权重覆盖**：未出现在 checkpoint 中的基座参数保持不变；意外键、索引遗漏或重复、形状不符会报错。

```bash
# 只读：解析实际入口；info / sha256 会读取所有权重字节计算指纹，但不加载模型。
python -m utils.vista4d_checkpoint resolve checkpoints/vista4d/384p49_step=30000
python -m utils.vista4d_checkpoint info checkpoints/vista4d/384p49_step=30000_sharded
```

- 单个 `.pth` / `.safetensors`：指纹仍是文件 SHA-256，旧 `.pth` 报告和契约保持兼容。
- 分片包：指纹覆盖索引及全部引用分片的名称、大小和 SHA-256；大小为索引加引用分片的总大小。不包含 `config.yaml` 或转换报告。
- 转换报告的 `source_sha256` 是追溯信息，不被当作分片包的真实性证明。
- **同一权重换存储格式后，包指纹会改变。** 不会自动将旧 `.pth` 生成结果视为新包结果；保留旧实验原样，新包使用新输出目录或独立实验命名，不要直接覆盖报告里的 hash。
- 高级选项 `--vista4d_checkpoint_sha256` / `VISTA4D_CHECKPOINT_SHA256` 是已有的可信缓存接口；手工使用时必须填写当前包的真实指纹，不能沿用旧文件 hash。

## 4. CPU 校验

测试只在临时目录生成小型模拟 checkpoint，不读取真实模型、不运行 CUDA 推理：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_vista4d_checkpoint tests.test_video_experiment -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

覆盖多片/单片/旧 `.pth`、字节一致性、部分覆盖后的 CPU forward 一致性、目标 dtype 保持、嵌套 offload 名称映射、索引损坏和路径越界、哈希与复用校验、转换失败不发布等。真实权重转换及 GPU smoke 是后续独立验证步骤。
