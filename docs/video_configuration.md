# 视频 YAML 配置与统一执行入口

每个视频用一份配置，例如 [1776148878076.yaml](../configs/1776148878076.yaml)。新视频可复制 [example.yaml](../configs/example.yaml)，修改 `video.path`、分辨率、prompt 和分割关键词。环境安装仍见 [README.md](../README.md)，方法原理见 [READMEv2.md](../READMEv2.md)。

已提供三个完整视频配置：[1776148878076](../configs/1776148878076.yaml)、[1778135019043](../configs/1778135019043.yaml)、[1778134948379](../configs/1778134948379.yaml)。均默认使用 384p、50 steps、seed=10027、baseline/matching-only/t*=0.5/0.6。它们是新运行配置，不是对历史实验参数的追溯记录。

YAML 流程与旧的直接 Bash 流程都保留，但**结果路径不同，不能混用命令或假定自动复用旧实验**。此文描述 YAML 流程；旧 README 中的底层命令用于调试和维护旧结果。

## 1. 配置边界与优先级

- 视频、相机、算法参数写入 YAML；Conda 环境、GPU、执行阶段放在命令行。
- 优先级：显式 CLI 参数 > YAML > 程序默认值。CLI 未提供的默认值不会覆盖 YAML。
- YAML 中所有相对路径都相对于**仓库根目录**，不是 `configs/`。不进行 `$VAR` / Shell 表达式替换，也不执行 YAML 中的代码。
- 旧 `SOURCE_VIDEO`、`EXAMPLE`、`RESOLUTION`、`OUTPUT_FOLDER` 等 Shell 流程变量不会覆盖 YAML。CUDA、Conda、缓存、代理等机器环境保留。
- 未知键、重复 YAML 键、错误类型和不支持的配置会报错。布尔值写 `true/false`，不要写字符串 `"false"`。
- 首版不支持 YAML 继承或任意 `--set`；用简单、完整的单文件配置，避免隐式覆盖。

## 2. 参数速查

| 配置 | 含义 / 约束 |
|---|---|
| `video.path` | 原始 MP4；总帧数和 FPS 自动读取，不手写 |
| `video.resolution` | `384p` → 672×384；`720p` → 1280×720；同步选择对应权重 |
| `video.mode` | `full` 或 `single_clip` |
| `single_clip.start_frame` | 单片段首帧，0-based；必须有连续 49 帧可用 |
| `single_clip.name` | 单片段的 `CLIP_NAME/EXAMPLE`；`null` 自动生成名称；完整视频不允许覆盖 |
| `preprocess.reconstruction` | 此配置接口当前支持 `da3`；SAM3 随重建阶段执行 |
| `preprocess.da3_process_res` | `auto` 使用目标宽度；也可指定正整数 |
| `preprocess.segmentation_keywords` | 非空列表，每项一个 token；映射到 SAM3 关键词 |
| `preprocess.save_visuals` | 重建可视化开关 |
| `camera.use_smoothed` | 完整视频必须为 `true`；单片段可关闭 |
| `camera.translation_sigma / rotation_sigma` | 相机平移/旋转平滑 sigma，按帧计，不是秒 |
| `render.mode` | `auto`：完整视频使用 `shared_static`，单片段使用 `single` |
| `render.static_frame_stride / chunk_size` | 全局 shared-static 渲染参数；单片段不使用这两个参数 |
| `render.save_visuals` | 单片段渲染可视化；完整视频不使用此开关 |
| `windows.*` | 当前只支持 49 帧、baseline overlap=5、FlowLong overlap=25、alignment=4、末帧 padding |
| `inference.seed / steps / cfg_scale / sigma_shift` | 公共生成参数；步数不再必须为 50，正式基准默认仍为 50/5/5 |
| `inference.tile_vae` | 单片段、baseline、FlowLong 都传递此开关 |
| `inference.microbatch_size` | FlowLong 窗口微批大小，不代表使用多张 GPU；单片段不使用 |
| `inference.prompt` | 支持 YAML 多行字符串，传递时作为一个完整参数 |
| `flowlong.baseline / matching_only` | 是否生成对应完整视频变体 |
| `flowlong.stochastic_thresholds` | 不重复的 `(0,1]` 数值列表，例如 `[0.5, 0.6]` 或 `[0.65]`；禁用随机采样用 `matching_only`，不能用阈值 0 |
| `models.vista4d` | `auto` 或含 `config.yaml` 和 `.pth` / safetensors 的权重目录 |
| `outputs.run_name` | 本次配置的命名空间，仅字母、数字、下划线和连字符 |

stride 不单独填写：baseline 为 49−5=44，FlowLong 为 49−25=24。任意其他窗口结构当前会明确报错，而不是静默忽略。

`TARGET` 默认是输入文件 stem。单片段名称默认是 `<TARGET>_frames<start>_<end>_<resolution>49`；完整视频每个窗口的 `EXAMPLE` 来自 manifest，不存在一个覆盖所有窗口的全局名称。

## 3. 完整视频逐步执行

以下在仓库根目录运行。示例 `--gpu 0` 指 `nvidia-smi` 的物理 GPU 0，请换成分配给自己的卡。

```bash
# 激活主环境；安装依赖时 requirements.txt 已包含 PyYAML。
conda activate vista4d-pgx

# 默认只读预览；不会加载模型、创建结果目录或启动 GPU 阶段。
python -m scripts.test_video.run_video_experiment \
  --config configs/1776148878076.yaml
```

```bash
# 生成 baseline / FlowLong 两套窗口。
python -m scripts.test_video.run_video_experiment \
  --config configs/1776148878076.yaml --gpu 0 --stages split --execute

# DA3 + SAM3，然后拼接重建、全局平滑和 shared-static 渲染。
python -m scripts.test_video.run_video_experiment \
  --config configs/1776148878076.yaml --gpu 0 \
  --stages recon stitch render --execute
```

```bash
# 执行 YAML 选中的 baseline、matching-only 和阈值变体。
python -m scripts.test_video.run_video_experiment \
  --config configs/1776148878076.yaml --gpu 0 --stages inference --execute
```

默认阶段和完整合法顺序为 `split → recon → stitch → render → inference`。可以省略已有产物的阶段；只选择下游阶段不会自动补跑前置阶段。

## 4. 单片段与调试配置

将 YAML 的 `video.mode` 改为 `single_clip`，设置 `single_clip.start_frame` 和可选名称；`render.mode` 用 `auto`。执行顺序为 `split recon smooth render inference`，不执行 FlowLong 阶段。

```bash
# 单片段分阶段执行；分辨率、起始帧、名称等从 YAML 读取。
python -m scripts.test_video.run_video_experiment \
  --config configs/my_single_clip.yaml --gpu 0 --stages split recon smooth render --execute
python -m scripts.test_video.run_video_experiment \
  --config configs/my_single_clip.yaml --gpu 0 --stages inference --execute
```

临时覆盖用显式 CLI，仍会写入最终配置快照：

```bash
# 仅预览 720p 配置，自动更新尺寸与 auto 权重目录。
python -m scripts.test_video.run_video_experiment \
  --config configs/1776148878076.yaml --resolution 720p

# 预览独立 1-step 调试运行；不会复用 default 目录的条件。
python -m scripts.test_video.run_video_experiment \
  --config configs/1776148878076.yaml --steps 1 --run-name smoke_1step
```

首版以安全隔离为先，不跨 `run_name` 自动复用条件。因此独立调试运行也需要其自身的前置产物；这里没有隐含的 GPU 实验。

## 5. 保存位置、复用与安全边界

YAML 流程全部产物位于：

```text
results/configured/<TARGET>/<RESOLUTION>/<run_name>/
├── resolved_config.yaml       # 解析后配置、源视频 SHA-256、视频元数据、派生路径
├── media/                     # baseline/flowlong/single 窗口与 manifest
├── reconstruction/            # 分窗重建及拼接后重切条件
├── full/                      # 全局重建、平滑轨迹、共享渲染
├── conditions/                # baseline/FlowLong 渲染条件
├── inference/                 # baseline / matching_only / flowlong_t0p*
├── logs/                      # 底层脚本日志（不是统一总日志）
└── .state/                    # 成功阶段记录
```

- 执行时冻结完整配置和源视频哈希；改变参数或源视频必须换 `run_name`。这是保守的**整次运行隔离**，不是自动计算最小失效阶段的缓存系统。
- 同一配置下，成功阶段依据命令签名及其产物文件大小/mtime 记录跳过；缺失/改变则报错。日志不参与此检查。这不是对所有产物重新做逐字节完整性验证。
- 阶段中途失败不会写成功标记。重试会使用底层脚本的恢复/拒绝覆盖策略；遇到部分 reconstruction 等产物需人工检查，或用新运行名，程序不会擅自删除。
- 没有快照的旧目录不自动接管；既有 `results/full`、`results/flowlong_eval` 不迁移、不覆盖。
- `.run.lock` 防止多个进程同时写同一 YAML 运行目录；不是 GPU 锁，不阻止不同运行竞争同一设备。
- 当前仓库的 GPU busy 拒绝逻辑由此前工作树注释禁用；本次没有恢复它。`--gpu` 仍负责选择设备，用户需确认设备分配。建议每条 GPU 命令显式传入相同 `--gpu`。
- 目前没有把全部基础模型内容哈希纳入运行级快照；不要在同一次运行中原地替换模型。更换模型目录或运行名，并查看下游 checkpoint 校验。

后台运行可在 tmux 内执行上述命令；统一入口不会自动创建会话。需要总日志时自行使用 `set -o pipefail` 和 `tee`。

## 6. 兼容方式

不带 `--config` 的原有统一入口和直接 Bash 命令继续使用旧默认值、旧路径。Bash 不直接读取 YAML；需要配置文件时统一从 `run_video_experiment --config` 进入。不要把 YAML 当作可 `source` 的 Shell 文件。

旧 Stage 4 默认仍检查 50 steps / CFG=5 / sigma_shift=5；YAML 入口显式开启自定义参数通道。自定义 1-step 或其他参数结果不是原来的 50-step 基准，报告应按真实设置解释。

精简后的配置结构与旧运行快照不同。已有运行目录保留原样；需要继续生成时，使用新的 `outputs.run_name` 准备完整流程。
