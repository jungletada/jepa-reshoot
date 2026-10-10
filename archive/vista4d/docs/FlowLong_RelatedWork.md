# FlowLong 相关工作调研

- 调研日期：2026-09-25
- 调研对象：[FlowLong: Inference-time Long Video Generation via Manifold-constrained Tweedie Matching](https://arxiv.org/abs/2605.20910)（arXiv v1，2026-05-20；Jangho Park、Geon Yeong Park、Gihyun Kwon、Jong Chul Ye；KAIST / Amazon）
- 关联文档：[FlowLong_Plan.md](./FlowLong_Plan.md)（实现设计）、[report.md](./report.md)（Vista4D × FlowLong 综述）

本文回答三个问题：FlowLong 引用了哪些工作、哪些工作引用了 FlowLong，以及引用关系之外还有哪些与它机制相同或目标相同的方法。第 6 节把这些方法对应到本项目的 Vista4D 长视频流程上。

文中数字分三类标注：**论文数字**直接摘自原文表格；**本文重算**是用论文表格数字重新计算的结果；**推断**是尚未在本项目实测的判断。

---

## 0. 结论速览

1. **引用 FlowLong 的工作目前只有 1 篇**：Vorch-Director（arXiv 2608.05776）。它只在 related work 中顺带提到 FlowLong，没有复现也没有对比。FlowLong 发布仅约 4 个月，这个数字预计还会增长。
2. **FlowLong 的 39 篇参考文献**可分为六组：理论根基（Tweedie、rectified flow、DDS/FlowDPS、MultiDiffusion）、训练无关的双向模型扩展（FIFO / RIFLEx / UltraViCo）、自回归蒸馏（CausVid / Self-Forcing 系列 / LongLive 等）、基础模型、相机控制与 4D 生成、3D 与评测。其中与本项目直接相关的是理论根基和**相机控制**这两组：FlowLong 在引言中把 ReCamMaster、TrajectoryCrafter、ReAngle-A-Video、InverseCrafter 列为长视频化的动机场景，也就是本项目所在的任务方向。
3. **FlowLong 没有引用、但机制最接近的是“同步扩散”一脉**：DiffCollage、Gen-L-Video、SyncTweedies、StochSync、SynCoS。其中
   - SyncTweedies 在多种任务上系统比较了同步位置，结论是“在 Tweedie（x0）输出上平均”效果最好，与 FlowLong 在 x0 上做 matching 的选择一致；
   - StochSync 的结论是：条件足够强时，确定性同步就够用；条件弱时才需要随机性。这可以用来解释 FlowLong 的 stochastic early phase；
   - SynCoS 用“局部 reverse 采样 + 全局优化采样 + 固定 baseline noise”处理远距离漂移，正对应 FlowLong 在结论里承认的局限：overlap 约束只作用于局部。
4. **论文 Table 1 的总分领先主要来自 Dynamic Degree 一项**（本文重算）：去掉 Dynamic Degree 后，FlowLong 在 30s 设置下排第 4，低于 LongLive、∞-RoPE 和 Deep-Forcing。本项目的运动由源视频决定，Dynamic Degree 不是目标指标，因此不能直接用论文的总分预期本项目的收益。
5. **读代码时的附带发现**：当前实现在关闭 stochastic（matching-only 消融）时，第 0 步会把各窗口独立采样的初始噪声线性混合，overlap 区噪声标准差最低降到 0.707。这会让 matching-only 与 full FlowLong 的 A/B 对比多出一个与 matching 无关的变量。详见 6.5 节。

---

## 1. 调研方法与覆盖范围

| 渠道 | 做法 | 结果 |
|---|---|---|
| FlowLong arXiv HTML | 下载 v1 HTML，解析 `ltx_bibitem` 和每条的 “Cited by: §x” | 39 条参考文献，逐条记录被引章节 |
| Semantic Scholar API | `paper/arXiv:2605.20910/citations` | 1 篇：Vorch-Director |
| OpenAlex | `filter=cites:W7161936546` | 0 篇（索引滞后） |
| Pith（pith.science） | FlowLong 页面的 forward citations | 1 篇：Vorch-Director |
| 人工全文检索 | 下载 11 篇 2026 年 3–8 月的长视频论文 HTML，grep `FlowLong` / `2605.20910` | 均未引用（名单见 3.2 节） |
| arXiv 摘要页 | 逐个核对本文列出论文的 arXiv 编号、标题和日期 | 全部核对通过 |

局限：

- Google Scholar 无法直接访问，所以可能漏掉没有被 S2 或 Pith 收录的引用，例如会议 camera-ready 版或非 arXiv 论文。
- 第 5 节的近邻方法由人工检索补充，没有穷举，按与 FlowLong 机制的接近程度挑选。

---

## 2. FlowLong 在方法谱系中的位置

FlowLong 由两个部件组成：

- **Tweedie matching**：每一步都把相邻窗口的 predicted-clean（x0）在 overlap 区按线性权重混合，写回一份全局 latent。
- **Stochastic early-phase sampling**：在高噪声阶段，matching 之后用新采样的噪声重新加噪；低噪声阶段恢复确定性 ODE。

按“在哪里融合窗口、用什么形式保证一致性”，FlowLong 与相关方法的谱系关系如下：

```text
多窗口 / 多视角同步扩散
├─ 在带噪状态 x_t 上融合：MultiDiffusion (2023) → Gen-L-Video (2023, 视频版 co-denoising)
├─ 在 score / 模型输出上按因子图组合：DiffCollage (2023)
├─ 在 x0 (Tweedie) 上融合：SyncTweedies (2024, 系统比较后选 x0)
│                           → FlowLong (2026, 线性 x0 blending + 全局 buffer)
├─ 同步 + 随机性：StochSync (2025) ──(类比)── FlowLong 的 stochastic early phase
└─ 局部同步 + 全局耦合：SynCoS (2025) ──(针对)── FlowLong 自述的“局部约束”局限

理论来源（Ye 组 inverse problem 系列）
  MCG (2022, manifold constraint) → DPS (2023) → DDS (2023, x0 上一步修正) → FlowDPS (2025, flow 版)
  └→ FlowLong 把长视频视为“多窗口轨迹对齐”的逆问题，一步梯度修正化简为 x0 线性插值
```

MCG（[2206.00941](https://arxiv.org/abs/2206.00941)）和 DPS（[2209.14687](https://arxiv.org/abs/2209.14687)）没有出现在 FlowLong 的参考文献中，但 “manifold-constrained” 的说法和在 x0 上施加约束的做法都源自这两篇，DDS 也是在它们的基础上发展出来的。

FlowLong 的消融实验直接对比了 “x_t matching”（相当于 MultiDiffusion 式融合，论文引用 Bar-Tal et al. 2023）和 x0 matching，见 4.2 节。

---

## 3. 引用 FlowLong 的工作（前向引用）

### 3.1 已确认的引用

| 论文 | arXiv | 日期 | 任务 / 主干 | 如何引用 FlowLong | 与本项目相关度 |
|---|---|---|---|---|---|
| Vorch-Director: Interactive World Story Model via Noise-Aware Error Rectification | [2608.05776](https://arxiv.org/abs/2608.05776) | 2026-08-06 | 多镜头、多主体、参考图引导的音视频长视频生成；LTX-2；训练式自回归续写 | §2.1 中与 HiAR、VideoAR 并列，归为 “inference-time trajectory and hierarchical-denoising tricks”；无对比实验 | 低：训练式 AR 路线，核心是按噪声级匹配注入预测残差以缓解 exposure bias |

同段被并列引用的两篇可以当作 FlowLong 的“同类”参考：

- **HiAR**（[2603.08703](https://arxiv.org/abs/2603.08703)）：自回归生成，但每个去噪步都对所有 block 做一次因果生成，让每个 block 的上下文与自身处于同一噪声级。这与 FlowLong“所有窗口在同一 timestep 联合前向”的想法一致，只是 HiAR 走的是 AR 蒸馏路线。
- **VideoAR**（[2601.05966](https://arxiv.org/abs/2601.05966)）：next-frame 与 next-scale 预测结合的自回归视频生成，与本项目关系较远。

### 3.2 已检查但未引用 FlowLong 的近期长视频论文

下列论文均在 FlowLong 发布前后出现、主题相近，但全文都不含 `FlowLong` 或 `2605.20910`。列在这里是为了说明检索覆盖范围。

| arXiv | 标题 | 路线 |
|---|---|---|
| [2608.05976](https://arxiv.org/abs/2608.05976) | Diff-VF: Training-free High-quality Long Video Generation | 训练无关，多窗口（见 5.2） |
| [2608.28460](https://arxiv.org/abs/2608.28460) | LayerRecall: A State-Conditioned Memory Router | 记忆 / AR |
| [2608.10439](https://arxiv.org/abs/2608.10439) | Stream Forcing | AR 训练 |
| [2607.18436](https://arxiv.org/abs/2607.18436) | Surprise Forcing | AR 记忆 |
| [2606.14732](https://arxiv.org/abs/2606.14732) | Steady-Forcing | AR |
| [2606.22370](https://arxiv.org/abs/2606.22370) | Towards Error-Free Long Video Generation | AR / exposure bias |
| [2605.18233](https://arxiv.org/abs/2605.18233) | MIGA: Enhancing Train-Free Infinite-Frame Generation | 训练无关，FIFO 系 |
| [2605.18733](https://arxiv.org/abs/2605.18733) | Training-Free Identity-Aware Memory | 训练无关，叙事长视频 |
| [2605.06509](https://arxiv.org/abs/2605.06509) | FreeSpec | 训练无关，注意力频谱 |
| [2605.31057](https://arxiv.org/abs/2605.31057) | LVSA: Training-Free Sparse Attention | 训练无关，稀疏注意力 |
| [2603.25209](https://arxiv.org/abs/2603.25209) | FreeLOC: Layer-Adaptive O.O.D Correction | 训练无关，位置编码 / 注意力 |

---

## 4. FlowLong 引用的工作（后向引用，共 39 篇）

“章节”一列取自 arXiv HTML 中每条参考文献的 “Cited by” 标注。“相关度”指与本项目（Vista4D 长视频重运镜）的相关程度。

### 4.1 按类别整理

**A. 方法的理论根基**

| # | 论文 | arXiv | 章节 | 在 FlowLong 中的作用 | 相关度 |
|---|---|---|---|---|---|
| 6 | Efron, Tweedie's formula and selection bias (JASA 2011) | — | §3 | x0 预测（Tweedie 估计）的出处 | 中 |
| 19 | Rectified flow (Liu et al.) | [2209.03003](https://arxiv.org/abs/2209.03003) | §3 | flow 模型与采样形式；与 Wan scheduler 一致 | 高 |
| 4 | DDS: Decomposed Diffusion Sampler (Chung, Lee, Ye) | [2303.05754](https://arxiv.org/abs/2303.05754) | §1, §4.1 | 在 x0 上做一步修正、再重新加噪的逆问题求解范式；Tweedie matching 由此推导 | 高 |
| 15 | FlowDPS (Kim, Kim, Ye; ICCV 2025) | [2503.08136](https://arxiv.org/abs/2503.08136) | §3, §4.2 | flow 模型上的 posterior sampling，分解 x0 / 噪声分量；stochastic 阶段的依据之一 | 高 |
| 2 | MultiDiffusion (Bar-Tal et al.) | [2302.08113](https://arxiv.org/abs/2302.08113) | §5.3 | 消融中 “x_t matching” 对照组的出处 | 高 |
| 21 | DiT (Peebles & Xie) | [2212.09748](https://arxiv.org/abs/2212.09748) | §1 | 背景 | 低 |
| 32 | DMD: Distribution Matching Distillation (CVPR 2024) | [2311.18828](https://arxiv.org/abs/2311.18828) | §2 | 自回归视频蒸馏的基础 | 低 |

**B. 训练无关的双向模型扩展（对比基线）**

| # | 论文 | arXiv | 章节 | 做法 | 相关度 |
|---|---|---|---|---|---|
| 16 | FIFO-Diffusion (NeurIPS 2024) | [2405.11473](https://arxiv.org/abs/2405.11473) | §1, §2 | 噪声级单调递增的对角队列，逐帧出队 | 中 |
| 38 | RIFLEx | [2502.15894](https://arxiv.org/abs/2502.15894) | §1, §2, §5.1 | 降低 RoPE 内禀频率，抑制时间重复；Table 1 基线 | 中 |
| 39 | UltraViCo | [2511.20123](https://arxiv.org/abs/2511.20123) | §1, §2, §5.1 | 压低训练窗口外 token 的注意力分数；Table 1 基线 | 中 |

这三种方法都需要修改注意力或位置编码，并且要把整段长序列送入 DiT。对 Wan 14B、720p 来说，显存和计算量远超窗口化方案。

**C. 自回归 / 蒸馏路线（对比基线或背景）**

| # | 论文 | arXiv | 章节 | 相关度 |
|---|---|---|---|---|
| 33 | CausVid (From Slow Bidirectional to Fast AR, CVPR 2025) | [2412.07772](https://arxiv.org/abs/2412.07772) | §1, §2, §5.1 基线 | 低 |
| 11 | Self Forcing | [2506.08009](https://arxiv.org/abs/2506.08009) | §1, §2, §5.1 基线 | 低 |
| 5 | Self-Forcing++ | [2510.02283](https://arxiv.org/abs/2510.02283) | §1, §2 | 低 |
| 18 | Rolling Forcing | [2509.25161](https://arxiv.org/abs/2509.25161) | §1, §2 | 低 |
| 31 | Deep Forcing（训练无关的 deep sink） | [2512.05081](https://arxiv.org/abs/2512.05081) | §1, §2, §5.1 基线 | 低 |
| 30 | ∞-RoPE (Infinity-RoPE) | [2511.20649](https://arxiv.org/abs/2511.20649) | §1, §5.1 基线 | 低 |
| 29 | LongLive | [2509.22622](https://arxiv.org/abs/2509.22622) | §5.1 基线 | 低 |
| 36 | FramePack (Frame context packing) | [2504.12626](https://arxiv.org/abs/2504.12626) | §2 | 低 |
| 37 | PFP (Pretraining frame preservation) | [2512.23851](https://arxiv.org/abs/2512.23851) | §2 | 低 |

这些方法大多需要从双向 teacher 蒸馏出因果学生模型。Vista4D 是在 Wan2.1-T2V-14B 上微调得到的条件模型，没有现成的因果版本，因此这条路线对本项目的直接价值有限。

**D. 基础模型与数据**

| # | 论文 | arXiv | 章节 | 备注 |
|---|---|---|---|---|
| 26 | Wan | [2503.20314](https://arxiv.org/abs/2503.20314) | §2, §4, §5 | FlowLong 的 T2V 实验用 Wan2.1-T2V-1.3B；3DGS 实验用 14B。Vista4D 同样基于 Wan2.1-T2V-14B |
| 9 | LTX-2 | [2601.03233](https://arxiv.org/abs/2601.03233) | §1, §2, §4.3, §5, §A.5 | 音视频联合生成实验 |
| 17 | HunyuanVideo | [2412.03603](https://arxiv.org/abs/2412.03603) | §2 | 背景 |
| 22 | Movie Gen | [2410.13720](https://arxiv.org/abs/2410.13720) | §5 | 评测 prompt 来源（MovieGen Bench） |

**E. 相机控制视频生成与 3D / 4D 生成（引言中的动机）**

| # | 论文 | arXiv | 章节 | 与本项目的关系 | 相关度 |
|---|---|---|---|---|---|
| 1 | ReCamMaster (ICCV 2025) | [2503.11647](https://arxiv.org/abs/2503.11647) | §1 | 单视频相机重运镜，微调 Wan；与 Vista4D 同任务 | 高 |
| 34 | TrajectoryCrafter (ICCV 2025) | [2503.05638](https://arxiv.org/abs/2503.05638) | §1 | 单目视频轨迹重定向，点云渲染作条件；与 Vista4D 思路最接近 | 高 |
| 13 | ReAngle-A-Video (ICCV 2025, Ye 组) | [2503.09151](https://arxiv.org/abs/2503.09151) | §1 | 把 4D 生成当作 video-to-video 翻译 | 中 |
| 10 | InverseCrafter (Ye 组) | [2512.05672](https://arxiv.org/abs/2512.05672) | §1 | 训练无关的新视角视频生成：在 latent 空间做 inpainting 式逆问题求解 | 中 |
| 20 | Zero4D (Park, Kwon, Ye) | [2503.22622](https://arxiv.org/abs/2503.22622) | §1 | 与 FlowLong 同一第一作者；训练无关，在时空采样网格上先生成关键帧再插值，得到多视角视频 | 中 |
| 25 | SV3D (ECCV 2024) | [2403.12008](https://arxiv.org/abs/2403.12008) | §1 | 背景 | 低 |
| 28 | CAT4D (CVPR 2025) | [2411.18613](https://arxiv.org/abs/2411.18613) | §1 | 背景 | 低 |
| 27 | 4Real-Video (CVPR 2025) | [2412.04462](https://arxiv.org/abs/2412.04462) | §1 | 背景 | 低 |
| 7 | VIST3A (ICLR 2026) | [2510.13454](https://arxiv.org/abs/2510.13454) | §1, §2, §4.3, §5 | Text-to-3DGS 实验的基线与框架 | 低 |

**F. 3D 重建、评测与 world model**

| # | 论文 | arXiv | 章节 | 备注 |
|---|---|---|---|---|
| 14 | AnySplat | [2505.23716](https://arxiv.org/abs/2505.23716) | §4.3, §5 | 3DGS 实验的重建器 |
| 35 | Prometheus | [2412.21117](https://arxiv.org/abs/2412.21117) | §5 | SceneBench prompt 来源 |
| 12 | VBench (CVPR 2024) | [2311.17982](https://arxiv.org/abs/2311.17982) | §5 | T2V 评测指标 |
| 8 | World Models (Ha & Schmidhuber) | [1803.10122](https://arxiv.org/abs/1803.10122) | §1 | 动机 |
| 3 | Genie (ICML 2024) | [2402.15391](https://arxiv.org/abs/2402.15391) | §1 | 动机 |
| 24 | Advancing open-source world models | [2601.20540](https://arxiv.org/abs/2601.20540) | §1 | 动机 |
| 23 | Grounding world simulation models in a real-world metropolis | [2603.15583](https://arxiv.org/abs/2603.15583) | §1 | 动机 |

### 4.2 论文实验中值得注意的数字

**Table 1（30s，Wan2.1-T2V-1.3B，100 条 MovieGen Bench prompt）**

表中的 Overall 是 7 项 VBench 指标的简单平均（本文重算后与表中数值一致）。下表把“去掉 Dynamic Degree 后的 6 项平均”也列出来：

| 方法 | Overall（论文数字） | 去掉 Dynamic 的 6 项平均（本文重算） | Dynamic Degree | Subject Consistency | Background Consistency |
|---|---:|---:|---:|---:|---:|
| RIFLEx | 0.6943 | 0.7950 | 0.08 | 0.97 | 0.97 |
| UltraViCo | 0.7508 | 0.7824 | 0.5612 | 0.8793 | 0.9348 |
| CausVid | 0.7760 | 0.8296 | 0.4545 | 0.8874 | 0.9037 |
| Self-Forcing | 0.7901 | 0.8308 | 0.5455 | 0.8760 | 0.9064 |
| Deep-Forcing | 0.8137 | 0.8399 | 0.6566 | 0.9019 | 0.9280 |
| ∞-RoPE | 0.7958 | 0.8434 | 0.5102 | 0.9128 | 0.9352 |
| LongLive | 0.7829 | **0.8545** | 0.3535 | **0.9294** | **0.9453** |
| **Wan2.1 + FlowLong** | **0.8233** | 0.8305 | **0.78** | 0.8751 | 0.9305 |

解读：

- FlowLong 的总分第一主要靠 Dynamic Degree，它的 0.78 明显高于其他方法。其余 6 项平均排第 4（与 Self-Forcing 基本持平）。Subject Consistency 在 8 个方法中最低。
- RIFLEx 的 Dynamic Degree 只有 0.08，接近静止画面，因此一致性指标虚高。这说明 VBench 的一致性指标会奖励“不动”，比较时必须和运动指标一起看。
- 60s 设置下 FlowLong 的 Overall 为 0.8251，是表中所有 60s 结果里最高的。论文没有给出双向基线的 60s 结果。
- LTX-2 上 FlowLong 的 Overall 为 0.7812，对照的滑窗基线为 0.7733，提升较小。
- **推断**：本项目的运动由源视频和目标相机决定，不追求 Dynamic Degree。我们关心的是 seam 处的跳变和外观漂移，这更接近 Subject/Background Consistency 与 Temporal Flickering。在这两类指标上，论文并没有显示出 FlowLong 相对 AR 方法的优势。本项目的实际收益仍需通过具体视频验证。

**Table 2（消融，论文数字）**

| 设置 | Consistency | Motion | Quality |
|---|---:|---:|---:|
| Full SDE | 0.9427 | 0.9449 | 0.5298 |
| Full ODE | 0.9604 | 0.9621 | 0.6075 |
| x_t matching（MultiDiffusion 式） | 0.9579 | **0.9690** | 0.5862 |
| FlowLong（x0 matching + early stochastic） | **0.9615** | 0.9685 | **0.6359** |

- x0 matching 相对 x_t matching 的主要收益在 Quality（+0.050），Consistency 仅 +0.004。正文说 FlowLong 在三项上都更高，但表中 Motion 一项 x_t matching 略高（0.9690 对 0.9685）。
- 相对 Full ODE，Consistency 只高 0.0011。论文用 Figure 7 的定性结果说明 ODE 生成的窗口“看起来彼此独立”，但这个差异在该表的数值上很小。
- 论文 v1 没有公布 stochastic 阶段的阈值 t*。本项目默认值 0.6 是自定的，见 FlowLong_Plan.md。

---

## 5. 引用图之外的近邻方法

### 5.1 同步扩散 / 多窗口融合：与 FlowLong 机制相同

| 方法 | arXiv | 年份 / 会议 | 核心做法 | 与 FlowLong 的关系 |
|---|---|---|---|---|
| DiffCollage | [2303.17076](https://arxiv.org/abs/2303.17076) | CVPR 2023 | 用因子图表示“片段 + 重叠”，并行聚合各片段的中间输出，生成任意尺寸的内容，不走自回归 | 最早的“并行多窗口 + 重叠聚合”框架之一，与 FlowLong 思想同源；区别在于它按因子图对 score 做“各片段相加、减去重叠部分”的组合，而不是在 x0 上做线性混合 |
| Gen-L-Video | [2305.18264](https://arxiv.org/abs/2305.18264) | 2023 | Temporal co-denoising：每步把重叠短片段的去噪结果加权融合，支持多 prompt | MultiDiffusion 在视频上的直接版本，属于 FlowLong 消融里 “x_t matching” 一类 |
| SyncTweedies | [2403.14370](https://arxiv.org/abs/2403.14370) | NeurIPS 2024 | 穷举多个扩散过程通过规范空间同步的各种位置，发现“对 Tweedie 输出（x0）取平均”质量最好、适用面最广 | **为 FlowLong 选择 x0 空间融合提供了独立的系统性证据**；FlowLong 未引用 |
| StochSync | [2501.15445](https://arxiv.org/abs/2501.15445) | ICLR 2025 | 揭示同步扩散与 SDS 之间的联系，并把随机性引入同步：条件强时，同步本身就能得到一致结果；条件弱时，需要随机性来保证一致 | **与 FlowLong 的 stochastic early phase 思路相同**；它关于“条件强弱”的结论对本项目有直接参考价值（见 6.2 节） |
| SynCoS | [2503.08605](https://arxiv.org/abs/2503.08605) | 2025 | 在局部 reverse 采样之外，加入全局 optimization-based 采样，并用 grounded timestep 和固定 baseline noise 让两者的去噪轨迹对齐，从而同时约束相邻帧和远距离帧 | **针对 FlowLong 结论中承认的“overlap 约束是局部的”这一局限**；FlowLong 未引用 |

### 5.2 训练无关的长序列扩展：改噪声、注意力或位置编码

| 方法 | arXiv | 做法 | 与本项目的适配性（推断） |
|---|---|---|---|
| FreeNoise (ICLR 2024) | [2310.15169](https://arxiv.org/abs/2310.15169) | 噪声重排：不为每帧独立初始化噪声，而是复用并打乱一段噪声序列，让远距离帧的噪声相关；时间注意力按窗口融合 | “窗口间共享、相关的初始噪声”可以直接用在本项目的全局 latent buffer 上，成本低（见 6.5 节） |
| Video-Infinity | [2406.16260](https://arxiv.org/abs/2406.16260) | 多 GPU 的 clip 并行，加上兼顾局部与全局的 dual-scope attention | 面向多卡，本项目当前在单卡 GB10 上运行，暂不适用 |
| FreeLong (NeurIPS 2024) / FreeLong++ | [2407.19918](https://arxiv.org/abs/2407.19918) / [2507.00162](https://arxiv.org/abs/2507.00162) | 全局注意力提供低频、局部窗口注意力提供高频，做频谱融合 | 需要整段长序列一起前向，14B 720p 下代价过高 |
| Ouroboros-Diffusion | [2501.09019](https://arxiv.org/abs/2501.09019) | 改进 FIFO：改进队尾采样，并加入主体感知的跨帧注意力与自回溯引导 | FIFO 系；对角去噪与“每个窗口一套条件”的 Vista4D 接口不太契合 |
| MIGA | [2605.18233](https://arxiv.org/abs/2605.18233) | 基于 Wan2.1 的 FIFO 系改进：两阶段对齐缩小噪声跨度，加上 self-reflection 与远程帧引导 | 同上 |
| LongDiff | [2503.18150](https://arxiv.org/abs/2503.18150) | 位置映射与信息帧选择，一次生成长视频 | 需要整段序列前向 |
| FreeLOC | [2603.25209](https://arxiv.org/abs/2603.25209) | 逐层探测 O.O.D 敏感度，对敏感层做相对位置重编码和分层稀疏注意力 | 需要整段序列前向，并修改注意力 |
| FreeSpec | [2605.06509](https://arxiv.org/abs/2605.06509) | 对全局 / 局部分支特征做 SVD，按低秩 / 高秩融合 | 同 FreeLong |
| Diff-VF | [2608.05976](https://arxiv.org/abs/2608.05976) | 混合噪声初始化（约束全局语义）、加权窗口采样（消除窗口间不连续），以及随 timestep 变化的时间扩展融合；在 LaVie 和 HunyuanVideo 上验证 | 与 FlowLong 同属“模型无关的多窗口采样器”，但**没有与 FlowLong 对比**；其中的混合噪声初始化与 6.5 节相关 |

### 5.3 自回归、但与“同噪声级联合去噪”相近

| 方法 | arXiv | 要点 |
|---|---|---|
| HiAR | [2603.08703](https://arxiv.org/abs/2603.08703) | 每个去噪步都对所有 block 做一次因果生成，让上下文与当前 block 处于同一噪声级，从而减少误差传播；可以流水线并行。思想上接近 FlowLong 的“所有窗口同 timestep 联合前向”，但需要蒸馏出因果模型 |
| Progressive AR Video Diffusion | [2410.08151](https://arxiv.org/abs/2410.08151) | 为不同帧分配递增的噪声级，逐步推进，是 FIFO 的训练版 |

---

## 6. 对本项目的启示（均为推断，未实测）

### 6.1 本项目与论文实验设置的差异

| 维度 | FlowLong 论文 | 本项目 |
|---|---|---|
| 任务 | T2V、音视频、Text-to-3DGS，条件只有文本 | 视频重运镜（V2V），每个窗口都有源视频与点云渲染作为逐帧条件 |
| 主干 | Wan2.1-1.3B / LTX-2 / Wan2.1-14B | Vista4D（Wan2.1-T2V-14B 微调） |
| 窗口间的跨窗约束 | 只有 x0 matching | 全局 Sim(3) 对齐、一次全局相机平滑、shared-static render，再加 x0 matching |
| 运动来源 | 模型自由生成（Dynamic Degree 是卖点） | 由源视频和目标相机决定 |

结论：本项目的条件远强于论文实验，几何在全局上已经由 shared-static 点云对齐，FlowLong 主要负责外观（纹理、亮度、色调）在窗口之间的同步。

### 6.2 stochastic early phase 的预期（依据 StochSync）

StochSync 的结论是：条件强时，确定性同步就能得到一致结果，随机性主要在条件弱时起作用，并且会损失细节。本项目的逐帧条件很强，因此：

- 预期 matching-only 与 full FlowLong 的差距小于论文 T2V 设置下的差距；t* 扫描的最优值可能偏低（随机阶段更短）。
- 已有的 `t* ∈ {0.5, 0.6, 0.7}` 扫描和 matching-only 对照可以直接检验这一点。检验前需要先处理 6.5 节的噪声方差问题，否则 matching-only 会带着一个额外的劣势参与对比。

### 6.3 全局一致性：FlowLong 的已知局限与候选改进

FlowLong 的约束只在相邻窗口之间传递，310 帧、12 个窗口时，窗口 0 与窗口 11 之间没有直接约束。几何由全局点云锚定，但亮度、色调这类外观属性仍可能缓慢漂移。

候选改进按实现成本从低到高排列：

1. **共享初始噪声（FreeNoise / Diff-VF 思路）**：在全局 latent 上一次性采样噪声，再切片给各窗口，见 6.5 节。几乎不增加计算。
2. **SynCoS 式全局耦合**：在 reverse 步之间加入对全局 x0 的优化步，并使用固定 baseline noise。需要额外前向，成本中等。
3. **远距离窗口参考**：MIGA / Ouroboros 类的远程帧引导需要修改注意力，与“不改 DiT”的非目标冲突，暂不考虑。

建议先用现有的首尾四分位亮度 / 饱和度漂移指标确认漂移确实存在，再决定是否引入第 2 项。

### 6.4 同方向的其他相机控制方法

FlowLong 在引言中列出的 ReCamMaster、TrajectoryCrafter、InverseCrafter 都是单窗口方法（通常 49–81 帧），论文没有在这些模型上做实验。目前唯一引用 FlowLong 的论文（Vorch-Director）也不属于相机控制方向，因此在本调研覆盖的范围内，还没有公开工作把 FlowLong 用于相机控制 / 重运镜模型。如果要写成论文，“强条件 V2V 下的多窗口同步”相对原论文的 T2V 设置有明确差异，值得单独分析。这一判断受检索范围限制，见第 7 节。

InverseCrafter 与 Zero4D 都来自 FlowLong 作者组，而且都训练无关：

- Zero4D 的时空网格采样可用于多条相机轨迹的联合生成；
- InverseCrafter 的 latent 逆问题形式与 FlowLong 的 x0 修正同源。

两者可作为后续扩展（多轨迹、无训练基线）的参考。

### 6.5 代码阅读发现：matching-only 模式下初始噪声方差塌缩

现象（读代码得出，未实测）：

- 各窗口的初始噪声按 `base_seed + index` 独立采样（`diffsynth/pipelines/wan_video_vista4d.py:640`）。
- 第 0 步 `global_xt` 由 `aggregate_window_values(initial_window_latents)` 线性混合得到（`diffsynth/pipelines/flowlong.py:320`）。两段独立的标准高斯按 `(1-λ, λ)` 混合后，方差变为 `(1-λ)² + λ²`。
- Wan scheduler 第 0 步 `sigma = 1.0`（`diffsynth/diffusion/flow_match.py:36-37`）。关闭 stochastic 时，确定性更新 `x1 = (xt - (1-t)·x0)/t`（`flowlong.py:289`）在 `t=1` 时就等于这份混合噪声，并且会沿确定性 ODE 一直保留下去。

overlap 区（O=7）每个 latent 帧的噪声标准差：

| overlap 内序号 i | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| λ = i/6 | 0 | 0.167 | 0.333 | 0.5 | 0.667 | 0.833 | 1 |
| 噪声标准差 | 1.000 | 0.850 | 0.745 | 0.707 | 0.745 | 0.850 | 1.000 |

由于 S=6 < O=7，几乎所有内部 latent 帧都落在某个 blend zone 内，所以方差不足以 6 个 latent 帧（24 个像素帧）为周期出现。

影响范围：

- **full FlowLong（t* = 0.6）不受影响**：第 0 步 t=1 ≥ t*，走 stochastic 分支，直接在全局 buffer 上采样新的标准噪声，不使用混合后的 `global_xt`。
- **只影响 `--flowlong_disable_stochastic` 的 matching-only 消融**：降方差的初始噪声通常会带来更平滑、细节更少的结果，而且与 matching 本身无关。在“stochastic 是否必要”的 A/B 中，matching-only 会因此多一个劣势。

候选修复：在全局 latent（`N_latent = 79` 帧）上一次性采样噪声，再用 `slice_global_latents` 切给各窗口。这样相邻窗口的 overlap 噪声完全相同，聚合后仍然是标准高斯。这与 FreeNoise 的“共享 / 相关初始噪声”思路一致。这样改会让第 0 步的模型输入在 overlap 区变成一致的，属于对论文“各窗口独立初始噪声”设定的偏离，需要作为单独的消融项记录。

---

## 7. 未覆盖 / TODO

| 项目 | 状态 | 说明 |
|---|---|---|
| 前向引用的完整性 | 未完全覆盖 | Google Scholar 不可访问；S2 / Pith 只收录 1 篇。建议 1–2 个月后复查 |
| FlowLong v2 或会议版 | 未检查 | 若有新版本，可能补充 t* 取值、更多基线（如 SynCoS / SyncTweedies） |
| 官方代码 | 截至 FlowLong_Plan.md 编写时只有 README | 未重新检查 [jhq1234/flowlong](https://github.com/jhq1234/flowlong) 是否已放出采样代码 |
| 6.2 节 stochastic 预期 | 未实测 | 依赖 t* 扫描和 matching-only A/B，且需先处理 6.5 节 |
| 6.5 节噪声方差问题 | 未实测 | 读代码得出；需要对比修复前后的 matching-only 结果，确认是否存在周期性模糊 |
| SynCoS 式全局耦合 | 未评估 | 需要先确认外观漂移的量级 |
| 其他相机控制长视频工作 | 未系统检索 | 本文只覆盖 FlowLong 引用图中的相机控制论文，未穷举同时期的长视频重运镜工作 |
