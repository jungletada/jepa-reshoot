# 执行摘要

Manifold4D 和 Vista4D 均针对“视频重拍”（video re-shooting）问题：给定一段单目动态视频及目标相机轨迹，从新视角重合成该场景。Vista4D基于4D点云条件，**显式**地将源视频与点云渲染（Render）作为双重输入，利用视频扩散模型生成目标视频；其优点是保持了几何一致性，但缺乏对点云置信度的调节，动态目标偶有漂移。Manifold4D创新地借鉴 **SDEdit** 概念，将几何信息“注入”到扩散起始噪声中（即在**扩散初始状态**加入部分目标视角渲染结果），而只将源视频作为唯一显式条件，从而解决了Vista4D的“信任困境”。这使生成过程从携带几何结构的**几何噪声流形**出发，网络无需重复学习读取几何信息。

两者实验证明：Manifold4D在相机控制误差（旋转/平移）指标上超越Vista4D约25–32%，同时保持相当的视频质量。在用户研究中，Manifold4D在轨迹跟随与动态一致性上获得明显优势。本文报告将详细对比Manifold4D与Vista4D在条件输入、几何保真、时间一致性、遮挡处理、计算成本、可扩展性、失败模式等方面的异同，并提出将SDEdit式**部分噪声注入**融合进Vista4D流程的设计与实验验证方案。

# 论文元数据

- **Vista4D**: Lin 等人，《Video Reshooting with 4D Point Clouds》, CVPR 2026.  
- **Manifold4D**: Mao 等人，《Denoising on Point Cloud Rendered Manifolds for Video Re-shooting》, 2026（arXiv）。 

# 核心问题

**视频重拍**要求“忠实重现已观测部分”与“合理生成未观测部分”之间的平衡。Vista4D通过构建**4D点云**（静态点和动态点分离）显式提供几何先验，手段包括：

- 对源视频每帧估计深度，后向投影到世界坐标系形成**4D点云**；使用时，将静态点积累为全局地图，动态点按时间帧分离。  
- 通过渲染（Rasterize）沿目标轨迹生成**目标视角渲染帧**及其覆盖掩码（mask）。  
- 在生成网络中，将源视频帧和渲染帧**Patchify**后，与噪声一并输入扩散模型，由模型学习融合几何和外观。  

Vista4D的难点在于：点云渲染和源视频作为两组输入“并列”，模型必须在每一步分辨哪一方可信。这导致了所谓“信任困境”（trust dilemma）——在未见过的大运动或点云伪影场景中，模型可能**过度或不足地**遵从点云，导致视觉漂移或几何失真。

Manifold4D的核心假设是：对于已对齐到目标视角的渲染，可**一次性注入**而非条件输入，使生成过程自带几何结构。具体做法是仿照SDEdit思想，将渲染结果与噪声相加得到扩散初始状态，从而避免后续步骤中源视频与渲染图像的竞争。

# 背景：扩散模型与SDEdit

扩散模型（Score-based/Flow Matching生成）通常定义正向过程：$x(0)\sim p_{\rm data}$，逐渐加噪至 $x(T)\approx \mathcal{N}(0,I)$。常见SDE形式包括**VE-SDE**（方差爆炸）和**VP-SDE**（方差保持）。训练时学习一个score或速度场使逆向过程恢复纯净数据。

- **VE-SDE**: $\mathbf{x}(t)=\mathbf{x}(0)+\sigma(t)z$（信号幅度不变，噪声增加）。  
- **VP-SDE**: $\mathbf{x}(t)=\alpha(t)\mathbf{x}(0)+\sigma(t)z$，满足 $\alpha^2+\sigma^2=1$（信号衰减，最终趋近高斯噪声）。  
- **Flow Matching**: 等价地，也可视为以ODE形式沿直线路径运输数据和噪声。如Wan2.1-T2V模型采用flow matching，把目标样本 $x_{\rm tgt}$ 和高斯噪声 $\varepsilon$ 插值：$x_t=(1-t)x_{\rm tgt}+t\varepsilon$，训练网络预测速度 $\varepsilon - x_{\rm tgt}$。

**SDEdit**（Meng 2022）提出：给定现有信号（如草图、掩码、图片等），先加噪到某个中间时间点后再逆向去噪，结果既保留原结构又提升真实感。直观地，SDEdit从非纯高斯噪声（有用户输入的结构信息）作为起点启动生成。Manifold4D即是将此理念扩展到视频重拍：将点云渲染加入初始噪声，而非以渲染作为条件向网络提供几何信息。

# 方法对比

## Vista4D 样本流程

Vista4D训练和采样流程主要为（图1）：

1. **构建4D点云**：源视频每帧深度估计，并用视频分割提取静态/动态像素，将静态像素累积形成全局静态点云 $\mathcal{P}_{\rm static}$，动态部分按帧 $\mathcal{P}_{\rm dyn}^{(i)}$ 保存。  
2. **渲染与编码**：沿目标轨迹对静态+动态点云进行光栅化，得到渲染帧 $X^{\rm src \to tgt}$ 及掩码 $M$。对源视频 $X^{\rm src}$ 和渲染帧 $X^{\rm src\to tgt}$ 分别经过VAE编码、Patchify得到token序列。  
3. **扩散采样**：以纯高斯噪声 $Z\sim\mathcal{N}(0,I)$ 作为初始状态，将编码的源视频和渲染token按帧拼接后，与噪声一起输入网络；同时注入目标相机轨迹参数（Plücker嵌入）。网络基于flow matching或DDPM生成目标视频。

Vista4D**条件类型**上：源视频和渲染帧都是显式条件，它们在token级别与噪声拼接。这种并列关系可保留外观与几何，但正如文中所述，模型需要不断“权衡”两者信息。在训练时，Vista4D也会随机drop掉部分条件（包括渲染和源视频）以增强鲁棒性。

## Manifold4D 样本流程

Manifold4D在Vista4D基础上引入**部分噪声注入**（SDEdit式）。流程如下（图2）：

1. **4D点云重建与渲染**：与Vista4D相同，重建4D点云并渲染得到目标视角的 $x_{\rm render}$ 和覆盖掩码 $\alpha\in[0,1]$（掩码下采样到token级）。  
2. **几何噪声注入**：采样高斯噪声 $\varepsilon\sim\mathcal{N}(0,I)$，并构造**几何感知初始状态** $x_1$：  

   $$
   \tilde{x}_1 = x_{\rm render} + \sigma\,\varepsilon,\quad
   x_1 = \alpha \odot \tilde{x}_1 + (1-\alpha)\odot \varepsilon.
   \tag{公式1}
   $$

   其中 $\sigma$ 为噪声强度超参数，表明**完整覆盖区域**（$\alpha=1$）从渲染加噪中得到起始状态，**无覆盖区域**（$\alpha=0$）则为纯噪声，中间区域按 $\alpha$ 线性插值。这种方式等价于**给覆盖区域的token赋予一个等效的“SDE时间”**，类似于图像SDEdit给整图加噪再去噪。文中选择 $\sigma=0.3$（也做过 $\sigma$ 扫描）。  
3. **联合编码与去噪**：将得到的几何噪声状态 $x_1$ 和源视频的patch序列拼接（token级）输入视频扩散模型，同时嵌入相机轨迹。网络此时**不再接收渲染作为输入**，只需专注于源视频的外观信息。扩散过程沿 $t=1\to 0$ 的路径进行，学习将“带几何结构的噪声”变换为真实视频。

图示上，Vista4D的三流输入（源视频、渲染、噪声）在网络级别并重，而Manifold4D将渲染只用于构造初始噪声，后续只以噪声+源视频双流输入。这种设计核心差异是：**显式条件（render as condition）⇄隐式注入（render as initial state）**。Manifold4D证明了，该注入操作经fine-tune后并未破坏扩散先验。

```mermaid
flowchart LR
  subgraph Vista4D_基础流程
    SV[源 视频] --> ESV(编码并分Patch)
    SV --> Depth[深度估计 & 4D 重建]
    Depth --> PC[4D 点云]
    PC --> R[目标视角渲染 + 掩码]
    R --> ER(编码渲染帧)
    Z[随机噪声] --> Diffuse1[视频扩散网络]
    ESV --> Diffuse1
    ER --> Diffuse1
    Pose(相机轨迹嵌入) --> Diffuse1
    Diffuse1 --> Out[输出 视频]
  end

  subgraph Manifold4D_SDEdit注入
    SV2[源 视频] --> ESV2(编码并分Patch)
    SV2 --> Depth2[深度估计 & 4D 重建]
    Depth2 --> PC2[4D 点云]
    PC2 --> R2[目标视角渲染 + 掩码]
    R2 --> ER2(编码渲染帧)
    Z2[随机噪声] --> Inject[生成$x_1 = \alpha(x_{\rm render}+\sigma\epsilon) + (1-\alpha)\epsilon$]
    ER2 --> Inject
    Inject --> Diffuse2[视频扩散网络]
    ESV2 --> Diffuse2
    Pose2(相机轨迹嵌入) --> Diffuse2
    Diffuse2 --> Out2[输出 视频]
  end
```
*图1. Vista4D 与 Manifold4D 的采样流程对比。Vista4D将渲染帧和源视频作为平行条件输入，Manifold4D则用SDEdit式注入将渲染信息合并到初始噪声中，仅用源视频作为显式条件。*

## 算法要点及关键公式

- **Manifold4D注入公式**：对token级覆盖掩码 $\alpha$（4D渲染可见像素）和噪声 $\varepsilon$，构造初始状态 $x_1$：  

  $$
  \tilde{x}_1 = x_{\rm render} + \sigma\,\varepsilon,\quad
  x_1 = \alpha \odot \tilde{x}_1 + (1-\alpha)\odot \varepsilon.
  $$

  等价于：*覆盖token从渲染加上强度 $\sigma$ 的噪声获得起始值，未覆盖token保持纯噪声，中间程度按 $\alpha$ 混合*。随后采用flow matching训练：目标路径从 $x_1$ 到真实目标 $x_{\rm tgt}$。网络训练时只最小化速度场损失，无需额外损失项。

- **Vista4D训练目标**：对带噪目标视频 $\mathbf{X}_t^{\rm tgt}$ 和条件（渲染、源视频、相机）进行flow matching：  

  $$
  \mathcal{L} = \Big\|\epsilon_{\theta}(X_t^{\rm tgt},\,X^{\rm src\to tgt},\,M,\,X^{\rm src},\,C^{\rm tgt},\,t) - (X^{\rm tgt}-\epsilon)\Big\|^2,
  $$

  网络输出与真实噪声的误差度量。三者并列输入，模型学习共同影响输出。

- **Vista4D 架构**：基于Wan2.1-T2V-14B大模型，通过finetune加入源视频、渲染视频和相机条件；采用流匹配或类似DDPM训练。Vista4D也使用token拼接而非cross-attention来融合条件。

- **关键超参数**：Manifold4D的 $\sigma$ 控制注入噪声强度，实验选用0.3，展示了更大 $\sigma$ 会提升图像质量但略微降低几何精准度。Manifold4D未明确采用重复注入（$K>1$），Vista4D也无此设定。采样步数方面，Wan2.1-14B的采样往往使用100+步，但Manifold4D和Vista4D报告使用49帧、384×672分辨率下微调30K步。Vista4D还引入first-frame anchor策略以增强长视频一致性。

- **掩码与遮挡**：两者都使用渲染产生的掩码 $\alpha$ 来区分“需生成”区域。Vista4D将掩码输入网络以指导生成（文中Eq2中的 $M^{\rm src\to tgt}$），Manifold4D将 $\alpha$ 用于融合噪声和渲染，并在训练中对缺失区域调整loss权重。

- **采样对比**：Vista4D采样从纯噪声开始，迭代去噪；Manifold4D采样从几何感知噪声 $x_1$ 开始，路径更短。两者皆为前向微分方程求解（如流匹配ODE）。Vista4D常规采样结果依赖网络对条件权衡，Manifold4D采样从几何良好的起点使生成过程更直接地满足相机控制。

# Manifold4D vs Vista4D 对比

| 维度                 | Vista4D                  | Manifold4D                                              |
|--------------------|------------------------------------------------------|----------------------------------------------------------------------------------|
| **条件类型**        | 将源视频和点云渲染平行拼接为条件（Token级）         | 仅将源视频作为显式条件；渲染仅用于构建初始噪声（SDEdit注入），不再作为输入 |
| **几何保真**        | 几何误差低（高精度相机控制），但动态对象可能轻微偏离（受错误深度影响）  | 进一步降低几何误差：旋转/平移误差接近点云渲染下限，动态对象跟踪更稳定            |
| **时间一致性**      | 使用多帧Transformer自回归生成，Vista4D还引入首帧记忆机制，长视频连贯 | 同样使用Transformer；暂无公开长视频方案（当前侧重49帧片段）。一致性好；用户研究显示动态物体稳定。 |
| **遮挡/离屏**       | 静态4D重建帮助场景外观连贯；依赖渲染掩码条件生成离屏区         | 静态4D重建＋注入机制：对遮挡区域从噪声生成，源视频外观引导细节，鲁棒性较高              |
| **计算成本**        | 基于Wan2.1-14B，384×672分辨率下微调30K步；推理时间~22s/Clip（见附录） | 同样Wan2.1-14B，计算量相近；仅修改采样起始（基本无额外开销）。总体训练采样成本与Vista4D相近。             |
| **扩展性**          | 支持扩展应用：长视频（首帧条件）、动态场景扩展、多视角重组 | 目前聚焦单视图短视频场景；可拓展性主要取决于基础模型，与Vista4D类似，但需额外处理长视频记忆。               |
| **失败模式**        | 点云误差大时几何失真（动态物体漂移）；对条件权衡不当会丢失细节 | 低 $\sigma$ 时几何保真好但视觉粗糙，$\sigma$ 过大时可能抹去细节；如果源视频缺失信息，可能出现一般扩散模糊。 |

两者实验证据：Vista4D在多项Camera-Control指标中领先其它基线，而Manifold4D更进一步，在旋转/平移误差上优于Vista4D。视觉质量（PSNR/LPIPS）上，Manifold4D与Vista4D不相上下。用户研究显示Manifold4D在**轨迹跟随**和**动态一致性**上远胜Vista4D。但Vista4D在一般视觉指标（FID/FVD）上也表现优秀。

---

# 官方 Manifold4D 核心代码解读

本节追加于 **2026-10-10**，依据当前仓库引入的官方提交 [cadeac5a6a7341de1ba76f770b3d07a020611fff](https://github.com/ManifoldTechLtd/Manifold4D/tree/cadeac5a6a7341de1ba76f770b3d07a020611fff)。文件来源及校验值见 [源码清单](../upstream/manifold4d.json)。以下聚焦根目录的官方 manifold4d 实现；此前的 DiffSynth 移植代码已进入 archive/vista4d，不作为本节的模型依据。

这是一份源码解读：实际调用可以确认的行为、由公式推导的结论和后续需要运行验证的边界会分别说明。本次没有执行模型训练或完整视频推理。前文保留为早期论文笔记，涉及实现细节时以本节的源码核对为准。

## 1. 先澄清前文中容易误读的表述

| 前文表述 | 官方代码中的准确含义 |
| --- | --- |
| “源视频是唯一显式条件” | 源视频是唯一独立输入的 RGB 视频条件流；渲染覆盖 / 运动掩码、两条流的相机和文本仍在每一步参与计算。 |
| “网络不再接收渲染输入” | DiT forward 不接收独立的 render RGB latent；render RGB 经 VAE 编码后进入初始状态，render mask 仍进入输出流的 patch embedding。 |
| “覆盖率对应不同 token 的 SDE 时间” | 覆盖率改变起始分布的均值和噪声标准差。输出流的所有位置仍使用同一个 flow 时间；源流另有符合其噪声状态的时间编码。 |
| “几何起点使积分路径更短” | 发布采样器仍沿全局时间网格从接近 1 积分到 0。更好的起点不等价于自动减少积分区间或采样步数。 |
| “模型预测真实噪声” | Wan 调度器使用 flow_prediction；在本节采用的时间约定下，预测对象是起点减去干净目标的速度。 |
| “视频 Transformer 自回归生成” | 发布模型对双流视频 token 做联合注意力，默认没有因果时间掩码，也没有逐帧自回归生成循环。 |
| “只修改采样起点即可得到 Manifold4D” | 发布权重包含更新后的 self-attention、四个 patch embedding、相机编码器和 projector。对应生成器已适配几何起始分布，需要同时使用这些增量权重。 |

上述判断主要来自 [run_diffusion](../../manifold4d/infer.py#L105)、[模型 forward](../../manifold4d/model14b/manifold4d_14b.py#L228) 和 [注意力 block](../../manifold4d/model14b/wan21_t2v.py#L178)。前文关于其他方法的性能与训练配方属于论文层面的记录，本节不以静态源码推导实验指标。

## 2. 核心文件与调用关系

| 文件 / 函数 | 在算法中的职责 |
| --- | --- |
| [generate.py](../../manifold4d/generate.py)：load_scene_sample | 对齐源帧、深度、动态掩码与相机，构造源观测点云和目标轨迹条件。 |
| [gpu_renderer.py](../../manifold4d/rendering/gpu_renderer.py)：render_video_gpu | 输出目标视角的粗 RGB、覆盖掩码和动态点投影掩码。 |
| [render_cond.py](../../manifold4d/condition/render_cond.py)：RenderConditioner | 用冻结 VAE 编码视频，把像素掩码池化到 latent 网格。 |
| [infer.py](../../manifold4d/infer.py)：load_model / run_diffusion | 加载基座和三份增量权重，构造起点，执行文本 CFG 与 Wan UniPC。 |
| [manifold4d_14b.py](../../manifold4d/model14b/manifold4d_14b.py)：Manifold4DModel14B | 构造输出 / 源视频双流、相机射线、逐 token 时间嵌入，并调用 40 个 block。 |
| [wan21_t2v.py](../../manifold4d/model14b/wan21_t2v.py)：WanModel21T2V / WanAttentionBlock | Wan T2V 基座及其双流扩展：相机注入、RoPE、projector、文本注意力与 FFN。 |
| [camera_encoder.py](../../manifold4d/model/camera_encoder.py)、[helpers_14b.py](../../manifold4d/model14b/helpers_14b.py) | 根据相机与内参构造 Plücker 射线，定义相机编码器、初始化和冻结策略。 |
| [epipolar_bias.py](../../manifold4d/model/epipolar_bias.py) | 当前发布路径实际提供双流 offset RoPE 与普通联合 self-attention。 |
| [demo.py](../../manifold4d/demo.py) | 从已有 source / render / mask / camera / text 条件直接进入生成器，省去重建与渲染。 |

完整预处理由 [run_vggt_caption_t5_sam3_traj.py](../../scripts/preprocess/run_vggt_caption_t5_sam3_traj.py) 编排：VGGT-Omega 重建、Qwen caption、T5 编码、SAM3 分割、目标轨迹构建与产物检查。其作用是准备条件数据，不包含扩散模型训练。

~~~mermaid
flowchart TD
  SV["源视频"] --> GEO["VGGT-Omega 深度与源相机"]
  SV --> SEG["SAM3 动态掩码"]
  GEO --> PC["带时间和动态标签的点云"]
  SEG --> PC
  PC --> R["目标相机下渲染"]
  R --> RV["render RGB"]
  R --> RM["覆盖与运动掩码"]
  SV --> VS["冻结 VAE：source latent"]
  RV --> VR["冻结 VAE：render latent"]
  RM --> MP["掩码池化"]
  VR --> INIT["构造几何起始状态"]
  MP --> INIT
  EPS["共享 Gaussian 噪声"] --> INIT
  INIT --> STATE["当前输出 latent"]
  STATE --> DIT["输出与源视频双流 DiT"]
  VS --> DIT
  MP --> DIT
  SEG --> SM["源覆盖与运动掩码"]
  SM --> DIT
  CAM["源与目标相机的 Plücker 射线"] --> DIT
  TXT["T5 文本条件"] --> DIT
  DIT --> V["预测 flow 速度"]
  V --> SOLVER["Wan UniPC 更新"]
  SOLVER --> STATE
  SOLVER --> DONE["积分完成后 VAE 解码"]
~~~

图中 render RGB 通过初始化进入采样状态，渲染掩码则同时参与初始化和每一步 DiT 计算。这里的 source mask 实际由源 alpha=1 和池化后的源动态掩码组成。

## 3. 几何条件怎样从源视频产生

### 3.1 帧、相机和尺度必须先对齐

[load_scene_sample](../../manifold4d/generate.py#L227) 读取 predictions.npz 的深度、内参，以及 source / target 两份 trajectory.npz。源与目标相机使用同一组 frame_ids；相机矩阵由 R_world_from_cam 与 centers 直接组成 camera-to-world，而不是把保存的 world-to-camera extrinsic 当作 c2w 使用。

[select_frame_ids](../../manifold4d/generate.py#L204) 提供两种采样：linspace 覆盖整段视频，window 截取连续窗口；帧数要求为 4n+1。采用哪种采样会改变条件的时间网格，后续监督目标也必须使用相同 frame_ids，并记录 fps。相机 latent 采样的 [0,4,8,...] 指的是这组已选帧的局部索引。

RGB 分为两种尺寸：点云颜色使用与深度相同的网格，源视频的 VAE 输入使用生成尺寸。main 通过 fit_dimensions_to_area 保持源宽高比，以指定画布面积为目标，令输出宽高对齐到 16 的倍数；因此最终尺寸可能与 CLI 输入不同。

几何重建和目标轨迹需要处于同一个世界坐标与尺度。代码用轨迹元数据中的 scene_scale 与当前点云范围做粗略检查；重新生成深度后继续使用旧轨迹，可能改变这一对应关系。

### 3.2 点云携带时间标签，静态与动态点采用不同累积方式

[backproject_colored_points](../../manifold4d/generate.py#L119) 对有效深度反投影：

$$
\mathbf p_c
=d\left[
\frac{u-c_x}{f_x},
\frac{v-c_y}{f_y},
1
\right]^\top,\qquad
\mathbf p_w=R_{cw}\mathbf p_c+\mathbf o.
$$

每个点同时保存颜色、选帧后的帧号和动态标签。动态标签来自分割掩码，默认先膨胀 3 个像素，以减少人体边界附近的动态点被归入静态背景。深度置信度过滤发生在反投影之前。

[render_video_gpu](../../manifold4d/rendering/gpu_renderer.py#L443) 的发布默认路径允许所有静态点参与各帧目标渲染，动态点只参与相同局部帧号的渲染。这样可以用跨时间静态观测补全背景，又避免把人物不同姿态同时叠到目标帧。

### 3.3 渲染器怎样定义覆盖与运动

[render_frame_softsplat](../../manifold4d/rendering/gpu_renderer.py#L26) 将点投影到四个相邻像素，按双线性权重分配贡献，再结合深度权重与 log-depth z-buffer 的容差完成遮挡和颜色融合。覆盖掩码由有效累积权重是否超过阈值决定；它首先表示“有点投影到这里”，并不直接表示几何重建正确。

RGB 在深度网格渲染后使用双线性插值调整尺寸，覆盖掩码使用最近邻。运动掩码来自动态点子集的第二次渲染，并采用同样的时间门控；它表示动态点的投影覆盖，不是光流，也不是预测误差或可靠性分数。第二次渲染只包含动态点，因此其覆盖还需要与完整场景的遮挡关系区别理解。

源 motion 通道直接使用原始分割掩码，点云的动态标签则来自膨胀后的掩码。因此源 motion 与 render motion 除了投影坐标不同，主体边界的定义也可能存在差异。

渲染器还保留 nearest_kf、freeze_dynamic、foreground_masking 和动态点裁剪等接口，但标准 generate 调用没有启用这些额外模式。函数中存在增强接口，不代表仓库已包含使用它们的完整训练数据管线。当前 soft-splat 后端的 point_radius 参数也已标为弃用，默认 z-buffer 渲染不会因调整它而改用更大的方形点块。

## 4. VAE latent、掩码与 token 的具体形状

发布入口在 [load_manifold4d_model](../../manifold4d/generate.py#L379) 中创建 z_dim=16 的 Wan2.1 VAE，并明确设置 RenderConditioner 的 stride 为 (4,8,8)。源视频和目标视角 render 分别经过同一个冻结 VAE；其编码结果使用 Wan VAE 的通道 mean / std 归一化。后面的噪声混合发生在这个归一化 latent 空间。

令输入为 T=4n+1 帧，图像尺寸为 H×W：

$$
T_z=\frac{T-1}{4}+1,\qquad
H_z=\frac H8,\qquad W_z=\frac W8.
$$

DiT 再使用 kernel=stride=(1,2,2) 的 Conv3d patchify，得到：

$$
H_p=\frac{H_z}{2}=\frac H{16},\qquad
W_p=\frac{W_z}{2}=\frac W{16},\qquad
N=T_zH_pW_p.
$$

下面以论文常用的 49 帧、384×672 为例，列出单样本形状；这不是 generate CLI 的默认尺寸。

| 张量 | 含义 | 例子形状 |
| --- | --- | --- |
| source / render RGB | VAE 输入，范围 [-1,1] | [3,49,384,672] |
| source / render latent | 16 通道归一化编码 | [16,13,48,84] |
| render_mask | alpha 与 motion 两通道 | [2,13,48,84] |
| source_mask | alpha=1 与 source motion 两通道 | [2,13,48,84] |
| 单流 patch 特征 | 5120 维 token | [1,13104,5120] |
| 双流联合特征 | [output \| source] | [1,26208,5120] |
| 双流相机射线 | 每 token 一个六维 Plücker 向量 | [1,26208,6] |
| 文本 embedding | 进入文本投影前 | [L,4096]，L≤512 |
| 预测速度 | 只返回输出流 | [16,13,48,84] |

[avgpool_mask](../../manifold4d/condition/render_cond.py#L105) 对首帧单独做 8×8 空间池化；后续按每 4 帧、8×8 空间块一起池化，再拼接首帧。这与 Wan VAE 的首帧独立时间布局一致。不能把完整 49 帧直接做普通 stride=4 池化，否则会得到不同的帧数与时间分组。

发布配置采用 avgpool，因此 alpha 与 motion 都是 [0,1] 的分数值。初始化用的是 VAE latent 网格上的 alpha；DiT 随后再通过掩码卷积映射到 patch token。把覆盖率一概称为“token mask”容易漏掉这两个网格的区别。

RenderConditioner 的旧注释仍写有 48 通道、空间 stride=16，且提供沿通道拼接 RGB 的 concat_conditions。实际发布调用使用 16 通道、空间 stride=8，并由双流模型分别 patchify；这些旧注释和辅助接口不能作为当前架构定义。代码也保留 allpack 掩码分支，但发布权重的掩码卷积输入是两通道，不能仅切换 mask_pack_mode 就认为权重仍然适配。

## 5. Manifold 初始化实际定义了什么分布

核心实现位于 [run_diffusion](../../manifold4d/infer.py#L105)。发布 avgpool 模式下，它从 render_mask 的 alpha 通道得到 anchor0，再使用同一份 eps0 构造初始状态：

~~~python
# 核心公式的等价写法；以下变量均处于 VAE latent 网格。
alpha = render_mask[:1].clamp(0, 1)
z_init = alpha * (z_render + prior_sigma * eps) + (1 - alpha) * eps
~~~

记 alpha 为 $\alpha$，prior_sigma 为 $\sigma_G$，可整理为：

$$
z_{\rm init}
=\alpha\odot z_{\rm render}
+\bigl[1-(1-\sigma_G)\alpha\bigr]\odot\epsilon.
$$

条件于给定 render 与 alpha，其逐元素均值和方差是：

$$
\mu=\alpha\odot z_{\rm render},\qquad
\operatorname{Var}(z_{\rm init}\mid z_{\rm render},\alpha)
=\bigl[1-(1-\sigma_G)\alpha\bigr]^2.
$$

| alpha | 初始化均值 | sigma_G=0.3 时的噪声标准差 |
| --- | --- | --- |
| 0 | 0 | 1 |
| 0.5 | 0.5 × render latent | 0.65 |
| 1 | render latent | 0.3 |

**两项使用同一份噪声很关键。**如果改为独立噪声，方差会变成 $\alpha^2\sigma_G^2+(1-\alpha)^2$，与当前实现不同。覆盖完整的位置仍有修正空间；空洞位置回到 Gaussian 起点；部分覆盖位置同时衰减几何均值并增加随机性。

这里 alpha 的数值来自覆盖池化，不是“几何正确概率”。错误深度也可能产生 alpha≈1；这是后续引入可靠性估计和 JEPA 预测时需要解决的具体问题。

还有一个容易误读的变量：[generate](../../manifold4d/generate.py#L506) 和 [demo](../../manifold4d/demo.py#L138) 都创建 target_latent=zeros_like(source_latent)。run_diffusion 只用它确定输出形状和序列长度，不读取真实目标内容。这个名字不表示推理阶段获得了目标视频真值。

## 6. 从几何起点到输出：flow 速度、CFG 与 UniPC

### 6.1 时间与速度的约定

令 z* 为干净目标，z_init 为构造的起点。与当前 Wan flow_prediction 相容的直线路径为：

$$
z_\tau=(1-\tau)z^\star+\tau z_{\rm init},\qquad
\frac{\mathrm dz_\tau}{\mathrm d\tau}=z_{\rm init}-z^\star.
$$

因此训练需要预测的速度是：

$$
v^\star=z_{\rm init}-z^\star.
$$

这是依据 flow 路径和调度器约定得到的训练目标，**发布仓库没有实现对应训练循环**。采样从接近 $\tau=1$ 向 0 积分；时间增量为负，速度符号应与上述定义一致。换了初始化分布却继续使用 Gaussian endpoint 的速度目标，会使训练和采样不匹配。

发布采样器使用 [Wan FlowUniPCMultistepScheduler](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers_unipc.py)，其 flow 输出转换中有 z0_pred=z_tau−sigma_t×v。这也说明模型输出是速度，而不是可以直接代换的 epsilon prediction。

### 6.2 三个容易混淆的参数

| 参数 | 实际作用 |
| --- | --- |
| prior_sigma，默认 0.3 | 初始分布在覆盖区域保留多少随机噪声。 |
| shift，默认 5 | 改变全局采样时间网格，使用 sτ / [1+(s−1)τ] 的变换。 |
| num_steps，默认 50 | UniPC 迭代次数；不是 num_train_timesteps=1000。 |

调度器的时间编码约为 1000τ，并转换为整数 timestep。初始网格接近 1，再逐步降至 0；**prior_sigma=0.3 不表示从全局 flow 时间 0.3 开始采样**。该差异是“几何起始分布”与直接截断原 Gaussian 采样时间的重要区别。

### 6.3 CFG 只改变文本

每一步先计算正文本条件的速度 v_c，需要 CFG 时再计算负文本条件的 v_u：

$$
v_{\rm cfg}=v_u+g(v_c-v_u).
$$

两次 forward 共用 source latent、render mask、source mask、相机与当前输出 latent。所谓 unconditional 分支在这里是负文本分支，不是删除源视频和几何条件。默认 g=5；g 接近 1 时只运行正文本分支。

负提示词优先使用预编码的 [neg_prompt_emb.pt](../../configs/neg_prompt_emb.pt)。缺失时采样器回退到零 embedding；正文本缺失时 generate 也会采用零 embedding。代码能够继续运行，不代表这些输入等价于预期的完整 baseline 条件。文本传入后会补零至 512 个位置，再经过基座的 text_embedding；caption.txt 本身主要用于记录，生成器读取的是 T5 embedding。

### 6.4 每一步究竟更新什么

每次 forward 都从固定 source latent 重新建立源流，从当前 z_tau 建立输出流。源 token 可以在同一次 40 层计算中被更新，用来交换上下文，但其最终状态不会成为下一次采样的源视频。UniPC 只更新输出 latent，最后经过冻结 VAE decode 得到 RGB。

代码另有 alpha 门控的 SDE churn：在去噪步骤后为覆盖区域再加入随机扰动。默认 churn_gmax=0，generate / demo 的标准调用也没有开启它，因此默认没有周期性重新加噪或把 render RGB 再写回输出的操作。

## 7. 双流模型的 forward：四个卷积与两种时间

### 7.1 四个 patch embedding 分工

[Manifold4DModel14B.__init__](../../manifold4d/model14b/manifold4d_14b.py#L78) 包装 WanModel21T2V，并新增四个 Conv3d：

| 模块 | 输入 | 初始化 |
| --- | --- | --- |
| output_rgb_patch_embed | 当前输出 z_tau，16 通道 | 复制 Wan 原生 patch_embedding |
| output_anchor_patch_embed | render alpha + motion，2 通道 | 零权重、零 bias |
| source_rgb_patch_embed | 源视频 latent，16 通道 | 复制 Wan 原生 patch_embedding |
| source_mask_patch_embed | source alpha + motion，2 通道 | 零权重、零 bias |

这里模块名中的 RGB 指视频来源；卷积实际处理的是 VAE latent，不是像素 RGB。output_anchor 处理的是两通道掩码，也不是 render RGB。

两条流分别建立：

$$
X_o=\operatorname{Patch}_o(z_\tau)+\operatorname{Patch}_{ma}(M_r),\qquad
X_s=\operatorname{Patch}_s(z_{\rm src})+\operatorname{Patch}_{ms}(M_s).
$$

再沿序列维拼接 $X=[X_o\mid X_s]$。seq_len 表示单流长度 N，进入注意力后的有效长度为 2N。源和输出必须具有一致的 patch 网格；forward 会检查实际序列长度相等。

发布默认尺寸为 dim=5120、40 层、40 个 attention head、每 head 128 维、FFN dim=13824、文本输入 dim=4096。网络输出只截取前 N 个 token，经过 Wan head 与 unpatchify 恢复 16 通道速度；源流不经过最终输出解码。

### 7.2 源视频为什么也需要时间编码

[forward 的时间处理](../../manifold4d/model14b/manifold4d_14b.py#L359) 将时间编码展开到每个 token，并区分输出与源流：

| cond_stream_t | 输出流时间 | 源流时间 |
| --- | --- | --- |
| shared | 当前输出时间 | 当前输出时间 |
| zero | 当前输出时间 | 固定为 0 |
| honest，发布默认 | 当前输出时间 | cond_t_source；没有传入时为 0 |

标准推理的 source latent 是干净编码，所以 honest 下源时间为 0。若后续训练把源条件替换为 Gaussian 噪声，源时间也应表达该噪声状态；不能只改 source latent，仍让网络把它当作干净观测。

模型经 time_embedding 和 time_projection 得到逐 token 的六组调制量，分别用于 self-attention / FFN 的 shift、scale 和 gate。输出流内部共享一个时间，源流内部共享其条件时间；这与依据 alpha 给每个输出 patch 设置不同 timestep 是两种设计。

## 8. 每个 DiT block 如何融合相机、视频和文本

核心计算位于 [WanAttentionBlock.forward](../../manifold4d/model14b/wan21_t2v.py#L178)。将时间调制量记为 b1、s1、g1、b2、s2、g2，相机射线为 P，可概括为：

$$
U=\operatorname{LN}_1(X)\odot(1+s_1)+b_1+E_{\rm cam}(P),
$$

$$
X'=X+g_1\odot
\operatorname{Projector}\!\left(
\operatorname{SelfAttn}_{\rm RoPE}(U)
\right),
$$

$$
X''=X'+\operatorname{CrossAttn}(\operatorname{LN}_3(X'),E_{\rm text}),
$$

$$
X_{\rm next}
=X''+g_2\odot
\operatorname{FFN}\!\left(
\operatorname{LN}_2(X'')\odot(1+s_2)+b_2
\right).
$$

每个 block 都有独立的相机编码器和 projector。相机编码器为 Linear(6,5120)，从零初始化；projector 为 Linear(5120,5120)，从单位矩阵和零 bias 初始化。这样的初始化使新增几何路径在初始阶段尽量温和，但完整双流架构并不能因此被断言与原单流 Wan 输出完全相同。

相机特征加在进入 self-attention 的特征 U 上，所以同时影响后续 Q、K 和 V。CameraEncoder 注释中关于“Q/K bias”的描述不能替代这里的实际调用。两条视频流通过联合 self-attention 交换信息；文本通过 cross-attention 进入两条流，默认是开启的。

Self-attention 对 Q/K 使用 RMSNorm，对 Q/K 应用 RoPE，再执行 attention 与输出投影。默认 window_size=(-1,-1)，没有因果时间屏蔽；epipolar_bias 这个文件名也不意味着该路径实现了逐射线 epipolar attention mask。几何关系通过相机特征参与学习，发布 attention 本身仍是双流联合注意力。

## 9. 相机 Plücker 射线与双流 RoPE

### 9.1 六维向量表示目标网格上的一条世界射线

[compute_plucker](../../manifold4d/model/camera_encoder.py#L58) 先将内参缩放至 DiT patch 网格，使用半像素中心 u+0.5、v+0.5 构造单位方向，再转至世界坐标：

$$
\mathbf d_c
=\operatorname{normalize}\!\left(
\left[
\frac{u+0.5-c_x}{f_x},
\frac{v+0.5-c_y}{f_y},
1
\right]^\top\right),
\qquad
\mathbf d_w=R_{cw}\mathbf d_c,
$$

$$
\mathbf m=\mathbf o\times\mathbf d_w,\qquad
P=[\mathbf m,\mathbf d_w]\in\mathbb R^6.
$$

前三维为 moment，后三维为 direction，顺序需要与权重一致。位姿采用 OpenCV camera-to-world 约定。世界平移与场景尺度会影响 moment，因此深度、源位姿和目标轨迹要使用一致坐标。

[build_plucker_2stream_14b](../../manifold4d/model14b/helpers_14b.py#L67) 对输出流使用 target c2w，对源流使用 source c2w；两条流共用传入的 K。相机按 latent 时间网格选取局部原视频帧 [0,4,8,...]，并在对应序列位置拼接为 [target rays | source rays]。模型 forward 强制要求源位姿、目标位姿和 K，不能把源流射线简单替换成目标流射线。

### 9.2 offset=31 是位置空间中的流区分

[rope_apply_offset](../../manifold4d/model/epipolar_bias.py#L17) 分别生成时间、纵向和横向 RoPE。输出流使用时间位置 [0,T_z)，源流使用 [offset,offset+T_z)，两者的空间网格一致。发布配置 offset=31；49 帧对应 T_z=13，所以时间位置分别为 [0,13) 和 [31,44)。

31 是 latent 时间位置的偏移，不是原视频时间延迟，也不表示把源视频整体推迟 31 帧。它帮助模型区分两条流，避免把源 token 当成输出视频的自然时间延续，同时仍允许跨流 attention。

固定 offset 的含义依赖片段长度。后续扩展到更长视频时，需要保证时间位置范围、RoPE 表长度及流区分约定一致；源码注释要求 offset≥T_z，但当前函数没有显式执行这一检查。

## 10. 冻结策略与三份增量权重

[unfreeze_trainable_14b](../../manifold4d/model14b/helpers_14b.py#L159) 先冻结整个 Wan 基座，再开放各 block 的完整 self-attention 参数；新增模块保持可训练。具体范围是：

| 范围 | 梯度策略 |
| --- | --- |
| self-attention Q/K/V/O、Q/K RMSNorm | 训练完整参数 |
| 40 个 camera encoder、40 个 projector | 训练 |
| 四个 stream patch embedding | 训练 |
| 基座 FFN、文本 cross-attention、block modulation 与 norm 的参数 | 冻结 |
| 基座 text / time embedding、time projection、输出 head、原 patch_embedding | 冻结 |
| VAE 与用于文本缓存的 T5 | 作为冻结编码器 |

冻结参数不等于前向中跳过该模块；FFN、文本注意力和输出 head 仍参与计算，训练时还需要通过它们传播激活梯度。unfreeze_trainable_14b 返回的是被开放的参数张量数量，不是 optimizer 参数列表。

按发布的 40 层、5120 维结构及默认非共享相机编码器静态计算，这一可训练范围约为 **5,246,504,960 个参数**。主要来自完整 self-attention 和每层的全维 projector；这是完整子模块微调，代码没有以 LoRA 替代这些矩阵。

三份增量文件承担不同角色：

| 文件 | 内容 | 保留哪部分基座 |
| --- | --- | --- |
| self_attn_full.pt | 各 block 的 Q/K/V/O 和 Q/K RMSNorm | 其余 Wan block 参数来自基座 |
| camera_encoder.pt | 相机共享模式标记，以及相机 Linear 参数 | 模型在基座之外新增相机路径 |
| conditioning_modules.pt | 四个 patch embedding、各层 projector、schema_version | 保留基座的文本、时间与输出模块 |

必须先加载 Wan2.1-T2V-14B，再应用这些增量；三份文件不构成一个独立的完整生成器。当前仓库使用 checkpoints/manifold4d 与 checkpoints/wan/Wan2.1-T2V-14B。

官方加载器较宽松：self-attention 会筛选兼容 key / shape 后以 strict=False 加载；conditioning 部分缺少模块时保留初始化；某些文件缺失只打印提示。相机编码器另外检查共享模式。**没有异常不代表发布权重已经全部生效**，后续正式 baseline 入口应核验三份文件、加载数量和未匹配项目，避免误用 Wan 权重或零 / 单位初始化。

## 11. 静态审查发现的实现边界

以下事项依据当前版本的调用链提出，未在完整 14B 视频运行中验证，也没有在本次文档修改中改动官方模型。

### 11.1 内参分辨率应与相机射线和渲染分别对应

generate 中的 sample K 被说明为深度网格分辨率的内参，渲染也在该网格使用它，再把 RGB 和 mask 调整到输出尺寸。但 [run_manifold4d_generation](../../manifold4d/generate.py#L543) 把同一 K 传入相机 helper 时，pixel_height / pixel_width 使用了输出尺寸；在这条路径中未见同步缩放 K。

当深度网格与输出网格尺寸不同，这可能造成 Plücker 内参与目标 render 的视场不一致。按 _intrinsics_from_K 的定义，有两种一致的做法：保留 K_depth 并传入深度网格尺寸，或者先缩放成 K_output 再传入输出尺寸。例如：

$$
f_x^{out}=f_x^{depth}\frac{W_{out}}{W_{depth}},\qquad
c_x^{out}=c_x^{depth}\frac{W_{out}}{W_{depth}},
$$

纵向同理。还需要核对 demo 导出时的相机内参分辨率；[export_demo_scene.py](../../scripts/export_demo_scene.py#L199) 保存了裁切帧后的轨迹内参，demo 使用首帧 K 与示例视频尺寸构造射线。该约定应通过数据元信息和几何检查确认，不能只因 K 的 shape 是 3×3 就认为正确。

此外，完整 generate 会选取一个 target 首选帧 K，缺失时取源内参中位数；两条流共享它。接入不同焦距、不同裁剪的源 / 目标视频时，必须显式决定是遵守这一 baseline 约定还是扩展为分别传入两套内参，并把实验设置记录清楚。

### 11.2 List 接口不等于任意变长 batch 已经得到正确处理

forward 将两条流分别 padding 到 seq_len，再拼接，但 seq_lens_joint 写为 2×out_lens，offset RoPE 也按两段紧邻的有效 token 组织。发布采样 B=1、seq_len=N 时，这些布局相容。

若某个样本有效长度 L 小于 padding 长度 N，布局实际为 [output 的 L 个有效 token | padding | source 的 L 个有效 token | padding]。此时简单的连续长度 2L 无法同时表达两段有效区域，RoPE 的 source 起点也要相应处理。因此后续训练宜先使用同形状 batch；变长 batch 需要重新设计或核对 padding、attention mask、相机序列和 RoPE，不能只把样本装进 List。

### 11.3 混合精度的意图需要与加载后的实际 dtype 一起核对

模型代码对 time embedding / modulation 的结果有 FP32 断言，RoPE 使用高精度复数计算，sampler 默认在 bf16 autocast 下调用模型。与此同时，load_model 会把整个模型转为指定 dtype，默认为 bf16；当前实现未见对 time_embedding、time_projection 或 head 单独保留 FP32 参数的处理。

因此“代码写了 FP32 autocast 和断言”并不足以证明完整混合精度路径已正确。后续运行机器需要通过小模型前向检查这些参数和输入 dtype 是否相容，尤其是嵌套 autocast 的实际行为；训练接入也应保持时间调制的数值约定。这里记录的是静态疑点，不宣称已测得完整模型运行失败。

### 11.4 配置中的开关应追踪到实际 forward

当前 wrapper 写死 use_plucker=True 和 camera_injection=input_x_add。load_model 的 use_plucker 参数主要影响相机权重加载分支，没有在 forward 中实现完整的相机禁用路径；camera_injection 参数也没有切换到另一种注入实现。

同样，修改 mask_channels 只设置属性，不能自动改变已经建立的两通道掩码卷积。正式消融应确认配置实际改变了所需计算，并检查与发布权重的兼容性。

### 11.5 训练可复用模型前向，不能复用整个推理函数反向传播

Manifold4DModel14B.forward 本身保留 autograd，可以作为训练核心；run_diffusion 则被 no_grad 装饰，应保留为推理路径。block 通过正常的 module 调用执行，使 per-block FSDP 的 pre-hook 有机会生效，但这只是训练友好的调用方式，并不是已经发布了 FSDP 训练系统。

当前仍缺少官方训练数据集 / collator、loss 实现、optimizer、checkpoint 保存和恢复、分布式入口与验证协议。论文中的区域平衡、动态权重、条件 dropout、点云增强等配方，需要后续明确实现；不能从模型的冻结策略直接推断这些配方已经存在于仓库。

## 12. 对本项目 JEPA 初始化研究的直接启示

当前 [研究计划](../reshoot-vjepa2.1-plan2.md) 将 JEPA 预测主要放到初始化分布。官方模型已经提供了合适的分工：源视频流保留外观条件，相机与掩码保留目标视角约束，初始化负责提供目标网格上的结构起点。

后续开发可以先把 run_diffusion 中的初始 latent 构造抽出为训练和推理共用的接口，再加入目标视角预测 latent 与可靠性权重。按照当前计划，扩展后的起点可写为：

$$
z_{\rm init}
=w_G\odot(z_{\rm render}+\sigma_G\epsilon)
+w_J\odot(\widehat z_J+\sigma_J\epsilon)
+w_U\odot\epsilon,
\qquad w_G+w_J+w_U=1.
$$

其中各项使用共享 epsilon。令 wG=alpha、wJ=0、wU=1−alpha 即退回当前官方初始化，适合做受控 baseline 检查。JEPA 预测的结构必须先映射到同一 Wan VAE latent 网格与数值空间，不能把 JEPA feature 仅做尺寸对齐就直接与 render latent 相加。

训练同时采用新的 z_init 构造 z_tau 和 v*=z_init−z*；真实目标 RGB 仅用于 teacher、VAE target 与损失，学生侧点云、可见性和预测输入只来自源观测。默认先保留官方 DiT 的源流、相机、掩码、文本 CFG 和采样协议，才能把实验变化明确归因于初始化构造及其训练适配。

进入这一步之前，应该先确定上述相机 / 内参约定、固定形状数据接口和权重完整加载，并建立官方 baseline 的固定样本验证。本节为开发提供代码依据，不在这里执行研究计划或替换官方算法。
