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
