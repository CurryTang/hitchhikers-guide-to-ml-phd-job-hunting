> 更好的阅读体验欢迎前往 GitHub → Currytang → hitchhikers-guide-to-ml-phd-job-hunting
> 这一讲会特别详细，因为会做相关的research:)

# Low-bit Quantization 核心方法详解

## 背景与发展脉络

模型量化的研究可以追溯到深度学习兴起之前。早在信号处理和通信领域，量化就是将连续信号离散化的基本手段（如 PCM 编码），Shannon 的率失真理论为其奠定了信息论基础。进入深度学习时代后，2015-2016 年 BinaryConnect、BNN、XNOR-Net 等工作率先探索了极端低比特（1-bit）的权重和激活表示，试图用 XNOR + popcount 替代浮点乘法，但精度损失过大、且在现代 GPU Tensor Core 上并无实质加速优势。随后 2018-2019 年，PACT 和 LSQ 开创了 QAT（训练时量化）范式，通过将 clip 范围和量化步长设为可学习参数，在 STE 框架下使 4-bit 甚至 2-bit 训练变得可行。与此同时，混合精度训练（FP16/BF16 + FP32 master weights）成为标配，Micikevicius 等人的工作奠定了"低精度计算 + 高精度累积"的基本范式。
2020-2022 年，后训练量化（PTQ）在 CNN 上逐步成熟：AdaRound 将取整方向建模为可优化变量，BRECQ 引入逐 block 二阶重构，QDrop 通过随机跳过激活量化来平坦化损失面，将 CNN 的低 bit PTQ 推进到实用水平。这些方法为后续 LLM 量化提供了关键的方法论积累。
LLM 的量化研究从 2022 年开始爆发，核心驱动力是模型规模急剧膨胀带来的显存和推理成本压力。Dettmers 等人首先发现了大模型激活中的 emergent outlier 现象（少数通道值比其余大 100 倍以上），提出 LLM.int8() 用向量级混合精度应对。SmoothQuant 则通过等价数学变换将激活的量化难度预先迁移到权重侧，实现了真正的 W8A8 部署。在更激进的 4-bit 方向，GPTQ 将 OBQ 的二阶框架扩展到百亿参数规模，AWQ 发现保护与 salient activation 对应的权重列是关键。2024 年，QuaRot 和 SpinQuant 通过正交旋转从根源消除离群值，使全链路 W4A4+KV4 成为可能；KIVI 则专攻长上下文场景下 KV cache 的 2-bit 压缩。与此并行，FP8（E4M3/E5M2）作为 Hopper/Blackwell 架构的原生支持格式，正在成为训练和推理的新基线精度。

---

## 1 量化基础

### 1.1 均匀量化公式

将浮点张量 $x$ 映射到 $b$-bit 整数：

$$x_q = \text{clamp}\!\Big(\!\left\lfloor \frac{x}{s} \right\rceil + z,\; 0,\; 2^b - 1\Big)$$

反量化：

$$\hat{x} = s \cdot (x_q - z)$$

其中两个关键参数：

- **Scale $s$**（缩放因子）：控制量化步长，即相邻两个量化值之间的实际间距。$s$ 越大，可覆盖的浮点范围越广，但相邻量化级之间的间距也越大，精度越低；$s$ 越小，精度越高，但能表示的范围越窄，超出范围的值会被 clamp 截断。$s$ 本质上决定了"用有限的量化格子覆盖多大的数值区间"。
- **Zero-point $z$**（零点偏移）：整数域中对应浮点 0 的值。$z = 0$ 时浮点 0 恰好映射到整数 0（对称量化）；$z \neq 0$ 时整数范围可以偏移，使量化区间不必关于 0 对称，从而更紧凑地覆盖单侧偏斜的分布（如 ReLU 后全为正值的激活）。

两种常见配置：

- **对称量化**：$z = 0$，$s = \max(|x|)\;/\;(2^{b-1} - 1)$。整数范围关于 0 对称，实现简单，但若分布严重偏斜则浪费量化位。
- **非对称量化**：$z \neq 0$，$s = (x_{\max} - x_{\min})\;/\;(2^b - 1)$，$z = \lfloor -x_{\min}/s \rceil$。$z$ 将整数零点对齐到实际分布的下界，更紧凑地利用全部 $2^b$ 个量化级。

### 1.2 术语与记号

**量化对象**：模型中可以被量化的张量主要有以下几类，量化难度和收益各不相同：

- **权重（Weight）**：模型参数，推理时固定不变。分布相对稳定、易于离线分析，是最容易量化的对象。Weight-only quantization（只量化权重）是最常见的起点，GPTQ、AWQ 都属于此类。主要收益是压缩模型体积、减少权重加载的显存带宽瓶颈。
- **激活（Activation）**：每层的输入/输出，随输入数据动态变化。分布不稳定且容易出现离群值，量化难度远高于权重。但如果能同时量化权重和激活，GEMM 就可以完全在低 bit 整数上执行（如 INT8 × INT8），获得真正的计算加速而非仅带宽节省。
- **KV Cache**：Transformer 推理时缓存的 Key 和 Value 张量，长上下文场景下显存占比可远超模型权重本身（如 LLaMA-2-7B 在 128K context 下 KV cache 达数十 GB）。KIVI 等方法专攻此处，可压缩到 2-bit。
- **梯度（Gradient）**：训练时反向传播的梯度。FP8 训练中梯度通常用 E5M2 格式（更大动态范围）存储，分布对称但含稀疏大值，量化策略与权重/激活不同。
- **优化器状态（Optimizer State）**：Adam 的一阶动量 $m$ 和二阶动量 $v$ 各占模型大小的 FP32 存储，是训练显存的大头。bitsandbytes 的 8-bit optimizer 将其压缩到 INT8，节省约 75% 显存。

**WxAy 记号**：基于上述分类，量化领域用 W 代表 Weight、A 代表 Activation、数字代表比特数的缩写来描述量化配置。例如：

- **W8A8**：权重 8-bit，激活 8-bit。SmoothQuant 的典型配置，权重和激活都用 INT8 做 GEMM。
- **W4A16**：权重 4-bit，激活保持 FP16。GPTQ、AWQ 的典型配置，推理时权重 dequant 回 FP16 再做矩阵乘（weight-only quantization）。
- **W4A4**：权重和激活都 4-bit。QuaRot/SpinQuant 追求的全链路低 bit 配置，对 kernel 支持要求最高。

有时还会看到 **KV4**（KV cache 4-bit）、**W1.58**（BitNet b1.58 的三值权重）等写法。

**为什么量化会损害模型精度？** 根本原因是信息损失。将一个 FP32 浮点数（约 $4.2 \times 10^9$ 种可能取值）压缩到 INT8（仅 256 种取值）或 INT4（仅 16 种取值），大量原始数值被映射到同一个量化桶（quantization bin），产生不可逆的舍入误差。具体来说：

- **舍入误差累积**：单个权重的舍入误差可能很小，但矩阵乘会将误差沿计算图传播和放大。层数越深、模型越大，累积效应越明显。
- **离群值问题**：如果张量中存在少数极大值（outlier），量化的 scale 被拉大，导致大量正常值挤在很窄的量化区间内，精度急剧下降。这正是 LLM 量化的核心难点——Transformer 激活中的 emergent outlier 使得朴素量化直接崩溃。
- **动态范围不匹配**：低 bit 整数的可表示范围有限，超出范围的值被 clamp 截断，造成信息永久丢失。

量化研究的核心目标就是在尽可能低的 bit 数下，通过各种技巧（更好的 scale 选择、误差补偿、离群值处理等）将这些精度损失控制在可接受范围内。

### 1.3 量化粒度

| 粒度 | 含义 | 典型场景 |
|---|---|---|
| Per-tensor | 整个张量共享 $(s, z)$ | INT8 推理 |
| Per-channel | 权重每个输出通道一组 $(s, z)$ | CNN / Transformer 权重 |
| Per-group | 每 $g$ 个连续元素一组（$g$ 常取 128） | GPTQ / AWQ 的 4-bit 权重 |
| Per-token | 激活矩阵每行一组 | SmoothQuant W8A8 |
| Per-block | 固定大小 tile（如 32）一组 | FP8 microscaling / MXFP8 |

粒度越细，精度越高，但 scale/zero-point 的存储与计算开销也越大；硬件能否高效访问这些元数据是关键。

### 1.4 Straight-Through Estimator (STE)

**为什么需要 STE？** 量化的核心操作是取整 $\lfloor \cdot \rceil$，这是一个阶梯函数——输出在整数点跳变，其余地方完全平坦。数学上，它的梯度几乎处处为 0。这意味着如果我们想在训练中引入量化（即 QAT），反向传播到量化节点时梯度直接"断掉"，权重无法通过梯度下降更新，训练完全失效。STE 是解决这一矛盾的标准手段：前向传播老老实实做量化，反向传播时假装量化操作不存在，让梯度直接穿过去。

QAT 中用 STE 绕过：

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \hat{x}} \cdot \mathbf{1}\{x \in [\text{lo},\, \text{hi}]\}$$

clip 范围内梯度直通，范围外梯度为 0。**所有 QAT 方法**（PACT、LSQ、BitNet …）都建立在 STE 之上。

### 1.5 混合精度训练

> Micikevicius et al., 2018 · ICLR

**为什么不能直接用纯 FP16 训练？** 朴素地将所有张量从 FP32 换成 FP16 会遇到两个致命问题：(1) **梯度下溢**——FP16 最小正次正规数约 $6 \times 10^{-8}$，很多层的梯度绝对值小于此值，直接变 0，权重停止更新；(2) **权重更新消失**——即使梯度没有下溢，当 `learning_rate × gradient` 远小于权重本身时，FP16 的有限精度（10 bit 尾数）会让 `weight + update` 的加法舍入掉 update，等于没更新。混合精度训练正是为了解决这两个问题而设计的。

**核心思想**："**低精度计算 + 高精度累积**"——用 FP16/BF16 做前向和反向的 GEMM 以获得 2× 显存压缩和计算加速，同时保留 FP32 master weights 保证训练稳定性。

**三板斧**：

1. **FP16 存储与计算**：权重 / 激活 / 梯度存 FP16，前向和反向的 GEMM 用 FP16 Tensor Core 执行（A100 FP16 算力 312 TFLOPS vs FP32 19.5 TFLOPS，16× 差距）。显存 2× 压缩。
2. **FP32 master weights**：优化器维护一份 FP32 的权重副本。每步训练：FP32 master weights → cast 到 FP16 做前向/反向 → FP16 梯度 cast 回 FP32 → 在 FP32 上做 optimizer update。这样权重更新的微小增量不会被舍入吞掉。
3. **Loss scaling**：loss 乘以常数 $S$（如 1024），等价于所有梯度左移 10 bit，将原本下溢到 0 的小梯度拉入 FP16 可表示范围。optimizer update 前除以 $S$ 还原。实践中常用 **dynamic loss scaling**——从大 $S$ 开始，如果检测到梯度出现 inf/NaN 就减小 $S$，否则逐步增大。

```
混合精度训练一步的数据流：

FP32 master weights ──cast──→ FP16 weights
                                    │
                              前向传播 (FP16 Tensor Core GEMM)
                                    │
                                    ▼
                              loss × S  (loss scaling)
                                    │
                              反向传播 (FP16)
                                    │
                                    ▼
                          FP16 gradients / S ──cast──→ FP32 gradients
                                                          │
                                                   Optimizer update
                                                   (FP32 精度累积)
                                                          │
                                                          ▼
                                                FP32 master weights (更新后)
```

**显存分析**：以 1B 参数模型为例（每个参数 4 bytes in FP32）：

| 组件 | 纯 FP32 | 混合精度 |
|---|---|---|
| 模型权重 | 4 GB (FP32) | 2 GB (FP16) + 4 GB (FP32 master) = 6 GB |
| 激活 | 4× 看 batch | 2× 看 batch（FP16，减半） |
| 优化器 (Adam) | 8 GB (m + v, FP32) | 8 GB (m + v, 仍 FP32) |

看起来混合精度反而多了 FP16 权重副本？但激活是大头——激活显存随 batch size 和 seq len 线性增长，FP16 激活节省的显存远超多存一份 FP16 权重的开销。而且计算速度的 2× 提升是最大收益。

**BF16 vs FP16**：

| | FP16 | BF16 |
|---|---|---|
| 指数位 | 5 bit（最大 65504） | 8 bit（最大 $3.4 \times 10^{38}$，同 FP32） |
| 尾数位 | 10 bit | 7 bit |
| 需要 loss scaling？ | 必须 | 几乎不需要（动态范围足够大，梯度极少下溢） |
| 精度 | 更高 | 略低（但实践中对训练收敛影响很小） |
| 硬件支持 | 所有现代 GPU | A100+（V100 不支持） |

BF16 因为省去了 loss scaling 的复杂性且训练更稳定，已成为大模型训练的事实标准。PyTorch 中只需 `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` 即可启用。

### 1.6 低精度数值格式一览

量化和低精度训练涉及多种浮点/整数格式。下表列出常见格式的构成与可表示范围，帮助直观理解不同精度的"信息容量"差异：

| 格式 | 构成（符号+指数+尾数） | 最大值 | 最小正次正规数 | 精度 | 典型用途 |
|---|---|---|---|---|---|
| **FP32** | 1 + 8 + 23 | $3.4 \times 10^{38}$ | $\sim 1.4 \times 10^{-45}$ | ~7 位十进制 | Master weights、优化器状态 |
| **FP16** | 1 + 5 + 10 | 65504 | $6.0 \times 10^{-8}$ | ~3 位十进制 | 混合精度训练（需 loss scaling） |
| **BF16** | 1 + 8 + 7 | $3.4 \times 10^{38}$ | $\sim 9.2 \times 10^{-41}$ | ~2 位十进制 | 大模型训练标配（动态范围同 FP32） |
| **FP8 E4M3** | 1 + 4 + 3 | 448 | $\sim 0.002$ | 3 bit 尾数 | 前向：权重 / 激活 |
| **FP8 E5M2** | 1 + 5 + 2 | 57344 | $\sim 6.1 \times 10^{-8}$ | 2 bit 尾数 | 反向：梯度（需更大动态范围） |
| **FP4 E2M1** | 1 + 2 + 1 | 6.0 | 0.5 | 1 bit 尾数 | 实验性，MXFP4 |
| **INT8** | 8 bit 整数 | 127（有符号） | 1 | 均匀间距 | PTQ 推理（W8A8） |
| **INT4** | 4 bit 整数 | 7（有符号） | 1 | 仅 16 级 | 权重压缩（W4A16） |

**关键对比**：
- **FP16 vs BF16**：同为 16-bit，FP16 精度更高（10 bit 尾数），但动态范围窄（最大 65504），大值易溢出需 loss scaling。BF16 动态范围与 FP32 一致，训练更稳定，是当前大模型首选。
- **E4M3 vs E5M2**：同为 FP8，E4M3 精度高但范围小（适合权重/激活），E5M2 范围大但精度低（适合梯度这类需要大动态范围的张量）。
- **浮点 vs 整数**：浮点格式的量化间距是非均匀的（小值处更密），整数量化间距均匀。对于近似正态分布的权重，非均匀量化（如 NF4 码本，§9.2）理论上更优。

---

## 2 量化方法总览：QAT vs PTQ

**为什么需要区分 QAT 和 PTQ？** §1.1 的量化公式看起来很简单——算个 scale，取个整就完了。但核心问题是：**scale 怎么选才最优？** 如果我们用 $\max(|x|)$ 来定 scale，那些罕见的极端值会拉大 scale，浪费大部分量化级；如果我们 clip 掉极端值收紧 scale，又会截断重要信息。更根本的矛盾是，量化会改变每一层的输出分布，这个误差沿着网络逐层传播和累积——单独优化每一层的 scale 并不能保证全局最优。

面对这个矛盾，有两条路：

- **QAT**：既然量化误差会传播，那就让模型在训练时就"看到"量化误差，通过端到端梯度下降自动调整权重分布去适应量化。代价是需要完整的训练流程。
- **PTQ**：模型已经训练好了，不动权重本身，而是用校准数据分析每层的分布特征，选择最优的量化参数（scale、取整方向等）来最小化输出误差。代价是没有训练的纠错能力，精度上限低于 QAT。

模型量化根据**是否需要训练**分为这两大范式。本节给出总览，具体 LLM 量化方法按问题递进在 §3-§6 展开。

### 2.1 QAT（Quantization-Aware Training）— 训练时量化

> QAT 的经典方法（PACT、LSQ 等）诞生于 CNN 时代（2018-2019），针对 ResNet、MobileNet 等视觉模型。LLM 场景的 QAT 代表是 BitNet（§7），走"原生低 bit 从头训练"路线。

**核心思路**：在训练中插入**伪量化节点（fake quantization）**——前向传播时对权重和激活做 quantize → dequantize（数据类型仍为 FP32，但值已含量化误差），让模型通过梯度下降学习对量化友好的分布。反向传播时用 STE（§1.4）让梯度穿过量化节点，正常更新 FP32 master weights。训练结束后导出真正的低 bit 权重。

```
┌─────────────────────────── 前向传播 ───────────────────────────┐
│                                                                │
│  x (FP32)                                                      │
│    │                                                           │
│    ▼                                                           │
│  ┌──────────────────┐                                          │
│  │  伪量化 (Fake Q)  │  quantize → dequantize                  │
│  │  x → x_q → x̂    │  数据类型仍为 FP32，但值已含量化误差      │
│  └────────┬─────────┘                                          │
│           ▼                                                    │
│  ┌──────────────────┐     w (FP32 master weight)               │
│  │  Linear(x̂, ŵ)   │ ◄── ŵ = dequant(quant(w))  同样伪量化    │
│  └────────┬─────────┘                                          │
│           ▼                                                    │
│         loss                                                   │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────────── 反向传播 ───────────────────────────┐
│                                                                │
│  ∂L/∂loss                                                      │
│    │                                                           │
│    ▼                                                           │
│  ∂L/∂ŵ ──→ 遇到伪量化节点 ──→ STE 直通（§1.4）                  │
│    │                                                           │
│    ▼                                                           │
│  用 FP32 梯度正常更新 master weights                             │
└────────────────────────────────────────────────────────────────┘
```

**关键可学习参数**：QAT 的核心在于把量化公式中的超参数变为可学习的：

| 可学习对象 | 思路 | 代表工作 |
|---|---|---|
| Clip 范围 $\alpha$ | 学习每层最优截断范围，避免截断信息或浪费比特 | PACT (2018) |
| 量化步长 $s$ | Step size 作为可学习参数 + 梯度缩放保证稳定 | LSQ (2019) |
| 两者结合 | 同时学习 clip 范围和步长 | LSQ+ |

### 2.2 PTQ（Post-Training Quantization）— 后训练量化

> **LLM 量化的主流路线**。模型训练完成后，不再更新权重，仅用少量校准数据（几百条无标注样本）确定量化参数。GPTQ、AWQ、SmoothQuant 等 LLM 量化方法都属于 PTQ。

**基本流程**：加载预训练模型 → 校准数据前向推理收集统计信息 → 确定各层 scale/zero-point → 量化权重（部分方法也量化激活） → 评估精度。

**PTQ 的三个技术层次**：

| 层次 | 思路 | 精度 |
|---|---|---|
| **Round-to-Nearest (RTN)** | 直接最近整数取整，无优化 | 8-bit 可用，4-bit 通常崩溃 |
| **逐层/逐块重构** | 最小化输出重构误差 $\min \|\|Wx - \hat{W}x\|\|_F^2$，可优化取整方向 | 4-bit 可用 |
| **二阶误差补偿** | 用 Hessian 信息逐列量化 + 补偿残差到未量化列 | 4-bit 甚至 3-bit 可用 |

**重构优化的核心思想**：量化不一定要 round-to-nearest——某些权重 round up 比 round down 对输出误差更小。把取整方向建模为可优化变量，逐层最小化 $\min_{\hat{W}} \|Wx - \hat{W}x\|_F^2$（$x$ 来自校准数据）。实际中逐 block 重构是性价比最优的选择。CNN 时代的 AdaRound、BRECQ、QDrop 等工作建立了这套方法论，被 GPTQ、AWQ 等 LLM 方法直接继承。

### 2.3 QAT vs PTQ 对比

| | QAT | PTQ |
|---|---|---|
| 需要训练数据？ | 是（完整训练集） | 否（几百条无标注校准样本） |
| 计算成本 | 完整训练周期 | 分钟到小时 |
| 适用场景 | 极低 bit（2-3 bit）、从头训练 | 部署压缩、4-bit+、**LLM 主流** |
| 精度 | 更高（尤其低 bit） | 略低但通常足够 |
| LLM 代表 | BitNet（从头训练） | GPTQ / AWQ / SmoothQuant |

对 LLM 而言，全参数 QAT 的成本接近从头训练，因此实践中几乎所有"压缩已有模型"的工作都走 PTQ 路线。QAT 仅在 BitNet 等"原生低 bit 从头训练"的方向有应用。

---

## 3 LLM 量化的核心挑战：离群值

在 §1-§2 中我们建立了量化的基础工具箱。现在进入 LLM 量化的实战——但在看具体方法之前，必须先理解 **为什么 LLM 量化比 CNN 难得多**。

**Emergent Outlier 现象**（Dettmers et al., 2022）：Transformer 模型参数量超过约 6.7B 后，激活张量中会涌现出 **少数固定通道的值比其余通道大 100 倍以上**。这些离群值（outlier）是"涌现"的——小模型没有，大模型才有，且集中在特定的隐藏维度上，跨不同输入 token 稳定存在。

**离群值为什么会涌现？** 这个问题尚未完全解决，但有几个有影响力的解释：

- **LayerNorm + 残差连接的交互**：Bondarenko et al. (2023, "Quantizable Transformers") 提出，离群值本质上是 attention head 的 **"no-op"信号**。当某个 attention head 对当前 token 没有有意义的信息需要提取时，它需要一种方式"什么都不做"——但 softmax 强制注意力权重归一化，无法输出全零。模型学到的策略是：将注意力集中在某些固定维度的大值上，这些大值经过后续处理后对最终输出贡献可控。LayerNorm 的存在使得模型可以安全地在少数通道上使用极大值而不影响其余通道的数值稳定性。
- **Massive Activations**：Sun et al. (2024, "Massive Activations in Large Language Models") 系统研究了大激活值现象，发现它们出现在**固定位置**（如序列开头的 token）和**固定维度**，并且承担功能性角色——移除它们会严重损害模型性能。这说明离群值不是训练的"bug"，而是模型学到的一种**信息编码方式**。
- **规模涌现假说**：与大模型的其他涌现能力（in-context learning、chain-of-thought 等）类似，一种观点认为离群值是模型达到一定规模后才出现的相变现象——小模型的表示能力不足以"负担"这种将信息集中在少数通道的编码策略，大模型则可以利用冗余维度实现更高效的信息路由。
![Pasted image 20260306201502.png](./assets/Pasted_image_20260306201502.png)
> 这里我们在qwen2.5的0.5b和7b两个scale上面复现了这个现象

**为什么离群值让朴素量化崩溃？** 回顾 §1.1 的量化公式，scale $s$ 由张量的最大绝对值决定。假设一个激活向量大部分值在 $[-1, 1]$，但有一个通道的值为 100：

- $s = 100 / 127 \approx 0.79$（INT8 对称量化）
- 正常值 $0.5$ 被量化为 $\lfloor 0.5 / 0.79 \rceil = 1$，反量化回 $0.79$，误差 58%
- 如果没有离群值：$s = 1 / 127 \approx 0.008$，$0.5$ 被量化为 $63$，反量化回 $0.50$，误差 < 1%

**一个离群值拉大了 scale，毁掉了其余 99.9% 正常值的量化精度。**

这就是 LLM 量化的核心矛盾：离群值不能丢（它们对模型输出至关重要），但它们的存在让整个张量难以用统一的 scale 量化。后续 §4-§6 的所有方法，本质上都在用不同策略解决这个矛盾。

```
LLM 量化的递进路线图：

§4  先解决能不能量化 ──→  W8A8：8-bit 推理
    策略：处理离群值         LLM.int8()（运行时分离离群）
                            SmoothQuant（预处理消除离群）
                                │
                                ▼
§5  再追求更高压缩比 ──→  W4A16：4-bit 权重量化
    策略：更智能的取整       GPTQ（二阶误差补偿）
                            AWQ（激活感知缩放）
                                │
                                ▼
§6  最终目标：全链路低 bit ──→  W4A4 + KV4
    策略：从根源消除离群      QuaRot（正交旋转分散离群）
                              SpinQuant（学习最优旋转）
                              KIVI（KV cache 专攻）
```

---

## 4 W8A8：8-bit 推理量化

> 第一步目标：让 LLM 能用 INT8 做 GEMM，获得 2× 带宽节省 + 计算加速。核心难点是如何处理激活中的离群值。

### 4.1 LLM.int8() — 运行时分离离群

> Dettmers et al., 2022 · NeurIPS

**思路**：既然离群值只占少数通道，就在运行时把它们分出来单独用 FP16 算，其余走 INT8。

```
输入 X ∈ ℝ^{n×d}, 权重 W ∈ ℝ^{d×m}

1. 检测离群维度  O = { j : max_i |X_ij| > τ },  τ = 6.0
2. 分离：
     X_out = X[:, O],  W_out = W[O, :]   →  FP16 GEMM
     X_reg = X[:, Ō],  W_reg = W[Ō, :]   →  INT8 absmax GEMM
3. Y = Y_out + Y_reg
```

- 离群维度通常占 0.1%–1%，绝大部分计算走 INT8。
- 无校准、无训练。175B 模型 PPL 增量 < 0.1。
- 局限：运行时检测离群 → kernel 需要支持分解拼接；FP16 部分限制整体带宽收益。

### 4.2 SmoothQuant — 预处理消除离群

> Xiao et al., 2022 · ICML 2023

**LLM.int8() 的问题**：虽然精度近乎无损，但它的运行时分离机制带来了显著的工程和性能代价：

1. **两次 GEMM 开销**：每个线性层都要做一次 INT8 GEMM + 一次 FP16 GEMM + 结果拼接，kernel launch 和同步的开销在小 batch 推理时占比显著。实测中 LLM.int8() 的推理速度经常**比纯 FP16 更慢**（尤其在单条推理时），因为节省的计算量被额外的 kernel 开销吞掉了。
2. **动态分支不友好**：离群维度的数量和位置需要运行时逐层检测，这种数据依赖的动态分支对 GPU 的并行执行和编译优化都不友好，难以被 torch.compile 等框架优化。
3. **无法用标准 INT8 kernel**：标准的 INT8 GEMM kernel（如 cuBLAS INT8）要求整个矩阵统一精度，LLM.int8() 的混合精度分解需要定制 kernel，限制了在不同硬件平台上的可移植性。

**SmoothQuant 的思路**：能不能在部署前就**预处理掉离群值**，让推理时直接全走 INT8，用标准 kernel 就能加速？

**核心洞察**：权重平滑（易量化），激活含离群（难量化）。用**数学等价变换**把难度从激活搬到权重。

对线性层 $Y = XW$，引入逐通道缩放 $\mathbf{s} \in \mathbb{R}^d$：

$$Y = \underbrace{(X\,\text{diag}(\mathbf{s})^{-1})}_{\hat{X}} \;\cdot\; \underbrace{(\text{diag}(\mathbf{s})\,W)}_{\hat{W}}$$

选择 $\mathbf{s}$ 平衡 $\hat{X}$ 与 $\hat{W}$ 的量化难度：

$$s_j = \frac{\big(\max_i |X_{ij}|\big)^{\alpha}}{\big(\max_k |W_{jk}|\big)^{1-\alpha}}, \qquad \alpha \in [0.5, 0.75]$$

**为什么这个公式能消掉离群值？** 用一个具体例子说明。假设某层激活 $X$ 的第 $j$ 通道是离群通道，$\max |X_{:,j}| = 100$，而对应的权重列 $\max |W_{j,:}| = 0.5$。取 $\alpha = 0.5$：

$$s_j = \frac{100^{0.5}}{0.5^{0.5}} = \frac{10}{0.707} \approx 14.1$$

变换后：
- 激活第 $j$ 通道：$\hat{X}_{:,j} = X_{:,j} / s_j = X_{:,j} / 14.1$，原来的 100 变成 ~7.1，**离群值被压下来了**
- 权重第 $j$ 行：$\hat{W}_{j,:} = s_j \cdot W_{j,:} = 14.1 \cdot W_{j,:}$，原来的 0.5 变成 ~7.1，**权重被放大了**

关键在于：**激活的离群通道恰恰对应权重中数值较小的行**（这是 LLM 中普遍观察到的现象——模型用大激活值乘以小权重来编码信息）。所以 $s_j$ 大 → 激活除以大数被压平 → 权重乘以大数被放大，但因为权重原本就小，放大后仍在合理范围内。变换后激活和权重的数值范围趋于接近，两边都变得"好量化"了。

数学上，$s_j$ 的公式就是在做**几何均衡**：$\alpha$ 控制把多少量化难度从激活搬到权重。$\alpha = 1$ 时完全按激活缩放（激活完全平滑，但权重可能爆炸）；$\alpha = 0$ 时不动激活。实践中 $\alpha \in [0.5, 0.75]$ 取得最佳平衡——因为权重分布比激活平滑得多，能"承受"更多难度转移。

**部署**：$s^{-1}$ 融合进前一层 LayerNorm/bias，$s$ 吸收进 $W$。推理时零额外计算，直接 **W8A8 INT8 GEMM**。

- 与 LLM.int8() 的区别：**预处理消除**离群，不在运行时分离。
- 局限：主要面向 INT8，向 4-bit 推进时平滑不够。

**校准过程详解**：SmoothQuant 需要用校准数据确定每个通道的缩放因子 $s_j$，具体流程：

1. **准备校准集**：从训练集或公开数据（如 Pile、C4）中随机抽取几百条无标注文本（通常 128–512 条），截断到固定长度（如 2048 tokens）。这些数据不需要标签，只需要能代表模型实际输入的分布。
2. **前向收集统计**：将校准数据逐条送入模型做前向推理（不做反向传播），在每个线性层的输入端记录激活张量的**逐通道最大绝对值** $\max_i |X_{ij}|$。由于不同输入的离群通道位置高度一致（§3 讨论过的 emergent outlier 特性），几百条数据的统计已经非常稳定。
3. **计算缩放因子**：结合激活统计和权重本身的逐通道最大值，按公式 $s_j = (\max_i |X_{ij}|)^\alpha / (\max_k |W_{jk}|)^{1-\alpha}$ 计算每层每个通道的 $s_j$。$\alpha$ 是唯一的超参数，通常在 $[0.5, 0.75]$ 范围内网格搜索，以校准集上的量化输出误差为指标选择。
4. **吸收缩放到权重**：$\text{diag}(\mathbf{s})$ 乘进权重矩阵 $W$，$\text{diag}(\mathbf{s})^{-1}$ 融合进前一层的 LayerNorm 参数（$\gamma \leftarrow \gamma / s$）。这步是离线完成的，修改后的模型保存为新的权重文件。
5. **确定量化参数**：对平滑后的权重和激活分别计算 INT8 量化的 scale/zero-point（权重 per-channel，激活 per-token），完成量化。

整个过程**只需前向推理**，不更新任何权重，耗时通常在几分钟到十几分钟。OPT-175B 上近乎无损。

这个校准范式也被 GPTQ、AWQ 等方法继承——它们同样用少量校准数据前向推理收集统计（GPTQ 收集 Hessian $H = 2X^TX$，AWQ 收集激活幅度 $\text{mean}_i |X_{ij}|$），区别只在于拿这些统计做什么优化。

### 4.3 W8A8 小结：基本已解决的问题

W8A8 在当前 LLM 量化中可以认为是**基本做到头了**。SmoothQuant 在主流模型（OPT、LLaMA、Mistral 等）上 PPL 增量通常 < 0.1，downstream 任务精度损失可忽略。剩余的边角问题包括：

- **$\alpha$ 调参**：不同模型、不同层的最优 $\alpha$ 可能不同，SmoothQuant 原版用全局统一 $\alpha$，后续工作（如 OS+）尝试逐层自适应 $\alpha$，但提升有限，说明全局 $\alpha$ 已经足够好。
- **特殊架构适配**：GQA（Grouped Query Attention）等新架构中 KV 投影的通道数不同，平滑因子的传播路径需要适配，但这是工程问题而非方法论瓶颈。
- **FP8 的替代**：H100+ 硬件原生支持 FP8 Tensor Core（§8.1），在很多场景下 FP8 推理比 INT8 更方便（不需要平滑变换，直接 cast），精度也够用。FP8 正在逐步取代 INT8 成为 8-bit 推理的默认选择。

**结论**：W8A8 的精度问题已经被 SmoothQuant 有效解决，研究前沿已全面转向更激进的 4-bit 甚至更低比特量化。W8A8 剩下的主要是工程落地和新硬件适配的工作。

---

## 5 W4A16：4-bit 权重量化

> W8A8 解决了"能不能量化"的问题。下一步自然是：**能不能压得更狠？** 4-bit 权重 = 模型体积再砍一半。这里只量化权重（weight-only），激活保持 FP16，推理时权重 dequant 回 FP16 做 GEMM。核心难点从"离群值"变成了"如何在只有 16 个量化级的情况下最小化权重量化误差"。

**W8A8 vs W4A16：怎么选？** 两者的设计目标不同，没有绝对的优劣，取决于部署场景：

| | W8A8 (SmoothQuant) | W4A16 (GPTQ/AWQ) |
|---|---|---|
| **精度** | 近乎无损（PPL 增量 < 0.1） | 略有损失（PPL 增量 0.3-0.5） |
| **模型体积** | 原始的 ~50%（权重+激活都 8-bit） | 原始的 ~25%（权重 4-bit） |
| **GEMM 计算** | INT8 × INT8 → 真正加速计算 | INT4→dequant→FP16 × FP16 → 计算仍是 FP16 |
| **加速来源** | 带宽 2× + 计算 2× | 主要是带宽 4×，计算不变 |

**速度对比的关键：取决于 batch size 和瓶颈类型**：

- **小 batch decode（memory-bound）**：瓶颈是从 HBM 加载权重。W4A16 的权重只有 W8A8 的一半大小，**加载速度快 2×**。虽然 W4A16 的 GEMM 计算仍是 FP16，但小 batch 下计算不是瓶颈，所以 **W4A16 更快**。这是 LLM 推理最常见的场景（autoregressive decode，batch 较小）。
- **大 batch prefill（compute-bound）**：瓶颈是 GEMM 计算。W8A8 的 INT8 × INT8 Tensor Core 算力是 FP16 的 2×，而 W4A16 仍然做 FP16 计算，所以 **W8A8 更快**。
- **显存受限场景**：W4A16 的模型体积更小（4-bit vs 8-bit 权重），同等 GPU 可以跑更大的模型或更长的 context。

**目前的使用趋势**：
- **开源社区/本地部署**：W4A16（GPTQ/AWQ）占主导。原因是大多数用户在单卡上跑中等大小模型，典型场景是小 batch decode（memory-bound），W4A16 的 4× 带宽节省直接转化为加速，而且模型体积小可以在消费级 GPU 上运行。
- **生产 serving**：逐步转向 **FP8**（H100+ 硬件原生支持），精度接近 FP16，不需要校准和平滑变换，运维最简单。大 batch serving 场景下 W8A8/FP8 的计算加速优势更明显。
- **边缘/移动端**：W4A16 甚至更低 bit（llama.cpp 的 Q4_K_M），追求最大压缩比。

**简单决策规则**：追求精度 → W8A8/FP8；追求最小体积 → W4A16；有 H100+ → 直接 FP8 最省事。

### 5.1 GPTQ — 近似二阶权重量化

> Frantar et al., 2023 · ICLR

**为什么 RTN 在 4-bit 下崩溃？** Round-to-Nearest（直接最近整数取整）在 8-bit 下工作良好，但到 4-bit 时精度急剧下降。根本原因是：INT4 只有 16 个量化级，单个权重的舍入误差相对值很大（最坏情况下误差 = 半个量化步长，而步长 = range/15）。更关键的是，矩阵乘 $Y = WX$ 中每个输出元素是数千个权重的加权和，独立的舍入误差不会完美抵消，而是按 $\sqrt{d}$ 量级累积。RTN 对每个权重独立取整，完全不考虑它们之间的误差相关性。

**GPTQ 的核心思想**：量化不应该是独立的——量化一个权重后产生的误差，可以通过**调整还没量化的权重**来补偿。这个思想来自 OBS（Optimal Brain Surgeon）框架：用 Hessian 矩阵 $H$ 描述每个权重对输出的影响，误差补偿的方向和大小由 $H^{-1}$ 给出。

**从 OBQ 到 GPTQ 的演进**：

OBQ（Optimal Brain Quantizer, Frantar & Alistarh 2022）是 GPTQ 的前身，思路完全一致——逐个量化权重 + Hessian 误差补偿。但 OBQ 每步要**贪心选择**量化哪个权重（选误差最小的），复杂度 $O(d^3)$，量化一个 BERT 就要几小时，完全无法扩展到 LLM。GPTQ 的关键贡献是发现：**固定量化顺序（从左到右）几乎不损失精度**，但把复杂度降到了 $O(d^2)$ 并允许大规模并行。

**核心算法**：对权重矩阵 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，逐行独立处理。对每一行，从左到右逐列量化。量化第 $i$ 列时：

$$\hat{w}_i = Q(w_i)$$

$$\delta_i = w_i - \hat{w}_i$$

$$w_{j>i} \;\leftarrow\; w_{j>i} - \delta_i \cdot \frac{[H^{-1}]_{ij}}{[H^{-1}]_{ii}}$$

**直觉解释**：$\delta_i$ 是第 $i$ 列的量化误差。$[H^{-1}]_{ij} / [H^{-1}]_{ii}$ 描述的是"第 $i$ 列的误差对第 $j$ 列最优调整量的比例"——本质上是在 Hessian 定义的二次误差面上做最优投影。$H = 2X^TX$ 是该行权重对应的 Hessian（$X$ 是校准数据的激活矩阵），只需前向一次即可计算。

**为什么误差补偿有效？** 考虑一个简化例子：两个权重 $w_1 = 0.3, w_2 = 0.7$，对应激活 $x_1, x_2$，输出 $y = w_1 x_1 + w_2 x_2$。如果 $w_1$ 量化后变成 $\hat{w}_1 = 0$（误差 0.3），RTN 下 $w_2$ 仍然是 1（量化到最近整数），输出误差 = $0.3 x_1$。但如果我们把 $w_1$ 的误差补偿到 $w_2$ 上——根据 $x_1, x_2$ 的相关性，适当增大 $w_2$，就能让 $\hat{w}_1 x_1 + \hat{w}_2 x_2 \approx w_1 x_1 + w_2 x_2$。Hessian 矩阵正是编码了这种"哪些权重之间可以互相补偿"的信息。

**工程优化**：

| 优化 | 效果 |
|---|---|
| 固定列序（0→d-1） | 避免 OBQ 逐步贪心排序的 $O(d^3)$ 开销，实测精度几乎不变 |
| Cholesky 预算 $H^{-1}$ | 对 $H^{-1}$ 做 Cholesky 分解，逐列量化时直接读取对应行，无需重复求逆 |
| Block 量化（128 列一批） | 每 128 列为一组，组内逐列量化 + 补偿，组间一次性更新残差。降低显存峰值，利用 GPU 并行 |
| Group quantization（g=128） | 每 128 列共享 scale/zero-point，比 per-tensor 精度高很多，额外存储仅 ~0.5 bit/weight |
| 阻尼项 $H \leftarrow H + \lambda I$ | 防止 $H$ 病态（对角元素太小导致补偿爆炸），$\lambda$ 通常取 $0.01 \cdot \text{mean}(\text{diag}(H))$ |

**数字**：128 条校准样本，单 GPU 几小时量化 175B。4-bit group (g=128) PPL 增量 < 0.5（LLaMA-65B）。3-bit 也能用但精度下降明显。

**局限**：
- **Weight-only**：只量化权重，激活仍 FP16，GEMM 实际是 INT4×FP16 混合计算，端到端加速取决于专用 kernel（如 MARLIN，§9.1）。
- **大 batch 瓶颈**：小 batch 推理是 memory-bound（受限于权重加载带宽），4-bit 权重压缩直接带来加速。但大 batch 推理是 compute-bound，dequant 回 FP16 的开销不可忽视。
- **校准数据敏感性**：Hessian 的质量取决于校准数据的代表性。如果校准数据分布与实际使用场景差异大，量化效果会下降。

### 5.2 AWQ — 激活感知权重量化

> Lin et al., 2024 · MLSys (Best Paper)

**思路的递进**：GPTQ 用二阶 Hessian 信息做误差补偿，虽然效果好但计算量不小（需要求逆、逐列补偿）。AWQ 换了个角度——与其优化取整方式，不如先**保护重要权重**再量化。
**核心洞察**：不是所有权重同等重要。观察 $Y = XW$，输出误差 $\Delta Y = X \cdot \Delta W$，第 $j$ 列权重的量化误差 $\Delta w_j$ 对输出的影响与对应激活 $X_{:,j}$ 的幅度成正比。如果第 $j$ 通道的激活值普遍很大（salient channel），那么即使 $\Delta w_j$ 很小，$X_{:,j} \cdot \Delta w_j$ 也会很大——这些通道的权重是"重要权重"，量化误差会被**激活幅度放大**。**一个自然的想法——为什么不直接跳过重要权重？** 最直觉的做法是：找到 top-1% 的 salient channels，这些通道的权重保持 FP16 不量化，其余量化。但这会破坏矩阵乘的硬件效率——混合精度的列需要特殊处理（类似 LLM.int8() 的问题）。
**AWQ 的解决方案——等价缩放**：借鉴 SmoothQuant 的思路，用数学恒等变换保护重要权重，而不是真的跳过它们。
**算法**：
```
1. 校准数据统计激活幅度：  s_j = mean_i |X_{ij}|   （衡量每个通道的重要性）
2. 对每个通道，按重要性做等价缩放：
      Ŵ_{:,j} = W_{:,j} · α_j       （放大重要权重 → 量化相对误差变小）
      X̂_{:,j} = X_{:,j} / α_j       （缩小对应激活 → 保持恒等）
   数学恒等变换：X̂Ŵ = XW
3. Grid search 找最优 α_j：
      目标：min_{α_j} ‖Q(Ŵ_{:,j}) · X̂_{:,j} − W_{:,j} · X_{:,j}‖
      搜索空间：α_j ∈ [1, max_scale]，按通道独立搜索
```
**为什么放大权重能保护它？** 量化的绝对误差上界约为半个量化步长 $s/2$，而 $s = \max|w| / (2^{b-1}-1)$。当权重乘以 $\alpha > 1$ 后，虽然 $s$ 也会变大，但权重值本身增大了 $\alpha$ 倍，**相对误差** $|\Delta w| / |w|$ 降低了。对于 salient channel，激活幅度大会放大绝对误差，所以降低相对误差特别重要。
**与 SmoothQuant 的联系与区别**：

|        | SmoothQuant            | AWQ                          |
| ------ | ---------------------- | ---------------------------- |
| 目标     | 平滑激活离群值，让 W8A8 可行      | 保护重要权重，提升 W4A16 精度           |
| 变换     | 缩小激活离群通道，放大对应权重        | 放大重要权重，缩小对应激活                |
| 缩放因子来源 | 激活 max 和权重 max 的几何均衡   | 激活 mean（衡量重要性） + grid search |
| 吸收位置   | $s^{-1}$ 融合进 LayerNorm | $\alpha^{-1}$ 融合进前一层         |

本质上都是用**逐通道等价缩放**改变量化难度分布，但优化目标不同：SmoothQuant 追求"激活和权重一样好量化"，AWQ 追求"重要权重的量化误差最小"。

**数字**：
- 校准几分钟，无反向传播，比 GPTQ 快很多。
- 4-bit group (g=128) 在 LLaMA 系列上**优于 GPTQ**（PPL 更低）。
- 配套 TinyChat runtime：端到端 4-bit 推理，边缘设备（Jetson Orin）3×+ 加速。

**局限**：
- **Weight-only**：同 GPTQ，激活仍 FP16，不加速计算本身。
- **不同硬件最优配置不同**：$\alpha$ / group size 的最佳选择与硬件的内存带宽、compute 能力相关。
- **与 GPTQ 互补**：AWQ 保护重要通道，GPTQ 做误差补偿，理论上可以组合使用（先 AWQ 缩放，再 GPTQ 补偿），部分开源实现已支持。

### 5.3 MR-GPTQ — FP4 微缩格式专用量化

> Egiazarian, Panferov et al., 2026 · ICLR 2026

**背景：为什么需要 FP4 专用量化？** NVIDIA Blackwell/AMD 新一代 GPU 原生支持 MXFP4 和 NVFP4 两种微缩浮点格式（microscaling FP4），理论算力翻倍。但直接将 INT4 时代的量化方法（GPTQ、QuaRot 等）套用到 FP4 格式上效果很差——MXFP4 的 E8M0 幂次 scale 精度太粗导致 ~10% 精度下降，而 NVFP4 的极小 group size (16) 使得传统离群值缓解技术失效。MR-GPTQ 是第一个针对 FP4 微缩格式特性定制的量化算法。

**MXFP4 vs NVFP4 两种微缩格式**：

| | MXFP4 | NVFP4 |
|---|---|---|
| 元素格式 | FP4 E2M1 | FP4 E2M1 |
| Block 大小 | 32 | 16 |
| Scale 格式 | E8M0（纯指数，无尾数） | E4M3（标准 FP8） |
| 额外 Scale | 无 | Per-tensor FP32 |
| 每元素平均 bit | 4.25 | 4.5 |
| 优点 | 更省空间，硬件乘法简单 | Scale 精度高，离群保持好 |
| 缺点 | Power-of-2 scale 误差大 | 略多存储 |

**核心发现——旋转对 FP4 的效果与 INT4 不同**：

MR-GPTQ 首先从理论上分析了 Hadamard 旋转对两种 FP4 格式的影响：

- **MXFP4**：旋转有益。E8M0 scale 的精度瓶颈在于 power-of-2 量化，旋转使权重分布更均匀后，scale 的粗粒度影响降低。
- **NVFP4**：旋转**可能有害**。NVFP4 的小 group size (16) 天然提供了较好的离群值保持（top element 被"提升"到 E4M3 精度），旋转反而破坏了这一优势，把 top element 的误差扩散到整个 group。
- **关键结论**：存在一个"交叉点"——group size 较小时 Laplace（原始）分布的 MSE 低于 Normal（旋转后）分布，group size 较大时反过来。

**MR-GPTQ 的三个核心改进**：

1. **Block-wise Hadamard 旋转**：不用全局 Hadamard，而是用与 quantization group size 匹配的 block-diagonal Hadamard 矩阵（如 $H_{32}$），在 group 内部做旋转。权重侧离线融合进权重，激活侧在线计算。由于 block size ≤ 128，变换是 memory-bound 的，任何旋转矩阵（不限于 Hadamard）的开销都相同。

2. **MSE-optimized scale search**：针对 NVFP4 的 per-group + per-tensor 双层 scale 结构，用交替优化（alternating optimization）搜索最小 MSE 的 scale 组合，而非简单取 absmax。对 MXFP4，由于旋转后分布均匀，固定 static scale 即可。

3. **Static activation reordering**：原始 GPTQ 的 "act-order"（按 Hessian 对角线排序列）虽然提升精度，但需要运行时动态 shuffle 列，带来 10-20% 推理开销。MR-GPTQ 改为：先固定 scale 和 grid，shuffle 列做 GPTQ 补偿，再 shuffle 回来——获得同等精度收益但**零运行时开销**。

**GPU kernel 支持（QuTLASS）**：MR-GPTQ 配套发布了 Blackwell 架构优化 kernel 库 QuTLASS：
- 轻量级 fused kernel 做在线旋转 + 量化 + scale 计算，延迟可忽略
- 支持 SM100 (B200) 和 SM120 (RTX5090) 两种 compute capability
- MXFP4 kernel 吞吐量甚至**超过理想 NVFP4** 矩阵乘

**实验结果**（Llama-3.1-8B-Instruct, W4A4）：

| 方法 | 格式 | Avg. Recovery |
|---|---|---|
| RTN | NVFP4 | 94.7% |
| SmoothQuant | NVFP4 | 96.5% |
| GPTQ | NVFP4 | **97.2%** |
| MR-GPTQ | NVFP4 | **97.0%** |
| RTN | MXFP4 | 88.4% |
| QuaRot | MXFP4 | 91.3% |
| MR-GPTQ | MXFP4 | **95.2%** |

- MXFP4 上 MR-GPTQ 将精度恢复从 88% 提升到 95%，接近 NVFP4 水平
- 端到端推理加速：B200 上 **2.2×**，RTX5090 上 **4×**（对比 FP16 baseline）
- Layer-wise 加速：B200 **3.6×**，RTX5090 **6×**

**与 GPTQ/AWQ 的关系**：MR-GPTQ 是 GPTQ 在 FP4 微缩格式上的自然扩展。传统 GPTQ 针对均匀 INT 格式设计，MR-GPTQ 通过 block-wise 旋转和格式感知优化适配了 FP4 的非均匀量化特性。

---

## 6 W4A4 + KV4：全链路低 bit 量化

> W4A16 只量化了权重，激活仍用 FP16——GEMM 还是 INT4×FP16 混合计算，加速有限。终极目标是 **权重、激活、KV cache 全部 4-bit 甚至更低**，让 GEMM 完全在低 bit 整数上执行。但 §3 分析过，激活中的离群值是全链路量化的最大障碍。SmoothQuant 的平滑技巧在 4-bit 下不够用了——需要更强的手段。

### 6.1 QuaRot — 正交旋转分散离群

> Ashkboos et al., 2024

**为什么 SmoothQuant 的思路在 4-bit 下不够用了？** SmoothQuant 用逐通道缩放把激活离群值"搬"到权重，在 8-bit 下效果很好。但 4-bit 只有 16 个量化级，即使平滑后激活的动态范围仍然太大——缩放只能改变每个通道的幅度，不能改变能量在维度间的分布。需要更强的变换。

**核心思想**：离群值本质是**能量集中在少数维度**。如果用**正交变换旋转坐标系**，把能量均匀分散到所有维度，离群就消失了——而且正交变换不改变向量范数和内积，模型输出不变。

**数学原理**：对任意正交矩阵 $R$（$R R^T = I$），有：

$$Y = XW = (XR^T)(RW)$$

设 $\tilde{X} = XR^T$，$\tilde{W} = RW$，则 $Y = \tilde{X}\tilde{W}$，输出完全不变。关键性质是：**随机正交变换（如 Hadamard 矩阵）具有"民主化"效应**——它把集中在少数维度的能量均匀分散到所有维度。直觉上，Hadamard 矩阵的每一行都是 $\pm 1/\sqrt{d}$ 的等幅组合，旋转后每个新维度都是原始所有维度的等权混合，任何单一维度的极端值都被"稀释"了。

**为什么用 Hadamard 矩阵而不是任意正交矩阵？** 一般正交矩阵乘法需要 $O(d^2)$ 复杂度，与 GEMM 同量级，作为预处理太贵。Hadamard 矩阵有快速算法（类似 FFT），复杂度只有 $O(d \log d)$，远小于 GEMM 的 $O(d^2)$。

**在 Transformer 中的插入位置**：不能简单地对整个模型做一次旋转了事——Transformer 中有 LayerNorm、softmax 等非线性操作会"打断"旋转的可吸收性。QuaRot 的关键工程贡献是识别出哪些旋转可以离线吸收进权重，哪些必须在线计算：

```
Transformer Block 中的旋转插入：

1. 线性层之间：R 可吸收进权重（离线）
   W_q, W_k, W_v ← R · W_q, R · W_k, R · W_v
   W_o ← W_o · Rᵀ
   → 推理时零开销

2. LayerNorm 之后：RMSNorm(x) · W = RMSNorm(x) · Rᵀ · (R · W)
   RMSNorm 对旋转不可交换 → 需在线计算 xRᵀ
   → 用快速 Hadamard 变换，O(d log d)

3. Attention 内部：Q, K 需要同时旋转保持内积不变
   Q' = QRᵀ_head,  K' = KRᵀ_head  （per-head 旋转）
   → Q'K'ᵀ = QRᵀR Kᵀ = QKᵀ  ✓
   → KV cache 也被旋转 → 离群被分散 → KV cache 也能低 bit 量化
```

**效果**：旋转后激活的通道间方差显著降低，原本集中在 0.1% 通道的离群值被分散到所有维度。在此基础上，直接用简单的 RTN 或 GPTQ 就能实现 W4A4+KV4 全链路量化。

**数字**：LLaMA-2-70B W4A4 量化，PPL 增量约 0.5-1.0，远优于不旋转直接量化（崩溃）。

**局限**：
- **在线 Hadamard 变换开销**：虽然 $O(d \log d)$ 远小于 GEMM，但需要定制 CUDA kernel，对现有推理框架有侵入性。
- **随机种子敏感**：随机 Hadamard 矩阵的选择影响最终精度，不同种子结果方差可达 0.2-0.3 PPL，实践中需多次试验取最佳。
- **需修改 attention kernel**：KV cache 被旋转后，attention 计算的内存布局和 kernel 都需要适配。

### 6.2 SpinQuant — 学习最优旋转

> Liu et al., 2025 · ICLR

**思路的递进**：QuaRot 用随机 Hadamard 矩阵，效果依赖随机种子的运气。自然的问题：能不能**学习一个最优的旋转矩阵**，而不是随机抽一个？

**核心挑战**：旋转矩阵 $R$ 必须是正交的（$RR^T = I$），这是一个约束优化问题。如果直接用梯度下降更新 $R$，更新后 $R$ 不再正交，模型输出就会改变。SpinQuant 用 **Cayley 参数化**巧妙解决了这个问题：

$$R = (I + A)^{-1}(I - A), \qquad A = -A^T \text{（反对称矩阵）}$$

当 $A$ 是反对称矩阵时，上述 Cayley 变换的输出 $R$ **自动满足正交性**。这样只需要对 $A$ 做无约束优化（$A$ 的上三角元素是自由参数），梯度下降自然保证 $R$ 始终正交。

**优化目标**：注意 SpinQuant **只优化旋转矩阵 $R$，不优化模型权重本身**——它不是端到端的 QAT，而是在 PTQ 之前寻找最佳的预处理旋转。给定校准数据，最小化"旋转 + 量化"后的输出重构误差：

$$\min_A \sum_{\text{layers}} \|\text{Quant}(\tilde{W}_l) \cdot \tilde{X}_l - W_l \cdot X_l\|_F^2$$

其中 $\tilde{W}_l = R_l W_l$，$\tilde{X}_l = X_l R_l^T$ 是旋转后的权重和激活，$\text{Quant}(\cdot)$ 是模拟量化。优化变量只有 $A$（决定 $R$），模型权重 $W$ 完全冻结。优化结束后，得到最优旋转 $R^*$，然后再用 GPTQ、AWQ 或 RTN 等标准 PTQ 方法对旋转后的权重 $R^* W$ 做实际量化。也可以直接优化端到端 loss（更贵但更准），但本质上仍然只调 $R$。

**训练成本**：几十到几百步梯度下降（只优化 $R$ 的参数 $A$，不动模型权重），远小于 QAT（不需要完整训练），但高于纯 PTQ（需要反向传播来计算 $\partial \mathcal{L}/\partial A$）。可以只对部分"敏感"层学习旋转，其余层用随机 Hadamard，进一步节省成本。

**与 QuaRot 的对比**：

| | QuaRot | SpinQuant |
|---|---|---|
| 旋转矩阵 | 随机 Hadamard | 学习（Cayley 参数化） |
| 优化成本 | 零（直接用） | 几十到几百步梯度下降 |
| 精度 | 好，但方差大 | 更好，方差小 |
| W4A4 场景 | 可用但精度有波动 | 稳定优于 QuaRot 0.2-0.5 PPL |

**局限**：
- **成本介于 PTQ 和 QAT 之间**：需要反向传播来优化 $A$，比纯 PTQ（如 GPTQ/AWQ）慢，但远快于全参数 QAT。
- **旋转结构需 kernel 配合**：同 QuaRot，在线 Hadamard 变换需要定制 kernel。
- **与现有量化方法组合**：SpinQuant 解决的是"消除离群值"，消除后仍需搭配 GPTQ/AWQ/RTN 做实际的权重量化。

### 6.3 KIVI — KV Cache 2-bit 量化

> Liu et al., 2024 · ICML

**问题转向**：前面解决了权重和激活的全链路量化。但长上下文场景下还有一个显存大户—— **KV cache**。

**为什么 KV cache 是长上下文的瓶颈？** LLM 推理时，每一层的 attention 都需要缓存所有历史 token 的 Key 和 Value 向量。以 LLaMA-2-7B 为例：

- 每层 KV cache：$2 \times \text{seq\_len} \times d_{\text{head}} \times n_{\text{heads}} \times 2\text{B}$（FP16）
- 32 层 × 128K context：$2 \times 128000 \times 128 \times 32 \times 32 \times 2 \approx 32\text{GB}$

这远超模型权重本身（~14GB in FP16），而且 **KV cache 随 seq_len 线性增长**——context 越长，KV cache 越大，最终成为显存瓶颈，限制了最大 batch size 和最大 context length。

**为什么不能直接对 KV cache 用统一的量化方法？** KIVI 的关键贡献是发现 **Key 和 Value 的离群结构完全不同**，需要不同的量化策略：

| 张量 | 离群特征 | 原因 | 适合的量化粒度 |
|---|---|---|---|
| **Key** | 离群集中在**固定通道**（跨 token 稳定） | Key 投影权重的某些输出通道天然产生大值，与输入 token 无关 | **Per-channel** |
| **Value** | 离群在 **token 维**变化，不同 token 的 Value 幅度差异大 | Value 的幅度与 token 的语义重要性相关，如 BOS token 通常有异常大的 Value | **Per-token** |

如果对 Key 用 per-token 量化，固定通道的离群值会拉大每个 token 的 scale，浪费其余通道的精度；如果对 Value 用 per-channel 量化，某些异常 token 会拉大该通道的 scale。KIVI 的做法是"**按离群结构选粒度**"。

**方法详解**：

**Step 1：观察 Key 和 Value 的离群模式差异**

KIVI 首先在多个模型（LLaMA-2-7B/13B、Mistral-7B 等）上系统分析了 KV cache 的数值分布：

- **Key 矩阵** $K \in \mathbb{R}^{T \times d}$：对每个通道 $j$ 统计所有 token 的值 $K_{:,j}$，发现**少数固定通道的值始终比其余通道大 10-100×**，而且这些"离群通道"跨不同输入 token 几乎不变——这与 §3 讨论的 emergent outlier 现象一致，本质是 Key 投影权重 $W_K$ 的某些输出通道天然产生大值。
- **Value 矩阵** $V \in \mathbb{R}^{T \times d}$：同样分析发现 Value 的离群**不在固定通道，而在特定 token 上**——某些 token（如 BOS、标点符号、高频功能词）的 Value 向量整体幅度远大于普通 token，而同一通道内不同 token 的值变化很大。

这个不对称性决定了不能用同一种量化粒度处理两者。

**Step 2：选择匹配的量化粒度**

```
Key cache K ∈ ℝ^{T×d}（离群在通道维，跨 token 稳定）:
  → Per-channel 量化：对每个通道 j，计算 scale 和 zero-point
    s_j = (max(K_{:,j}) - min(K_{:,j})) / (2^b - 1)
    z_j = round(-min(K_{:,j}) / s_j)
  → 固定通道的大值被该通道自己的 s_j 覆盖，不影响其余通道

Value cache V ∈ ℝ^{T×d}（离群在 token 维，跨通道变化）:
  → Per-token 量化：对每个 token t，计算 scale 和 zero-point
    s_t = (max(V_{t,:}) - min(V_{t,:})) / (2^b - 1)
    z_t = round(-min(V_{t,:}) / s_t)
  → 异常 token 的大值被该 token 自己的 s_t 覆盖，不影响其余 token

均使用非对称量化（z ≠ 0），精度 2-bit，group size 可调（32/64/128）
```

**Step 3：Residual token 机制处理增量更新**

自回归推理时，每步生成一个新 token，其 Key/Value 向量需要追加到 cache 中。这带来一个问题：**新 token 的值可能超出已有 cache 的量化范围**，尤其是 Key 的 per-channel 量化——新 token 的某通道值可能比历史所有 token 都大，需要更新该通道的 $s_j$，这意味着要对整个 Key cache 重新量化，开销极大。

KIVI 的解决方案是 **residual token buffer**：

```
KV cache 结构：
┌─────────────────────────┐  ┌─────────────┐
│  量化 cache（2-bit）      │  │ 残差 buffer   │
│  token 1 ~ token T-R     │  │ (FP16, 最近R个)|
│  已量化，不再修改          │  │  token T-R+1  │
│                          │  │  ...          │
│                          │  │  token T      │
└─────────────────────────┘  └─────────────┘

每当残差 buffer 满（积累 R 个 token）→ 批量量化这 R 个 token
→ 追加到量化 cache → 清空 buffer
```

- 最近的 $R$ 个 token 保持 FP16 不量化（$R$ 通常取 128）——这些 token 通常是 attention 权重最大的部分（局部性），保持高精度最重要
- 只有"历史"token 被量化，且一旦量化就不再修改——避免了反复更新 scale 的问题
- Key 的 per-channel scale 在每次批量量化时，基于当前 buffer 中 $R$ 个 token 计算（而非全局更新）

**Step 4：量化后的 Attention 计算**

量化后 Attention 的计算需要分成两部分——量化部分和 FP16 残差部分：

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q \cdot [K_{\text{quant}}; K_{\text{res}}]^T}{\sqrt{d}}\right) \cdot [V_{\text{quant}}; V_{\text{res}}]$$

其中 $K_{\text{quant}}, V_{\text{quant}}$ 是 2-bit 量化部分，$K_{\text{res}}, V_{\text{res}}$ 是 FP16 残差 buffer。实际实现中：

1. 计算 $Q \cdot K_{\text{quant}}^T$：需要对 2-bit Key 做 dequant（乘 scale + 减 zero-point），然后与 Q 做矩阵乘
2. 计算 $Q \cdot K_{\text{res}}^T$：标准 FP16 矩阵乘
3. 拼接后过 softmax
4. $\text{softmax\_weights} \cdot [V_{\text{quant}}; V_{\text{res}}]$：同样分两部分 dequant + 乘

这个分段计算需要定制的 attention kernel 支持（标准 FlashAttention 不支持混合精度 KV cache）。

**为什么能做到 2-bit？** 关键在于 KV cache 量化的"容错空间"比权重量化大得多：

1. **Key 侧**：Attention score 是 $QK^T / \sqrt{d}$，经过 softmax 后动态范围被大幅压缩。Key 的小量化误差导致 attention score 的微小偏移，但 softmax 的"赢者通吃"特性使得排名几乎不变——真正重要的 token 的 attention 权重仍然最大。
2. **Value 侧**：$\text{softmax}(QK^T) \cdot V$ 是加权平均，attention 权重本身是稀疏的（大部分 token 权重接近 0），所以只有少数"被关注"的 token 的 Value 精度真正影响输出。而这些重要 token 往往在残差 buffer 中（最近的 token），保持了 FP16 精度。
3. **非对称量化的额外收益**：2-bit 对称量化只有 {-1, 0, +1, 2} 四个值，而非对称量化通过 zero-point 偏移可以更灵活地覆盖任意区间，对 KV cache 中常见的偏斜分布（如 Value 中某些 token 全为正值）特别有效。

**显存分析**（LLaMA-2-7B, 4K context, per head）：

| 精度 | 单层 KV cache | 32 层总计 | 128K context |
|---|---|---|---|
| FP16 | $2 \times 4096 \times 128 \times 2\text{B} = 2\text{MB}$ | 64 MB | ~2 GB |
| KIVI 2-bit | $2 \times 4096 \times 128 \times 0.25\text{B} + \text{scales} \approx 0.28\text{MB}$ | ~9 MB | ~0.28 GB |
| 节省 | **~7×** | | |

加上 residual buffer（$R=128$ tokens, FP16）的固定开销约 2MB/层，总节省仍然显著。实际测试中 128K context 的峰值显存从 >32GB 降到 ~12GB。

**数字**：
- Tuning-free，无需校准数据——scale/zero-point 完全在线计算。
- 2-bit KV 在 LLaMA-2-7B/13B、Mistral-7B 上 PPL 增量通常 < 0.2。
- 峰值显存降 **2.6×+**，最大 batch size 显著提升（长上下文场景下收益更明显）。
- 可与权重量化（GPTQ/AWQ）正交使用——权重用 4-bit，KV cache 用 2-bit，叠加压缩。

**局限**：
- **Attention kernel 适配**：2-bit KV cache 需要定制的 packing/unpacking 逻辑和内存布局，标准 FlashAttention 无法直接使用。需要实现混合精度 attention kernel（量化部分 + FP16 残差部分的分段计算）。
- **Key 的 per-channel 统计依赖 batch 量化**：每次批量量化 $R$ 个 token 时，per-channel scale 基于这 $R$ 个 token 计算。如果这批 token 不具有代表性（如前 128 个 token 与后续分布差异大），可能导致 scale 不够准确。实际中 $R=128$ 通常足够稳健。
- **与 GQA 的交互**：GQA 中多个 query head 共享同一组 KV head，KV cache 已经被压缩（如 LLaMA-3 的 8 KV heads vs 32 query heads），进一步量化到 2-bit 的精度影响需要更仔细评估——每个 KV head 服务更多 query head，其误差被放大的机会也更多。
- **Residual buffer 大小的选择**：$R$ 太小则最近 token 也被量化，损害局部 attention 精度；$R$ 太大则 FP16 部分占用过多显存，削弱量化收益。实际中 $R = 128$ 是一个经验性的平衡点。

### 6.4 MambaQuant — 非 Transformer 架构的量化挑战

> Xu, Yue et al., 2025 · ICLR 2025

**为什么要关注 Mamba 量化？** 前面 §4-§6 的所有方法都假设目标架构是 Transformer。但 Mamba（基于 Selective State Space Model）正在成为 Transformer 的重要竞争者——在长序列任务上具有线性复杂度优势。当我们尝试将 Transformer 上成功的量化方法（如 QuaRot）直接应用到 Mamba 时，却发现**精度崩溃**：QuaRot 在 Vim-T 上 W8A8 精度下降 21%。MambaQuant 是第一个系统研究 Mamba 量化并提出解决方案的工作。

**Mamba 量化的三大挑战**：

1. **Gate/Output 投影层的离群值**：Mamba block 中 gate projection 的权重和 output projection 的输入激活存在显著离群值，类似 Transformer 但分布模式不同。

2. **Parallel Scan (PScan) 放大离群**：Mamba 的核心操作是 PScan——连续对固定参数矩阵 $A$ 做自乘：$h(t) = A h(t-1) + B x(t)$。高值通道在 PScan 中被反复放大，低值通道被抑制，导致输出的通道间方差差异**远大于 Transformer**。

3. **Hadamard 变换失效**：在 Transformer 上 Hadamard 旋转能有效均匀化 max value（QuaRot 的核心）。但 MambaQuant 从理论上证明：Hadamard 变换**无法保证通道方差一致**。具体来说，Hadamard 变换后第 $l$ 个通道的方差为：

$$(\mathbf{C}_{\mathbf{X}\mathbf{H}})_{ll} = \frac{1}{n-1}\sum_{j=1}^{m}\left(\sum_{i=1}^{m}H_{il}K_{ij}\right)^2\lambda_j$$

由于 $H$ 是固定矩阵而 $K$（特征向量）和 $\lambda$（特征值）随输入变化，Hadamard 无法适应不同的通道分布，导致旋转后方差仍然不一致。

**MambaQuant 的解决方案**：

**Offline 模式——KLT 增强旋转**：将 Karhunen-Loève 变换（KLT）与 Hadamard 矩阵组合。KLT 通过特征值分解找到数据的主成分方向，组合矩阵 $H_K = KH$（先 KLT 再 Hadamard）使得：

$$(\mathbf{C}_{\mathbf{X}\mathbf{H}_K})_{ll} = \frac{1}{(n-1)m}\sum_{j=1}^{m}\lambda_j$$

**所有通道方差完全相同**——等于特征值的均值。KLT 利用校准数据离线计算，不增加推理开销。应用于 LoRA 模块和 block 间连接（output/gate/state projection）。

**Online 模式——Smooth-Fused 旋转**：对无法离线吸收的位置（如 PScan 输出），在 Hadamard 变换前先做 SmoothQuant 式的通道缩放来均匀化方差，再用 Hadamard 均匀化 max value。创新点是将 smoothing 参数巧妙融合进 Mamba 结构：
- **Output projection**：将 SiLU 扩展为 Smooth-SiLU（S-SiLU），smoothing 因子吸收进 gate 和 output 权重
- **Matrix multiplication**：smoothing 因子分别吸收进 B projection 和 C projection 权重，对 $\Delta$ 的指数运算做 addcmul 处理

**实验结果**：

| 模型 | 方法 | W8A8 | W4A8 |
|---|---|---|---|
| Vim-T (76.1%) | QuaRot | 59.3% | 52.7% |
| Vim-T (76.1%) | **MambaQuant** | **75.6%** | **72.1%** |
| Vim-S (80.5%) | QuaRot | 73.8% | 72.0% |
| Vim-S (80.5%) | **MambaQuant** | **80.3%** | **79.4%** |
| Mamba-LLM | QuaRot | 显著退化 | — |
| Mamba-LLM | **MambaQuant** | <1% 精度损失 | ~1% 精度损失 |

**关键启示**：**量化方法不能盲目跨架构迁移**。Transformer 上的最佳实践在 SSM/Mamba 架构上可能完全失效——需要理解目标架构的特有数值特征（如 PScan 的放大效应）并定制方案。这对未来混合架构（如 Jamba = Mamba + Attention）的量化也有启示。

---

## 7 极端路线：1-bit / 三值网络

> 前面 §4-§6 都是压缩已有模型（PTQ）。另一条完全不同的路线：**从头训练就用极低 bit**，让模型天生适应量化。这是 QAT（§2.1）在 LLM 时代的激进应用。

**为什么要走这条路？** PTQ 方法（GPTQ、AWQ 等）本质上是在"抢救"一个按 FP16/BF16 训练好的模型——训练时权重分布是为高精度优化的，硬塞到 4-bit 不可避免地丢失信息。一个自然的想法：如果模型**从训练开始就在低 bit 下优化**，权重分布会自然适应低精度表示，理论上能获得比 PTQ 更好的精度。代价是需要完整的训练流程。

**重要澄清：这里的"低 bit 训练"是模拟量化，不是真正的低精度计算。** §2.1 介绍的 QAT 框架在这里同样适用——前向传播通过伪量化节点（quantize → dequantize）让模型"看到"量化误差，但**实际的 GEMM 仍然用 FP16/FP32 Tensor Core 执行**，梯度也在 FP32 latent weights 上累积。低 bit 权重只在推理时导出使用。换句话说，BitNet 的训练成本与 FP16 模型相当——它节省的是**推理**而非训练的计算量。真正降低训练计算精度的方法见 §8（FP8/FP4 训练）。

### 7.1 经典路线（CNN 时代）

| 方法 | 年份 | 权重 | 激活 | 核心操作 | 关键思想 |
|---|---|---|---|---|---|
| BinaryConnect | 2015 | {-1, +1} | 实值 | 乘法 → 符号翻转 | 首次证明二值权重可以训练 |
| BNN | 2016 | {-1, +1} | {-1, +1} | XNOR + popcount | 权重和激活都二值化 |
| TWN | 2016 | {-α, 0, +α} | 实值 | 阈值置零 + 缩放 | 加入零值，稀疏化 |
| XNOR-Net | 2016 | {-1, +1} | {-1, +1} | XNOR + popcount + 通道级 scale | 加入实值 scale 因子恢复精度 |

**这些方法的共同框架**：训练时维护一份 FP32 的 latent weights（真正的可训练参数）。前向传播时将 latent weights 量化为二值/三值，然后**立即 dequant 回 FP16/FP32**再做标准 GEMM——这就是 §2.1 介绍的伪量化（fake quantization），模型"看到"了量化误差但实际计算仍在浮点精度下进行，**训练时没有速度收益**。反向传播用 STE（§1.4）让梯度穿过 sign 函数，更新 FP32 latent weights。推理时丢弃 latent weights，导出真正的二值/三值权重，此时才用专用 kernel（如 XNOR+popcount 或 bitnet.cpp）获得加速。

**为什么在 CNN 上没有成功？** 理论上 32× 压缩（FP32 → 1-bit），但实际问题很多：
- **精度鸿沟**：ImageNet 上 ResNet-18 二值化后 top-1 精度下降 10-15%，大型网络（ResNet-50+）下降更多。
- **硬件没有真正加速**：XNOR + popcount 在 CPU 上确实快，但现代 GPU 的 Tensor Core 专为 FP16/INT8 矩阵乘优化，二值运算反而因为缺乏硬件支持而更慢。
- **训练不稳定**：STE 是一个非常粗糙的梯度近似（sign 的真实梯度是 0，STE 假装是 1），训练过程中容易出现梯度震荡和收敛困难。

### 7.2 BitNet — LLM 时代的 1-bit 网络

> Wang et al., 2023 (Microsoft Research)

BitNet 是首个将极低 bit 训练扩展到 LLM 规模的工作。核心改造是用 **BitLinear** 替代所有 `nn.Linear`：

**权重二值化**：
$$\hat{W} = \text{Sign}(W - \mathbb{E}[W])$$

先减去均值（中心化），再取符号。减均值很关键——如果权重分布不对称（均值 ≠ 0），直接 Sign 会让 +1 和 -1 的数量严重不平衡，浪费表达能力。

**激活量化**：激活量化到 $b$-bit（论文中用 8-bit），使用 absmax 对称量化：
$$\hat{X} = \text{Quant}(X) = \text{Clip}\!\left(\left\lfloor \frac{X}{\max|X|} \cdot (2^{b-1} - 1) \right\rceil, -2^{b-1}+1, 2^{b-1}-1\right)$$

**完整的 BitLinear 前向**：
```
训练时的 BitLinear 前向（伪量化）：
1. LayerNorm(x)                         ← 稳定输入分布
2. W_bin = Sign(W - mean(W))            ← 权重二值化（值为 ±1，但数据类型仍 FP16）
3. X_q = absmax_quant(x, 8bit)          ← 激活量化再 dequant（值含量化误差，类型仍 FP16）
4. Y = X_q @ W_bin                      ← 标准 FP16 GEMM（训练时无加速）
5. Y = Y · (β · γ / Q_max)             ← rescale 还原数值范围
   其中 β = mean(|W|), γ = max(|X|)

推理时的 BitLinear（真正低 bit）：
  导出的 W_bin 为真正的 1-bit packed 格式
  → 专用 kernel 实现加减法替代乘法 → 获得实际加速
```

**训练**：STE + FP32 latent weights。与经典方法相同：前向用 {-1, +1} 权重，反向传播梯度到 FP32 latent weights 更新。

**关键发现**：BitNet 展示了 **scaling law 对 1-bit 模型依然成立**——随着模型增大，1-bit 和 FP16 之间的精度差距在缩小。这暗示足够大的 1-bit 模型可能匹配较小的 FP16 模型性能。

### 7.3 BitNet b1.58 — 从 {-1, +1} 到 {-1, 0, +1}

> Ma et al., 2024 (Microsoft Research)

BitNet b1.58 是 BitNet 的关键改进：权重从二值 {-1, +1} 扩展到三值 {-1, 0, +1}，每个权重的信息量为 $\log_2 3 \approx 1.58$ bit。

**权重量化**：

$$\hat{W} = \text{RoundClip}\!\left(\frac{W}{\gamma},\,-1,\,1\right), \qquad \gamma = \frac{\|W\|_1}{nm}$$

$\gamma$ 是权重绝对值的均值（L1 归一化因子）。归一化后权重大部分落在 $[-1, 1]$ 内，取整到 {-1, 0, +1}。

**0 的引入为什么是关键改进？**

1. **矩阵乘变成纯加减法**：$y_i = \sum_j w_j x_j$，当 $w_j \in \{-1, 0, +1\}$ 时，$w_j x_j$ 要么是 $+x_j$，要么是 $-x_j$，要么是 0（跳过），完全消除了乘法运算。相比 BitNet 的 {-1, +1}（只有加减），0 引入了**稀疏性**——部分权重直接跳过计算。
2. **更好的特征选择**：0 允许模型"忽略"某些输入维度，相当于隐式做了特征选择。{-1, +1} 强制每个输入都参与计算，即使某些输入对当前输出不重要。
3. **精度显著提升**：同等模型大小下，b1.58 的精度大幅优于 BitNet（1-bit），在 3B 参数规模上已经可以**匹配 FP16 LLaMA 的性能**。

**与 PTQ 方法的根本区别**：

| | PTQ（GPTQ/AWQ） | BitNet b1.58 |
|---|---|---|
| 起点 | FP16 训练好的模型 | 从头训练 |
| 权重分布 | 为 FP16 优化，强行压缩到低 bit | 训练过程中自然适应三值 |
| 精度 | 4-bit 较好，3-bit 以下崩溃 | 1.58-bit 可匹配 FP16 |
| 成本 | 几分钟到几小时 | 完整训练（与 FP16 训练同等成本） |
| 推理硬件 | 现有 GPU + 专用 kernel | 需要全新的硬件/kernel 支持 |

**挑战与未解决问题**：
- **需要专用硬件/kernel**：现有 GPU 的 Tensor Core 不支持三值矩阵乘。虽然理论上加减法比乘法快得多，但没有硬件原生支持就无法兑现加速优势。Microsoft 已发布 bitnet.cpp 推理引擎。
- **训练基础设施**：训练时仍需 FP32/FP16 latent weights + STE，训练成本与 FP16 模型相当。优势只在推理侧。
- **对齐/RLHF 兼容性**：大模型训练后还需要 SFT、RLHF、DPO 等对齐步骤。这些步骤在三值权重约束下的稳定性和效果尚未充分验证。
- **Scaling 到更大模型**：目前公开的实验最大到 3B 参数，百亿以上规模的验证仍缺失。

---

## 8 训练侧低精度

> §4-§6 聚焦 PTQ 推理量化，§7 的 BitNet 走的是"从头用低 bit 训练"路线——但 BitNet 的目标仍是获得一个低 bit **推理**模型，训练过程本身仍用 FP16/FP32。本节关注的是另一个方向：**训练计算本身的低精度化**——不是为了得到低 bit 模型，而是为了**让训练过程更快、更省显存**。混合精度训练的基础已在 §1.5 介绍，本节聚焦更激进的 FP8/FP4 训练和低精度优化器。

### 8.1 FP8 训练与 Scaling

> Micikevicius et al., 2022

**为什么要从 FP16/BF16 进一步降到 FP8？** 混合精度训练已经把 GEMM 从 FP32 降到了 FP16/BF16（§1.5），但训练规模还在指数增长。FP8 Tensor Core 的算力是 FP16 的 **2×**（H100：FP8 1979 TFLOPS vs FP16 990 TFLOPS），同时显存带宽占用也减半。如果能把训练的 GEMM 进一步降到 FP8，理论上可以再获得 2× 加速。

**为什么 FP8 训练比 FP16 训练难？** FP8 只有 8 bit，能表示的数值范围和精度都很有限（§1.6）。关键矛盾是训练中不同张量的数值特征差异很大：

- **权重**：分布相对集中，绝对值通常在 $[0, 几]$ 范围内，需要精度 → 用 **E4M3**（3 bit 尾数，范围 ±448）
- **激活**：分布类似权重但可能有离群值 → 也用 **E4M3**
- **梯度**：分布极端——大部分接近 0，少数极大值（尤其训练早期和尾部层），需要大动态范围 → 用 **E5M2**（2 bit 尾数，范围 ±57344）

| 格式 | 指数 | 尾数 | 动态范围 | 典型用途 |
|---|---|---|---|---|
| **E4M3** | 4 | 3 | ±448 | 前向：权重 / 激活 |
| **E5M2** | 5 | 2 | ±57344 | 反向：梯度 |

**Scaling 是 FP8 训练的核心技术**。即使选对了 E4M3/E5M2 格式，FP8 的动态范围仍然有限。如果张量的实际值远小于 FP8 最大值，大部分 FP8 量化级被浪费（利用率低）；如果实际值超过 FP8 最大值，直接溢出到 NaN/Inf。Scaling 就是在 cast 到 FP8 之前先缩放到合适的范围：

$$x_{\text{fp8}} = \texttt{cast\_fp8}(x \,/\, s), \qquad s = \frac{\max|x|}{f_{\max}}$$

其中 $f_{\max}$ 是 FP8 格式的最大可表示值（E4M3: 448, E5M2: 57344）。反量化时乘以 $s$ 还原。

**Scaling 策略对比**——粒度越细精度越高，但开销也越大：

| Scaling 策略 | 说明 | 精度 | 开销 | 代表 |
|---|---|---|---|---|
| **Per-tensor** | 整张量一个 $s$ | 低（离群值拉大 s） | 最低 | Transformer Engine 默认 |
| **Per-token × per-channel** | 激活按 token 一个 $s$，权重按 output channel 一个 $s$ | 中（分维度适配） | 中 | 部分学术工作 |
| **Per-block (MXFP8)** | 固定 tile（如 32 元素）一个 $s$，$s$ 本身用 8-bit 存储 | 高（局部精度好） | 较高 | Blackwell MXFP8 |
| **Delayed scaling** | 不用当前 step 的 $\max\lvert x \rvert$（需要额外 pass），而是用前几步的 amax 估计 $s$ | 中 | 低（但有时序依赖） | NVIDIA recipe |

**Delayed scaling 详解**：Per-tensor scaling 的一个实际问题是——要算 $s = \max|x| / f_{\max}$，需要先跑一遍前向得到 $x$，才能算 $s$，然后再用 $s$ 做量化跑第二遍前向。为了避免两遍开销，delayed scaling 用前几步的历史 $\max|x|$ 来估计当前 step 的 $s$。假设相邻 step 的统计变化不大（通常成立），这种"延迟"一步的近似足够好。

**端到端 FP8 训练的完整 recipe**：

```
前向传播：
  权重 cast FP8 E4M3 (per-tensor scaling)
  激活 cast FP8 E4M3 (per-tensor scaling)
  GEMM: FP8 × FP8 → FP16/FP32 累积（Tensor Core 内部用高精度累加）

反向传播：
  梯度 cast FP8 E5M2 (per-tensor scaling)
  权重梯度 GEMM: FP8 × FP8 → FP16/FP32 累积

参数更新：
  FP32 master weights + FP32 optimizer states（同 FP16 混合精度）
```

**硬件支持现状**：
- **H100 (Hopper)**：原生 FP8 Tensor Core，支持 E4M3 和 E5M2，per-tensor scaling。NVIDIA Transformer Engine 库提供开箱即用的 FP8 linear layer。
- **B200 (Blackwell)**：进一步支持 MXFP8（per-block scaling，block size 32），精度更高，且 $s$ 用 8-bit 共享指数存储，元数据开销可控。
- **AMD MI300X**：支持 FP8 但生态和优化程度落后于 NVIDIA。

**瓶颈**：FP8 目前主要加速 GEMM 操作。但训练中还有大量非 GEMM 算子（LayerNorm、softmax、残差连接、激活函数等）仍用 FP16/BF16。如果这些算子占比较高（尤其在小 batch 或 attention-heavy 的场景下），FP8 GEMM 的加速会被非 GEMM 部分稀释——Amdahl 定律。

### 8.2 低精度优化器

> Dettmers et al., 2021 (bitsandbytes 8-bit Adam)

**问题**：训练一个 LLM，显存的大头不是模型权重，而是**优化器状态**。以 Adam 训练 7B 模型为例：

| 组件 | 精度 | 显存 |
|---|---|---|
| 模型权重 | BF16 | 14 GB |
| 梯度 | BF16 | 14 GB |
| FP32 master weights | FP32 | 28 GB |
| Adam $m$（一阶动量） | FP32 | 28 GB |
| Adam $v$（二阶动量） | FP32 | 28 GB |
| **合计** | | **~112 GB** |

优化器状态（$m + v$）占了 56 GB，超过模型权重本身的 4 倍。如果能把 $m, v$ 从 FP32 压缩到 INT8，就能节省 42 GB（75%）。

**为什么不能直接把 $m, v$ 存成 FP16？** $m$ 和 $v$ 的分布特征与权重不同——$m$ 是梯度的指数移动平均，可正可负，分布相对集中；$v$ 是梯度平方的移动平均，全为正，且跨参数的幅度差异可达数个数量级。FP16 的 5 bit 指数覆盖的动态范围有限，某些参数的 $v$ 值可能下溢到 0，导致 Adam 更新步长爆炸（$\text{update} \propto m / \sqrt{v}$，$v \to 0$ 时分母趋零）。

**bitsandbytes 的做法——block-wise dynamic INT8 量化**：

```
每步 Adam 更新：

1. 反量化：m_fp32 = dequant_int8(m_int8),  v_fp32 = dequant_int8(v_int8)
2. 标准 Adam 更新（FP32）：
     m_fp32 = β1 · m_fp32 + (1-β1) · grad
     v_fp32 = β2 · v_fp32 + (1-β2) · grad²
     weight -= lr · m_fp32 / (√v_fp32 + ε)
3. 重新量化：m_int8 = quant_int8(m_fp32),  v_int8 = quant_int8(v_fp32)
```

**关键设计——为什么 block-wise？** 如果对整个 $m$ 或 $v$ 向量用一个 scale 做 INT8 量化，少数极大值会拉大 scale，挤压其余参数的精度（和 §3 离群值问题一模一样）。Block-wise（block size = 2048）让每个 block 独立计算 scale，局部大值只影响局部精度，不会污染全局。

**Stable Embedding**：Embedding 层的特殊性在于梯度极稀疏——每个 batch 只有被采样到的 token embedding 有非零梯度，其余全是 0。这导致 $m, v$ 的更新极不规律（长时间为 0 突然跳变），INT8 量化后容易丢失小更新。解决方案很简单：**embedding 层保留 FP32 optimizer state**，其余层用 INT8。Embedding 参数量相对小，FP32 的额外显存可接受。

**数字**：
- 显存节省约 75%（optimizer state 从 FP32 → INT8 + 少量 scale 元数据）。
- 在 GPT-2、RoBERTa、BLOOM 等模型上收敛曲线几乎无损，最终精度与 FP32 Adam 相当。
- 使用极简：`bnb.optim.Adam8bit` 替换 `torch.optim.Adam` 即可，接口完全兼容。

**后续发展**：
- **4-bit 优化器**（Dettmers et al., 2024）：进一步压缩到 4-bit，使用非均匀量化（类似 NF4），显存节省更多但需要更仔细的超参数调整。
- **Galore**（Zhao et al., 2024）：从另一个角度压缩——将梯度投影到低秩子空间，减少优化器状态的维度而非精度。与 8-bit optimizer 正交，可组合使用。

### 8.3 Quartet — FP4 预训练

> Panferov, Chen et al., 2025 · ICML 2025

**为什么要比 FP8 更低？** §8.1 介绍了 FP8 训练，Blackwell 架构已经原生支持。但 Blackwell 同时还支持 **MXFP4** 格式的 Tensor Core 运算——理论上比 FP8 再快 2×。问题是：FP4 只有 7 个非零可表示值（E2M1 格式），训练中如此粗糙的表示能否收敛？Quartet 证明答案是肯定的，并给出了具体方法。

**MXFP4 格式回顾**：每 32 个 FP4 元素共享一个 E8M0 scale（power-of-2），每个元素 1 sign + 2 exp + 1 mantissa。可表示的正值仅为 {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}，加上 0 和对应负值，共 15 个值。

**核心方法——前向后向不对称量化**：

Quartet 的关键洞察是：**前向和后向传播对量化的需求完全不同**。

- **前向传播**（权重 × 激活）：需要高精度，因为直接决定模型输出质量。使用 **QuEST**（Quantization with Error-aware Stochastic Training）——在 Hadamard 旋转后寻找最小 MSE 的 clipping ratio 做确定性量化，精度高但计算略重。
- **后向传播**（梯度 × 激活/权重）：可以容忍更多噪声（梯度本身就是随机估计），使用 **随机取整（Stochastic Rounding, SR）**——将量化噪声变成零均值随机变量，保证梯度期望无偏。SR 计算简单但单次精度差。

```
Quartet 一步训练流程：

前向传播：
  X̃ = Hadamard(X)    # 旋转消除离群
  W̃ = Hadamard(W)
  Y = QuEST_FP4(X̃) × QuEST_FP4(W̃)    # 确定性最优量化

后向传播：
  ∂L/∂X = SR_FP4(∂L/∂Y) × W̃ᵀ          # 随机取整，无偏
  ∂L/∂W = X̃ᵀ × SR_FP4(∂L/∂Y)

优化器更新：FP32 master weights（同混合精度训练）
```

**低精度 Scaling Law**：Quartet 提出了考虑精度的 scaling law：

$$L(N, D, P_{fwd}, P_{bwd}) = \frac{A}{(\text{effN})^\alpha} + \frac{B}{(\text{effD})^\beta} + E$$

其中 $\text{effN} = N \cdot \rho(P_{fwd})$ 为"有效参数量"（低精度降低了参数的信息容量），$\text{effD} = D \cdot \eta(P_{bwd})$ 为"有效数据量"（低精度梯度降低了每个 token 的学习效率）。核心发现：

- **前向精度影响参数效率**：FP4 前向的 $\rho$ 约 0.69-0.78，即 FP4 模型需要约 1.3-1.45× 参数才能匹配 BF16 同等精度
- **后向精度影响数据效率**：FP4 后向的 $\eta$ 约 0.85，需要约 1.18× 数据补偿
- **前向比后向更敏感**：这解释了为什么 Quartet 在前向用更精确的 QuEST 而后向用简单的 SR

**实验结果**：
- 在 Llama 架构上训练 60M-1.3B 模型，FP4 训练精度与 FP8 baseline 差距在 **0.5-1.0 PPL** 以内
- 相比 BF16 baseline，FP4 需要约 1.3× 参数才能匹配同等精度
- **硬件加速**：在 Blackwell GPU 上，FP4 GEMM 吞吐量为 FP8 的 **2×**，end-to-end 训练加速约 **2×**

**局限**：目前仅验证了预训练阶段；FP4 的极粗粒度在 fine-tuning（学习率小、梯度信号弱）阶段可能更具挑战。

### 8.4 HALO — Hadamard 辅助低精度微调

> Ashkboos et al., 2025

**动机**：§8.1-§8.3 讨论的是预训练阶段的低精度化。但大模型部署前还需要 fine-tuning（SFT/RLHF），对于大多数用户来说这才是训练的主要场景。Fine-tuning 的特殊性在于：(1) 学习率比预训练低 10-100×，梯度信号更弱，对精度更敏感；(2) 通常在消费级 GPU（如 RTX 4090）上进行，显存更紧张。HALO 专门为 fine-tuning 场景设计低精度方案。

**核心思想——两级量化方案**：

HALO 提出两个级别，用户根据硬件约束选择：

**HALO-1（FP6 级别）**：权重和激活量化到 FP6，前向 GEMM 用 FP6 计算。这是一个"保守"方案，精度损失极小，主要收益是 1.5× 显存节省。

**HALO-2（INT8 级别）**：权重和激活量化到 INT8，前向和部分后向 GEMM 都用 INT8 计算。更激进，2× 显存节省和计算加速。

**关键技术——右手侧 vs 左手侧 Hadamard 旋转**：

QuaRot 等方法只在"右手侧"做旋转：$Y = (XR^T)(RW) = \tilde{X}\tilde{W}$，分散激活和权重的离群值。但 HALO 发现 fine-tuning 中的**误差梯度**（error gradient $\partial L / \partial Y$）也存在严重离群值——如果只旋转了前向的 $X$ 和 $W$ 而不处理反向的 $\partial L / \partial Y$，后向传播的量化误差会很大。

HALO 的创新是同时在"左手侧"做旋转：

$$Y = (Q \cdot X \cdot R^T) \cdot (R \cdot W \cdot P^T)$$

- $R$：右手侧旋转，分散激活和权重的离群
- $Q$：左手侧旋转，分散误差梯度 $\partial L / \partial Y$ 的离群
- $P$：输出侧旋转，处理输出方向的离群

反向传播时：$\partial L / \partial X = Q^T \cdot \text{quant}(\partial L / \partial Y_{\text{rotated}}) \cdot R \cdot W$，$\partial L / \partial Y$ 先被 $Q$ 旋转均匀化再量化，精度更高。

**HQ-FSDP：量化分布式通信**：

Fine-tuning 大模型通常使用 FSDP（Fully Sharded Data Parallelism），通信量是一个瓶颈。HALO 将 FSDP 的 all-gather（分发权重）和 reduce-scatter（聚合梯度）操作中传输的张量也做量化压缩：
- All-gather：传输量化后的权重，接收端反量化
- Reduce-scatter：梯度量化后传输，聚合后反量化
- 通信量减少 2-4×，在多卡训练时显著提速

**实验结果**：
- 在 Llama-2-7B fine-tuning 上，HALO-2 精度损失 <0.5%，训练速度提升 **1.41×**（RTX 4090 单卡）
- HQ-FSDP 在 4× RTX 4090 上额外带来 15-20% 加速
- 与 QLoRA 对比：HALO 做全参数低精度 fine-tuning，QLoRA 做低秩 + 量化；两者互补，可组合使用

### 8.5 Flash Attention 低精度训练的隐患

> Qiu & Yao, 2025 — "Why Low-Precision Transformer Training Fails"

**问题**：BF16 Flash Attention 训练在某些情况下会突然 loss 爆炸——例如 GPT-2 在 ~6600 步后训练发散。这一现象在 nanoGPT、flash-attention 等项目的 issue 中被反复报告，但此前缺乏机理解释。本节剖析根因并给出修复方案。

**故障链路追踪**：

作者通过系统排查，将问题定位到 Flash Attention 反向传播中的一个关键中间量：

$$\delta[T] = \text{rowsum}(dO \odot O)[T]$$

其中 $O = \text{softmax}(QK^\top)V$ 是注意力输出，$dO$ 是输出梯度。在低精度下 $\delta_{\text{lp}}$ 与高精度 $\delta_{\text{hp}}$ 之间产生了**系统性偏差**（非随机噪声），导致权重梯度累积低秩误差矩阵，谱范数持续增长直到训练爆炸。

**两个根因的叠加**：

| 根因 | 机制 | 后果 |
|---|---|---|
| **相似低秩表示** | 训练过程中，注意力矩阵 $P$、投影矩阵 $K$、隐层 $X$ 在 token 间形成相似的低秩结构 $R$ | 梯度误差 $dW_{\text{hp}} - dW_{\text{lp}} \approx \alpha \sum (\delta_{\text{lp}} - \delta_{\text{hp}})[T] \cdot R$，不会因 token 平均而消除 |
| **BF16 有偏舍入** | Safe softmax 中，当 score 行内存在多个相同最大值时，$\bar{P}$ 中出现精确等于 1 的元素。后续 $\bar{P}V$ 的 BF16 累加因 significand overflow 产生**系统性负偏差** | $O$ 偏负 → $(\delta_{\text{lp}} - \delta_{\text{hp}})[T]$ 系统性偏正 → 权重谱范数单调增长 |

**为什么 $\bar{P}$ 出现精确的 1 是危险的？**

Safe softmax 计算 $\bar{P} = \exp(S - m)$，其中 $m = \text{rowmax}(S)$。当一行中有多个 token 的 score 等于行最大值时，$S[T,t] - m = 0$，$\exp(0) = 1.0$。此时 $\bar{P}[T,t] \times V[t,i]$ 的 BF16 乘加运算会因 significand 对齐和舍入产生有偏误差——与 $\bar{P} < 1$ 时误差统计对称的情况完全不同。

**与 attention sink 的联系**：Attention sink 现象（少数 token 吸引极高注意力分数）使得 $\bar{P} = 1$ 的情况更频繁出现，这从数值算术层面直接解释了为什么 attention sink 会加剧训练不稳定性。这也与 §3 中讨论的离群值问题形成呼应——离群值不仅影响推理量化，也通过注意力集中效应威胁训练稳定性。

**修复：动态最大值调整（Stabilized Flash Attention）**

核心思想：利用 softmax 的平移不变性 $\text{softmax}(z) = \text{softmax}(z - c)$，在检测到行内重复最大值时动态调大归一化常数 $m$，确保 $\bar{P}$ 的所有元素严格小于 1：

$$r_m = \text{rowmax}(S), \quad r_s = \text{rowsum}(S \equiv r_m)$$
$$m = \begin{cases} \beta \cdot r_m & \text{if } r_m > 0 \wedge r_s > 1 \\ 0 & \text{if } r_m < 0 \wedge r_s > 1 \\ r_m & \text{otherwise} \end{cases}$$

其中 $\beta \in [2, 8]$。这样 $\max(S - m) < 0$，因此 $\max(\bar{P}) < 1$，消除了有偏舍入的触发条件。

**设计要点**：
- **为什么不用固定偏移？** 固定减去小常数会在 $\bar{P}$ 的 BF16 转换中引入新的系统性舍入误差，无法根治问题
- **为什么条件触发？** 无条件调整 $m$ 在 $r_m$ 很大时会导致 $\exp(S - \beta r_m)$ 下溢为 0，引发除零错误
- **仅修改前向传播**：调整只在 softmax 的 tiling 算法中加入一行判断（Algorithm 1 第 8-9 行），反向传播无需改动

**实验验证**：
- GPT-2 BF16 训练：原始 Flash Attention 在 ~6600 步 loss 爆炸，Stabilized Flash Attention（$\beta=7$）训练全程稳定
- 修复在 NVIDIA A100、RTX 4090、Huawei Ascend 910B 上均有效
- 数学上精确等价于标准注意力（只是选择了不同的平移常数 $c$），不影响模型精度

**启示**：这项工作揭示了一个容易被忽视的事实——**低精度训练的不稳定性不一定来自显式的量化操作，也可能藏在基础算子（如 Flash Attention）的数值实现细节中**。对于 §8.1-§8.4 中讨论的各种低精度训练方法，Flash Attention 的 BF16 舍入问题同样适用，需要配合使用稳定化方案。

### 8.6 优化器选择如何影响量化？

> Vlassis, Ashkboos et al., 2025 · "Beyond Outliers"

前面 §3 介绍了离群值是 LLM 量化的核心挑战，§4-§6 介绍了各种 PTQ 方法来应对离群值。一个被忽视的问题是：**所有量化方法都假设"给定一个训练好的模型"，但不同优化器训好的模型，量化友好程度一样吗？** Beyond Outliers 首次系统研究了优化器（AdamW、Muon、PSGD、Shampoo、SOAP、Scion）与量化（PTQ + QAT）的交互关系，在 50M-1.5B 参数规模上实验，得出了多个反直觉结论。

**核心发现 1：离群值指标不能预测 PTQ 精度**

§3 中我们介绍了离群值（outlier）是 LLM 量化的核心难点。直觉上，离群值越大量化越难——传统工作用 MMR（max/median ratio）或 Kurtosis 来衡量离群程度，并以此指导量化方案设计（如 §4.2 SmoothQuant 的目标就是降低激活的离群值）。但 Beyond Outliers 发现，**跨优化器比较时 MMR 和 Kurtosis 与 PTQ 后精度几乎无相关性**（$\rho = 0.62$ 和 $\rho = -0.89$ 对 760M 模型）：

- **Muon** 的 MMR 最低（离群最小），但 PTQ 后精度**下降最严重**（760M: 64.63% → 50.00%）
- **Shampoo** 的 MMR 最高（离群最大），但 PTQ 后精度**保持最好**（760M: 63.05% → 59.26%）

**为什么 MMR 失败？** 论文给出了理论分析——ABC 分解框架。MMR 是**单层**度量，但量化误差的总效应取决于误差如何**逐层传播和放大**。具体来说，将第 $\ell$ 层的量化误差 $\Delta h_\ell = h_\ell^q - h_\ell$ 分解为：

$$R_\ell = A_\ell + B_\ell + C_\ell$$

- $A_\ell$：前面层累积误差经当前层传播（**主导项**——类似 §6.1 QuaRot 讨论的"误差沿计算图传播"）
- $B_\ell$：当前层新引入的量化误差（MMR 能部分预测的部分）
- $C_\ell$：两者的交互项

关键发现：$R_\ell$ 几乎完全由 $A_\ell$ 主导。即使 MMR 能预测单层误差 $B_\ell$，总误差取决于**增益** $G_\ell = A_\ell / R_{\ell-1}$——量化误差通过每一层的放大倍数：

$$G_\ell = G_{1,\ell} \cdot G_{2,\ell}$$

$G_{1,\ell}$ 是谱范数比（量化前后权重谱范数变化，各优化器接近 1），$G_{2,\ell}$ 是对齐比（量化误差方向与权重主奇异方向的对齐程度）。**Muon 的 $G_\ell$ 在线性层最高**——量化误差恰好指向权重放大最强的方向，导致误差快速积累；Shampoo 和 AdamW 的 $G_\ell$ 最低。

论文提出的新指标 $R_L$（最终层累积量化误差）与 PTQ 精度高度相关（$\rho = 0.70$）。

**核心发现 2：QAT 最佳优化器 ≠ 全精度最佳优化器**

在全精度训练中 Muon 表现最好（参考 §2.1 讨论的 QAT 需要重新训练），但 4-bit QAT（使用 §8.3 提到的 QuEST 方案）下各优化器排名完全改变。**Shampoo 在 QAT 中精度退化最小**——如 760M: Shampoo 仅 -0.46%，而 Muon -3.57%。

**核心发现 3：QAT 的 scaling law**

类似 §8.3 Quartet 提出的低精度 scaling law（用 effN 描述精度对参数效率的影响），Beyond Outliers 推导了 QAT 下的 optimizer-aware scaling law：$L = A' / (N \cdot \rho)^\alpha + E$，其中 $\rho$ 是"参数效率"——4-bit QAT 模型的等效参数量为 $\rho \cdot N$。各优化器的 $\rho_{4bit}$：

| 优化器 | $\rho_{4bit}$ | 含义 |
|---|---|---|
| **Shampoo** | **0.879** | 4-bit 保留 87.9% 参数效率 |
| AdamW | 0.863 | |
| Scion | 0.856 | |
| Muon | 0.852 | |
| SOAP | 0.822 | |

**实践启示**：如果最终目标是部署量化模型，训练阶段的优化器选择也很重要——Shampoo 训出的模型量化更鲁棒。对于 PTQ，传统的离群值度量（MMR、Kurtosis）可能误导优化方向，应关注误差传播特性而非单层统计。**评测量化友好性需要全局视角，不能只看局部指标**。

---

## 9 量化工程实现

> 量化算法最终需要工程实现才能落地。本节以 bitsandbytes 为例介绍量化库的实现细节，并讨论量化库与推理框架在架构上的本质区别。

### 9.1 bitsandbytes 量化实现

bitsandbytes 是 §1.1 均匀量化公式的典型工程实现，分 Python 接口和 CUDA kernel 两层。

**Python 层调用**：

```python
import bitsandbytes.functional as F

x_q, state = F.quantize_blockwise(x_fp32)   # 量化
x_deq = F.dequantize_blockwise(x_q, state)  # 反量化
```

`state`（`QuantState` 对象）保存 `absmax`（每 block 的最大绝对值 = scale）、`code`（码本）、`blocksize` 等。

**8-bit 对称量化（blockwise）**——对应 §1.1 的对称公式，但按 block（默认 2048 元素）独立计算 scale：

$$s_{\text{block}} = \frac{\text{absmax}_{\text{block}}}{127}, \qquad x_q = \left\lfloor \frac{x}{s_{\text{block}}} \right\rceil, \qquad \hat{x} = s_{\text{block}} \cdot x_q$$

分 block 的好处：避免全局少数大值拉高 scale，挤压其余正常值的量化精度。

**4-bit NF4 量化**（用于 QLoRA）——假设权重近似正态分布，预计算 16 个最优量化点作为码本：

```python
# NF4 码本（硬编码，信息论最优于 N(0,1)）
code = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0]

# 量化：归一化后找最近码本索引
x_norm = x / absmax_per_block
x_q = argmin_i |x_norm - code[i]|     # 4-bit index (0~15)

# 反量化
x_hat = absmax_per_block * code[x_q]
```

与均匀量化不同，NF4 的量化点分布是非均匀的，中心更密、尾部更稀，匹配正态分布的概率密度。

**CUDA kernel 层**（`csrc/kernels.cu`）关键实现片段：

**1. Block 内 absmax 归约**——使用 CUB 的 `BlockReduce` 原语，比手写 shared memory 归约更高效：

```cuda
// 每个线程处理 NUM_PER_TH 个元素，先求局部 abs max，再用 CUB 做 block 级归约
local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, BNB_MAX_OP, valid_items);

if (threadIdx.x <mark> 0) {
    smem_absmax_value[0] = 1.0f / local_abs_max;  // 存倒数，后续用乘法替代除法
    absmax[i / BLOCK_SIZE] = local_abs_max;
}
```

**2. NF4 量化——决策树替代线性搜索**：16 个非均匀量化点的最近邻查找被编译为二叉判断树（$O(\log 16) = 4$ 次比较），避免了遍历码本的开销：

```cuda
__device__ unsigned char dQuantizeNF4(float x) {
    // 编译期展开为 4 层 if-else，每个叶节点返回 4-bit 索引
    if (x > 0.03979014977812767f)
        if (x > 0.3893125355243683f)
            if (x > 0.6427869200706482f)
                if (x > 0.8614784181118011f) return 0b1111;
                else return 0b1110;
            else if (x > 0.5016634166240692f) return 0b1101;
            else return 0b1100;
        // ... 共 16 个叶节点，阈值为相邻码本值的中点
}
```

**3. 4-bit packing——两个值压入一个字节**：

```cuda
// NF4: 每个 byte 存两个 4-bit 量化值，高 4 位 + 低 4 位
for (int j = 0; j < NUM_PER_TH / 2; j++) {
    qvals[j]  = dQuantizeNF4(((float)vals[2*j])   * local_abs_max) << 4;
    qvals[j] |= dQuantizeNF4(((float)vals[2*j+1]) * local_abs_max);
}
```

**4. 反量化——查表 + 位操作**：

```cuda
__device__ __forceinline__ float dDequantizeNF4(unsigned char val) {
    return nf4_dequantization_lut[val & 0x0F];  // 16 entry LUT，预计算好的码本值
}
```

**主要工程挑战**：
- **归约效率**：每个 block（2048 元素）需要求 absmax，朴素 shared memory 归约有 bank conflict 和同步开销。bitsandbytes 依赖 CUB 的 `BlockReduce`，内部使用 warp shuffle 指令避免 shared memory 访问
- **4-bit packing/unpacking 的对齐问题**：两个 4-bit 值共享一个字节，读写时需要位移和掩码操作。当 block 边界不对齐时需要特殊处理（`valid_items` 参数）
- **小 block 场景的 warp 利用率**：当 blocksize 很小（如 64）时，标准 kernel 的线程块过大导致浪费。bitsandbytes 为此实现了 `kQuantizeBlockwiseSmall`，用 `WarpReduce` 替代 `BlockReduce`，一个线程块处理多个 block
- **反量化的访存模式**：`dDequantizeNF4` 的 LUT 仅 16 个 float，常驻 L1 cache。但量化值的随机访问模式可能导致 cache miss——实际实现中通过 `__ldg()`（read-only cache）和连续 tile 加载来缓解

**整体架构**：

```
Python API (functional.py)
  ├── quantize_blockwise()   →  对称 INT8，blocksize=2048
  ├── dequantize_blockwise()
  ├── quantize_nf4() / quantize_fp4()  →  4-bit 非均匀码本
  └── QuantState  ← 保存 absmax, code, blocksize, dtype
        │
CUDA kernels (csrc/kernels.cu)
  ├── kQuantizeBlockwise / kDequantizeBlockwise   ← 8-bit
  └── kQuantize4bit / kDequantize4bit             ← NF4/FP4
```

总结：bitsandbytes 在基础公式上加了两个关键工程优化：**Blockwise**（分 block 各算 scale，精度更高）和 **NF4 码本**（4-bit 时用正态分布最优量化点替代均匀量化，匹配权重分布）。

### 9.2 量化库 vs 推理框架：两种落地路径

bitsandbytes 代表的是**量化工具库**路径——提供量化/反量化原语，用户在 PyTorch 训练或推理流程中调用。但在生产部署中，主流做法是使用 vLLM、TensorRT-LLM 等**推理框架**，它们将量化深度融合到推理 kernel 中。两者的本质区别：

| 维度 | 量化工具库（bitsandbytes, TorchAO） | 推理框架（vLLM, TensorRT-LLM, SGLang） |
|---|---|---|
| **量化位置** | Python 层显式调用量化/反量化 | 量化逻辑融合在 CUDA kernel 内部 |
| **计算流程** | 量化存储 → 反量化回 FP16 → 标准 GEMM | 量化权重直接参与计算，dequant 与 GEMM 流水线化（如 MARLIN kernel 将 INT4 解包隐藏在访存延迟中） |
| **性能瓶颈** | 反量化是独立步骤，有额外开销 | dequant 开销被计算完全掩盖，接近理论带宽上限 |
| **典型加速** | 主要节省显存，速度提升有限 | W4A16 可达 ~3.9× 加速（vs FP16），接近 4× 理论值 |
| **适用场景** | 研究实验、QLoRA 微调、快速原型 | 生产部署、高吞吐在线服务 |

**为什么推理框架能做到接近理论加速？** 以 vLLM 默认使用的 MARLIN kernel（Frantar et al., 2024）为例：LLM 小 batch 推理是 memory-bound（瓶颈在从 HBM 加载权重），4-bit 权重理论上减少 4× 带宽占用。MARLIN 通过异步 dequant + 计算 pipeline（第 $i$ 批权重从 global mem 异步拷贝到 shared mem 时，Tensor Core 同时执行第 $i-1$ 批的 FP16 GEMM），将 dequant 完全隐藏在访存延迟中，A100 上实测达到理论带宽上限的 >95%。

**实践选择**：研究阶段用 bitsandbytes/TorchAO 快速验证量化方案 → 确定方案后用推理框架的融合 kernel 部署。两者不是替代关系，而是量化工作流的不同阶段。

### 9.3 AWQ Triton Kernel 解读

> Example: vLLM AWQ Triton kernel（`quantization/test_awq_triton.py`）。这个真实推理 kernel 展示了 §9.2 中"量化融合到 kernel"的具体路径。

§5.2 介绍了 AWQ 算法（激活感知权重量化），本节聚焦它的 kernel 实现。AWQ 将 FP16 权重量化为 4-bit 非对称格式，8 个 4-bit 值打包进一个 int32：

```
原始权重:  W (K, N) float16     ← 占 K×N×2 bytes
量化后:
  qweight: (K, N/8)   int32     ← 每个 int32 = 8 个 4-bit 权重，压缩 4×
  zeros:   (K/G, N/8) int32     ← 每组零点，同样打包
  scales:  (K/G, N)   float16   ← 每组缩放因子
```

反量化公式（非对称量化）：$W_{\text{fp16}} = (W_{\text{int4}} - Z_{\text{int4}}) \times S_{\text{fp16}}$

#### 9.3.1 Kernel 1: `awq_dequantize_kernel`

将 `(K, N/8)` 的 int32 打包权重解压为 `(K, N)` 的 FP16 矩阵。

**AWQ 的特殊打包顺序**：

```python
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]           # 打包时第 i 个 weight 放在第 AWQ_ORDER[i] 个 nibble
AWQ_REVERSE_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]   # 解包时的逆映射
```

这个顺序不是随意的——它配合 Triton 的 `tl.interleave` 硬件指令，使解包零开销。

**核心解包流程**：

```python
# 1. 加载 (BY, BX) 个 int32
iweights = tl.load(qweight_ptr + offsets)

# 2. 三次 interleave 把每个 int32 复制 8 份 → (BY, BX*8)
iweights = tl.interleave(iweights, iweights)   # [a,b] → [a,a,b,b]         ×2
iweights = tl.interleave(iweights, iweights)   # → [a,a,a,a,b,b,b,b]       ×4
iweights = tl.interleave(iweights, iweights)   # → [a×8, b×8, ...]          ×8

# 3. 构造 shift 值：[0, 16, 4, 20, 8, 24, 12, 28] — 每个 nibble 的 bit 偏移
reverse_awq_order_tensor = (
    (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
).reshape(8)                                    # = [0, 4, 1, 5, 2, 6, 3, 7]
shifts = reverse_awq_order_tensor * 4           # = [0, 16, 4, 20, 8, 24, 12, 28]

# 4. 移位提取 nibble
iweights = (iweights >> shifts) & 0xF
```

对 8 份相同的 int32 各右移不同位数，再 `& 0xF` 取低 4 位：

```
int32 值: 0x76543210

位置 0: >> 0  → nibble 0 → weight 0    (AWQ_ORDER[0] = 0)
位置 1: >> 16 → nibble 4 → weight 1    (AWQ_ORDER[1] = 4)
位置 2: >> 4  → nibble 1 → weight 2    (AWQ_ORDER[2] = 1)
...
结果: 8 个 weight 按正确顺序排列
```

**为什么 AWQ 用这个奇怪的顺序？** `interleave` ×3 产生的重复模式，配合 shift 序列 `[0,16,4,20,8,24,12,28]`，恰好让 8 个 nibble 按正确顺序输出。AWQ 的打包顺序是解包 shift 模式的**逆**。打包只做一次（离线），解包每次推理都做——复杂性放在打包端。

**零点和缩放因子**：

```python
# zeros (K/G, N/8) 的解包方式与权重完全相同
zeros = tl.load(zeros_ptr + zero_offsets)
zeros = tl.interleave(zeros, zeros)   # ×3 → 解包
zeros = (zeros >> shifts) & 0xF

# scales (K/G, N) 已经是逐元素的，不需要解包
scales = tl.load(scales_ptr + scale_offsets)

# 反量化
iweights = (iweights - zeros) * scales    # int32 → float → float16
```

零点每 `group_size` 行共享一行，通过 `pid_y * BLOCK_SIZE_Y // group_size` 索引到对应 group。

#### 9.3.2 Kernel 2: `awq_gemm_kernel` — 融合反量化 + GEMM

这是 §9.2 中"融合 kernel"的具体实现：$C_{M \times N} = A_{M \times K} \times \text{dequant}(B_{K \times N})$。

**Grid 设计**：

```python
pid = tl.program_id(axis=0)         # M×N 维度的线性 CTA id
pid_z = tl.program_id(1)            # Split-K 维度

pid_m = pid // num_pid_n            # 行方向 tile 索引
pid_n = pid % num_pid_n             # 列方向 tile 索引
```

```
输出矩阵 C (M × N):                    K 维度拆分:
     pid_n=0  pid_n=1  pid_n=2          pid_z=0 处理 K[0:BK]
    ┌──────┬──────┬──────┐              pid_z=1 处理 K[BK:2BK]
m=0 │pid=0 │pid=1 │pid=2 │              ...
    ├──────┼──────┼──────┤              最后 result.sum(0) 归约
m=1 │pid=3 │pid=4 │pid=5 │
    └──────┴──────┴──────┘
```

**主循环——边解包边算**：

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
    # 加载 A: (BM, BK) float16
    a = tl.load(a_ptrs, mask=masks_a)

    # 加载 B: (BK, BN/8) int32 → interleave ×3 → (BK, BN) int32
    b = tl.load(b_ptrs, mask=masks_b)
    b = tl.interleave(b, b)
    b = tl.interleave(b, b)
    b = tl.interleave(b, b)

    # 加载当前 K 位置对应的 zeros 和 scales（按 group_size 索引）
    zeros = tl.load(zeros_ptrs)     # (1, BN/8) → interleave ×3 → broadcast → (BK, BN)
    scales = tl.load(scales_ptrs)   # (1, BN) → broadcast → (BK, BN)

    # 融合解包 + 反量化
    b = (b >> shifts) & 0xF
    zeros = (zeros >> shifts) & 0xF
    b = (b - zeros) * scales        # → float16

    # 矩阵乘累加
    accumulator = tl.dot(a, b, accumulator)

    # 推进 K 指针（步长 = BK × SPLIT_K，跳过其他 pid_z 的条带）
    a_ptrs += BLOCK_SIZE_K * SPLIT_K
    b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)
```

数据流：`b_int32 (BK, BN/8) → interleave ×3 → shift & mask → (b-zeros)*scales → dot(a, b) → accumulate`

**Split-K 的作用**：当 M 很小（decode 阶段 batch=1）时，M 维度的 tile 数太少，SM 闲置。Split-K 把 K 维度也拆给多个 CTA 并行，最终由 Python 端 `result.sum(0)` 归约。代价是额外的 reduce 操作，但在小 batch 时远小于并行度提升的收益。

**写回**：每个 `pid_z` 写到输出 tensor 的第 `pid_z` 个 slice（`c_ptr + pid_z * N * M`）。

#### 9.3.3 性能分析

**Dequantize Kernel**（A5000 实测）：

| K | N | Group Size | Latency | 带宽利用率 |
|---|---|---|---|---|
| 4096 | 4096 | 128 | 0.064 ms | 85% |
| 4096 | 11008 | 128 | 0.169 ms | 87% |
| 8192 | 8192 | 128 | 0.250 ms | 88% |

A5000 理论峰值 768 GB/s，kernel 达到 ~88% 利用率，接近 memory-bound 极限。

**Fused GEMM vs 分步流水线**（A5000）：

| M | K×N | Split-K | cuBLAS (FP16) | deq+mm | fused | fused vs deq+mm |
|---|---|---|---|---|---|---|
| 1 | 4096×4096 | 4 | 0.052 ms | 0.117 ms | 0.052 ms | **2.25×** |
| 16 | 4096×4096 | 1 | 0.058 ms | 0.123 ms | 0.063 ms | **1.97×** |
| 32 | 4096×11008 | 1 | 0.135 ms | 0.305 ms | 0.147 ms | **2.08×** |
| 64 | 4096×4096 | 1 | 0.056 ms | 0.123 ms | 0.110 ms | 1.12× |
| 128 | 4096×4096 | 1 | 0.065 ms | 0.129 ms | 0.208 ms | 0.62× |

关键观察：
- **M≤32（decode 阶段）**：Fused 比 deq+mm 快 **1.8-2.3×**。memory-bound 场景下读 4-bit 打包数据比读 16-bit 展开数据节省 4× 带宽，且省掉中间 tensor 分配
- **M=1 + Split-K=4**：达到与 cuBLAS（预解压 FP16）**持平**的性能，但显存仅需 1/4
- **M≥64（prefill 阶段）**：Fused 反而更慢——变成 compute-bound，cuBLAS 的 Tensor Core 利用率远高于边解包边算的开销
- **结论**：AWQ fused GEMM 的甜区是**小 batch decode（M≤32）**，正是 LLM 自回归推理最常见的场景

#### 9.3.4 设计总结

| 设计选择 | 原因 |
|---|---|
| 4-bit 打包到 int32 | 减少 8× 显存占用和带宽需求 |
| 非对称量化（有零点） | 4-bit 只有 16 个 level，零点避免浪费精度 |
| AWQ 特殊 nibble 顺序 | 配合 `tl.interleave` 硬件指令，解包零开销 |
| Per-group scales/zeros | 比 per-tensor 精度高，比 per-channel 存储小 |
| Fused dequant + GEMM | 避免中间 FP16 权重 tensor 的显存分配和读写 |
| Split-K | 小 batch 时增加 K 维度并行度，提升 SM 利用率 |
| FP16 累加器 | 牺牲精度换速度（与标准 FP32 累加不同） |

### 9.4 SageAttention — INT8 量化注意力

> Example: SageAttention INT8 attention path（`quant_per_block.py` + `attn_qk_int8_per_block.py`）。§9.3 展示权重量化的 kernel 融合，本节展示另一个融合方向：**将注意力计算中的 QK^T 矩阵乘量化为 INT8**，在 FlashAttention 框架内实现加速。

**核心思路**：将 Q、K 按 block 量化为 INT8，利用 INT8 Tensor Core（吞吐量约为 FP16 的 2×）加速 $QK^T$ 计算，V 保持 FP16 不量化。

```
整体流程:
Q, K (FP16) ──per_block_int8()──► Q_int8, K_int8, Q_scale, K_scale ──forward()──► O (FP16)
```

#### 9.4.1 Per-Block INT8 量化

**量化 kernel `quant_per_block_int8_kernel`**：每个 CTA 处理一个 `[BLK, head_dim]` 的子矩阵。

```python
# 每个 CTA 的坐标
off_blk = tl.program_id(0)    # 序列维度的 block 索引
off_h = tl.program_id(1)      # head 索引
off_b = tl.program_id(2)      # batch 索引

# 加载 [BLK, head_dim] 的数据
x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
x = x.to(tl.float32)
x *= sm_scale                  # Q 侧乘以 1/√d × log₂(e)，K 侧 sm_scale=1.0

# 对称量化：整个 block 共享一个 scale
scale = tl.max(tl.abs(x)) / 127.

# 手动四舍五入（Triton 的 to(int8) 是截断）
x_int8 = x / scale
x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)   # +0.5 再截断 = round
x_int8 = x_int8.to(tl.int8)

tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
tl.store(scale_ptrs, scale)    # 每个 block 一个 FP32 scale
```

**sm_scale 融入量化的技巧**：Q 量化时预乘 `sm_scale × log₂(e) = 1/√d × 1.44269504`。这样反量化后 $Q_\text{int8} \cdot K_\text{int8}^T \times q_\text{scale} \times k_\text{scale}$ 直接得到 $QK^T / \sqrt{d} \times \log_2 e$，后续可以用硬件更快的 `exp2` 替代 `exp`：

$$e^{x/\sqrt{d}} = 2^{x/\sqrt{d} \cdot \log_2 e}$$

**与 AWQ 量化的对比**：

| 维度 | SageAttention (Q/K 量化) | AWQ (权重量化) |
|---|---|---|
| 量化对象 | 运行时激活（Q, K） | 离线权重 |
| 量化方式 | 对称（无零点） | 非对称（有零点） |
| 粒度 | Per-block（128/64 tokens × head_dim） | Per-group（128 个权重） |
| 不量化的部分 | V（保持 FP16） | 激活（保持 FP16） |

Q/K 经过 LayerNorm 后分布近似对称，所以对称量化足够；AWQ 的权重分布不一定对称，需要零点。

#### 9.4.2 INT8 注意力 kernel

采用 FlashAttention 的 Q-stationary tiling 策略：外层固定一个 Q block，内层遍历所有 K/V block。

**核心计算——INT8 矩阵乘 + 反量化**：

```python
# q: [BLOCK_M, HEAD_DIM] int8     k: [HEAD_DIM, BLOCK_N] int8（已转置）
qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
```

这一行完成了：
1. `tl.dot(q, k)`：INT8 Tensor Core 矩阵乘，结果为 INT32
2. `.to(tl.float32)`：转为 FP32
3. `× (q_scale × k_scale)`：反量化为真实的 attention score（已包含 $1/\sqrt{d} \times \log_2 e$）

**Online softmax（FlashAttention 算法）**：

```python
# 初始状态: m_i = -inf, l_i = 1.0, acc = 0

for each K/V block:
    # 1. 加载 K block → INT8 matmul → 反量化
    qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

    # 2. 更新行最大值
    m_ij = tl.maximum(m_i, tl.max(qk, 1))

    # 3. 修正历史累积（最大值变化时需要缩放）
    alpha = tl.math.exp2(m_i - m_ij)       # 修正因子
    l_i = l_i * alpha + tl.sum(tl.math.exp2(qk - m_ij[:, None]), 1)
    acc = acc * alpha[:, None]

    # 4. P @ V（FP16 Tensor Core，不用 INT8）
    p = tl.math.exp2(qk - m_ij[:, None])
    v = tl.load(V_ptrs)                     # V 保持 FP16
    acc += tl.dot(p.to(tl.float16), v)

    m_i = m_ij

# 最终归一化
output = acc / l_i[:, None]
```

Online softmax 的精髓：当最大值从 $m_\text{old}$ 更新为 $m_\text{new}$ 时，历史累积需要乘以修正因子 $\alpha = 2^{m_\text{old} - m_\text{new}}$。如果最大值没变则 $\alpha = 1$；如果增大了则 $\alpha < 1$，缩小历史值。

**GQA 支持**：K/V 的 head 索引用 `off_h // num_kv_groups`，多个 Q head 自然共享一个 KV head。

**Mask 优化**：

```python
if mask.dtype </mark> tl.int1:           # Bool mask
    if tl.max(mask_block) == 0:     # 整个 block 全被 mask
        skip = True                  # 跳过全部计算！
    else:
        qk += tl.where(mask_block, 0, -1.0e6)
else:                                # Float mask（加性）
    qk += mask_block
```

Bool mask 支持 block 级跳过——当整个 `[BLOCK_M, BLOCK_N]` 都被 mask 时直接跳过，对稀疏 mask 有显著加速。

#### 9.4.3 设计总结

| 设计选择 | 原因 |
|---|---|
| Q/K 量化为 INT8，V 保持 FP16 | $QK^T$ 经过 softmax 平滑，对量化误差不敏感；V 的误差直接影响输出 |
| Per-block 量化（BLKQ=128, BLKK=64） | 与 FlashAttention 分块大小对齐，每个 block 独立 scale |
| 对称量化（无零点） | Q/K 经 LayerNorm 后分布对称，零点意义不大 |
| sm_scale 融入 Q 量化 | 避免 attention kernel 中额外浮点乘法 |
| `exp2` 代替 `exp` | 硬件原生支持 `exp2` 更快，配合 `log₂(e)` 预乘 |
| Bool mask block 级跳过 | 整个 block 被 mask 时跳过全部计算，稀疏 mask 加速显著 |
| P@V 用 FP16 而非 INT8 | 平衡精度与速度，只在 $QK^T$ 阶段使用 INT8 加速 |

---

## 课后练习题

### 练习 1：weight-only quantization 的收益边界

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么 W4A16 对 decode 有用，但对大 batch prefill 不一定最快？</span></summary>

小 batch decode 常受权重加载带宽限制，4-bit 权重能显著减少 HBM traffic，即使要 dequant 也可能更快。大 batch prefill 更 compute-bound，W4A16 解包后仍然做 FP16/BF16 compute，可能不如 W8A8 直接用 INT8 Tensor Core。

</details>

### 练习 2：KV cache 量化的风险

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么 KV cache 量化比权重量化更敏感？</span></summary>

权重是固定的，可以离线校准和逐层修正；KV cache 是每个请求动态生成的，误差会沿 decode 时间传播，并直接影响 attention score 和上下文读取。长上下文下 KV cache 量化收益很高，但必须控制 scale 粒度、分组方式和 residual window。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">W4A16、W8A8、FP8 三者的系统差别是什么？</span></summary>

W4A16 主要压权重带宽，计算通常还要反量化到 FP16/BF16，适合小 batch decode；W8A8 权重和激活都量化，更可能用 INT8 Tensor Core，适合吞吐场景；FP8 是训练/推理都常用的低精度格式，需要 scale 管理和数值稳定策略，硬件支持很关键。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">为什么 per-channel/per-block scale 往往比 per-tensor scale 更准？</span></summary>

不同 channel 或 block 的数值范围可能差很多。per-tensor scale 被最大 outlier 决定，会浪费大部分量化 bins；per-channel/per-block scale 能贴近局部分布，减少误差。代价是多存 scale，并在 kernel 中多做 scale load 和反量化。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">KV cache 量化的收益怎么估算？</span></summary>

先写 KV bytes：`layers * tokens * kv_heads * head_dim * 2(K,V) * dtype_bytes`。把 BF16/FP16 的 2 bytes 换成 INT8 的 1 byte，理论上 KV 存储和读带宽约减半；实际还要加 scale、residual window、反量化和 kernel 支持成本。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">为什么 attention 里常量化 Q/K，而对 V 更谨慎？</span></summary>

QK 误差经过 softmax 后有一定平滑和归一化；V 是最后被 attention probability 加权求和的内容本身，误差更直接进入输出 hidden state。很多实现会先量化 Q/K 加速 score 计算，而让 V 保持 FP16/BF16 来保护质量。

</details>
