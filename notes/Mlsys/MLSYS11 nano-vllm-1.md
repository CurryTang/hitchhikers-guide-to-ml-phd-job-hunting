# MLSYS11 · nano-vllm 精读 (1)

## nano-vllm part 1

**nano-vllm** 是一个用 ~1000 行纯 Python/PyTorch 实现的最小化 LLM 推理引擎,项目目标是用最少的代码还原 vLLM 的核心设计，便于学习和研究。

| 特性 | 说明 |
|------|------|
| 模型支持 | Qwen3 系列（可扩展） |
| 推理优化 | Paged KV Cache、Continuous Batching |
| 并行策略 | Tensor Parallelism（列并行 + 行并行） |
| 代码量 | ~1000 行，无复杂依赖 |

### Revisit transformers

Transformer 是 2017 年由 Google 在论文 *"Attention Is All You Need"* 中提出的架构，成为了现代大语言模型（如 GPT、LLaMA、Claude）的基础。核心创新：**Self-Attention**（直接关注序列任意位置）、**并行计算**（无需 RNN 的顺序依赖）、**位置编码**。

#### Transformer Decoder 层结构

```
Input [B,T,D]
  │
  ├─ RMSNorm → MHA (Q/K/V/O proj, Params: 4D²) ─┐
  └────────────────────────────── (+) residual ←┘
  │
  ├─ RMSNorm → FFN (Up→Act→Down, Params: 3DF) ──┐
  └────────────────────────────── (+) residual ←┘
  │
Output [B,T,D]
```


MHA 允许模型同时从不同"表示子空间"学习信息，三个核心向量：**Q（我在找什么）**、**K（我有什么标签）**、**V（我的内容是什么）**。

#### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

计算：① $QK^T$ 得相似度矩阵 → ② 除以 $\sqrt{d_k}$ 防梯度消失 → ③ softmax → ④ 加权求 V。MHA 将输入分多头并行学习不同模式：$\text{MultiHead} = \text{Concat}(\text{head}_i)W^O$，其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$。

三种变体：**MHA**（$N=K$，标准）、**MQA**（$K=1$，极省 KV cache）、**GQA**（$K$ 整除 $N$，折中，Qwen3 使用）。



### 分析理解transformer的roofline性质

符号：$x, y \in [P]$（向量），$A \in [N, P]$，$B \in [P, M]$（矩阵）。

#### FLOPs 计算规则

| 操作 | FLOPs | 数据传输 | 说明 |
|------|-------|----------|------|
| 向量点积 $x \cdot y$ | $2P$ | $2P$ | P 次乘法 + P 次加法 |
| 矩阵-向量乘 $Ax$ | $2NP$ | $NP + P$ | N 个点积 |
| 矩阵-矩阵乘 $AB$ | $2NPM$ | $NP + PM$ | M 个矩阵-向量乘 |

矩阵乘法：计算 $O(N^3)$、数据传输 $O(N^2)$，矩阵越大越容易达到 compute-bound，这也是深度学习大量使用矩阵乘法的原因。

#### 通用 Einsum 规则

对于高维张量收缩，我们需要区分三类维度：
- ==红色== **收缩维度**：在两个张量中都出现，在输出中消失（被求和掉）
- ==蓝色== **批处理维度**：在两个张量和输出中都出现（独立并行计算）
- **普通维度**：只在一个输入张量和输出中出现

**示例解析**：
$$
C[\textcolor{blue}{GH}IJ\textcolor{red}{KL}] \cdot D[\textcolor{blue}{GH}MN\textcolor{red}{KL}] \rightarrow E[\textcolor{blue}{GH}IJMN]
$$

用 einsum 表示：`einsum('ghijkl,ghmnkl->ghijmn', C, D)`

```
张量 C: [G, H, I, J, K, L]
        ↑  ↑  ↑  ↑  ↑  ↑
        批 批 普 普 收 收
        处 处 通 通 缩 缩

张量 D: [G, H, M, N, K, L]
        ↑  ↑  ↑  ↑  ↑  ↑
        批 批 普 普 收 收
        处 处 通 通 缩 缩

输出 E: [G, H, I, J, M, N]  ← K, L 被求和消失
```

**计算过程理解**：
- 对于输出的每个位置 `E[g,h,i,j,m,n]`，需要计算：
  $$E[g,h,i,j,m,n] = \sum_{k,l} C[g,h,i,j,k,l] \times D[g,h,m,n,k,l]$$
- 这是一个 $K \times L$ 次乘法和加法

**FLOPs 计算**：
$$
\text{FLOPs} = 2 \times \textcolor{blue}{G} \times \textcolor{blue}{H} \times I \times J \times M \times N \times \textcolor{red}{K} \times \textcolor{red}{L}
$$

- 系数 2：每个元素需要 1 次乘法 + 1 次加法
- 所有维度相乘，但每个维度只计一次（不管它出现在几个张量中）

**记忆**：FLOPs = 2 × 所有维度之积（包括收缩维度）

---

### 前向与反向传播的 FLOPs

设 $A[N,P]$，$B[P,M]$，$C=AB$：

| 阶段 | 操作 | FLOPs |
|------|------|-------|
| 前向传播 | $C = AB$ | $2NPM$ |
| 反向传播 (权重梯度) | $\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$ | $2NPM$ |
| 反向传播 (输入梯度) | $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T$ | $2NPM$ |
| **总计** | - | ==$6NPM$== |

**推理** = $2NPM$（仅前向）；**训练** = $6NPM$（前向 + 两个反向），训练是推理的 3 倍。

### 符号定义

$B$=batch，$T/S$=序列长（Q/KV），$D$=model dim，$F$=FFN dim，$N$=Q heads，$K$=KV heads，$H$=head dim，$L$=层数，$V$=词表大小。

---

### MLP 层的计算

现代 Transformer 用 **Gated MLP**（SwiGLU）：$W_{out} \cdot [\sigma(W_{in1} x) \odot W_{in2} x]$，比传统 2 矩阵多 1 个矩阵（参数 +50%），表达能力更强。常见变体：GLU（sigmoid）、GEGLU（GELU）、SwiGLU（SiLU，LLaMA/Qwen3 使用）。

| 操作 | 训练 FLOPs | 参数量 |
|------|------------|--------|
| $A[B,T,\textcolor{red}{D}] \cdot W_{in1}[\textcolor{red}{D}, F]$ | $6BTDF$ | $DF$ |
| $A[B,T,\textcolor{red}{D}] \cdot W_{in2}[\textcolor{red}{D}, F]$ | $6BTDF$ | $DF$ |
| $\sigma(A_{in1}) \odot A_{in2}$ (激活+门控) | $O(BTF)$ 可忽略 | - |
| $A[B,T,\textcolor{red}{F}] \cdot W_{out}[\textcolor{red}{F}, D]$ | $6BTDF$ | $DF$ |
| **MLP 总计** | ==$\approx 18BTDF$== | ==$3DF$== |

*无 Gating 的传统 MLP：2 个矩阵，参数量 $2DF$。现代模型（LLaMA、DeepSeek 等）均使用 Gating 变体。*

---

### Attention 层的计算

#### QKVO 投影矩阵

| 操作 | 训练 FLOPs | 参数量 |
|------|------------|--------|
| $A[B,T,\textcolor{red}{D}] \cdot W_Q[\textcolor{red}{D}, N, H]$ | $6BTDNH$ | $DNH$ |
| $A[B,T,\textcolor{red}{D}] \cdot W_K[\textcolor{red}{D}, K, H]$ | $6BTDKH$ | $DKH$ |
| $A[B,T,\textcolor{red}{D}] \cdot W_V[\textcolor{red}{D}, K, H]$ | $6BTDKH$ | $DKH$ |
| $A[B,T,\textcolor{red}{N},\textcolor{red}{H}] \cdot W_O[\textcolor{red}{N},\textcolor{red}{H}, D]$ | $6BTDNH$ | $DNH$ |
| **QKVO 总计** | ==$12BTD(N+K)H$== | ==$2D(N+K)H$== |

#### Dot-Product Attention

| 操作 | 训练 FLOPs |
|------|------------|
| $Q[\textcolor{blue}{B}, T, \textcolor{blue}{K}, G, \textcolor{red}{H}] \cdot K[\textcolor{blue}{B}, S, \textcolor{blue}{K}, \textcolor{red}{H}]^T \rightarrow S[B,T,S,N]$ | $6BTSNH$ |
| $\text{softmax}_S(S) \rightarrow P$ | $O(BTSN)$ 可忽略 |
| $P[\textcolor{blue}{B}, T, \textcolor{red}{S}, \textcolor{blue}{K}, G] \cdot V[\textcolor{blue}{B}, \textcolor{red}{S}, \textcolor{blue}{K}, H] \rightarrow O[B,T,N,H]$ | $6BTSNH$ |
| **Attention 总计** (self-attention: S=T) | ==$\approx 12BT^2NH$== |

*注：Decoder-only 的 causal attention 只算下三角，实际 FLOPs 减半，但需要 Flash Attention 等专用 kernel 才能利用。*

---

### 6ND 法则

忽略 dot-product attention（短上下文合理），整个模型的 FLOPs：

$$
\boxed{\text{Total FLOPs} = 6 \times \text{num\_tokens} \times \text{num\_parameters}}
$$

MLP 和 QKVO 投影的 FLOPs 都是 **6 × BT × 参数量**（MLP: $18BTDF = 6BT \cdot 3DF$；QKVO: $12BTD(N+K)H = 6BT \cdot 2D(N+K)H$），因此忽略 attention 后：

$$
\text{FLOPs} = 6 \times BT \times \underbrace{(3DF + 2D(N+K)H) \times L}_{\text{总参数量}} = 6 \times N_{\text{tokens}} \times N_{\text{params}}
$$

系数 6 = 2（乘加）× 3（前向 + 权重梯度 + 输入梯度），每个参数在训练时被"使用" 6 次。

> [!example] 使用 6ND 法则
> 这个法则让估算训练成本变得非常简单：
> 
> ```
> 训练 FLOPs ≈ 6 × 参数量 × 训练 token 数
> ```
> 
> **例如**：训练一个 70B 参数模型，使用 2T tokens：
> ```
> FLOPs = 6 × 70×10⁹ × 2×10¹² = 8.4×10²³ FLOPs
> ```

#### Attention vs MLP：何时 attention 开始主导？

典型配置：$F = 4D$，$D = NH$，$N = K$

$$
\frac{\text{Attention FLOPs}}{\text{Matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{T}{8D}
$$

> [!important] 关键结论
> Dot-product attention 的 FLOPs 只有在 ==$T > 8D$== 时才会开始主导。
> 
> 对于 $D = 8192$ 的模型，这意味着 **~65K tokens**。
> 
> ==对于大模型，attention 的二次复杂度实际上并没有那么可怕！==



### 进阶话题

#### Mixture of Experts (MoE)

MoE 将单个 dense MLP 替换为多个独立的 "expert" MLP，通过 router 动态选择激活哪些 expert。

**核心思想**：增加模型容量（参数量）而不成比例增加计算量。

```
Dense:  x → MLP → y

MoE:    x → Router（选 top-k）→ E_1, E_2, ..., E_k → 加权求和 → y
```

**Vanilla MoE 公式 (LLM中具体design有所不同)**：

1. **Router 计算**（选择哪些 expert）：
$$
G(x) = \text{softmax}(\text{TopK}(x \cdot W_r))
$$
其中 $W_r \in \mathbb{R}^{D \times E}$ 是 router 权重，TopK 保留前 k 个最大值，其余置为 $-\infty$。

2. **MoE 输出**（加权组合）：
$$
\text{MoE}(x) = \sum_{i=1}^{E} G(x)_i \cdot \text{Expert}_i(x)
$$
由于 TopK，实际只有 k 个 expert 被激活（其余 $G(x)_i = 0$）。

3. **负载均衡损失**（防止 expert 使用不均）：
$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i
$$
其中 $f_i$ 是分配给 expert $i$ 的 token 比例，$p_i$ 是 router 对 expert $i$ 的平均概率。

#### MoE 参数

| 参数 | 说明 | 示例 (DeepSeek v3) |
|------|------|-------------------|
| **E** | Expert 数量 | 256 |
| **k** | 每个 token 激活的 expert 数 | 8 |
| **Sparsity** | $E / k$ | 32 |

#### 计算特点

- 总参数量增加 $O(E)$ 倍
- 每 token 激活参数仅增加 $k$ 倍
- 引入 AllToAll 通信开销

要达到 compute-bound，需要 $B > 120E/k$（DeepSeek E=256, k=8 时约 3840），推理时是相当大的 batch size。

---

### 梯度检查点 (Gradient Checkpointing)

保存中间激活避免 $O(L^2)$ 重计算，但内存开销极大：

> [!warning] 激活内存示例
> 对于 $BT = 4M$ tokens，$L = 64$ 层，$D = 8192$：
> ```
> 激活内存 ≈ 2 × 20 × BT × D × L = 84 TB (bf16)!
> ```

#### 检查点策略

| 策略 | 保存内容 | FLOPs 开销 |
|------|----------|------------|
| **Block Remat** | 每层输入 (1 checkpoint/层) | 从 $6ND$ 增加到 $8ND$ |
| **Big Matmuls Only** | 大矩阵乘法的输出 (7/层) | 避免重计算大矩阵乘法 |

---

### KV Cache

LLM 推理的两个阶段：

1. **Prefill**：处理 prompt，保存 K/V 到 cache
2. **Decode**：逐 token 生成，复用 KV cache

$$
\text{KV Cache Size} = 2 \times S \times L \times K \times H
$$

> [!example] KV Cache 大小示例
> 8K context，64 层，$KH = D = 8192$：
> ```
> KV Cache = 2 × 8192 × 64 × 8192 = 8 GiB (int8)
> ```
> ==这就是为什么 GQA ($K \ll N$) 如此重要！==


### 典型模型配置参考

| 参数 | 7B | 13B | 70B |
|------|-----|------|------|
| D (model dim) | 4096 | 5120 | 8192 |
| L (layers) | 32 | 40 | 80 |
| N (heads) | 32 | 40 | 64 |
| F (FFN dim) | 11008 | 13824 | 28672 |
| KV Cache/token (int8) | ~256KB | ~400KB | ~1.3MB |

## Back to nano-vllm

> 这一部分主要讲每个 **module 的设计**（`models/qwen3.py`、`layers/` 下各子模块），包括 attention、linear、embedding、sampler 等的实现细节。下一篇（Part 2）会讲 **system 的设计**：KV cache 管理、scheduler、continuous batching 等推理引擎核心。

接下来回到nano-vllm，从models/qwen3.py开始

这个文件的核心是如下的 Attention 设计：

### 1. Grouped Query Attention (GQA)

`num_heads (Q) >> num_kv_heads (K/V)`：Q 头数远多于 K/V 头数（例如 32Q heads vs 8KV heads），K/V 权重在多个 Q head 之间共享，大幅减少 KV Cache 显存。

```python
# qwen3.py:30-38
self.num_heads    = self.total_num_heads    // tp_size  # Q heads per GPU
self.num_kv_heads = self.total_num_kv_heads // tp_size  # KV heads per GPU
self.q_size  = self.num_heads    * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim        # kv_size << q_size
```

### 2. QK Norm（RMSNorm on Q and K）

Qwen3 的独特设计：在 RoPE 之前对 Q、K 各做一次 RMSNorm（per head_dim）：

```python
# qwen3.py:81-83
if not self.qkv_bias:
    q = self.q_norm(q)  # RMSNorm on each head's q
    k = self.k_norm(k)  # RMSNorm on each head's k
```

**为什么需要 QK Norm？**

- 防止 attention logit（$QK^T / \sqrt{d}$）数值爆炸，训练更稳定
- 替代了传统的 `qkv_bias`（当 `qkv_bias=False` 时启用）
- 与 Llama 等模型的区别：Qwen3 对每个 head 单独做 norm，粒度更细

![[assets/Pasted image 20260227222651.png]]

### 3. SwiGLU 激活函数

**SwiGLU** 是 GLU（Gated Linear Unit）的变体，公式为：

$$\text{SwiGLU}(x_1, x_2) = \text{SiLU}(x_1) \times x_2$$

两个优势：**门控机制**（$x_2$ 动态控制每个维度信息流通量，表达能力更强）；**SiLU 平滑**（$x \cdot \sigma(x)$，比 ReLU 梯度更稳定）。Google 在 *GLU Variants Improve Transformer (2020)* 中实验证明 SwiGLU 在语言模型上效果最好，此后成为主流。

**`layers/activation.py` forward 逐行解读**：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)  # 沿最后一维切成两半
    return F.silu(x) * y    # 对前半做 SiLU，再和后半逐元素相乘
```

| 步骤 | 操作 | 结果 shape |
|------|------|-----------|
| 输入 | Linear 将 hidden 投影到 2× 宽度 | `[B, L, 2H]` |
| `x.chunk(2, -1)` | 最后一维切成两半，得到 x 和 y | 各 `[B, L, H]` |
| `F.silu(x)` | 激活：$x \cdot \sigma(x)$ | `[B, L, H]` |
| `* y` | 与门控信号 y 逐元素相乘 | `[B, L, H]` |

> [!note] 为什么 Linear 投影到 2H？
> SwiGLU 需要两路输入，FFN 第一个 Linear 输出设为 `intermediate_size * 2`，forward 中 chunk 成两份，一份激活、一份门控。这正是 `layers/activation.py:13` 的作用。

### 4. LayerNorm vs RMSNorm（`layers/layernorm.py`）

| | 公式 | 特点 |
|---|---|---|
| **LayerNorm** | $(x - \mu) / \sqrt{\sigma^2 + \varepsilon} \cdot \gamma + \beta$ | 先中心化再归一化 |
| **RMSNorm** | $x / \sqrt{\text{mean}(x^2) + \varepsilon} \cdot \gamma$ | 去掉减均值步骤，更快 |

RMSNorm 省去了均值计算，实验表明效果与 LayerNorm 相当，因此成为现代 LLM（LLaMA、Qwen3）的默认选择。

### 5. RoPE 位置编码（`layers/rope.py`）

**核心思想**：对 head 中每对维度 $(x_1, x_2)$ 做旋转，频率 $\theta_i = 1 / \text{base}^{2i/d}$（base 通常 10000，Qwen 用更大值）：

$$y_1 = x_1\cos\theta - x_2\sin\theta, \quad y_2 = x_2\cos\theta + x_1\sin\theta$$

`apply_rotary_emb` 直接实现上式，无额外变换。

**三个工程优化**：

**① 预计算 cache**（`__init__` 里）：

```python
inv_freq = 1.0 / (base ** (arange(0, d, 2) / d))     # 频率向量
freqs    = einsum("i,j->ij", positions, inv_freq)      # 外积 [max_pos, d/2]
cache    = cat(cos, sin, dim=-1).unsqueeze_(1)         # [max_pos, 1, d]
```

把所有位置的 cos/sin 提前算好存表，推理时直接 gather，避免重复计算。

**② 按位置索引**（`forward` 里）：

```python
cos_sin = self.cos_sin_cache[positions]   # 直接 gather，支持非连续位置
cos, sin = cos_sin.chunk(2, dim=-1)
```

支持 prefill/decode 混合场景中的非连续 token 位置。

**③ `lru_cache(1)` 单例化**：

```python
@lru_cache(1)
def get_rope(...):
    assert rope_scaling is None   # 只支持标准 RoPE
```

全局只创建一个 RoPE 实例，节省显存；`assert` 明确不支持 YaRN / 线性插值等扩展。

### 6. 采样策略（`layers/sampler.py`）

原始的nano-vllm采用的是普通的贪心采样，这里可以简单扩展讨论一下top-p采样。
采样分三步：温度缩放 → Top-p 过滤 → Gumbel-max 采样。

#### Step 1 — Temperature Scaling（第 13 行）

```python
logits = logits.div_(temperatures.unsqueeze(1))
```

温度的本质是"拉伸或压缩 logit 之间的差距"：

| 温度 | 效果 | 用途 |
|------|------|------|
| T < 1（如 0.3） | 差距放大，高分 token 概率更集中 | 输出更确定、保守 |
| T = 1 | 不变 | 模型原始分布 |
| T > 1（如 2.0） | 差距缩小，概率趋于均匀 | 输出更多样、创意 |

*例：logits = [10, 5, 1]，T=0.5 → [20, 10, 2]，softmax 后最大值概率更高。*

#### Step 2 — Top-p 过滤（第 15–20 行）

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = softmax(sorted_logits).cumsum(-1)
# 累计概率超过 p 之后的 token 屏蔽为 -inf
to_remove = (cumulative_probs - softmax(sorted_logits)) >= top_ps
logits = logits.masked_fill(to_remove, float('-inf'))
```

```
token:   A     B     C     D     E
prob:   0.5   0.3   0.1   0.06  0.04
cumsum: 0.5   0.8   0.9   0.96  1.0
                    ↑ top_p=0.9 恰好在此截断，D/E 被屏蔽
```

`cumsum - prob`（加入当前 token *之前*的累计概率）确保恰好触碰到 p 的那个 token 被保留。

#### Step 3 — Gumbel-max 采样（第 22–23 行）

```python
probs.div_(torch.empty_like(probs).exponential_(1)).argmax()
```

从过滤后的分布中随机采样。随机性来自为每个 token 独立采样的 $\text{Exp}(1)$ 随机变量，再用 $p_i / E_i$ 取 argmax。

**为什么等价于按概率采样？**（Gumbel-Max Trick）

从分类分布 $p$ 采样等价于：

$$\text{argmax}\bigl(\log p_i + G_i\bigr), \quad G_i \sim \text{Gumbel}(0,1)$$

利用 $G_i = -\log E_i,\ E_i \sim \text{Exp}(1)$，代入得：

$$\text{argmax}\bigl(\log p_i - \log E_i\bigr) = \text{argmax}\bigl(\log(p_i / E_i)\bigr) = \text{argmax}(p_i / E_i)$$

即代码中的写法，在数学上严格等价于从原始分布采样。

### 7. 列并行与行并行 Linear（`layers/linear.py`, `embed_head.py`）

![[assets/Pasted image 20260302111018.png]]

对线性层 $Y = XW$，TP 有两种切法：

**列并行（Column Parallel，按输出维切 W）**：每卡算一段输出通道 $Y_i = XW_i$，各卡输出天然不重叠。
- Forward：无通信（各卡独立算，后续需完整 Y 时 concat/all-gather）
- Backward：算 $dX = \sum_i dY_i W_i^\top$ 时需要 all-reduce

**行并行（Row Parallel，按输入维切 W）**：每卡拿到一段输入 $X_i$，算部分和 $Y_i = X_i W_i$，最终 $Y = \sum_i Y_i$。
- Forward：需要 all-reduce（把各卡部分和加起来）
- Backward：$dX_i = dY W_i^\top$ 各卡各算，不需要 all-reduce

**一句话**：forward 通信上，列并行是 gather、行并行是 reduce；但列并行的 backward 仍需 all-reduce。

> [!note] vLLM 是推理系统，无 backward
> 推理时只跑 forward：列并行完全无通信，行并行仍需 all-reduce（把各卡的部分 logit 求和）。所以 vLLM 中 FFN 的 gate/up 矩阵用列并行（free），down 矩阵用行并行（one all-reduce per layer）。

**所有 Linear 类继承自 `LinearBase`**，核心差异在权重如何切分（`weight_loader`）和 forward 是否需要通信：

| 类 | weight 形状（单卡） | 通信 | 用途 |
|---|---|---|---|
| `ReplicatedLinear` | `[O, I]`（完整复制） | 无 | RMSNorm weight 等小参数 |
| `ColumnParallelLinear` | `[O/tp, I]`（按输出维切） | 无 | FFN up/gate（后接 Row） |
| `MergedColumnParallelLinear` | `[(O1+O2)/tp, I]` | 无 | FFN gate+up 合并一次 kernel |
| `QKVParallelLinear` | `[(Nq+2Nkv)·H/tp, I]` | 无 | Q/K/V 合并，GQA 时 K/V shard 更小 |
| `RowParallelLinear` | `[O, I/tp]`（按输入维切） | all_reduce | FFN down，各卡部分和求和 |

**关键实现细节**：

`weight_loader` 以属性挂载到 `Parameter` 上，checkpoint 加载时统一调用，每个类自己决定怎么切分——无需修改加载逻辑。

`RowParallelLinear` bias 只在 rank 0 加：`bias if tp_rank == 0 else None`，避免 all_reduce 后 bias 被累加 tp_size 倍。

**数学等价性**（Column + Row 配对）：
```
完整: y = x @ W.T
分片: x 完整，W 按输出维切 → 各卡 y_i = x @ Wi.T，concat 得 y   ← ColumnParallel
      x 分片，W 按输入维切 → 各卡 y_i = xi @ Wi.T，all_reduce sum ← RowParallel
```

### 8. VocabParallelEmbedding 与 ParallelLMHead（`embed_head.py`）

#### VocabParallelEmbedding

**`__init__`**：词表按 TP 均分，每卡持有 `vocab/tp` 个向量：

```python
self.vocab_start_idx = num_embeddings_per_partition * tp_rank
self.vocab_end_idx   = vocab_start_idx + num_embeddings_per_partition
self.weight = nn.Parameter(torch.empty(num_embeddings_per_partition, embedding_dim))
```

**`forward`**：查表 + all_reduce

```python
mask = (x >= vocab_start_idx) & (x < vocab_end_idx)
x    = mask * (x - vocab_start_idx)  # 范围外归零（避免越界崩溃）
y    = F.embedding(x, self.weight)   # 安全查表，范围外结果无意义
y    = mask.unsqueeze(1) * y         # mask 抹掉无意义结果
dist.all_reduce(y)                   # 每个 token 只有一卡非零，sum = 正确向量
```

为什么不直接跳过范围外的 token？GPU 上 if/else 很贵，用 "先安全查、再 mask 清零" 的 trick 避免条件分支。

数值示例（tp=2, vocab=10）：
```
x = [2, 7, 0, 5]
GPU 0 (id 0~4): y = embed([2,0,0,0]) * mask → [vec2, 0, vec0, 0]
GPU 1 (id 5~9): y = embed([0,2,0,0]) * mask → [0, vec7, 0, vec5]
all_reduce sum → [vec2, vec7, vec0, vec5] ✓
```

#### ParallelLMHead

继承 `VocabParallelEmbedding` **共享权重**，但 forward 完全不同：

```python
def forward(self, x):
    # Prefill 只取每段最后一个 token（只需预测下一个 token）
    if context.is_prefill:
        x = x[context.cu_seqlens_q[1:] - 1].contiguous()

    logits = F.linear(x, self.weight)   # [batch, vocab/tp]

    # gather 到 rank 0 拼接完整 logits（只有 rank 0 做采样）
    if tp_size > 1:
        all_logits = [torch.empty_like(logits) for _ in range(tp_size)] if rank == 0 else None
        dist.gather(logits, all_logits, dst=0)
        logits = torch.cat(all_logits, dim=-1) if rank == 0 else None
    return logits
```

| | VocabParallelEmbedding | ParallelLMHead |
|---|---|---|
| 通信 | all_reduce（所有卡都需要结果） | gather → rank 0（只有 rank 0 采样） |
| 原因 | 所有卡的下一层都要用 embedding | 采样只在 rank 0 做 |

**Prefill 只算最后一个 token**：输入序列 [tok1...tokN] 只需 tokN 的 logits，LM Head 计算量从 `seq_len × vocab` 降为 `batch × vocab`。



---

## GQA 实现要点（面试重点）

**核心思想**：多个 Q head 共享同一组 K/V head，KV cache 大小 ∝ `num_kv_heads`（而非 `num_heads`）。

```
MHA (28Q, 28KV): Q0↔KV0, Q1↔KV1, ..., Q27↔KV27
GQA (28Q,  4KV): Q0~Q6↔KV0, Q7~Q13↔KV1, Q14~Q20↔KV2, Q21~Q27↔KV3
节省倍数 = num_heads / num_kv_heads = 28 / 4 = 7×
```

**5 步实现**（以 Qwen3 + nano-vllm 为例）：

**① 声明不对称 head 数**
```python
self.num_heads    = total_num_heads    // tp_size   # Q heads/卡（大）
self.num_kv_heads = total_num_kv_heads // tp_size   # KV heads/卡（小）
self.q_size  = self.num_heads    * head_dim
self.kv_size = self.num_kv_heads * head_dim
```

**② QKV 投影输出不对称维度**
```python
# output = (28 + 2×4) × head_dim = 36 × head_dim
output_size = (total_num_heads + 2 * total_num_kv_heads) * head_dim
```

**③ split 按不同大小切分**
```python
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
q = q.view(-1, num_heads,    head_dim)   # [N, 28/tp, d]
k = k.view(-1, num_kv_heads, head_dim)   # [N,  4/tp, d]
v = v.view(-1, num_kv_heads, head_dim)   # [N,  4/tp, d]
```

**④ Flash Attention 原生支持 GQA**：`flash_attn_varlen_func` 当 `num_heads != num_kv_heads` 时自动广播 K/V，无需手动 `repeat_kv`。

**⑤ KV cache 按 kv_heads 分配**
```python
# kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
#                                                          ↑ 只存 4 份，不是 28 份
```

**面试高频问题**：

> **Q: GQA 节省的是什么内存？节省多少？**
> KV cache（推理时的主要内存瓶颈）。节省倍数 = `num_heads / num_kv_heads`。Qwen3-8B：28Q/8KV = 3.5×；LLaMA-3-70B：64Q/8KV = 8×。模型权重本身（W_K, W_V）也等比减小。

> **Q: GQA 和 MQA 的区别？**
> MQA（Multi-Query Attention）是 GQA 的极端情况（`num_kv_heads=1`），所有 Q head 共享同一个 K/V。GQA 是折中方案，效果接近 MHA，KV cache 接近 MQA。

> **Q: GQA 在 TP 下如何切分？**
> Q/K/V 都按 `// tp_size` 切分，但需保证 `num_kv_heads` 能被 `tp_size` 整除。如果 `num_kv_heads < tp_size`，部分卡没有 KV head，需要额外的 broadcast 逻辑（不常见）。

> **Q: 为什么不需要手动 repeat K/V？**
> Flash Attention kernel 内部直接根据 `num_heads / num_kv_heads`（group size）做 broadcast，在寄存器级别完成，比显式 repeat 节省显存且更高效。

## 参考资料

- [How To Scale Your Model - Part 4: Transformers](https://jax-ml.github.io/scaling-book/transformers/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- https://github.com/GeeeekExplorer/nano-vllm

---

## 课后练习题

### 练习 1：prefill 和 decode 的差异

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么 LLM serving 要区分 prefill 和 decode？</span></summary>

prefill 一次处理 prompt 的多个 token，矩阵较大，更像 compute-bound GEMM；decode 每步只生成一个新 token，batch 形状小，常受 KV cache 读取、调度和 memory bandwidth 影响。continuous batching 的核心就是把许多 decode request 拼起来，提高 GPU 利用率。

</details>

### 练习 2：Paged KV Cache

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">PagedAttention 为什么要把 KV cache 分页？</span></summary>

不同请求长度不同，直接为每个请求预留最大长度会浪费显存。Paged KV Cache 把 KV 切成固定大小 block，用 block table 映射逻辑 token 到物理 cache block。这样能减少碎片，支持动态增长、复用和 eviction。

</details>
