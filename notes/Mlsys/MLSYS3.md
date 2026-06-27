# Roofline Analysis

> 核心问题：为什么一个算法跑 50ms 而不是 5ms 或 500ms？瓶颈到底在哪里？
## 1. Motivation

在深度学习中，我们经常遇到这样的困惑：
- 增大 batch size 有时能提速，有时却没用
- 同样的模型在不同硬件上表现差异巨大
- 某些算子（如 attention）特别慢，但矩阵乘法却很快
**Roofline 分析**提供了一个简洁的框架来回答这些问题：它告诉你当前的瓶颈是**算力**还是**带宽**，以及如何优化。
## 2. 核心定义

任何计算都可以分解为两部分时间：
$$T_{\text{math}} = \frac{\text{FLOPs}}{\text{Accelerator FLOPs/s}}$$
$$T_{\text{comms}} = \frac{\text{Bytes}}{\text{Bandwidth (Bytes/s)}}$$

| 符号        | 含义     | 例子 (TPU v5e)                 |
| --------- | ------ | ---------------------------- |
| FLOPs/s   | 芯片峰值算力 | $1.97 \times 10^{14}$ (bf16) |
| Bandwidth | HBM 带宽 | $8.2 \times 10^{11}$ bytes/s |

### 算术强度 (Arithmetic Intensity)
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}$$
这是 roofline 分析的核心概念：**每搬运一个 byte 的数据，能做多少次浮点运算**。
### 临界强度 (Critical Intensity)
$$\text{Critical Intensity} = \frac{\text{Peak FLOPs/s}}{\text{Peak Bandwidth}}$$
对于 TPU v5e MXU：
$$\frac{1.97 \times 10^{14}}{8.2 \times 10^{11}} \approx 240 \text{ FLOPs/byte}$$
### 2.4 Compute-bound vs Memory-bound

| 条件                                                              | 状态                         | 含义           |
| --------------------------------------------------------------- | -------------------------- | ------------ |
| $\text{Intensity}_{\text{algo}} > \text{Intensity}_{\text{hw}}$ | **Compute-bound**          | 算力被充分利用 ✓    |
| $\text{Intensity}_{\text{algo}} < \text{Intensity}_{\text{hw}}$ | **Memory/Bandwidth-bound** | 算力在等数据，被浪费 ✗ |

## 3. 分析方法与例子

### 3.1 Roofline 分析的系统方法

进行 Roofline 分析需要遵循以下步骤：
**Step 1: 确定硬件参数**
首先需要查阅目标硬件的规格：

| 硬件      | HBM 容量 | HBM 带宽 $\beta$           | bf16 算力 $\pi$         | int8 算力               | 临界强度 $I_c = \pi/\beta$ |
| ------- | ------ | ------------------------ | --------------------- | --------------------- | ---------------------- |
| TPU v5e | 16 GB  | $8.1 \times 10^{11}$ B/s | $1.97 \times 10^{14}$ | $3.94 \times 10^{14}$ | 243                    |
| TPU v5p | 96 GB  | $2.8 \times 10^{12}$ B/s | $4.59 \times 10^{14}$ | $9.18 \times 10^{14}$ | 164                    |
| TPU v6e | 32 GB  | $1.6 \times 10^{12}$ B/s | $9.20 \times 10^{14}$ | $1.84 \times 10^{15}$ | 575                    |

**Step 2: 计算算法的 FLOPs 和 Bytes**

对于给定算法，分别计算：
- $W$：总计算量（FLOPs）
- $Q$：总数据传输量（Bytes）= 读取 + 写回

**Step 3: 计算算术强度**
$$I_{\text{algo}} = \frac{W}{Q} \quad \text{(FLOPs/Byte)}$$
**Step 4: 判断瓶颈类型**

比较 $I_{\text{algo}}$ 与 $I_c$：
$$T_{\text{actual}} = \max\left( \underbrace{\frac{W}{\pi}}_{T_{\text{compute}}}, \underbrace{\frac{Q}{\beta}}_{T_{\text{memory}}} \right)$$
- 若 $I_{\text{algo}} > I_c$：**Compute-bound**，$T_{\text{actual}} = T_{\text{compute}}$
- 若 $I_{\text{algo}} < I_c$：**Memory-bound**，$T_{\text{actual}} = T_{\text{memory}}$

**Step 5: 计算硬件利用率**
$$\text{Efficiency} = \frac{\text{Achieved FLOPs/s}}{\pi} = \frac{W / T_{\text{actual}}}{\pi}$$
### 3.2 Roofline 图的理解
**两个区域的物理意义**：
- **斜线区域（Memory-bound）**：数据搬运是瓶颈，计算单元在"等数据"
  - 实际吞吐 = $I_{\text{algo}} \times \beta$（随强度线性增长）
- **平台区域（Compute-bound）**：计算是瓶颈，已达峰值算力
  - 实际吞吐 = $\pi$（不再增长）

![[assets/Pasted image 20251216210603.png]]
> *展示了两种不同运算强度的算法（算法 1 和算法 2）及其在不同带宽（BW1 和 BW2）下的理论峰值吞吐量。红色区域表示算法在两种带宽下均受限于带宽，浪费了硬件峰值 FLOPs/s 的一部分。黄色区域表示算法仅在较低带宽（BW1）下受限于带宽。绿色区域表示算法在所有带宽下均受限于计算能力。此处，我们已充分利用了加速器的峰值 FLOPs/s，增加带宽或提高运算强度均无益处。*

### 3.3 例子：Dot Product（向量点积）

**问题**：计算 `x · y`，其中 `x, y ∈ bf16[N]`，输出 `bf16[1]`

| 项目              | 计算过程                                                 | 结果           |
| --------------- | ---------------------------------------------------- | ------------ |
| 读取              | `x` 需要 $2N$ bytes，`y` 需要 $2N$ bytes (bf16 = 2 bytes) | $4N$         |
| 写回              | 输出 1 个 bf16 标量                                       | $2$          |
| **总 Bytes $Q$** | $4N + 2$                                             | $\approx 4N$ |
| FLOPs           | $N$ 次乘法 + $(N-1)$ 次加法                                | $\approx 2N$ |

$$I_{\text{dot}} = \frac{W}{Q} = \frac{2N}{4N + 2} \xrightarrow{N \to \infty} \frac{1}{2}$$
对于 TPU v5e，$I_c = 243$，而 $I_{\text{dot}} = 0.5 \ll 243$
**结论**：向量点积**永远是 memory-bound**，无论 $N$ 多大。这是因为每个元素只被使用一次（没有数据复用），算术强度有上界。
> [!warning]
> 这解释了为什么 elementwise 操作（如 ReLU、LayerNorm）通常需要通过 **kernel fusion** 来优化——单独执行时几乎总是 memory-bound。
### 3.4 例子：Matrix Multiplication（矩阵乘法）⭐

**问题**：计算 `C = A @ B`，其中 `A ∈ bf16[M, K]`，`B ∈ bf16[K, N]`，输出 `C ∈ bf16[M, N]`

| 项目              | 计算过程                                  | 结果          |
| --------------- | ------------------------------------- | ----------- |
| 读取 A            | $M \times K$ 个 bf16                   | $2MK$ bytes |
| 读取 B            | $K \times N$ 个 bf16                   | $2KN$ bytes |
| 写回 C            | $M \times N$ 个 bf16                   | $2MN$ bytes |
| **总 Bytes $Q$** | $2(MK + KN + MN)$                     |             |
| FLOPs           | 每个输出元素需要 $K$ 次乘法 + $K$ 次加法，共 $MN$ 个输出 | $2MNK$      |

> [!note] 为什么是 2MNK？
> 矩阵乘法 $C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}$ 对每个输出元素做 $K$ 次 multiply-add。
> 一次 multiply-add = 2 FLOPs，故总共 $2 \times M \times N \times K$ FLOPs。

$$I_{\text{matmul}} = \frac{2MNK}{2(MK + KN + MN)} = \frac{MNK}{MK + KN + MN}$$
**特殊情况分析**（设 $M = B$（batch），$K = D$（hidden），$N = F$（output））：

| 情况 | 条件 | 近似强度 | 物理意义 |
|------|------|----------|----------|
| Batch 推理 | $B \ll D, F$ | $I \approx B$ | batch size 决定是否 compute-bound |
| 方阵乘法 | $M = K = N$ | $I \approx \frac{N}{3}$ | 维度越大越好 |
| GEMV | $N = 1$ | $I \approx 1$ | 向量-矩阵乘，几乎总是 memory-bound |

对于 `bf16[B, D] @ bf16[D, F] → bf16[B, F]`（典型的 FFN 层）：
当 $B \ll D, F$ 时：
$$I \approx \frac{BDF}{DF} = B$$
在 TPU v5e 上，**Batch size $B > 243$** 时，matmul 变成 compute-bound。

**常见矩阵乘法 FLOPs 速查表**：

| 操作 | Shape | FLOPs | 说明 |
|------|-------|-------|------|
| GEMM | `[M,K] @ [K,N]` | $2MNK$ | 通用矩阵乘 |
| GEMV | `[M,K] @ [K,1]` | $2MK$ | 矩阵-向量乘，$I \approx 1$ |
| 方阵乘法 | `[N,N] @ [N,N]` | $2N^3$ | $I \approx N/3$ |
| Batch GEMM | `[B,M,K] @ [B,K,N]` | $2BMNK$ | 批量矩阵乘 |

### 3.5 例子：完整 Roofline 计算

**问题**：在 TPU v5e 上计算 `bf16[256, 4096] @ bf16[4096, 4096]`，分析其性能。

**已知参数**：
- $\pi = 1.97 \times 10^{14}$ FLOPs/s（bf16 峰值算力）
- $\beta = 8.1 \times 10^{11}$ B/s（HBM 带宽）
- $I_c = \pi / \beta = 243$ FLOPs/Byte

**Step 2: 计算量**
- $M = 256, K = 4096, N = 4096$
- $W = 2MNK = 2 \times 256 \times 4096 \times 4096 = 8.59 \times 10^9$ FLOPs
- $Q = 2(MK + KN + MN) = 2(256 \times 4096 + 4096 \times 4096 + 256 \times 4096)$
  $= 2(1.05 \times 10^6 + 1.68 \times 10^7 + 1.05 \times 10^6) = 3.77 \times 10^7$ Bytes

**Step 3: 算术强度**
$$I = \frac{8.59 \times 10^9}{3.77 \times 10^7} = 228 \text{ FLOPs/Byte}$$
**Step 4: 判断瓶颈**
- $I = 228 < I_c = 243$ → **Memory-bound**（略低于临界点）

**Step 5: 计算时间和效率**
- $T_{\text{compute}} = W / \pi = 8.59 \times 10^9 / 1.97 \times 10^{14} = 43.6 \mu s$
- $T_{\text{memory}} = Q / \beta = 3.77 \times 10^7 / 8.1 \times 10^{11} = 46.5 \mu s$
- $T_{\text{actual}} = \max(43.6, 46.5) = 46.5 \mu s$
- Efficiency = $43.6 / 46.5 = 93.8\%$

**结论**：虽然略微 memory-bound，但效率已达 94%，接近最优。

### 3.6 例子：Int8 量化 Matmul

**问题**：`int8[B, D] @ int8[D, F] → int8[B, F]`

**变化分析**：

| 项目 | bf16 版本 | int8 版本 | 变化 |
|------|-----------|-----------|------|
| 数据类型大小 | 2 bytes | 1 byte | $\times 0.5$ |
| 总 Bytes $Q$ | $2(BD + DF + BF)$ | $BD + DF + BF$ | $\times 0.5$ |
| 峰值算力 $\pi$ | $1.97 \times 10^{14}$ | $3.94 \times 10^{14}$ | $\times 2$ |
| FLOPs $W$ | $2BDF$ | $2BDF$ | 不变 |

**新的算术强度**：
$$I_{\text{int8}} = \frac{2BDF}{BD + DF + BF}$$
当 $B \ll D, F$ 时：
$$I_{\text{int8}} \approx \frac{2BDF}{DF} = 2B$$
**新的临界强度**：
$$I_c^{\text{int8}} = \frac{3.94 \times 10^{14}}{8.1 \times 10^{11}} = 486$$
**Compute-bound 条件**：
$$2B > 486 \implies B > 243$$
**结论**：
- Int8 的临界 batch size **仍然约为 243**（与 bf16 相同！）
- 但达到 compute-bound 后，**吞吐量翻倍**

> [!tip]
> 量化的主要收益是在 compute-bound 区域提升吞吐，而非改变临界点。

### 3.7 例子：混合精度（Int8 权重 + BF16 激活）

**问题**：`bf16[B, D] @ int8[D, F] → bf16[B, F]`

这种方案常用于推理优化：权重量化为 int8，但激活保持 bf16 精度。

**分析**：

| 项目 | 计算 |
|------|------|
| 读取激活 A | $2BD$ bytes (bf16) |
| 读取权重 B | $DF$ bytes (int8) |
| 写回输出 C | $2BF$ bytes (bf16) |
| **总 Bytes $Q$** | $2BD + DF + 2BF$ |
| FLOPs | $2BDF$（仍按 bf16 计算） |

**算术强度**：
$$I_{\text{mixed}} = \frac{2BDF}{2BD + DF + 2BF}$$
当 $B \ll D, F$ 且 $D \approx F$ 时：
$$I_{\text{mixed}} \approx \frac{2BDF}{DF} = 2B$$
**Compute-bound 条件**（使用 bf16 算力 $\pi = 1.97 \times 10^{14}$）：
$$2B > \frac{1.97 \times 10^{14}}{8.1 \times 10^{11}} = 243 \implies B > 122$$
**结论**：混合精度方案只需 **$B > 122$** 即可 compute-bound，比纯 bf16（$B > 243$）更容易达到！

### 3.8 不同内存层级的影响

TPU 有多层内存，带宽差异巨大：

| 内存类型 | 带宽 | 相对 HBM | 典型用途 |
|----------|------|----------|----------|
| VMEM (SRAM) | ~18 TB/s | 22× | Tile 内部计算 |
| HBM | ~0.8 TB/s | 1× | 主存储 |
| ICI (芯片间) | ~0.09 TB/s | 0.1× | 多芯片通信 |
| PCIe | ~0.015 TB/s | 0.02× | Host-Device 传输 |

**例子**：`int8[B, 4096] @ int8[16384, 4096]`

| 内存来源 | 临界 Batch Size |
|----------|-----------------|
| HBM | $B > 271$ |
| VMEM | $B > 11$ |

**结论**：如果权重可以 fit 进 VMEM，临界点降低 25 倍！这就是为什么 **tiling** 和 **weight caching** 如此重要。
### 3.9 Batch-Specific 权重矩阵（反面教材）

**问题**：如果每个 batch 元素有不同的权重矩阵：
`int8[B, D] @ int8[B, D, F] → int8[B, F]`

求算术强度。

**分析**：

这种情况在某些特殊场景出现（如 LoRA 的极端情况、per-sample adaptation）。

| 项目 | 标准 matmul | Batch-specific 权重 |
|------|-------------|---------------------|
| 读取 X | $BD$ | $BD$ |
| 读取 Y | $DF$ | $\mathbf{BDF}$ |
| 写回 Z | $BF$ | $BF$ |
| **总 Bytes** | $BD + DF + BF$ | $BD + BDF + BF$ |
| FLOPs | $2BDF$ | $2BDF$ |

**算术强度**：
$$I = \frac{2BDF}{BD + BDF + BF}$$
分母中 $BDF$ 占主导（因为 $D, F$ 通常很大）：
$$I \approx \frac{2BDF}{BDF} = 2$$
**结论**：
$$\boxed{I \approx 2 \text{ (常数)}}$$
> [!warning] 这是一个反面教材！
>
> 算术强度为常数（约为 2）意味着：
> - **永远是 memory-bound**，无论 batch size 多大
> - 每个权重元素只被使用一次，没有数据复用
> - 硬件利用率极低：$\text{Efficiency} = \frac{2}{486} \approx 0.4\%$
>
> **避免这种模式的方法**：
> - 尽量共享权重（标准 matmul）
> - 如果必须用不同权重，考虑分组/分块复用
> - 使用 LoRA 等低秩适配方法

### 3.10 GPU (H100) Roofline 分析

**问题**：使用 NVIDIA H100 的规格，计算临界 batch size。

**H100 SXM 规格**：

| 参数 | 数值 | 说明 |
|------|------|------|
| bf16 Tensor Core FLOPs | $1.979 \times 10^{15}$ | **带稀疏性** |
| 实际 bf16 FLOPs | $\sim 1 \times 10^{15}$ | 无稀疏性（除以 2） |
| HBM3 带宽 | 3.35 TB/s = $3.35 \times 10^{12}$ B/s | |
| HBM 容量 | 80 GB | |

> [!note] 关于稀疏性
> NVIDIA 宣传的 Tensor Core FLOPs 包含 2:4 结构化稀疏加速。
> 实际使用中，如果模型没有稀疏化，需要将官方数字除以 2。

**临界强度**：
$$I_c = \frac{\pi}{\beta} = \frac{1 \times 10^{15}}{3.35 \times 10^{12}} = 298 \text{ FLOPs/byte}$$
**临界 Batch Size**（当 $B \ll D, F$）：
$$B > I_c \implies \boxed{B > 298}$$
**与 TPU v5e 对比**：

| 硬件 | 峰值算力 $\pi$ | HBM 带宽 $\beta$ | 临界强度 $I_c$ | 临界 B |
|------|----------------|------------------|----------------|--------|
| TPU v5e | $1.97 \times 10^{14}$ | $8.1 \times 10^{11}$ | 243 | ~243 |
| H100 SXM | $1.0 \times 10^{15}$ | $3.35 \times 10^{12}$ | 298 | ~298 |
| **比值** | 5× | 4× | 1.2× | ~1.2× |

> [!important] 关键发现
> 尽管 H100 的绝对算力和带宽都远超 TPU v5e，但**临界 batch size 几乎相同**！
>
> 这是因为两者的"算力/带宽"比例接近（约 240-300 FLOPs/byte）。
> 这个比例由芯片架构决定，是现代 AI 加速器的共同特征。
## 4. 实践：代码与工具

### 4.1 PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

def torch_roofline(B, D, F, device='cuda'):
    x = torch.randn(B, D, dtype=torch.bfloat16, device=device)
    y = torch.randn(D, F, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(10):
        _ = x @ y
    torch.cuda.synchronize()
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    ) as prof:
        for _ in range(100):
            _ = x @ y
        torch.cuda.synchronize()
    
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=10
    ))
    
    # 导出 Chrome trace
    prof.export_chrome_trace("torch_trace.json")

torch_roofline(256, 4096, 4096)
```

### 4.2 NVIDIA Nsight 分析（需要root权限）

```bash
# 收集 roofline 数据
ncu --set roofline -o profile ./your_program

# 查看报告
ncu-ui profile.ncu-rep
```

### 4.3 分析hello world kernel/matmul kernel的roofline


![[assets/Pasted image 20251223165208.png]]
所有ops都在同一个位置


![[assets/Pasted image 20251223164911.png]]

这个是matmul的roofline curve，可以看到随着scale增大，逐渐从memory bound成为了compute bound(这里会跑到线上去,为什么？因为这张图其实是错的，这是cuda core的图，但是bf16的matmul会用到的是tensor core！)
### 5 小结

![[assets/Pasted image 20251223105527.png]]

* 点相对于 Ridge Point 的位置
	* 点在 Ridge Point 左侧（AI < Ridge Point）：
		* 算法处于 Memory-Bound 状态。性能瓶颈是内存带宽，计算单元在等待数据。理论最大性能 = 带宽 × AI。此时增加计算能力没有意义，因为数据供应不上。
		* 优化方向：减少内存访问（算子融合、量化、稀疏化）或提高数据复用（改变算法）。
	* 点在 Ridge Point 右侧（AI > Ridge Point）：
		* 算法处于 Compute-Bound 状态。性能瓶颈是计算能力，内存带宽有富余。理论最大性能 = 峰值算力。此时增加内存带宽没有意义，因为计算跟不上。
		* 优化方向：使用更高效的计算指令（Tensor Core）、提高并行度、减少指令依赖。
* 点相对于 Roofline 线的位置
	* 点在线上（效率 > 80%）：
		* 实现已经接近硬件极限，当前 AI 下几乎没有优化空间。如果还想提升性能，必须改变算法本身来提高 AI（比如算子融合），或者换更强的硬件。
	* 点在线下（效率 < 80%）：实现没有充分利用硬件，存在优化空间。需要诊断具体原因。
		* 如果在 Memory-Bound 区域且效率低，可能是：内存访问不连续（non-coalesced）、cache 命中率低、存在 bank conflict、数据对齐问题。
		* 如果在 Compute-Bound 区域且效率低，可能是：occupancy 不足、寄存器溢出、没有使用 Tensor Core、存在指令依赖导致流水线停顿。
	* 点在线上方：理论上不可能。如果测量结果显示点在 roofline 上方，说明测量有误或者 AI 计算错误。常见原因包括：没有算上 cache 效应导致实际内存访问量小于理论值、FLOPs 统计有遗漏、计时不准确。

Ref: 
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.

---

## 课后练习题

### 练习 1：判断 memory-bound / compute-bound

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">给定 FLOPs、Bytes、peak FLOPs 和 bandwidth，怎么判断瓶颈？</span></summary>

先算 arithmetic intensity 等于 FLOPs 除以 Bytes，再算硬件 ridge point 等于 peak FLOPs 除以 bandwidth。如果前者小于后者，性能上界由带宽决定，属于 memory-bound；如果前者大于后者，性能上界接近峰值算力，属于 compute-bound。注意 BF16 matmul 应该用 Tensor Core peak。

</details>

### 练习 2：Roofline 误用

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么只看 achieved TFLOPs 可能误判？</span></summary>

memory-bound kernel 的 achieved TFLOPs 天然低，因为上界由带宽决定。正确做法是同时看 achieved bandwidth、arithmetic intensity、理论 roofline 上界和实际运行时间。reduce 的 TFLOPs 很低可能已经接近带宽上限；GEMM 的 TFLOPs 低才更可能表示 Tensor Core 没吃满。

</details>
