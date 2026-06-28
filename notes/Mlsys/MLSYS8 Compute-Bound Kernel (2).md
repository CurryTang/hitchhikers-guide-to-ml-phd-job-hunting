# MLSYS8 · Compute-Bound Kernel (2)

## 4. 卷积 (Convolution) 优化

### 4.1 核心思想：将卷积映射为矩阵乘法

#### 4.1.1 卷积的数学定义

标准二维卷积（Conv2D）：
$$\text{Output}[n, k, p, q] = \sum_{c=0}^{C-1} \sum_{r=0}^{R-1} \sum_{s=0}^{S-1} \text{Input}[n, c, p \cdot s_h + r - \text{pad}_h, q \cdot s_w + s - \text{pad}_w] \times \text{Weight}[k, c, r, s]$$

| 张量 | 形状 | 含义 |
|------|------|------|
| Input | `[N, C, H, W]` | 批量大小 N，输入通道 C，高 H，宽 W |
| Weight | `[K, C, R, S]` | 输出通道 K，输入通道 C，卷积核高 R，宽 S |
| Output | `[N, K, P, Q]` | 批量大小 N，输出通道 K，输出高 P，输出宽 Q |

朴素实现的伪代码一目了然地揭示了卷积的计算结构：

```python
# 朴素卷积：7 层嵌套循环
for n in range(N):              # 遍历 batch
    for k in range(K):          # 遍历输出通道
        for p in range(P):      # 遍历输出高度
            for q in range(Q):  # 遍历输出宽度
                acc = 0.0
                for c in range(C):      # 归约：输入通道
                    for r in range(R):  # 归约：卷积核高度
                        for s in range(S):  # 归约：卷积核宽度
                            h = p * stride_h + r - pad_h
                            w = q * stride_w + s - pad_w
                            if 0 <= h < H and 0 <= w < W:
                                acc += Input[n, c, h, w] * Weight[k, c, r, s]
                Output[n, k, p, q] = acc
```

**关键观察**：外层 4 层循环 `(n, k, p, q)` 枚举所有输出位置，内层 3 层循环 `(c, r, s)` 做归约求和。这个结构与矩阵乘法 $C_{ij} = \sum_k A_{ik} B_{kj}$ 完全同构——外层枚举输出坐标 `(i, j)`，内层在 `k` 维度归约。

#### 4.1.2 Implicit GEMM：零开销的卷积→矩阵乘法映射

传统 im2col 方法需要显式构建一个 `(N·P·Q) × (C·R·S)` 的展开矩阵，内存开销可达原始输入的数十倍。**Implicit GEMM 的核心突破是：不构建展开矩阵，而是在访问时动态计算地址。**

> [!tip] im2col (Image to Column)
> im2col 将每个卷积窗口"拉直"成矩阵的一行：对输入的每个输出位置 $(p, q)$，提取 $C \times R \times S$ 个元素排成一行，所有位置堆叠成 $(N \cdot P \cdot Q) \times (C \cdot R \cdot S)$ 的矩阵 A；权重 reshape 为 $(K) \times (C \cdot R \cdot S)$ 的矩阵 B。卷积变为标准 GEMM：$\text{Output} = A \times B^T$。
> **优点**：直接复用高度优化的 cuBLAS GEMM，实现简单。
> **缺点**：展开矩阵的冗余极大——相邻窗口共享 $(R-1)(S-1)/(RS)$ 比例的元素（3×3 卷积约 55%），但 im2col 会完整复制，内存膨胀倍数 $= R \times S$（3×3 即 9 倍）。这就是 Implicit GEMM 要解决的问题。

映射规则：

$$\underbrace{(N \times P \times Q)}_{M} \times \underbrace{K_{\text{out}}}_{N_{\text{gemm}}} = \underbrace{(N \times P \times Q)}_{M} \times \underbrace{(C \times R \times S)}_{K_{\text{gemm}}} \cdot \underbrace{(C \times R \times S)}_{K_{\text{gemm}}} \times \underbrace{K_{\text{out}}}_{N_{\text{gemm}}}$$

- **M 维度** = N × P × Q（输出空间所有位置展平）
- **N_gemm 维度** = K（输出通道数）
- **K_gemm 维度** = C × R × S（归约维度）

**坐标解码**是整个实现的核心。GEMM 迭代中的 `(m, k_gemm)` 坐标需要反向解码为张量坐标：

```python
# M 维度解码：展平的输出索引 → (n, p, q)
n = m // (P * Q)
p = (m % (P * Q)) // Q
q = m % Q

# K 维度解码：展平的归约索引 → (c, r, s)
c = k // (R * S)
r = (k % (R * S)) // S
s = k % S

# 从 (n, p, q, c, r, s) 计算输入张量坐标
h = p * stride_h + r - pad_h
w = q * stride_w + s - pad_w
# 若 h, w 越界，则该位置为零填充（通过 mask 实现）
```

这些坐标计算在寄存器中完成，完全省去了 im2col 缓冲区的全局内存开销。

---

### 4.2 Triton 实现（完整版）

> CUTLASS 与 Triton 的核心思想完全一致——implicit GEMM 的坐标映射、tile 分块、归约循环在两套框架里没有本质区别。差异在于：CUTLASS 要达到同等性能，需要手动配置 TMA 描述符、warp 特化流水线、smem swizzle 等大量底层细节；Triton 由编译器自动处理这些。这里先用 Triton 版本把算法讲清楚。

Triton 实现采用完全相同的映射策略，以 Python DSL 表达。编译器自动处理共享内存分配、数据预取、指令调度等底层细节。

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 配置 1: 大 tile，适合大规模卷积（大 batch、大特征图）
        # 128×128 输出 tile + 3 级流水线 + 8 个 warp (256 线程)
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=3, num_warps=8
        ),
        # 配置 2: 中等 tile，寄存器压力更低，允许更深流水线
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=4, num_warps=4
        ),
        # 配置 3: 非对称 tile，适合输出通道数中等但空间维度大的情况
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=3, num_warps=4
        ),
        # 配置 4: 小 tile，适合小 batch 或小特征图
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=5, num_warps=2
        ),
    ],
    key=['M', 'N_gemm', 'K_total'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    # ---- 张量指针 ----
    input_ptr,    # 输入张量 [N, C, H, W] (NCHW 布局)
    weight_ptr,   # 权重张量 [K_out, C, R, S]
    output_ptr,   # 输出张量 [N, K_out, P, Q]
    # ---- 张量维度 ----
    batch, C_in, H, W, K_out, R, S, P, Q,
    # ---- 卷积参数 ----
    pad_h, pad_w, stride_h, stride_w,
    # ---- 隐式 GEMM 维度（显式传入，而非在 kernel 内部推算）----
    #
    # 为什么不直接在 kernel 里算 M = batch*P*Q、K_total = C*R*S？
    #
    # 1. 解耦卷积语义与 GEMM 语义
    #    implicit GEMM 把 (n,p,q) 展平为一维 m，kernel 内部只在 [0,M) 上
    #    切块、做 mask、做边界处理。显式传入 M 等于声明："本次 GEMM 的 M
    #    轴长度就是这个值"，kernel 不必关心它从哪几个卷积维度推导出来。
    #
    # 2. 边界 mask 需要 M，且越早越好
    #    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    #    mask_m = offs_m < M
    #    直接用 M 判断尾块边界，一次比较即可。若改成 offs_m < batch*P*Q，
    #    则每个线程额外做一次乘法链，且把"边界定义"绑死到 batch,P,Q 的
    #    具体关系上，阻碍下面两种常见变体。
    #
    # 3. 允许更一般的 M（split-K / padding / 不同 layout）
    #    - split-M pipeline：本次 kernel 只算 M 轴的某个子范围，M 是
    #      本次参与 GEMM 的局部长度，不等于全局 batch*P*Q。
    #    - 对齐/向量化需要 padding：逻辑上输出是 P×Q，但 GEMM 视角的
    #      有效 M 可能是 padded 后的长度。
    #    - NHWC / blocked / fused epilogue 等不同 layout：展平后的
    #      leading dimension 不再能简单写成 N*P*Q。
    #    显式传参让 host 侧统一控制 M 的含义，kernel 保持通用。
    M,            # = N * P * Q（GEMM 的 M 维度）
    N_gemm,       # = K_out（GEMM 的 N 维度）
    K_total,      # = C * R * S（GEMM 的 K 维度）
    # ---- 张量步长 ----
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_k, stride_wt_c, stride_wt_r, stride_wt_s,
    stride_out_n, stride_out_k, stride_out_p, stride_out_q,
    # ---- 编译期常量 ----
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    隐式 GEMM 卷积核。

    GEMM 映射:
      M 轴 (行): 展平 (n, p, q) → 枚举所有输出空间位置
      N 轴 (列): 枚举输出通道 k
      K 轴 (归约): 展平 (c, r, s) → 枚举输入通道×卷积核空间

    每个 program 实例计算一个 BLOCK_M × BLOCK_N 的输出 tile。
    """

    # ================================================================
    # 1. Program ID → tile 坐标
    # ================================================================
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    # 列优先序：相邻 pid 访问相邻的 M tile → L2 cache 共享输入数据
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # ================================================================
    # 2. M 索引解码 → (n, p, q) 坐标
    # ================================================================
    # 这是 implicit GEMM 的核心：将展平索引反向解码为张量坐标
    n_idx = offs_m // (P * Q)
    residual = offs_m % (P * Q)
    p_idx = residual // Q
    q_idx = residual % Q

    # ================================================================
    # 3. K 维度迭代（核心归约循环）
    # ================================================================
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # K 索引解码 → (c, r, s) 坐标
        c_idx = offs_k // (R * S)
        rs_residual = offs_k % (R * S)
        r_idx = rs_residual // S
        s_idx = rs_residual % S

        # ---- 加载输入 tile A: (BLOCK_M, BLOCK_K) ----
        # 从 (n, p, q) 和 (c, r, s) 计算输入坐标 (n, c, h, w)
        # 注意广播: p_idx 是 [BLOCK_M], r_idx 是 [BLOCK_K]
        h_in = p_idx[:, None] * stride_h + r_idx[None, :] - pad_h
        w_in = q_idx[:, None] * stride_w + s_idx[None, :] - pad_w

        # 边界检查：越界位置对应零填充
        valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
        valid = valid & (offs_m[:, None] < M) & (offs_k[None, :] < K_total)

        a_ptrs = (input_ptr
                  + n_idx[:, None] * stride_in_n
                  + c_idx[None, :] * stride_in_c
                  + h_in * stride_in_h
                  + w_in * stride_in_w)
        a = tl.load(a_ptrs, mask=valid, other=0.0)

        # ---- 加载权重 tile B: (BLOCK_N, BLOCK_K) ----
        b_ptrs = (weight_ptr
                  + offs_n[:, None] * stride_wt_k
                  + c_idx[None, :] * stride_wt_c
                  + r_idx[None, :] * stride_wt_r
                  + s_idx[None, :] * stride_wt_s)
        mask_b = (offs_n[:, None] < K_out) & (offs_k[None, :] < K_total)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # ---- 矩阵乘累加 ----
        # a: (BLOCK_M, BLOCK_K) × b^T: (BLOCK_K, BLOCK_N) → (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, tl.trans(b))

    # ================================================================
    # 4. 写回输出
    # ================================================================
    out_ptrs = (output_ptr
                + n_idx[:, None] * stride_out_n
                + offs_n[None, :] * stride_out_k
                + p_idx[:, None] * stride_out_p
                + q_idx[:, None] * stride_out_q)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < K_out)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_out)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor,
                  padding: tuple = (0, 0), stride: tuple = (1, 1)) -> torch.Tensor:
    """
    Triton 隐式 GEMM 卷积封装。
    input:  [N, C, H, W], FP16, NCHW
    weight: [K, C, R, S], FP16
    返回:   [N, K, P, Q], FP16
    """
    N, C, H, W = input.shape
    K, C_w, R, S = weight.shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    P = (H + 2 * pad_h - R) // stride_h + 1
    Q = (W + 2 * pad_w - S) // stride_w + 1

    output = torch.empty((N, K, P, Q), dtype=torch.float16, device=input.device)
    M = N * P * Q
    N_gemm = K
    K_total = C * R * S

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N_gemm, meta['BLOCK_N']),)

    conv2d_implicit_gemm_kernel[grid](
        input, weight, output,
        N, C, H, W, K, R, S, P, Q,
        pad_h, pad_w, stride_h, stride_w,
        M, N_gemm, K_total,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    return output
```

#### 4.2.1 关键实现细节

**Padding 的零开销实现**：当 `h_in` 或 `w_in` 越界时，通过 `mask` 将加载值设为 0。这在数学上等价于零填充，但不需要分配填充后的张量。

**Autotune 配置的设计逻辑**：
- **BLOCK_M × BLOCK_N** 决定计算/访存比。大 tile（128×128）计算密集，小 tile（32×64）适合小问题。
- **BLOCK_K = 32** 是两次 Tensor Core 操作的量（K=16×2），在流水线效率和寄存器压力间平衡。
- **num_stages** 控制流水线深度。更多 stages 隐藏更长的全局内存延迟（~400-800 cycle），但增加 SMEM 占用。
- **num_warps** 影响线程级并行度。大 tile 需要更多 warps 填充计算流水线。

#### 4.2.2 kred 对齐的本质

`kred` 就是代码里的 `offs_k`——归约轴 `K_total = C * R * S` 上的坐标，对应 `(c, r, s)` 展平后的一维下标。每次 K 循环推进一个 `BLOCK_K`，`offs_k` 就是这一块归约维的坐标向量。

一个初学者容易卡住的问题是：**`A[m, kred]` 和 `B[kred, kout]` 的 `kred` 怎么保证指向同一组 `(c,r,s)`？**

答案分三步讲清楚。

**第一步：把卷积写成点积**

对固定的输出位置 $(n, oh, ow)$ 和输出通道 $kout$，卷积的定义是：

$$\text{out}[n, kout, oh, ow] = \sum_{c,r,s} \text{in}[n, c,\; oh \cdot s_h + r - p_h,\; ow \cdot s_w + s - p_w] \cdot w[kout, c, r, s]$$

右边是对 $(c,r,s)$ 的求和——**这就是两个长度为 $C \cdot R \cdot S$ 的向量的点积**。可以直接记成：

> 每个输出点 = "输入 patch 向量" · "卷积核向量"，二者长度均为 $C \cdot R \cdot S$

**第二步：展平定义 M / N / K**

Implicit GEMM 把两个维度展平：

| GEMM 轴 | 含义 | 展平规则 |
|---------|------|---------|
| M（行） | 输出位置 $(n, oh, ow)$ | $m = n \cdot OH \cdot OW + oh \cdot OW + ow$ |
| N（列） | 输出通道 $kout$ | 直接对应，$N_{gemm} = K_{out}$ |
| K（归约）| patch 下标 $(c, r, s)$ | $kred = c \cdot R \cdot S + r \cdot S + s$ |

于是两个矩阵（A 不真正存在，B 即权重）就是：

$$A[m, kred] = \text{in}[n, c, h(oh,r), w(ow,s)], \quad B[kred, kout] = w[kout, c, r, s]$$

$$\text{out}[m, kout] = \sum_{kred} A[m, kred] \cdot B[kred, kout]$$

**第三步：`offs_k` 同时驱动 A 和 B 的索引**

K 循环里做了一件关键的事：

```python
offs_k = k_start + tl.arange(0, BLOCK_K)

# 用同一套反解，还原出 (c, r, s)
c_idx   = offs_k // (R * S)
r_idx   = offs_k % (R * S) // S
s_idx   = offs_k % S
```

这个反解是 $kred = c \cdot RS + r \cdot S + s$ 的逆运算。之后：

- **A 的加载**用 `c_idx[None,:]`（广播到列）+ `r_idx/s_idx` 来计算 `h_in, w_in`，取 `input[n, c, h, w]`
- **B 的加载**用 `c_idx[None,:]`、`r_idx[None,:]`、`s_idx[None,:]`，取 `weight[kout, c, r, s]`

两者用的是 **同一个 `offs_k`，同一套反解，同一个 `(c,r,s)`**。`tl.dot(a, tl.trans(b))` 的归约轴就是这个 $kred$，因此每一项乘积 $a[i,j] \cdot b[t,j]$ 对应的就是卷积公式里同一个 $(c_j, r_j, s_j)$ 的项：

$$\text{in}[n_i,\, c_j,\, h(oh_i, r_j),\, w(ow_i, s_j)] \cdot w[kout_t,\, c_j,\, r_j,\, s_j]$$

把 K 循环从 0 跑到 `K_total`，就把所有 $(c,r,s)$ 全部累加完毕，得到完整的卷积结果。

> **脑内口诀**：行是输出位置 $(m \leftrightarrow n,oh,ow)$，K 是 patch 向量下标 $(kred \leftrightarrow c,r,s)$。只要两边用同一套展平规则，`kred` 永远对齐。

#### 4.2.3 Triton 心智模型

写完卷积 kernel 可以提炼出 Triton 编程的核心思维方式，它和 CUDA 有一个根本性的差异：

**CUDA 问的是"每个 thread 算哪个元素"；Triton 问的是"这个 program 覆盖的 tile 里，每个元素的坐标是什么"。**

Triton 的 thread 对用户是半透明的——你不手动绑定 `threadIdx`，硬件负责把坐标张量里的标量分发给各个 warp/lane。你只需要对两件事负责：

1. **每个坐标张量的 shape 是什么？**
2. **它的每个元素代表全局张量里的哪个逻辑坐标？**

**核心抽象：坐标张量 → 指针张量**

Triton kernel 本质上是在构造一张"坐标 → 地址"的地图：

```
program_id 选 tile → arange 造坐标张量 → 坐标张量乘以 stride → 指针张量 → load/store
```

每一个 `_ptrs` 变量都是这张地图的具现——它的 shape 和 `acc` 完全一致，一格对一格。`a_ptrs` 是 `[BM, BK]`，`b_ptrs` 是 `[BN, BK]`，`out_ptrs` 是 `[BM, BN]`，和累加器同形。

**通用 Kernel 骨架**

绝大多数 dense 算子（GEMM、卷积、attention）都能套进同一个框架：

```python
# 1. tile 坐标
pid_m, pid_n = ...
offs_m = pid_m * BM + tl.arange(0, BM)   # [BM]
offs_n = pid_n * BN + tl.arange(0, BN)   # [BN]

# 2. 语义解码（纯 GEMM 跳过；conv/attn 在这里把 m → (n,oh,ow) 等）
...

# 3. 归约循环
acc = tl.zeros((BM, BN), tl.float32)
for k_start in range(0, K_total, BK):
    offs_k = k_start + tl.arange(0, BK)
    # 归约坐标解码（conv: kred → (c,r,s)；GEMM: 直接用 offs_k）
    a_ptrs = base_a + ...   # [BM,BK] 指针
    b_ptrs = base_b + ...   # [BK,BN] 指针
    a = tl.load(a_ptrs, mask=..., other=0.)
    b = tl.load(b_ptrs, mask=..., other=0.)
    acc += tl.dot(a, b)

# 4. 写回
tl.store(out_ptrs, acc.to(out_dtype), mask=mask_out)
```

不同算子之间的差异几乎全在**第 2 步的语义解码**和**第 3 步内部的坐标→指针映射**。主框架不变，只插拔解码逻辑。卷积的 `kred → (c,r,s)` 是一个典型例子；attention 的 causal mask 是另一个。

---

---

## 课后练习题

### 练习 1：implicit GEMM

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">implicit GEMM 和 im2col 的区别是什么？</span></summary>

im2col 会显式把卷积输入展开成大矩阵，方便调用 GEMM，但会产生额外内存写入和更大的中间张量。implicit GEMM 不 materialize 展开矩阵，而是在 kernel 内通过索引计算按需读取对应元素，把卷积映射到 GEMM tile。

</details>

### 练习 2：Conv2D 映射到 GEMM

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">M/N/K 分别是什么？</span></summary>

常见映射是 M 对应输出位置 batch 乘输出高宽，N 对应输出通道，K 对应输入通道乘卷积核高宽。不同 layout 会调整具体顺序，但核心是把输出坐标压成矩阵行列，把输入通道和卷积核空间压成 GEMM 的归约维。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">implicit GEMM 里为什么要做坐标反解？</span></summary>

kernel 看到的是 GEMM tile 的行、列和 K 维下标，但卷积真实访问的是 `(n, c, h, w)` 和 `(kout, c, r, s)`。坐标反解就是把展平后的 `m` 和 `kred` 还原成输出位置、输入通道和卷积核位置，从而按需加载对应 patch，而不显式生成 im2col 矩阵。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">为什么 implicit GEMM 省内存，但索引逻辑更复杂？</span></summary>

它省掉了 im2col 中间矩阵的 HBM 写入和存储空间，但每次 load 都要根据 stride、padding、dilation、layout 计算真实地址，并处理边界 mask。优化难点从“调用 GEMM”转成“在 kernel 内高效构造地址和保持访存合并”。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">Triton 写卷积时，最容易错的三类索引是什么？</span></summary>

第一是输出行 `m` 到 `(n, oh, ow)` 的反解；第二是归约维 `kred` 到 `(c, r, s)` 的反解；第三是输入坐标 `ih = oh * stride + r * dilation - pad` 和 `iw` 的边界 mask。任何一个错了，结果可能只在特定 padding/stride 下暴露。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">什么时候不该手写 implicit GEMM 卷积？</span></summary>

如果 shape 常规、库已经有高度优化实现，手写 kernel 很难赢。自定义 kernel 更适合特殊 layout、小 batch 特定 shape、融合后处理、稀疏/动态 mask 或研究型算子。工程上要用 benchmark 证明收益，而不是默认手写更快。

</details>
