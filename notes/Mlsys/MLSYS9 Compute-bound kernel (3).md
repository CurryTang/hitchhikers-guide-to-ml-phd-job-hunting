# MLSYS9 · Compute-Bound Kernel (3)

## 5. Softmax 与归一化操作

Softmax 和归一化操作包含**归约（reduction）**步骤——每个输出元素依赖同一行所有输入元素的全局统计量，这天然地限制了并行度。本章深入讲解 Online Softmax 算法（理解 Flash Attention 的关键前置知识），然后用 Triton 实现 Softmax、LayerNorm、RMSNorm。

---

### 5.1 Online Softmax 算法原理

#### 5.1.1 标准 Softmax（3-pass 算法）

给定输入向量 $\mathbf{x} = [x_1, x_2, \dots, x_N]$：

**第一遍（Pass 1）—— 求最大值：**
$$m = \max_{i=1}^{N} x_i$$
**第二遍（Pass 2）—— 求指数和：**
$$\ell = \sum_{i=1}^{N} \exp(x_i - m)$$
**第三遍（Pass 3）—— 归一化输出：**
$$y_i = \frac{\exp(x_i - m)}{\ell}$$
减去最大值 $m$ 是为了数值稳定性：避免 $\exp(x_i)$ 溢出。

**问题**：三遍各需完整遍历整行数据，总共从 HBM 加载 $3N$ 个元素。GPU 上全局内存带宽是最昂贵的资源。

#### 5.1.2 Online Softmax（1-pass 统计量计算）

Online Softmax 的核心思想：**在单次遍历中同时维护 running max 和 running exp sum，遇到新最大值时通过校正因子修正之前的部分和。**

初始化：$m_0 = -\infty$，$\ell_0 = 0$

处理第 $j$ 个元素 $x_j$ 时：
$$m_j = \max(m_{j-1}, x_j)$$
$$\ell_j = \ell_{j-1} \times \exp(m_{j-1} - m_j) + \exp(x_j - m_j)$$
**正确性推导**：处理完前 $j-1$ 个元素后，$\ell_{j-1} = \sum_{i=1}^{j-1} \exp(x_i - m_{j-1})$。新元素到来后，新最大值 $m_j = \max(m_{j-1}, x_j)$，我们需要：
$$\ell_j = \sum_{i=1}^{j} \exp(x_i - m_j)$$
拆开：
$$= \underbrace{\sum_{i=1}^{j-1} \exp(x_i - m_{j-1})}_{\ell_{j-1}} \cdot \underbrace{\exp(m_{j-1} - m_j)}_{\text{校正因子}} + \exp(x_j - m_j)$$
注意当 $m_j = m_{j-1}$ 时，校正因子 $= 1$，退化为简单累加。

处理完所有元素后仍需第二遍计算 $y_i = \exp(x_i - m_N) / \ell_N$。**总结：3-pass → 2-pass，内存流量 $3N → 2N$。**

#### 5.1.3 Block-wise Online Softmax（Flash Attention 的核心）

GPU 上以 block 为单位处理数据。设当前全局统计量 $(m, \ell)$，处理新 block $B_j$：

**Step 1：block 内局部统计量**
$$m_j^{\text{local}} = \max_{x \in B_j} x, \quad \ell_j^{\text{local}} = \sum_{x \in B_j} \exp(x - m_j^{\text{local}})$$
**Step 2：更新全局统计量**
$$m^{\text{new}} = \max(m, m_j^{\text{local}})$$
$$\ell^{\text{new}} = \ell \cdot \exp(m - m^{\text{new}}) + \ell_j^{\text{local}} \cdot \exp(m_j^{\text{local}} - m^{\text{new}})$$
**Step 3：校正之前的输出**（Flash Attention 中累积的 $O$ 矩阵）
$$O^{\text{new}} = O \cdot \frac{\ell \cdot \exp(m - m^{\text{new}})}{\ell^{\text{new}}} + \frac{\exp(m_j^{\text{local}} - m^{\text{new}})}{\ell^{\text{new}}} \cdot P_j \cdot V_j$$
这个公式就是 Flash Attention 的核心：**不需要物化完整的注意力矩阵**，只需在 SRAM 中逐 block 处理 K/V 并用 online softmax 维护归一化。

---

### 5.2 Triton Grid 模式总览

不同类型的 kernel 对应不同的 grid 划分策略。核心原则：**有几个独立的并行维度，grid 就需要几维**。以下总结常见模式，后续各 kernel 实现可对照参考。

#### 1. 向量运算（elementwise）— 1D grid

$N$ 个元素，每个 program 处理 `BLOCK_SIZE` 个，最简单的情况。

```python
# Triton
grid = (triton.cdiv(N, BLOCK_SIZE),)

pid = tl.program_id(0)
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```
```c
// CUDA 等价
dim3 grid(cdiv(N, BLOCK_SIZE));
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

#### 2. 矩阵运算（GEMM: M×K @ K×N）— 2D grid

每个 program 负责输出矩阵的一个 `BLOCK_M × BLOCK_N` tile。M 和 N 方向的 tile 互相独立，K 方向在内层循环累加。

```python
# 方案 A: 2D grid
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

# 方案 B: 1D 展平（更常见，方便做 swizzle 优化）
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

pid = tl.program_id(0)
grid_n = triton.cdiv(N, BLOCK_N)
pid_m = pid // grid_n     # 输出的行 block
pid_n = pid % grid_n      # 输出的列 block
```

```
输出矩阵 C (M × N):
         N 方向 →
    ┌────┬────┬────┐
M   │0,0 │0,1 │0,2 │
方  ├────┼────┼────┤
向  │1,0 │1,1 │1,2 │
↓   ├────┼────┼────┤
    │2,0 │2,1 │2,2 │
    └────┴────┴────┘
每个格子 = 一个 program, 大小 BLOCK_M × BLOCK_N
内层循环沿 K 维度累加
```

#### 3. Batch GEMM — 加 batch 维度

```python
# 方案 A: 3D grid
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), batch)
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
pid_b = tl.program_id(2)

# 方案 B: 把 M×N tile 压入 axis=0，batch 放 axis=1（更常见）
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), batch)
pid   = tl.program_id(0)
pid_b = tl.program_id(1)
pid_m = pid // grid_n
pid_n = pid % grid_n
```

#### 4. Flash Attention — batch×heads 压扁

```python
grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

# axis 0: Q 的分块索引（内层循环遍历 K/V blocks）
# axis 1: batch 和 head 压扁成一维（它们之间完全独立）
```

为什么 `batch × heads` 压扁？它们之间完全独立，没必要占两个维度；Triton grid 最多 3 维，省一维给可能的扩展。

#### 5. 2D 卷积 — 3D grid

```python
grid = (
    triton.cdiv(out_H, BLOCK_H) * triton.cdiv(out_W, BLOCK_W),  # 空间维度压扁
    out_C,                                                         # 输出通道
    batch,                                                         # batch
)

pid_spatial = tl.program_id(0)
pid_oc      = tl.program_id(1)
pid_b       = tl.program_id(2)
pid_h = pid_spatial // grid_w
pid_w = pid_spatial % grid_w
```

#### 6. Reduction（softmax, layernorm）— 1D grid

每个 program 处理一整行的归约，行间独立并行。

```python
# softmax: 输入 (M, N), 每行独立归一化
grid = (M,)

pid = tl.program_id(0)    # 第几行
offs = tl.arange(0, BLOCK_N)
x = tl.load(X_ptr + pid * stride + offs, mask=offs < N)
# ... 整行的 max → exp → sum → normalize
```

#### 7. Batch LayerNorm — 2D grid

```python
# 输入 (batch, seq_len, hidden_dim), 每个 (batch, position) 独立归一化
grid = (seq_len, batch)

pid_s = tl.program_id(0)  # 序列位置
pid_b = tl.program_id(1)  # batch
# 在 hidden_dim 维度上做 reduce
```

**总结规律**：独立的输出 tile → grid 维度；归约/累加的维度 → 内层循环。当独立维度超过 3 个时，将完全独立的维度压扁（如 `batch × heads`）以适应 Triton 的 3 维 grid 限制。

#### Grid 维度设计方法论

Grid 的维度数和问题本身的维度数是**解耦**的——不是"问题几维 grid 就几维"，而是取决于哪些维度需要归约。

**Step 1：列出问题的所有维度，标记并行/归约**

| 维度角色 | 去哪里 | 原因 |
|---|---|---|
| **并行维**：各 block 之间不需要通信 | → Grid 维度 | 可以独立并行 |
| **归约维**：需要累加/比较/聚合 | → 内层循环 | 必须串行或协作完成 |

以几个核心 kernel 为例：

```
GEMM: C[M,N] = A[M,K] @ B[K,N]
  M → 并行（每行独立）
  N → 并行（每列独立）
  K → 归约（要累加）

Flash Attention: O[B,H,M,d] = softmax(Q @ K^T) @ V
  B → 并行
  H → 并行
  M → 并行（每个 query 位置独立）
  N → 归约（要遍历所有 key 做 softmax + 累加）
  d → 一个 block 内完整处理，不切分

LayerNorm: y[B,S,D] = norm(x[B,S,D], dim=-1)
  B → 并行
  S → 并行
  D → 归约（要算 mean/var）

2D Conv: out[B,OC,OH,OW]
  B  → 并行
  OC → 并行
  OH → 并行
  OW → 并行
  IC, KH, KW → 归约
```

**Step 2：把并行维映射到 grid（最多 3 维，多了就压扁）**

```
并行维数量     做法
─────────────────────────────────
  1 个        1D grid，直接映射
  2 个        2D grid，直接映射
  3 个        3D grid，直接映射
  ≥4 个       压扁：把某些维合并成一维
```

压扁策略——优先压扁语义相关或大小较小的维度：

```python
# Flash Attention: 4个并行维(B,H,M,d) 但 d 不切分，实际3个
# B 和 H 语义接近（都是"哪一组"），压扁
grid = (cdiv(M, BLOCK_M), B * H)            # 2D

# 2D Conv: 4个并行维(B,OC,OH,OW)
# OH 和 OW 是空间维度，压扁
grid = (cdiv(OH,BH) * cdiv(OW,BW), OC, B)   # 3D

# Batch GEMM: 3个并行维(B,M,N)
# M 和 N 压扁（方便做 swizzle 优化 L2 cache）
grid = (cdiv(M,BM) * cdiv(N,BN), B)          # 2D
```

**Step 3：归约维放进内层循环**

```python
# GEMM: K 是归约维
for k in range(0, K, BLOCK_K):
    a = load(A_block)   # 流式加载
    b = load(B_block)
    acc += dot(a, b)     # 累加

# Flash Attention: N(key 序列)是归约维
for start_n in range(0, N_CTX, BLOCK_N):
    k = load(K_block)   # 流式加载
    v = load(V_block)
    # online softmax + 累加
```

#### 实际工程考量

除了并行/归约的基本划分，还有三个因素会影响最终决策：

**1. SRAM 容量限制 → 决定哪些维度必须切分**

```
SRAM 大小有限（A100 ~192KB/SM），需要放得下：
  Q block:  BLOCK_M × d     = 128 × 64 × 2B = 16KB
  K block:  BLOCK_N × d     = 64 × 64 × 2B  = 8KB
  V block:  BLOCK_N × d     = 64 × 64 × 2B  = 8KB
  acc:      BLOCK_M × d     = 128 × 64 × 4B = 32KB
  ──────────────────────────────────────────────
  总共 ~64KB ✓ 放得下

如果 d=256，可能放不下 → 需要把 d 也变成归约维，加循环
```

**2. SM 利用率 → grid 总大小要足够**

A100 有 108 个 SM，grid 总 program 数要远大于 108 才能充分利用。

```
Flash Attention 例子:
  grid = (4, 16) = 64 个 program
  只用了 64/108 ≈ 59% 的 SM → 浪费

如果 seq_len 很短，可以把 BLOCK_M 调小让 axis 0 更多
```

**3. L2 Cache 友好性 → 影响压扁和遍历顺序**

```python
# GEMM 经典优化：swizzle 遍历顺序
# 不是简单的 row-major，而是按 "grouped" 顺序
# 让相邻 program 访问相邻的 K/V 数据，提高 L2 命中率

pid = tl.program_id(0)
# 不用 pid_m = pid // grid_n; pid_n = pid % grid_n
# 而是 swizzle:
GROUP_SIZE = 8
group_id = pid // (GROUP_SIZE * grid_n)
group_m  = group_id * GROUP_SIZE + (pid % GROUP_SIZE)
pid_n    = (pid % (GROUP_SIZE * grid_n)) // GROUP_SIZE
```

---

### 5.3 Triton Softmax

当一行数据能完全装入一个 `BLOCK_SIZE` 时（实践中最常见），可以做到**真正的单次遍历**：1 次 load + 1 次 store，理论最优内存流量 $2N$。

**为什么整行装得进才能 1-pass，装不进就必须 2-pass？** 关键在于 SRAM 能否同时持有原始数据和最终统计量。装得进时，整行 $x$ 一次性加载到 SRAM，在寄存器中依次完成 max → exp → sum → 归一化，计算 $y_i = \exp(x_i - m)/\ell$ 时 $x_i$、$m$、$\ell$ 三者同时可用。装不进时，SRAM 一次只能放一个 block，遍历所有 block 算出最终 $m$ 和 $\ell$ 时，之前 block 的原始数据 $x$ 已被后续 block 覆盖。而归一化公式 $y_i = \exp(x_i - m_{\text{final}})/\ell_{\text{final}}$ 既需要原始 $x_i$，又需要全局统计量——两者无法同时在 SRAM 中共存，只能重新从 HBM 加载 $x$，被迫多一遍扫描。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    单 block Softmax：整行数据一次性加载到 SRAM，在寄存器中完成全部计算。

    内存流量分析：
      本实现: N (load) + N (store) = 2N（理论最优）
      标准 3-pass: 3N (load) + N (store) = 4N
      Online 2-pass: 2N (load) + N (store) = 3N
    """
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    # 一次性加载整行到寄存器/SRAM
    # other=-inf 的选择：
    #   不影响 tl.max: max(x_valid, -inf) = x_valid
    #   不影响 tl.sum: exp(-inf) = 0
    #   保证输出为 0: exp(-inf - m) / l = 0
    x = tl.load(
        input_ptr + row_idx * input_row_stride + col_offs,
        mask=mask, other=float('-inf')
    )

    # Softmax 三步在寄存器中完成
    # tl.max 底层使用 warp shuffle 归约 (__shfl_xor_sync)
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)        # 数值稳定：减去 max
    x_sum = tl.sum(x_exp, axis=0)     # warp shuffle 归约求和
    result = x_exp / x_sum

    # 一次性写回
    tl.store(
        output_ptr + row_idx * output_row_stride + col_offs,
        result, mask=mask
    )


@triton.jit
def softmax_online_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online Softmax：当 n_cols > BLOCK_SIZE 时使用。
    第一遍用 online 算法计算 (m, l)，第二遍输出。总计 2-pass。
    """
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride

    # ========== 第一遍：Online 统计量 ==========
    m_i = float('-inf')
    l_i = 0.0

    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(row_start + col_offs, mask=col_offs < n_cols, other=float('-inf'))

        m_ij = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        # 核心：online softmax 更新
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    # ========== 第二遍：归一化输出 ==========
    out_start = output_ptr + row_idx * output_row_stride
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        x = tl.load(row_start + col_offs, mask=mask, other=float('-inf'))
        result = tl.exp(x - m_i) / l_i
        tl.store(out_start + col_offs, result, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """统一入口：根据行长度自动选择最优版本。"""
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(min(N, 65536))
    grid = (M,)

    if N <= 65536:
        softmax_kernel[grid](x, output, N, x.stride(0), output.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    else:
        BLOCK_SIZE = max(128, triton.next_power_of_2(min(N, 4096)))
        softmax_online_kernel[grid](x, output, N, x.stride(0), output.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return output
```

**为什么 Softmax kernel 不像 GEMM 那样做 2D tiling？** GEMM 的输出是 M×N 矩阵，各 output tile 互相独立，因此需要 `pid → (pid_m, pid_n) → (offs_m, offs_n)` 把一维 program ID 映射到二维 tile 坐标。而 Softmax 的每一行内部有归约依赖（max、sum 必须看完整行），**列方向不能拆成独立 tile**——一行只能由一个 program 负责。因此 grid 天然是 1D 的 `(M,)`（M = 行数），`program_id(0)` 直接就是行号，不需要 tile 坐标拆分。并行度仅存在于行与行之间。

**`tl.max` 和 `tl.sum` 的实现原理**：当 `BLOCK_SIZE = 1024` 时，一个 block 有 32 个 warp。`tl.max(x, axis=0)` 先在每个线程内对其元素求 max，再通过 `__shfl_xor_sync` 在 warp 内归约，最后通过 shared memory 在 warp 间归约。Triton 编译器自动生成全部代码。

**精度策略**：`tl.exp` 在 PTX 层面编译为 `ex2.approx.ftz.f32`（以 2 为底的近似指数），比精确 `expf` 快约 2 倍。对 softmax 的近似误差可忽略。

---

### 5.4 Triton LayerNorm
$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
```python
@triton.jit
def layer_norm_kernel(
    input_ptr, output_ptr,
    gamma_ptr, beta_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm：每个 program 处理一行。

    精度关键点：
    - mean/var 必须在 FP32 下计算。FP16 精度 2^{-10} ≈ 0.001，
      累加 4096 次后误差可达 4.096（灾难性取消）。
      FP32 精度 2^{-23}，累加 4096 次后误差仅 ~5×10^{-4}。
    - 使用两步法（先减 mean 再算 var），而非 E[x²]-E[x]²（数值不稳定）。
    """
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    # 加载并转为 FP32
    x = tl.load(input_ptr + row_idx * input_row_stride + col_offs,
                mask=mask, other=0.0).to(tl.float32)

    # 均值和方差
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # 归一化 + 仿射变换
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * inv_std
    gamma = tl.load(gamma_ptr + col_offs, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
    result = x_norm * gamma + beta

    tl.store(output_ptr + row_idx * output_row_stride + col_offs,
             result.to(tl.float16), mask=mask)
```

**`col_offs` 的作用与 Triton 线程分配机制**：`col_offs = tl.arange(0, BLOCK_SIZE)` 生成列索引向量 $[0, 1, \dots, \text{BLOCK\_SIZE}-1]$。一个 Triton program 对应一个 CUDA thread block（CTA），假设 `BLOCK_SIZE=1024`、`num_warps=4`，则 CTA 有 $4 \times 32 = 128$ 个线程，每个线程负责 $1024/128=8$ 个元素。`x`、`gamma`、`beta` 都用同一个 `col_offs` 索引加载，因此**同一线程持有的 `x[i]`、`gamma[i]`、`beta[i]` 天然对应同一列**。逐元素操作（`x_norm * gamma + beta`、`x - mean`、`exp`）各线程独立计算自己的元素，无需线程间通信。只有归约操作（`tl.sum`）需要协作：线程内先局部求和，再通过 warp shuffle 在 warp 内归约，最后通过 shared memory 在 warp 间归约，归约结果广播回所有线程。因此 `mean` 和 `var` 计算完后每个线程拿到相同的标量值，后续 `x - mean` 又回到各线程独立的逐元素操作。

**如果 `n_cols` 超过单个 BLOCK_SIZE 怎么办？** 当前实现假设 `BLOCK_SIZE >= n_cols`（整行一次加载），这对实践中常见的 hidden_dim（4096~8192，FP16 仅 8~16 KB）完全够用。若 `n_cols` 极大，需要分 block 循环处理，但**与 softmax 不同，LayerNorm 的累加不需要校正因子**。原因在于：softmax 的 $\ell = \sum \exp(x_i - m)$ 依赖全局 max $m$，新 block 可能刷新 max 导致之前的 exp-sum 全部失效，必须乘 $\exp(m_{\text{old}} - m_{\text{new}})$ 校正；而 LayerNorm 的 $\sum x_i$ 是简单加法，新 block 来了直接累加，之前的部分和不会因为新数据而"变错"。不过仍然需要多遍扫描——mean 和 var 之间有顺序依赖（var 需要先知道 mean），朴素实现需要 3 遍（第 1 遍求 mean，第 2 遍求 var，第 3 遍输出）。可优化为 2 遍：第 1 遍同时累加 $\sum x_i$ 和 $\sum x_i^2$，利用 $\sigma^2 = E[x^2] - (E[x])^2$ 一遍算出 mean 和 var（代价是数值稳定性略差），第 2 遍归一化输出。无论哪种方案，多遍扫描的根本原因与 softmax 一致：**全局统计量和原始数据无法同时驻留 SRAM。**

**LayerNorm 是纯 memory-bound 操作**：算术强度极低，性能完全由 HBM 带宽决定。单次遍历加载 $N$（input）+ $N$（gamma）+ $N$（beta），写出 $N$（output），总共 $4N$ 次内存操作。优化方向是减少遍历次数和最大化内存带宽利用率。

---

### 5.5 Triton RMSNorm
$$y = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2 + \epsilon}$$
相比 LayerNorm，RMSNorm **去掉了均值中心化和偏置**：少 1 次 sum 归约 + 少 1 次逐元素减法 + 少加载 $N$ 个 beta 参数，综合节省约 30% 计算量和 25% 内存流量。

```python
@triton.jit
def rms_norm_kernel(
    input_ptr, output_ptr,
    gamma_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    x = tl.load(input_ptr + row_idx * input_row_stride + col_offs,
                mask=mask, other=0.0).to(tl.float32)

    # RMS = sqrt(mean(x²) + eps)
    x_sq = tl.where(mask, x * x, 0.0)
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rms_inv = 1.0 / tl.sqrt(mean_sq + eps)

    gamma = tl.load(gamma_ptr + col_offs, mask=mask, other=1.0).to(tl.float32)
    result = x * rms_inv * gamma

    tl.store(output_ptr + row_idx * output_row_stride + col_offs,
             result.to(tl.float16), mask=mask)
```

现代大模型（LLaMA、Mistral 等）几乎全部使用 RMSNorm。研究表明均值中心化对模型性能贡献微乎其微，但占据了 LayerNorm 约 1/3 的计算开销。

---

## 6. Flash Attention

### 6.1 动机

标准自注意力 $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) \times V$ 需要物化 $N \times N$ 的注意力矩阵（$Q, K, V \in \mathbb{R}^{N \times d}$），内存 $O(N^2)$，带宽开销巨大。

Flash Attention 的核心思路：
1. **永远不物化 $N \times N$ 矩阵**：通过 tiling 在 SRAM 中逐块计算
2. **用 block-wise online softmax 增量式处理**（第 5 章已详述原理）
3. **中间结果保持在 SRAM**：GPU SRAM 带宽约 19 TB/s vs HBM 2 TB/s
4. **内存 $O(N^2) \to O(N)$**：只存输出 $O$ 和少量 softmax 统计量 $(m, \ell)$

---

### 6.2 Flash Attention 2 算法

**输入**：Q, K, V $\in \mathbb{R}^{N \times d}$，**输出**：O = softmax(QK^T / √d) × V

**分块**：Q 分为 $T_r$ 块（每块 $B_r$ 行），K/V 分为 $T_c$ 块（每块 $B_c$ 行）。

```
输入: Q_i ∈ R^{B_r × d}
初始化: O_i = 0, m_i = -∞, l_i = 0

对每个 K,V 块 j = 1, ..., T_c:
    S_ij = Q_i × K_j^T / √d              // 在 SRAM 中计算
    m_ij = rowmax(S_ij)                   // 当前块的行最大值
    m_new = max(m_i, m_ij)               // 更新全局最大值

    // 校正之前的结果（block-wise online softmax，见 5.1.3）
    l_i = l_i × exp(m_i - m_new)
    O_i = O_i × exp(m_i - m_new)

    P_ij = exp(S_ij - m_new)             // softmax 分子
    l_i = l_i + rowsum(P_ij)
    O_i = O_i + P_ij × V_j              // GEMM-II

    m_i = m_new

O_i = O_i / l_i                          // 最终归一化
```

**正确性**：修正因子 $\exp(m_{\text{old}} - m_{\text{new}})$ 将之前相对于 $m_{\text{old}}$ 计算的指数值转换为相对于 $m_{\text{new}}$ 的值：$\exp(x - m_{\text{old}}) \times \exp(m_{\text{old}} - m_{\text{new}}) = \exp(x - m_{\text{new}})$。展开后与标准公式完全一致。

**内存**：HBM 只需存储 Q, K, V（$3Nd$）、O（$Nd$）、统计量 m, l（$2N$）。SRAM 中临时存储 $O(B_r d + B_c d + B_r B_c)$，不随 $N$ 增长。

### 6.3 Triton Flash Attention 实现

**Stride 与地址计算**：Q 的 shape 是 `(batch, heads, seq_len, d_model)`，`stride` 是沿每个维度移动一个元素需要跳过的元素数。对行优先连续张量（`batch=2, heads=8, seq_len=512, d_model=64`）：

| stride | 对应维度 | 值 | 含义 |
|---|---|---|---|
| `stride_qb` | batch | $8 \times 512 \times 64 = 262144$ | 跳到下一个 batch |
| `stride_qh` | heads | $512 \times 64 = 32768$ | 跳到下一个 head |
| `stride_qm` | seq_len | $64$ | 跳到下一行（token） |
| `stride_qk` | d_model | $1$ | 跳到下一列（最内层连续） |

访问 `Q[b, h, m, k]` 的地址：`Q_ptr + b*stride_qb + h*stride_qh + m*stride_qm + k*stride_qk`。

代码中 `off_hz = tl.program_id(1)` 是 `batch × heads` 压扁后的索引，基地址用 `off_hz * stride_qh` 而非拆开算。这成立是因为连续 layout 下 `stride_qb = heads × stride_qh`，所以 `b * stride_qb + h * stride_qh = (b * heads + h) * stride_qh = off_hz * stride_qh`，一次乘法搞定。若 tensor 不连续（如做过 `transpose`），则需拆成 `off_hz // heads * stride_qb + off_hz % heads * stride_qh`。

**Q block 的指针矩阵构造**：`q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk` 构造了一个 `[BLOCK_M, BLOCK_DMODEL]` 的指针矩阵，本质是把多维索引 `Q[b,h,m,d]` 手动展开为一维地址。分三部分：

```
Q_ptr + q_offset                → 定位到 (batch, head) 块的起始地址
+ offs_m[:, None] * stride_qm   → 行偏移，[BLOCK_M, 1] 列向量
+ offs_d[None, :] * stride_qk   → 列偏移，[1, BLOCK_DMODEL] 行向量
```

两个偏移通过广播相加得到完整的 2D 指针矩阵（假设 `start_m=1`, `stride_qm=64`, `stride_qk=1`）：

```
行偏移 (×64)          列偏移 (×1)              结果（相对偏移）
┌──────┐              ┌──────────────┐         ┌─────────────────────┐
│ 8192 │              │ 0  1  ... 63 │         │ 8192  8193  ... 8255│ ← Q[128, 0:64]
│ 8256 │       +      │ 0  1  ... 63 │    =    │ 8256  8257  ... 8319│ ← Q[129, 0:64]
│  ... │              │     ...      │         │         ...         │
│16320 │              │ 0  1  ... 63 │         │16320 16321  ...16383│ ← Q[255, 0:64]
└──────┘              └──────────────┘         └─────────────────────┘
[128,1] 广播           [1,64] 广播              [128,64] 指针矩阵
```

每个格子是一个内存地址，`tl.load(q_ptrs)` 一次将整个 `[BLOCK_M, BLOCK_DMODEL]` 的 Q block 加载到 SRAM。Triton 没有多维数组原生支持，所以必须用指针算术 + 广播来模拟 2D 块加载。

```python
import triton
import triton.language as tl
import torch


@triton.jit
def flash_attention_forward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX, sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Flash Attention 2 Forward — Triton 实现。
    每个 program 处理一个 Q block (BLOCK_M 行) 在一个 (batch, head) 上。
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # 基地址偏移
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 加载 Q block（整个计算过程中驻留 SRAM）
    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 初始化 online softmax 状态
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 主循环：遍历 K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # 加载 K block
        k_ptrs = K_ptr + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # S = Q @ K^T * scale
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))
        s *= sm_scale
        s = tl.where(offs_m[:, None] < N_CTX, s, float('-inf'))
        s = tl.where((start_n + offs_n)[None, :] < N_CTX, s, float('-inf'))

        # Online softmax 更新（核心：block-wise 版本，见 5.1.3）
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)          # 校正因子
        p = tl.exp(s - m_new[:, None])       # softmax 分子
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]            # 校正之前的输出

        # 加载 V block 并累加
        v_ptrs = V_ptr + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_new

    # 最终归一化
    acc = acc / l_i[:, None]

    # 写回
    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


def flash_attention_triton(q, k, v):
    """
    Flash Attention forward. q, k, v: (batch, heads, seq_len, d_model) fp16.
    """
    BLOCK_M, BLOCK_N = 128, 64
    batch, heads, seq_len, d_model = q.shape
    o = torch.empty_like(q)
    sm_scale = 1.0 / (d_model ** 0.5)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    flash_attention_forward_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=d_model,
    )
    return o
```



### 6.4 反向传播

Flash Attention 的反向传播比前向更复杂，因为需要重新计算注意力矩阵（前向只存了 $O$, $m$, $\ell$，不存 $P$）。

#### 6.4.1 反向传播的数学

设 $P = \text{softmax}(S/\sqrt{d})$，$S = QK^T$，$O = PV$。给定上游梯度 $dO$：

**Step 1**：计算辅助量 $D_i = \sum_j dO_{ij} \cdot O_{ij}$（逐行点积）

**Step 2**：对每对 (Q block $i$, KV block $j$)：
- 重新计算 $S_{ij} = Q_i K_j^T / \sqrt{d}$
- 重新计算 $P_{ij} = \exp(S_{ij} - m_i) / \ell_i$（使用前向保存的 $m$, $\ell$）
- $dV_j \mathrel{+}= P_{ij}^T \cdot dO_i$
- $dP_{ij} = dO_i \cdot V_j^T$
- $dS_{ij} = P_{ij} \odot (dP_{ij} - D_i)$（逐元素，softmax 反传公式）
- $dQ_i \mathrel{+}= dS_{ij} \cdot K_j / \sqrt{d}$
- $dK_j \mathrel{+}= dS_{ij}^T \cdot Q_i / \sqrt{d}$

#### 6.4.2 Triton 反向传播（近似版本）

以下是反向传播的概念性 Triton 实现。实际生产代码通常分为两个 kernel（分别计算 dQ 和 dK/dV），此处合并展示核心逻辑。**注意：此版本为教学目的的简化版，可能存在边界处理和性能上的不足。**

```python
@triton.jit
def flash_attention_backward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    L_ptr, M_ptr,             # 前向保存的 log-sum-exp (l) 和 max (m)
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX, sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Flash Attention 反向传播 — 简化版。
    每个 program 处理一个 Q block，遍历所有 K/V block 计算 dQ。
    dK 和 dV 通过 atomic_add 累加（生产代码应使用专门的 kernel 避免 atomic）。
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 加载 Q block, O block, dO block
    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o = tl.load(o_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    do_ptrs = dO_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 加载前向保存的 softmax 统计量
    m_ptrs = M_ptr + off_hz * N_CTX + offs_m
    l_ptrs = L_ptr + off_hz * N_CTX + offs_m
    m_i = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
    l_i = tl.load(l_ptrs, mask=offs_m < N_CTX, other=1.0)

    # Step 1: D_i = rowsum(dO * O)
    D_i = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)  # (BLOCK_M,)

    # 初始化 dQ 累加器
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)

    # Step 2: 遍历 K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # 加载 K, V blocks
        k_ptrs = K_ptr + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V_ptr + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # 重新计算 S 和 P
        s = tl.dot(q, tl.trans(k)) * sm_scale                    # (BLOCK_M, BLOCK_N)
        p = tl.exp(s - m_i[:, None]) / l_i[:, None]              # 重新计算 softmax
        p = tl.where((start_n + offs_n)[None, :] < N_CTX, p, 0.0)

        # dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))                              # (BLOCK_M, BLOCK_N)

        # dS = P * (dP - D_i)  — softmax 反传公式
        ds = p * (dp - D_i[:, None]) * sm_scale                  # (BLOCK_M, BLOCK_N)

        # dQ += dS @ K
        dq += tl.dot(ds.to(tl.float16), k)

        # dV += P^T @ dO (简化: atomic_add，生产代码应避免)
        # dK += dS^T @ Q (简化: atomic_add)
        # 这里省略 dV/dK 的原子累加，实际实现需要专门处理

    # 写回 dQ
    dq_ptrs = dQ_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    tl.store(dq_ptrs, dq.to(tl.float16), mask=offs_m[:, None] < N_CTX)
```

**上述简化版的已知问题**：
1. **前向未保存统计量**：反向期望从 `M_ptr`/`L_ptr` 加载 $m$ 和 $\ell$，但前向 kernel 没有将它们写回 HBM。完整实现需要在前向最终归一化前增加 `tl.store` 写出 `m_i` 和 `l_i`。
2. **dK/dV 完全缺失**：代码只计算了 dQ，dK 和 dV 的累加在注释中提及但未实现。
3. **精度损失**：`ds.to(tl.float16)` 在 `tl.dot` 前将梯度截断为 FP16。反向传播中梯度值可能很小，FP16 截断会导致明显的数值误差。应保持 FP32 或使用 BF16。
4. **缺少编译器提示**：前向有 `tl.multiple_of(start_n, BLOCK_N)` 帮助编译器优化，反向遗漏了。

**反向传播的实现难点**：
1. **dK/dV 的累加**：多个 Q block 需要对同一个 K/V block 的梯度求和，生产代码通常使用专门的 kernel（外层循环遍历 K/V block，内层遍历 Q block）以避免 atomic 操作。
2. **重计算的代价**：反向需要重新计算 $S$ 和 $P$，计算量约为前向的 2.5 倍。但相比存储 $N \times N$ 的 $P$ 矩阵，重计算更划算。
3. **数值精度**：反向中 $\exp(S_{ij} - m_i) / \ell_i$ 的重计算必须与前向一致，包括相同的 max 和 sum 值。

---

### 6.5 Flash Attention 3：Hopper 架构优化

FA2 在 A100 上能达到 ~70% 利用率，但在 H100 上仅 ~35%。原因是 Hopper 引入了全新的硬件能力（异步 Tensor Core、TMA、FP8），FA2 的同步设计无法利用。FA3 是针对 Hopper 的完全重写。

#### 6.5.1 三大核心改进

**1. Warp 特化（Warp Specialization）**

FA2 中所有 warp 同质化工作——既做数据加载又做计算。FA3 将 warp 分为两类角色：

```
┌─────────────────── CTA (Thread Block) ───────────────────┐
│                                                           │
│  Producer Warps              Consumer Warps               │
│  ┌──────────────┐           ┌──────────────────────────┐  │
│  │ TMA load K_j │──SMEM──→ │ WGMMA: S = Q @ K^T       │  │
│  │ TMA load V_j │──SMEM──→ │ Softmax: P = softmax(S)  │  │
│  │ (环形缓冲)    │           │ WGMMA: O += P @ V        │  │
│  └──────────────┘           └──────────────────────────┘  │
│  寄存器少 ← setmaxnreg → 寄存器多（GEMM 累加器）           │
└───────────────────────────────────────────────────────────┘
```

- **Producer**：专门用 TMA（Tensor Memory Accelerator）从 HBM 异步加载数据到 shared memory 的环形缓冲区
- **Consumer**：专门用 WGMMA（warpgroup 级矩阵乘）做计算
- **`setmaxnreg`**：Hopper 新指令，动态重分配寄存器——producer 释放寄存器给 consumer，让 GEMM 累加器有更多空间

**2. 两级流水线：GEMM-Softmax 交叠**

FA2 的瓶颈：两个 GEMM（$S = QK^T$ 和 $O = PV$）之间隔着 softmax，严格串行。FA3 利用 WGMMA 的异步语义打破这个依赖：

```
时间 →
──────────────────────────────────────────────────
迭代 j:   [GEMM0: S_j=QK_j^T]  [softmax(S_j)]  [GEMM1: O+=P_j·V_j]
迭代 j+1:                       [GEMM0: S_{j+1}] [softmax(S_{j+1})]  [GEMM1]

↓ 交叠后：

迭代 j:   [GEMM0_j] [softmax_j + GEMM0_{j+1}] [GEMM1_j]
                     ↑ softmax 用标量单元(MUFU)
                     ↑ GEMM0 用 Tensor Core
                     → 两者可以同时执行！
```

核心：发射 `GEMM0_{j+1}` 后不等待完成（`commit_group` 但不 `wait_group`），立刻用 MUFU 做 `softmax_j`。两者使用不同的硬件单元，真正并行。

**3. Pingpong 调度（双 Warpgroup 交替）**

即使有了两级流水线，softmax 的 MUFU 吞吐量（~3.9 TFLOPS）比 GEMM 的 Tensor Core（~989 TFLOPS FP16）低 ~256 倍，仍可能占据可观的时钟周期。Pingpong 用两个 warpgroup 互相掩盖：

```
时间 →
WarpGroup 0: [GEMM] [softmax] [GEMM] [softmax] ...
WarpGroup 1:        [GEMM] [softmax] [GEMM] [softmax] ...
                     ↑ WG1 做 GEMM 时 WG0 做 softmax
                            → softmax 完全被隐藏
```

两个 warpgroup 通过 `bar.sync` 硬件屏障交替执行，softmax 延迟完全藏在对方的 GEMM 执行时间内。

#### 6.5.2 FP8 支持：Incoherent Processing

FP8 的问题：transformer 中 Q/K 常有少量"outlier"维度值特别大，FP8 的动态范围无法同时表示大值和小值。

**解决方案**：乘以随机正交矩阵 $M$（Hadamard + 随机符号翻转）"打散"outlier：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(QM)(KM)^T}{\sqrt{d}}\right) V$$
因为 $MM^T = I$，数学结果不变：$(QM)(KM)^T = QMM^TK^T = QK^T$。但乘以 $M$ 后各维度幅值趋于均匀，FP8 量化误差降低 2.6 倍。$M$ 的应用通过 Fast Walsh-Hadamard Transform 实现，复杂度 $O(d \log d)$，可与 RoPE 融合零额外开销。

#### 6.5.3 Triton 实现的可行性与局限

| FA3 技术 | Triton 可行性 | 说明 |
|---|---|---|
| Warp 特化 | **不可行** | Triton 抽象层隐藏了 warp 级控制，无法指定 producer/consumer 角色 |
| `setmaxnreg` | **不可行** | Triton 不暴露寄存器分配指令 |
| TMA | **部分可行** | Triton 3.0+ 开始支持 `tl.async_copy` 等 TMA 原语，但不如 CUDA 灵活 |
| WGMMA 异步语义 | **不可行** | Triton 的 `tl.dot` 是同步的，无法做 commit-without-wait 式流水线 |
| Pingpong 调度 | **不可行** | 需要 warpgroup 级屏障控制，Triton 无此抽象 |
| 两级 GEMM-softmax 流水线 | **有限** | Triton 编译器可能自动做一些指令级重排，但无法显式控制 |
| FP8 + block 量化 | **可行** | Triton 支持 FP8 类型和 `tl.dot` 的 FP8 操作数 |
| Incoherent processing | **可行** | Hadamard 变换是逐元素 + butterfly 操作，可用 Triton 实现 |
| 基本的 online softmax tiling | **可行** | 这是 FA2 的核心，Triton 已能良好支持（如前文实现） |

**结论**：FA3 的核心优势（warp 特化、异步流水线、pingpong）深度依赖 Hopper 的底层硬件原语，必须用 CUDA/CUTLASS 实现。Triton 能做的是 FA2 级别的 tiling + online softmax + FP8 量化，大约对应 FA3 性能的 50-60%。这也解释了为什么 FA3 是用 CUDA（基于 CUTLASS 3.x）而非 Triton 编写的。

---

## 7. 激活函数融合优化

### 7.1 融合的动机

以全连接层 `D = GELU(A × B + bias)` 为例：

```
未融合（3 个独立 kernel）：
  HBM → A, B → GEMM → C 写回 HBM     (2 读 + 1 写)
  HBM → C, bias → 加法 → C' 写回 HBM  (2 读 + 1 写)
  HBM → C' → GELU → D 写回 HBM         (1 读 + 1 写)
  总共: 5 次 HBM 读 + 3 次 HBM 写

融合（1 个 kernel）：
  HBM → A, B, bias → 寄存器中: GELU(A×B + bias) → D 写回 HBM
  总共: 3 次 HBM 读 + 1 次 HBM 写（省去 2 次中间结果的 HBM 往返）
```

中间结果 C（如 4096×4096 FP16 = 32 MB），一次写入+读取耗时约 32 μs（A100, 2 TB/s）——完全浪费的带宽。融合后 GEMM 结果直接在寄存器中经历 Bias 和 GELU，零额外 HBM 开销。

---

### 7.2 CUTLASS Epilogue 融合

CUTLASS 的 GEMM kernel 分三阶段：Prologue（加载 A, B）→ Mainloop（分块矩阵乘法）→ **Epilogue（后处理 + 写回）**。

Epilogue 阶段 GEMM 结果仍在寄存器中（fragment 形式），任何逐元素操作都可以零开销融合。CUTLASS 通过 epilogue functor 模板实现：

```cpp
// D = GELU(alpha * A×B + beta * C)
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::GELU,     // 激活函数
    cutlass::half_t,                     // 输出类型
    8,                                   // 每次向量化写回的元素数
    float,                               // 累加器类型 (FP32)
    float                                // 计算类型
>;

// 将 EpilogueOp 嵌入完整 GEMM 类型定义
using FusedGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,      // A
    cutlass::half_t, cutlass::layout::ColumnMajor,    // B
    cutlass::half_t, cutlass::layout::RowMajor,       // C/D
    float,                                            // 累加器
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,           // CTA tile
    cutlass::gemm::GemmShape<64, 64, 32>,             // Warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,              // MMA 指令
    EpilogueOp,                                       // ← 融合 GELU
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3                                                 // 流水线级数
>;
```

**Epilogue 执行原理**：当 Mainloop 计算完一个 output tile（如 128×128），结果以 FP32 fragment 分布在各线程的寄存器中。Epilogue functor 对每个线程持有的元素执行 `output[i] = GELU(alpha * acc[i] + beta * source[i])`，然后向量化写回 HBM。整个过程不涉及额外的 SMEM 或 HBM 访问。

Bias 可通过 C 矩阵接口传入：`{bias, stride=0}` 表示沿 M 维度广播，`beta=1.0` 表示 `D = GELU(A×B + bias)`。

---

### 7.3 Triton 融合 Kernel

Triton 的融合更直接——所有操作写在一个 kernel 内，无需特殊框架。

#### 7.3.1 GEMM + Bias + GELU（完整实现）

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_gelu_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    C = GELU(A @ B + bias)

    三步融合全部在寄存器中完成：
    1. GEMM 累加到 FP32 寄存器 (acc)
    2. bias 广播加到 acc（零 HBM 访问——bias 只有 N 个元素，一次加载）
    3. GELU 直接在 acc 上计算
    4. 一次性写回 HBM
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # ---- GEMM 主循环 ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_offset in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k_offset * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- Bias（寄存器内广播加法）----
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]  # (1, BLOCK_N) 广播到 (BLOCK_M, BLOCK_N)

    # ---- GELU（寄存器内计算）----
    # tanh 近似: GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    kAlpha = 0.7978845608028654   # √(2/π)
    kBeta = 0.044715
    inner = kAlpha * (acc + kBeta * acc * acc * acc)
    acc = 0.5 * acc * (1.0 + tl.math.tanh(inner))

    # ---- 写回（一次 store 包含 GEMM + Bias + GELU）----
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def gemm_bias_gelu(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    """A: (M,K) fp16, B: (K,N) fp16, bias: (N,). 返回: (M,N) fp16."""
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_gelu_kernel[grid](
        A, B, bias, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
    )
    return C
```

#### 7.3.2 SwiGLU 融合简述

SwiGLU（LLaMA 系列模型）：`output = Swish(X @ W_gate) ⊙ (X @ W_up)`

融合的关键洞察：**两个 GEMM 共享输入 X**。在 K 维度迭代循环中，X tile 只需加载一次，分别与 W_gate 和 W_up 做 `tl.dot`，节省 50% 的输入带宽。Swish 和逐元素乘法在累加完成后的寄存器中执行。

```python
# SwiGLU 核心循环（概念代码）
gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

for k_start in range(0, D_in, BLOCK_K):
    x = tl.load(...)           # X tile — 只加载一次
    wg = tl.load(...)          # W_gate tile
    wu = tl.load(...)          # W_up tile
    gate_acc += tl.dot(x, wg)  # 共享 x
    up_acc += tl.dot(x, wu)    # 共享 x

# Swish + 逐元素乘（寄存器中完成）
result = (gate_acc * tl.sigmoid(gate_acc)) * up_acc
tl.store(out_ptrs, result.to(tl.float16), mask=mask)
```

**融合策略原则**：优先融合 memory-bound 算子（激活函数、bias、残差连接），这些操作的计算/访存比极低，融合收益最大。两个大 GEMM 间的融合需谨慎——可能导致寄存器压力过大，降低 occupancy。

---

## 课后练习题

### 练习 1：online softmax 不变量

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么最大值变化时要缩放历史 sum？</span></summary>

历史 sum 是按旧最大值归一化的。当新最大值出现后，同一批历史项需要改写到新最大值的坐标系下，所以整体乘上 exp(old_max - new_max)。这就是 online softmax 能分块处理且保持数值稳定的关键。

</details>

### 练习 2：FlashAttention 的本质

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">FlashAttention 快在哪里？</span></summary>

它不把完整 attention matrix 写回 HBM，而是按 block 流式计算 QK、online softmax 和 PV，把中间状态留在 SRAM/register。它减少的是 HBM traffic，而不是改变 attention 的数学结果。对长序列来说，省掉 N by N attention matrix 的读写是核心收益。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">FlashAttention 为什么需要 online softmax，而不是普通 softmax？</span></summary>

普通 softmax 需要先看到整行 attention score 才能求最大值和归一化分母，会逼着你 materialize 或反复读完整矩阵。online softmax 维护当前最大值 `m`、分母 `l` 和累加输出 `acc`，每来一个 K/V block 就更新状态，因此可以分块流式计算。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">FlashAttention 减少的是计算量还是内存流量？</span></summary>

主要减少 HBM 流量。QK 和 PV 的数学计算仍然存在，但完整 `N x N` attention matrix 不再写回 HBM，也不需要再读出来做 softmax 和 PV。它的核心是 IO-aware，不是把 dense attention 变成 sparse attention。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">GEMM + Bias + GELU 融合的收益来自哪里？</span></summary>

GEMM 结果如果先写回 HBM，再由 bias/GELU kernel 读出来处理，会多一轮大张量读写。融合后 bias 和 GELU 在 accumulator/register 中完成，只写最终结果。收益主要来自减少 memory traffic 和 kernel launch，而不是减少 GEMM FLOPs。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">SwiGLU 融合为什么比普通 GELU 融合更容易带来资源压力？</span></summary>

SwiGLU 同时维护 gate 和 up 两套 accumulator，还要共享输入 tile 做两次 dot。这样能省输入读取，但寄存器、shared memory 和 instruction scheduling 压力更高。如果 tile 选得太大，可能 spill 或 occupancy 降低，抵消 fusion 收益。

</details>
