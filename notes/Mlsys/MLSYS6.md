# Memory-Bound Kernel 优化

本讲承接前两讲的 Reduce、Histogram、Scan 三个并行原语。这三者的共同特征是算术强度极低，均为 memory-bound kernel。对于这类 kernel，优化目标始终是**最大化有效带宽利用率**。

本讲将前几讲中零散使用的优化技巧提炼为一套通用的分析框架，使其可以迁移到任意 memory-bound kernel 上。

---

## 1. Roofline 分析：确定性能瓶颈

优化任何 kernel 的第一步是建立性能上界。Roofline 模型给出如下关系：

$$P \le \min(\pi,\ \beta \cdot I), \quad I = \frac{\text{FLOPs}}{\text{Bytes}}$$

- $\pi$：GPU 峰值算力（FLOP/s）
- $\beta$：显存带宽（Byte/s）
- $I$：算术强度（FLOP/Byte）
- $I_{ridge} = \pi / \beta$：屋脊点（ridge point）

当 $I \ll I_{ridge}$ 时，kernel 的性能受限于显存带宽而非算力。

> [!tip] Roofline 的意义
> Roofline 将优化问题转化为可量化的分析：当前性能距离理论上界有多远，瓶颈属于哪一类。没有这个基准，优化方向的选择缺乏依据。

**示例**：上一讲的 Reduce kernel 读取 N 个 float（4N 字节），执行 N-1 次加法，$I \approx 0.25$ FLOP/Byte。A100 的 $I_{ridge} > 100$，两者相差两个数量级，属于典型的 memory-bound。

---

## 2. Memory-Bound 优化的五条核心原则

将 transpose、stencil、SpMV、histogram、compaction 等各类 kernel 的优化经验归纳后，可以提炼出以下五条通用原则：

### 原则 A：字节账本（Byte Accounting）

优化前需要准确计算 kernel 的总内存流量。这一步决定了后续优化是否对准了真正的瓶颈。

计算规则：**将所有读、写操作的字节数累加**，其中 atomic 操作本质是 read-modify-write，其内存流量需按 2-3 倍计算。

> [!note] 常见误区
> 将 FLOPs 减半但未减少内存访问量，kernel 的执行时间不会有任何改善。更糟糕的情况是：为减少计算而引入额外的中间数组，反而增加了内存流量，导致性能下降。

**例：向量加法的字节账本**
```
// C[i] = A[i] + B[i]，N 个 float
// 读：A (4N bytes) + B (4N bytes) = 8N bytes
// 写：C (4N bytes)
// 总内存流量 = 12N bytes
// 峰值带宽 900 GB/s → 理论下界 = 12N / 900G 秒
```

**例：Histogram 的字节账本（容易算错）**
```
// 输入：N 个 int（读 4N bytes）
// 输出：bins[] 使用 atomicAdd
// atomic = read + modify + write → 每次 ≈ 3×4 = 12 bytes
// 总流量 = 4N + 12N = 16N bytes（而非天真以为的 4N + 4N）
```

---

### 原则 B：合并访存（Coalescing）

理想的访存形态：**一个 warp 的 32 个线程访问连续的 128 字节**（以 float 为例）。

具体要求：
- warp 内的 lane 映射到连续的内存地址
- 整体访问模式尽可能接近顺序 streaming

合并访存是所有其他优化的前提。如果 coalescing 未满足，有效带宽的上限将被大幅削减。

**例：矩阵转置中的 coalescing 问题**
```
// 反面：按列读取，warp 内线程访问步长为 N
out[j][i] = in[i][j]   // in 按行读（coalesced）✓
                        // out 按列写（strided）✗ → 带宽利用率骤降

// 正面：借助 shared memory 中转
tile[threadIdx.y][threadIdx.x] = in[row][col]   // coalesced 读
__syncthreads()
out[col][row] = tile[threadIdx.x][threadIdx.y]   // coalesced 写
```

**例：AoS vs SoA**
```
// AoS（Array of Structs）— warp 读 x 时跨步为 sizeof(Point)
struct Point { float x, y, z; };
Point pts[N];            // pts[tid].x → stride=12 bytes ✗

// SoA（Struct of Arrays）— warp 读 x 时连续
float px[N], py[N], pz[N];
px[tid]                  // stride=4 bytes，完美 coalesced ✓
```

---

### 原则 C：显式复用（Tiling）

当 kernel 存在邻域或数据复用结构（如 stencil、卷积、部分稀疏局部算子）时：

- 不应依赖硬件 L1/L2 cache 的隐式命中
- 应通过 tiling（shared memory 或 register）将复用转化为确定性行为

核心思路：**将数据从 HBM 加载到 SRAM 后进行多次复用，避免重复的 HBM 访问**。

> [!info] 什么是 Stencil？
> Stencil（模板计算）是一种常见的计算模式：每个输出元素由其**自身及固定邻域内的输入元素**加权求和得到。半径 R 的 1D stencil 意味着 `out[i]` 依赖 `in[i-R] ... in[i+R]` 共 2R+1 个元素。典型应用包括有限差分（CFD/PDE 求解）、图像模糊/锐化（2D stencil）、音频滤波等。由于相邻输出点的输入窗口高度重叠，stencil 是 tiling 优化的经典场景。

**例：1D Stencil — 无 tiling vs 有 tiling**
```
// 无 tiling：每个输出点从 HBM 读 2R+1 个邻居
// 相邻线程的读取大量重叠 → 依赖 cache 命中，不可控
out[i] = Σ w[k] * in[i-R+k],  k=0..2R

// 有 tiling：block 协作加载一段 tile（含 halo）到 shared memory
__shared__ float tile[BLOCK + 2*R];
tile[threadIdx.x + R] = in[blockStart + threadIdx.x];
if (threadIdx.x < R) {               // 加载左右 halo
    tile[threadIdx.x] = in[blockStart - R + threadIdx.x];
    tile[BLOCK + R + threadIdx.x] = in[blockStart + BLOCK + threadIdx.x];
}
__syncthreads();
out[i] = Σ w[k] * tile[threadIdx.x + k];  // 全部命中 SRAM
```

---

### 原则 D：减少同步与争用（Sync/Contention）

Memory-bound kernel 的性能瓶颈往往不在于带宽本身，而在于：
- `__syncthreads()` 调用过于频繁，将流水线吞吐转化为串行等待
- 原子操作竞争（histogram、scatter），将并行写退化为串行写

上一讲 Histogram 中的"层次化私有化"是这一原则的典型应用：将竞争范围从 global 逐级缩小到 block、再到 warp，从而降低争用开销。

**例：Histogram 层次化私有化**
```
// 级别 1 — 全局 atomic（最大争用）
atomicAdd(&global_bins[val], 1);           // 所有线程竞争同一组 bins

// 级别 2 — block 私有 bins → 最后归约
__shared__ int local_bins[NUM_BINS];       // 每个 block 一份
atomicAdd(&local_bins[val], 1);            // 争用缩小到 block 内
__syncthreads();
atomicAdd(&global_bins[tid], local_bins[tid]);  // 一次性归约

// 级别 3 — warp 私有 bins（寄存器/shared memory 分区）
// 争用进一步缩小到 32 个线程内，几乎无冲突
```

---

### 原则 E：延迟隐藏（Latency Hiding）

当内存访问的高延迟无法避免时（如 SpMV 的随机访问），可通过以下方式隐藏延迟：
- **提高 occupancy**：增加同时驻留的 warp 数量，使更多 warp 能在内存等待期间被调度执行
- **增加 ILP（指令级并行）**：通过 unroll 和一线程多元素策略，使单线程同时发起更多 load 请求

Grid-stride loop 是实现这一原则的通用工程化手段。

**例：Grid-stride loop + ILP unroll**
```
// 基础版：每线程处理一个元素，occupancy 是唯一延迟隐藏手段
for (int i = tid; i < N; i += gridDim.x * blockDim.x)
    out[i] = f(in[i]);

// ILP 版：每线程同时发起多个 load，隐藏内存延迟
for (int i = tid; i < N; i += stride * 4) {
    float a = in[i];
    float b = in[i + stride];
    float c = in[i + stride*2];
    float d = in[i + stride*3];    // 4 个 load 同时 in-flight
    out[i]            = f(a);
    out[i + stride]   = f(b);
    out[i + stride*2] = f(c);
    out[i + stride*3] = f(d);
}
```

---

## 3. Pattern Library：六类 Memory-Bound Kernel

以下将 memory-bound kernel 按访存模式归纳为六类。面对新的 kernel 时，首先判断其所属类别，然后按对应的原则组合进行优化。

---

### Pattern 1：Streaming（线性读写）

**典型场景**：`out[i] = f(in[i])`，elementwise map、向量加法

**访存特征**：顺序读取输入，顺序写入输出，无数据复用。

**适用原则**：B（coalescing）+ E（延迟隐藏）

**优化手段**：vectorized load/store（float4）、内存对齐、grid-stride loop、循环展开

```cpp
// Vectorized elementwise kernel
// 使用 float4 进行向量化加载和存储，每次内存事务搬运 16 字节
__global__ void vector_add_v4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n4  // n / 4，即 float4 元素个数
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n4; i += stride) {
        float4 va = a[i];  // 128-bit load
        float4 vb = b[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[i] = vc;  // 128-bit store
    }
}

//// float4 的定义（大致）
// struct float4 {
//    float x, y, z, w;
//};

// 启动配置
// n 为 float 元素总数，需为 4 的倍数（否则需额外处理尾部）
// vector_add_v4<<<(n/4 + 255) / 256, 256>>>(a4, b4, c4, n/4);
```

> [!note] float4 的优化原理
> 使用 float4 后，每条 load/store 指令搬运 16 字节而非 4 字节。这减少了所需的 load/store 指令总数，使编译器能够更有效地安排指令流水线，提高指令级并行度（ILP）。一个 warp 的 32 个线程同时执行 float4 load 时，产生 4 个 128B 内存事务，总计搬运 512 字节。

---

### Pattern 2：Reorder / Permutation（重排类）

**典型场景**：Matrix Transpose

**核心矛盾**：读取方向与写入方向正交。若读取是 coalesced 的，则写入必然是 strided 的，反之亦然。

**解决方案**：使用 shared memory 作为中间缓冲区。读取时按行方向 coalesce 加载到 shared memory，写入时从 shared memory 按列方向 coalesce 写出。

**适用原则**：B（读写两端均需 coalesce）+ C（shared memory 作为重排缓存）+ D（避免 bank conflict）

```cpp
// Shared memory tiled 矩阵转置
// 输入 [M x N]，输出 [N x M]
#define TILE_DIM 32
#define BLOCK_ROWS 8  // block 尺寸: TILE_DIM x BLOCK_ROWS
                      // 每个 block 处理 TILE_DIM x TILE_DIM 的 tile

__global__ void transpose_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    // 列数 +1 作为 padding，消除 bank conflict（详见下方说明）
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Step 1: 按行方向从 global memory 加载到 shared memory（coalesced read）
    // 每个线程负责加载 TILE_DIM / BLOCK_ROWS = 4 行
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }

    __syncthreads();

    // Step 2: 坐标互换后，按行方向从 shared memory 写入 global memory（coalesced write）
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < M && (y + j) < N) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// 启动配置
// dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
// dim3 block(TILE_DIM, BLOCK_ROWS);
// transpose_optimized<<<grid, block>>>(input, output, M, N);
```

> [!important] Padding 消除 bank conflict
> Shared memory 由 32 个 bank 组成，每个 bank 宽度为 4 字节。若 tile 的列数恰好为 32，则同一列的所有元素将映射到同一个 bank，导致列方向访问时产生 32 路 bank conflict。将列数设为 `TILE_DIM + 1 = 33` 后，相邻行中同一列位置的元素错开一个 bank，从而消除冲突。这一技巧在涉及 shared memory 列访问的 kernel 中被广泛使用。

---

### Pattern 3：Stencil / Neighborhood（邻域复用类）

**典型场景**：1D/2D stencil、图像卷积、图像处理滤波器

**核心特征**：每个输入元素被多个相邻输出点复用。以 1D 3-point stencil 为例，每个输入元素被左、中、右三个输出点各读取一次。

**适用原则**：C（tile + halo 实现确定性复用）+ B（halo 加载时保持 coalescing）

```cpp
// 1D Stencil: out[i] = c0*in[i-R] + c1*in[i-R+1] + ... + c2R*in[i+R]
// R = stencil 半径，本例取 R = 4（9-point stencil）

#define RADIUS 4
#define BLOCK_SIZE 256
// 每个 block 计算 BLOCK_SIZE 个输出点
// 需加载 BLOCK_SIZE + 2*RADIUS 个输入点（含两端 halo 区域）

__constant__ float coeff[2 * RADIUS + 1];  // stencil 系数存入 constant memory

__global__ void stencil_1d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float smem[BLOCK_SIZE + 2 * RADIUS];

    int gidx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lidx = threadIdx.x + RADIUS;  // shared memory 内的偏移索引

    // Step 1: 加载中间区域
    smem[lidx] = (gidx < n) ? input[gidx] : 0.0f;

    // Step 2: 加载左侧 halo（前 RADIUS 个线程负责）
    if (threadIdx.x < RADIUS) {
        int halo_idx = gidx - RADIUS;
        smem[threadIdx.x] = (halo_idx >= 0) ? input[halo_idx] : 0.0f;
    }

    // Step 3: 加载右侧 halo（后 RADIUS 个线程负责）
    if (threadIdx.x >= BLOCK_SIZE - RADIUS) {
        int halo_idx = gidx + RADIUS;
        smem[lidx + RADIUS] = (halo_idx < n) ? input[halo_idx] : 0.0f;
    }

    __syncthreads();

    // Step 4: 从 shared memory 计算 stencil，无 global memory 访问
    if (gidx < n) {
        float result = 0.0f;
        #pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++) {
            result += coeff[j + RADIUS] * smem[lidx + j];
        }
        output[gidx] = result;
    }
}
```

**内存流量对比**：

| 版本 | Global Memory 读取量 | 说明 |
|------|---------------------|------|
| Naive（无 shared memory） | $N \times (2R+1)$ | 每个输出点从 HBM 读取 $2R+1$ 个输入 |
| Tiled（使用 shared memory） | $N + 2R \times \text{num\_blocks}$ | 每个输入元素基本仅从 HBM 加载一次 |

当 stencil 半径 $R$ 越大时，数据复用程度越高，tiling 的收益越显著。

---

### Pattern 4：Indirection / Gather（读侧不规则）

**典型场景**：CSR 格式 SpMV、gather、embedding lookup

**核心困难**：间接寻址 `x[col_idx[j]]` 导致访问地址取决于数据内容，难以实现理想的 coalescing。

**适用原则**：A（将 index 的字节开销纳入计算）+ E（通过 warp-level 映射隐藏延迟）

稀疏矩阵-向量乘法 (SpMV)

```
CSR (Compressed Sparse Row) 用三个数组存储稀疏矩阵:
═══════════════════════════════════════════════════════════════════════════════

1. values[nnz]:   所有非零值，按行存储
2. col_idx[nnz]:  每个非零值对应的列号
3. row_ptr[M+1]:  每行在 values/col_idx 中的起始位置

对于上面的矩阵:

values[]:   [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
             ↑       ↑   ↑   ↑   ↑           ↑   ↑
            row0    row0 row1 row1 row2      row2 row3

col_idx[]:  [ 0,  2,  4,  1,  5,  0,  1,  2,  3,  5]
             对应每个非零元素的列号

row_ptr[]:  [ 0,  3,  5,  9, 10]
              ↑   ↑   ↑   ↑   ↑
             row0 row1 row2 row3 结束
             起始 起始 起始 起始

row_ptr[i] 到 row_ptr[i+1] 之间的索引就是第 i 行的非零元素
```


warp-per-row策略
```
核心思想: 一个 Warp (32个线程) 协作处理矩阵的一行
═══════════════════════════════════════════════════════════════════════════════

为什么不用 Thread-per-Row？
─────────────────────────────
如果一行有很多非零元素（比如 1000 个），单线程串行处理太慢

Warp-per-Row 的优势:
─────────────────────────────
1. 32 个线程并行处理一行的非零元素
2. 访问 values[] 和 col_idx[] 时地址连续 → Coalesced Access
3. 使用 Warp Shuffle 归约，不需要 Shared Memory
```

```cpp
// CSR 格式 SpMV: y = A * x
// CSR 存储: row_ptr[M+1], col_idx[nnz], values[nnz]
//
// 映射策略:
//   行长度较均匀: thread-per-row
//   行长度差异大: warp-per-row（本例采用此策略）

// Warp-per-row: 每个 warp（32 线程）处理矩阵的一行
// 优势:
//   1. warp 内线程访问连续的 col_idx 和 values（coalesced）
//   2. 使用 warp shuffle 归约，无需 shared memory 或 atomic
//   3. 32 个线程同时发起 load，有效隐藏延迟
__global__ void spmv_csr_warp_per_row(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= num_rows) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    float sum = 0.0f;

    // warp 内 32 个线程以 stride 32 遍历该行的非零元素
    for (int j = row_start + lane; j < row_end; j += 32) {
        // col_idx[j], values[j]: 连续访问，coalesced
        // x[col_idx[j]]: 随机访问，依赖 L2 cache 和延迟隐藏
        sum += values[j] * x[col_idx[j]];
    }

    // Warp-level 归约（warp shuffle），无需 shared memory 或 barrier
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[warp_id] = sum;
    }
}

// 启动配置
// int threads_per_block = 256;  // 每个 block 包含 8 个 warp
// int num_blocks = (num_rows * 32 + threads_per_block - 1) / threads_per_block;
// spmv_csr_warp_per_row<<<num_blocks, threads_per_block>>>(...);
```

如果需要所有 lane 都拿到结果，可以用 __shfl_sync 广播，或者用 __shfl_xor_sync 做 butterfly 归约。但这里只需要写一个 y[warp_id]，所以 Lane 0 有结果就够了。

> [!note] 稀疏 kernel 的带宽上界
> 对于稀疏类 kernel，实际可达的带宽上限往往低于 HBM 的理论峰值。`x[col_idx[j]]` 的访问模式由矩阵的稀疏结构决定，无法在 kernel 层面完全控制。优化方向因此转为"降低访问的随机性"，例如对矩阵列进行 reorder 以提高向量 x 的访问局部性。

---

### Pattern 5：Scatter / Atomic（写侧不规则 + 争用）

**典型场景**：histogram、scatter-add、部分图算法

**核心困难**：写入地址取决于数据内容，热点目标（如高频 bin）导致原子操作严重串行化。

**适用原则**：D（层次化私有化）

本类 kernel 的优化方法已在上一讲 Histogram 的三个版本中详细展示。核心策略如下：

```
Level 0: Global atomic — 所有线程竞争全局内存
  ↓ 私有化
Level 1: Block 级 shared memory atomic — 竞争范围缩小至 256 线程
  ↓ 进一步私有化
Level 2: Warp 级 / 寄存器本地累积 — 竞争缩小至 32 线程或完全消除
  ↓ 最终归约
写回 global memory — atomic 调用次数从 N 降至 num_bins × num_blocks
```

---

### Pattern 6：Filter / Compaction（条件过滤类）

**典型场景**：stream compaction、去零操作、predicate filter

#### Stream Compaction 是什么？

**从数组中过滤出满足条件的元素，紧凑存储**

```
输入:  [ 3, -1, 4, 0, -2, 5, 0, 1 ]
条件:  元素 > 0
输出:  [ 3, 4, 5, 1 ]
```

#### 为什么难并行？

每个线程不知道自己的输出位置——**用 Exclusive Prefix Sum 解决**：

```
flag:   [ 1,  0,  1,  0,  0,  1,  0,  1 ]   ← 标记满足条件的
scan:   [ 0,  1,  1,  2,  2,  2,  3,  3 ]   ← exclusive scan

scan[i] = "在我之前有几个满足条件的" = 我的输出位置
```

#### 三步流程

| 步骤 | 做什么 | 代码 |
|-----|-------|------|
| **Flag** | 标记满足条件的元素 | `flag = (val > 0) ? 1 : 0` |
| **Scan** | Exclusive prefix sum (Blelloch算法) | Up-sweep + Down-sweep |
| **Scatter** | 写到正确位置 | `output[offset + scan[tid]] = val` |


```cpp
// Stream Compaction: 筛选 input 中 > 0 的元素，紧凑写入 output
// Fused 实现: 在单个 kernel 内完成 flag、block-level scan、scatter

#define BLOCK_SIZE 256

__global__ void stream_compaction(
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ output_count,  // 输出的元素总数
    int n
) {
    __shared__ int scan[BLOCK_SIZE];
    __shared__ int block_output_offset;

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // Step 1: Flag — 标记满足条件的元素
    int val = 0;
    int flag = 0;
    if (gid < n) {
        val = input[gid];
        flag = (val > 0) ? 1 : 0;
    }
    scan[tid] = flag;
    __syncthreads();

    // Step 2: Block-level exclusive scan（Blelloch 算法）

    // Up-sweep
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < BLOCK_SIZE) {
            scan[ai] += scan[ai - offset];
        }
        __syncthreads();
    }

    // 提取 block 内满足条件的元素总数，并通过一次 atomic 分配输出空间
    if (tid == 0) {
        int block_total = scan[BLOCK_SIZE - 1];
        block_output_offset = atomicAdd(output_count, block_total);
        scan[BLOCK_SIZE - 1] = 0;
    }
    __syncthreads();

    // Down-sweep
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < BLOCK_SIZE) {
            int temp = scan[ai - offset];
            scan[ai - offset] = scan[ai];
            scan[ai] += temp;
        }
        __syncthreads();
    }

    // Step 3: Scatter — 将满足条件的元素写入输出数组的正确位置
    if (gid < n && flag) {
        output[block_output_offset + scan[tid]] = val;
    }
}
```

> [!tip] 关键优化
> `atomicAdd(output_count, block_total)` 的调用次数为 num_blocks 而非 N。这是原则 D（减少争用）和原则 A（降低 atomic 的内存开销）的结合：每个 block 内部通过 scan 计算局部偏移，最后仅需一次 atomic 操作分配输出空间。

---

## 4. 优化 Checklist

对 memory-bound kernel 进行优化时，建议按以下顺序依次排查：

| 步骤 | 内容 | 观测手段 |
|------|------|----------|
| 1 | 计算算术强度与带宽上限 | Roofline 模型，手动计算 |
| 2 | 测量有效带宽 | Bytes / kernel_time，与理论峰值对比 |
| 3 | 检查合并访存 | ncu: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` |
| 4 | 检查争用与同步开销 | ncu: atomic throughput、barrier wait time |
| 5 | 调整 occupancy 与 ILP | grid-stride loop、循环展开、一线程多元素 |

此顺序的重要性在于：若 coalescing 未满足就去调整 occupancy，所获得的收益将非常有限。应首先确保访存模式正确，再进行更高层次的调优。

---

## 5. 总结

| 原则 | 优化目标 | 典型适用 kernel |
|------|----------|----------------|
| A 字节账本 | 准确量化内存流量 | 所有 kernel |
| B 合并访存 | warp 内连续地址访问，减少内存事务数 | streaming、transpose |
| C 显式复用 | 数据从 HBM 加载到 SRAM 后多次复用 | stencil、部分 sparse |
| D 减少争用 | 降低 barrier 与 atomic 的串行化开销 | histogram、scatter、compaction |
| E 延迟隐藏 | 通过并行度与 ILP 掩盖内存延迟 | sparse、gather |

前两讲中 Reduce、Histogram、Scan 的优化已经覆盖了以上所有原则（coalescing、first add during load、warp shuffle、ILP、grid-stride loop）。本讲的作用是将这些分散在具体例子中的技巧抽象为通用的分析框架。

面对新的 memory-bound kernel 时，首先判断其所属的 Pattern（1-6），然后按对应的原则组合制定优化策略。

---

## 课后练习题

### 练习 1：字节账本

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">优化 memory-bound kernel 前为什么要先算 bytes？</span></summary>

memory-bound kernel 的时间下界近似是 bytes 除以 bandwidth。如果一个优化没有减少 HBM bytes，也没有提高访问合并度或 cache reuse，它很可能不会显著变快。字节账本还会暴露中间 tensor 写回、atomic read-modify-write、重复读取和 padding。

</details>

### 练习 2：什么时候不该 fusion？

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">kernel fusion 一定更快吗？</span></summary>

不一定。Fusion 能减少 HBM 往返和 launch overhead，但也可能增加 register pressure、降低 occupancy、破坏 library GEMM 的 Tensor Core 路径，或者让原本可复用的中间结果变成重复计算。判断标准是节省的 memory traffic 是否大于新增成本。

</details>
