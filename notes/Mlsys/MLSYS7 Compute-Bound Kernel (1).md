# MLSYS7 · Compute-Bound Kernel (1)

## 1. 引言：什么是 Compute-Bound Kernel

Compute-bound kernel 是指其性能瓶颈在于 GPU 的计算单元（ALU/FPU/Tensor Core）处理能力，而非内存带宽。这类 kernel 的特征是：

- **高算术强度（Arithmetic Intensity）**：每从内存加载一个字节的数据，需要执行大量的浮点运算
- **计算单元利用率是关键**：优化目标是最大化 SM（Streaming Multiprocessor）的计算吞吐量
- **典型代表**：矩阵乘法、卷积、注意力机制等

### 1.1 Compute-Bound 的判断标准

```
算术强度 = FLOPs / Bytes_Accessed

若 算术强度 > GPU峰值计算能力 / GPU峰值带宽
则该 kernel 为 compute-bound
```

以 NVIDIA A100 为例：
- 峰值计算能力（FP32）：19.5 TFLOPS
- 峰值带宽：2039 GB/s
- 临界算术强度 ≈ 9.6 FLOPs/Byte

### 1.2 常见Compute-Bound Kernel 类型

| Kernel 类型 | 算术强度 | 典型应用场景 |
|------------|---------|-------------|
| GEMM | O(N) | 全连接层、注意力投影 |
| 卷积 | O(K²) | CNN、图像处理 |
| Self-Attention | O(N) | Transformer |
| 复杂激活函数 | O(1) 但计算密集 | GELU、Swish、SiLU |
| 融合 Kernel | 变化大 | 各种算子融合 |

---

## 2. 矩阵乘法 (GEMM) 优化

矩阵乘法 `C = A × B` 是最经典的 compute-bound kernel，也是深度学习中最核心的操作。

### 2.1 GEMM 的计算特性

对于 `C[M,N] = A[M,K] × B[K,N]`：
- **FLOPs**: 2 × M × N × K
- **数据访问**: M×K + K×N + M×N（写回）
- **算术强度**: 约 2×M×N×K / (M×K + K×N + M×N) ≈ O(min(M,N,K))

当矩阵足够大时，GEMM 几乎总是 compute-bound。

### 2.2 朴素实现（Baseline）

```cuda
// 朴素 GEMM - 每个线程计算 C 的一个元素
__global__ void gemm_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**问题分析**：
1. 全局内存访问模式差（B 矩阵的列访问不连续）
2. 没有数据重用
3. 每个线程只计算一个输出元素，效率低

### 2.3 共享内存分块（Tiling）优化

---

> 💡 **核心技巧：Shared Memory Tiling（共享内存分块）**
>
> Tiling 是 CUDA 优化中**最重要的通用技巧之一**，几乎出现在所有高性能 kernel 中。核心思想只有一句话：**把数据从慢的全局内存（HBM, ~1TB/s）搬到快的共享内存（SRAM, ~19TB/s），然后在快的内存上重复使用多次**。

**为什么需要 Tiling？** 以 GEMM 为例：计算 $C_{ij} = \sum_k A_{ik} B_{kj}$，朴素实现中每个线程独立从全局内存读 A 的一行和 B 的一列。但相邻线程其实需要 A 的同一行或 B 的同一列——这些数据被反复读取却没有复用。Tiling 的做法是：一个 block 的所有线程**协作地**把 A 和 B 的一个小块（tile）搬到共享内存，然后每个线程从共享内存读数据计算。一份数据被 TILE_SIZE 个线程共享，全局内存访问量降低 TILE_SIZE 倍。

**Tiling 的通用模式（三步曲）**：

```
1. 协作加载：block 内所有线程各负责一部分，把 tile 从 global memory → shared memory
2. 同步屏障：__syncthreads()，确保所有线程加载完毕
3. 本地计算：每个线程从 shared memory 读数据做计算（命中 SRAM，延迟 ~20 cycles vs HBM ~400 cycles）
```

**Tiling 在不同场景中的应用**（参见 [MLSYS6.md](notes/Mlsys/MLSYS6.md) 的详细讨论）：

| 场景 | Tile 的作用 | 关键细节 |
|------|-----------|---------|
| **GEMM**（本节） | 复用 A 的行和 B 的列 | 每份数据被 TILE_SIZE 个线程复用 |
| **矩阵转置**（MLSYS6 Pattern 2） | 读写方向正交的中间缓冲 | 读时 coalesced → tile → 写时 coalesced；`tile[DIM][DIM+1]` 加 padding 消除 bank conflict |
| **Stencil/卷积**（MLSYS6 Pattern 3） | 邻域数据复用 + halo 区域 | tile 需要额外加载 ±R 的 halo 边界；复用倍数 = 2R+1（stencil 宽度）|
| **直方图**（MLSYS6 原则 D） | 私有 bin 减少 atomic 争用 | 每个 block 在 shared memory 维护私有直方图，最后合并到全局 |

**性能分析**：设 tile 大小为 $T$，原始每个元素被读 $R$ 次（复用因子），Tiling 后全局内存访问量降为 $\frac{1}{R}$（理想情况）。对 GEMM：$R = T$（TILE_SIZE）；对 stencil：$R = 2 \times \text{radius} + 1$。**tile 越大，复用越高，但受限于共享内存容量**（每个 SM 通常 48-228 KB）。这就是 tile size 选择的核心权衡：太小→复用不够，太大→shared memory 不够或 occupancy 降低。

**常见的 Tiling 进阶技巧**（后续小节会用到）：
- **Double buffering**（2.5 节）：用两个 tile buffer 交替，一个在计算、一个在预取，隐藏访存延迟
- **寄存器分块**（2.4 节）：在 shared memory tiling 之上，再把 tile 的一小块加载到寄存器，形成 global → shared → register 的三级层次
- **Padding**（2.5 节）：`tile[DIM][DIM + padding]` 避免 bank conflict，代价是多占少量共享内存
- **向量化加载**（2.4 节）：用 `float4` 一次加载 128 位，减少加载指令数量、提高带宽利用率

---

```cuda
#define TILE_SIZE 32

__global__ void gemm_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历所有 tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 协作加载 A 和 B 的 tile 到共享内存
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算部分和
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**优化点**：
1. 利用共享内存实现数据重用
2. 每个数据被重用 TILE_SIZE 次
3. 减少全局内存访问次数

### 2.4 寄存器分块 + 向量化访问

```cuda
#define BM 128  // Block tile size in M dimension
#define BN 128  // Block tile size in N dimension
#define BK 8    // Block tile size in K dimension
#define TM 8    // Thread tile size in M dimension
#define TN 8    // Thread tile size in N dimension

__global__ void gemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 共享内存声明
    __shared__ float As[BK][BM];  // 转置存储以避免 bank conflict
    __shared__ float Bs[BK][BN];
    
    // 计算 block 和 thread 的位置
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // 每个 block 有 (BM/TM) × (BN/TN) 个线程
    const int thread_x = tx;
    const int thread_y = ty;
    
    // 寄存器存储线程的计算结果
    float thread_results[TM][TN] = {0.0f};
    
    // 寄存器存储 A 和 B 的片段
    float reg_a[TM];
    float reg_b[TN];
    
    // A 和 B 的起始位置
    const float* A_ptr = A + by * BM * K;
    const float* B_ptr = B + bx * BN;
    
    // 计算加载索引
    const int num_threads = (BM / TM) * (BN / TN);
    const int thread_id = ty * (BN / TN) + tx;
    
    // 主循环：遍历 K 维度
    for (int k_base = 0; k_base < K; k_base += BK) {
        // ============ 协作加载 A tile 到共享内存 ============
        // 使用向量化加载 (float4)
        #pragma unroll
        for (int i = 0; i < BM * BK / (num_threads * 4); ++i) {
            int load_idx = thread_id + i * num_threads;
            int load_row = load_idx / (BK / 4);
            int load_col = (load_idx % (BK / 4)) * 4;
            
            if (by * BM + load_row < M && k_base + load_col < K) {
                float4 tmp = *reinterpret_cast<const float4*>(
                    &A_ptr[load_row * K + k_base + load_col]
                );
                As[load_col + 0][load_row] = tmp.x;
                As[load_col + 1][load_row] = tmp.y;
                As[load_col + 2][load_row] = tmp.z;
                As[load_col + 3][load_row] = tmp.w;
            }
        }
        
        // ============ 协作加载 B tile 到共享内存 ============
        #pragma unroll
        for (int i = 0; i < BK * BN / (num_threads * 4); ++i) {
            int load_idx = thread_id + i * num_threads;
            int load_row = load_idx / (BN / 4);
            int load_col = (load_idx % (BN / 4)) * 4;
            
            if (k_base + load_row < K && bx * BN + load_col < N) {
                float4 tmp = *reinterpret_cast<const float4*>(
                    &B_ptr[(k_base + load_row) * N + load_col]
                );
                reinterpret_cast<float4*>(&Bs[load_row][load_col])[0] = tmp;
            }
        }
        
        __syncthreads();
        
        // ============ 计算线程的 tile ============
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 从共享内存加载到寄存器
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_a[m] = As[k][thread_y * TM + m];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                reg_b[n] = Bs[k][thread_x * TN + n];
            }
            
            // 外积计算
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    thread_results[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // ============ 写回结果 ============
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int global_row = by * BM + thread_y * TM + m;
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int global_col = bx * BN + thread_x * TN + n;
            if (global_row < M && global_col < N) {
                float4 result;
                result.x = thread_results[m][n + 0];
                result.y = thread_results[m][n + 1];
                result.z = thread_results[m][n + 2];
                result.w = thread_results[m][n + 3];
                *reinterpret_cast<float4*>(&C[global_row * N + global_col]) = result;
            }
        }
    }
}
```

### 2.5 Bank Conflict 消除

共享内存被组织成 32 个 bank，连续的 4 字节（一个 float）分布在连续的 bank 中。当同一 warp 中的多个线程访问同一 bank 的不同地址时，会产生 bank conflict。

```cuda
// 有 bank conflict 的访问模式
__shared__ float smem[32][32];
// 线程 i 访问 smem[i][k]，所有线程访问同一列时会有冲突

// 解决方案 1：Padding
__shared__ float smem[32][33];  // 添加一列 padding

// 解决方案 2：交错访问（Swizzle）
__shared__ float smem[32][32];
// 访问时：smem[row][col ^ (row % 32)]
```

## 3. CUTLASS 与 Triton 编程范式

在前两章中，我们从零开始手写了 CUDA GEMM kernel，逐步引入了 shared memory tiling、register blocking、向量化访存、bank conflict 消除等优化技巧。这些底层优化让我们深刻理解了 GPU 的执行模型。然而，在生产环境中，直接手写 CUDA kernel 面临着诸多挑战。本章将介绍两种主流的高级编程范式——NVIDIA CUTLASS 和 OpenAI Triton，它们分别从 C++ 模板元编程和 Python 编译器的角度，提供了更高效、更可维护的 GPU kernel 开发方式。

---

### 3.1 为什么需要高级抽象

#### 3.1.1 手写 CUDA 的困境

回顾前两章的 GEMM 优化之旅，我们不得不手动管理以下所有细节：

- **Shared memory 管理**：手动计算所需大小、声明 `__shared__` 数组、处理 double buffering 的指针交换、确保 `__syncthreads()` 的正确放置。一个遗漏的同步就可能导致难以复现的竞态条件。
- **Register tiling**：手动将 warp 级别的计算分解到每个线程的寄存器中，精确计算每个线程负责的元素范围，确保寄存器使用量不超过硬件限制（否则发生 register spilling，性能断崖式下降）。
- **Bank conflict 消除**：shared memory 被划分为 32 个 bank，当同一 warp 中多个线程访问同一 bank 的不同地址时产生冲突。需要手动添加 padding 或调整访问模式来避免，这对代码可读性伤害极大。
- **向量化访存**：为了充分利用内存带宽，需要使用 `float4`、`int4` 等向量类型进行 128-bit 访存，手动处理指针对齐和类型转换。
- **指令级优化**：手动展开循环（`#pragma unroll`）、交错计算与访存指令以隐藏延迟、选择合适的 `mma` 指令变体。

#### 3.1.2 跨架构移植的难题

更严重的问题在于，最优的实现策略在不同 GPU 架构之间存在本质差异：

| 特性 | Volta (SM70) | Ampere (SM80) | Hopper (SM90) |
|------|-------------|---------------|---------------|
| Tensor Core 指令 | `wmma` | `mma` | `wgmma` |
| 最优 tile size | 128x128x32 | 128x256x64 | 256x256x64 |
| 异步拷贝 | 不支持 | `cp.async` | `cp.async.bulk` (TMA) |
| 软件流水线 stages | 2 | 3-4 | 4-8 |
| Shared memory 大小 | 96 KB | 164 KB | 228 KB |
| 集群调度 | 不支持 | 不支持 | Thread Block Cluster |

一份在 Ampere 上精心调优的 kernel，迁移到 Hopper 上可能完全无法利用新硬件特性（如 TMA、wgmma、Cluster），性能反而不如通用库。这意味着每一代新架构都需要几乎重写 kernel，维护成本极高。

#### 3.1.3 可组合抽象的需求

工业界需要的是一套**将算法逻辑与硬件映射分离**的可组合抽象：

- **算法层面**：定义"GEMM = 三层循环 + 累加 + epilogue"这样的数学语义
- **调度层面**：定义 tile 大小、流水线深度、warp 分配等策略
- **硬件层面**：映射到具体的指令（mma vs wgmma）、内存层次（shared vs register）、同步机制

CUTLASS 和 Triton 正是这两种不同哲学的代表：CUTLASS 通过 C++ 模板在编译期组合各层抽象；Triton 通过编译器自动推导底层实现。

---

### 3.2 CUTLASS 设计哲学与核心抽象

[CUTLASS](https://github.com/NVIDIA/cutlass)（CUDA Templates for Linear Algebra Subroutines）是 NVIDIA 开源的 C++ 模板库，提供了一套**层次化、可组合**的 GEMM 及相关运算的构建块。它的核心设计理念是：用 C++ 模板元编程将 GEMM 的每一层分解封装，开发者通过组合模板参数来定制 kernel，而无需从头编写。

#### 3.2.1 层次化分解（Hierarchical Decomposition）

CUTLASS 将 GEMM 计算严格分解为四个层次，每个层次对应 GPU 硬件的一个执行级别：

**Device 级别（Grid of Threadblocks）**

整个 GEMM 问题 $D = \alpha \cdot A \times B + \beta \cdot C$ 被划分为一个二维网格的 threadblock。每个 threadblock 负责计算输出矩阵 $D$ 的一个 tile（通常 128x128 或 256x128）。Threadblock 之间完全独立，无需同步。

CUTLASS 提供了 **Swizzle** 策略来控制 threadblock 到输出 tile 的映射顺序，优化 L2 cache 的局部性。例如，`GemmIdentityThreadblockSwizzle<8>` 会将相邻的 8 个 threadblock 映射到空间上相邻的输出 tile，使它们共享更多的 A 或 B 矩阵数据。

**Threadblock 级别（Shared Memory Tiling）**

每个 threadblock 协作地将 A 和 B 矩阵的 tile 从 global memory 加载到 shared memory，然后沿 K 维度迭代，每次处理一个 K-tile。这一层的核心是 **software pipelining**（软件流水线）：在计算当前 K-tile 的同时，异步预取下一个 K-tile 的数据。CUTLASS 通过 `num_stages` 参数控制流水线深度。

**Warp 级别（Tensor Core MMA）**

Shared memory 中的 tile 被进一步划分给各个 warp。每个 warp 使用 Tensor Core 的 MMA（Matrix Multiply-Accumulate）指令进行计算。在 Ampere 上使用 `mma.sync` 指令，每条指令计算如 16x8x16 的小矩阵乘法。CUTLASS 定义了 `MmaWarp` 抽象来封装 warp 级别的 tiling 和 MMA 调用。

**Thread 级别（Epilogue）**

当所有 K-tile 迭代完成后，每个线程持有累加器（accumulator）中的部分结果。Epilogue 阶段负责将累加结果从寄存器写回 global memory，同时执行后处理操作，如：缩放（$\alpha, \beta$）、添加偏置（bias）、激活函数（ReLU, GELU）等。

#### 3.2.2 核心抽象（Key Abstractions）

**1. Layout——张量内存布局**

Layout 描述了逻辑上的多维张量如何映射到一维的物理内存。CUTLASS 支持多种布局：

```cpp
// 行主序：逻辑坐标 (i, j) → 物理偏移 i * stride + j
using LayoutA = cutlass::layout::RowMajor;

// 列主序：逻辑坐标 (i, j) → 物理偏移 i + j * stride
using LayoutB = cutlass::layout::ColumnMajor;

// 卷积专用布局
using LayoutNHWC = cutlass::layout::TensorNHWC;

// 交错布局（用于 INT8/INT4 量化推理）
using LayoutInterleaved = cutlass::layout::ColumnMajorInterleaved<32>;
```

Layout 的选择直接影响内存访问的连续性，进而影响向量化加载的可行性和 shared memory 的 bank conflict 模式。CUTLASS 的 iterator 会根据 layout 自动生成最优的访存指令序列。

**2. TileDescription——分块大小描述**

Tile size 是 GEMM 性能最关键的超参数。CUTLASS 使用 `GemmShape` 模板在三个层次分别定义：

```cpp
// Threadblock 级别：每个 threadblock 计算 128x128 的输出 tile，每次迭代 K=32
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;

// Warp 级别：每个 warp 计算 64x64 的子 tile
// 因此每个 threadblock 有 (128/64) * (128/64) = 4 个 warp
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;

// Instruction 级别：Tensor Core MMA 指令的形状
// Ampere mma.sync.aligned.m16n8k16.f32.f16.f16.f32
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
```

这三个层次的 shape 之间存在整除约束：
- `ShapeMMAThreadBlock::kM` 必须被 `ShapeMMAWarp::kM` 整除
- `ShapeMMAWarp::kM` 必须被 `ShapeMMAOp::kM` 整除
- K 维度类似

违反这些约束会导致编译错误（CUTLASS 通过 `static_assert` 在编译期检查）。

**3. Epilogue——后处理操作**

CUTLASS 的一大亮点是将 epilogue 操作模板化，支持将常见的后处理融合到 GEMM kernel 中，避免额外的 kernel launch 和 global memory 往返：

```cpp
// 线性组合 + ReLU 激活：D = max(0, alpha * A@B + beta * C)
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    float,                                    // 输出元素类型
    128 / cutlass::sizeof_bits<float>::value,  // 每次向量化访问的元素数 (=4)
    float,                                    // 累加器类型
    float                                     // 计算类型
>;

// 线性组合 + GELU 激活
using EpilogueGelu = cutlass::epilogue::thread::LinearCombinationGELU<
    cutlass::half_t, 8, float, float
>;

// 纯线性组合（无激活）：D = alpha * A@B + beta * C
using EpilogueLinear = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 8, float, float
>;
```

Epilogue fusion 是 CUTLASS 相比于裸 cuBLAS 的一大优势：cuBLAS 的 GEMM 只能输出 $D = \alpha AB + \beta C$，后续的激活函数必须另起一个 kernel 完成；而 CUTLASS 可以将激活函数融合在 GEMM kernel 的写回阶段，一次完成所有操作。

**4. Iterator——内存访问模式抽象**

Iterator 是 CUTLASS 中最复杂但也最关键的抽象之一。它封装了从 global memory 到 shared memory、从 shared memory 到 register 的数据搬运逻辑：

- **Global Memory Iterator**：根据 tensor layout 和 alignment，选择最优的向量化加载方式（如 `LDG.128`）；处理边界条件（当 tile 超出矩阵边界时自动 mask）
- **Shared Memory Iterator**：生成无 bank conflict 的 shared memory 访问模式；配合 swizzle 策略重新排列数据布局
- **异步拷贝**：在 Ampere+ 架构上使用 `cp.async` 指令直接从 global memory 异步拷贝到 shared memory，绕过寄存器中转

开发者通常不需要直接操作 iterator，而是通过选择预定义的策略类来间接配置。


#### 3.2.3 CUTLASS 3.x 与 CuTe

CUTLASS 3.x 引入了全新的 **CuTe**（CUDA Tensors）抽象，是对 CUTLASS 2.x 的根本性重构。CUTLASS 2.x 的抽象体系（`GemmShape`、`Iterator`、`Policy` 等）虽然功能强大，但存在两个根本性问题：

1. **概念碎片化**：数据布局、线程映射、内存访问模式被分散在不同的抽象中，彼此的关系隐藏在复杂的模板特化里
2. **组合爆炸**：每种新的 MMA 指令、内存访问模式或布局都需要大量新的特化代码

CuTe 的核心洞察是：**GPU 编程中的几乎所有问题——数据布局、线程分配、分块策略、内存访问模式——本质上都是一个问题：整数坐标到整数偏移的映射**。CuTe 用一套统一的 Layout 代数来表达这一切。

##### Layout = (Shape, Stride)——CuTe 的第一性原理

一个 Layout 完全由两个元组定义：**Shape**（形状）和 **Stride**（步幅）。给定一个逻辑坐标，Layout 将其映射到一个一维的物理偏移：

```
offset = coord[0] * stride[0] + coord[1] * stride[1] + ...
```

```cpp
using namespace cute;

// 4x8 行主序矩阵：坐标 (i,j) → 偏移 i*8 + j
auto row_major = make_layout(make_shape(4, 8), make_stride(8, 1));
// row_major(0, 0) = 0,  row_major(0, 1) = 1,  row_major(1, 0) = 8

// 4x8 列主序矩阵：坐标 (i,j) → 偏移 i + j*4
auto col_major = make_layout(make_shape(4, 8), make_stride(1, 4));
// col_major(0, 0) = 0,  col_major(0, 1) = 4,  col_major(1, 0) = 1
```

关键特性：**Shape 和 Stride 可以是层次化的**（hierarchical），即元组的元素本身也可以是元组。这使得 CuTe 能用同一套语法表达从简单的行/列主序到复杂的分块交错布局：

```cpp
// 层次化 layout：一个 (4,8) 矩阵被分成 2x4 个 (2,2) 的小块
// Shape:  ((2,2), (2,4))  —— 外层是 tile 内坐标，内层是 tile 数量
// Stride: ((1,2), (4,16)) —— 对应的步幅
auto tiled_layout = make_layout(
    make_shape(make_shape(2, 2), make_shape(2, 4)),
    make_stride(make_stride(1, 2), make_stride(4, 16))
);
// 等价于 4x8 矩阵的 RowMajor，但"知道"它被分成了 2x2 的 tile
```

这一点至关重要——**tiling 不是数据操作，而是 layout 的重新解释**。`local_tile` 或 `logical_divide` 这类操作只是重组 Shape 和 Stride 的嵌套结构，不移动任何数据，编译期零开销。

##### Tensor = Pointer + Layout

CuTe 中的 Tensor 就是一个指针加上一个 Layout：

```cpp
// 从 global memory 指针构造 tensor
auto tensor_A = make_tensor(make_gmem_ptr(A_ptr),
    make_layout(make_shape(M, K), make_stride(K, Int<1>{})));

// 从 shared memory 构造 tensor
auto smem_A = make_tensor(make_smem_ptr(smem_ptr),
    make_layout(make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})));

// 从寄存器构造 tensor（register 也是一种"内存"）
auto reg_C = make_tensor<float>(make_shape(Int<8>{}, Int<4>{}));  // 在寄存器上分配
```

注意 `Int<128>{}` 这种编译期常量——CuTe 大量使用编译期 shape/stride，使编译器能在编译期完成所有地址计算，生成的 PTX 中只有真正的数据搬运和计算指令。

##### Layout 代数——核心操作

CuTe 的强大之处在于一组**闭合的 layout 变换操作**，每个操作的输入和输出都是 Layout，可以任意链式组合：

**1. `logical_divide`——逻辑分块**

将一个 layout 按指定的 tile shape 分割，结果是一个层次化 layout，外层索引 tile 内坐标，内层索引 tile 编号：

```cpp
auto layout = make_layout(make_shape(16, 32));  // 16x32 矩阵

// 按 (4, 8) 分块 → 结果 shape: ((4,8), (4,4))
//   第一维度 (4,8)：tile 内的行/列坐标
//   第二维度 (4,4)：tile 在矩阵中的行/列编号
auto tiled = logical_divide(layout, make_shape(4, 8));
```

**2. `local_tile`——提取特定 tile**

`logical_divide` 之后，用坐标索引取出特定的 tile：

```cpp
// 取第 (2, 1) 个 tile：从 16x32 矩阵中取行 8~11、列 8~15 的 4x8 子矩阵
auto tile_2_1 = local_tile(tensor_A,
    make_shape(Int<4>{}, Int<8>{}),   // tile shape
    make_coord(2, 1));                // tile 坐标
// tile_2_1 的 shape 是 (4, 8)，指向原始数据的对应区域
```

**3. `composition`——Layout 复合**

两个 layout 可以复合成一个新的 layout，数学上就是函数复合 $L_1 \circ L_2$：

```cpp
// L2 将逻辑坐标映射到中间索引，L1 再将中间索引映射到物理偏移
auto composed = composition(L1, L2);
// composed(coord) = L1(L2(coord))
```

这是实现 **Swizzle** 的基础——swizzle 就是在普通 layout 上复合一个 XOR 变换。

**4. `complement`——互补 Layout**

给定一个将 thread ID 映射到数据元素的 layout（"每个线程访问哪个元素"），`complement` 计算出剩余未被覆盖的元素的 layout（"还有哪些元素需要被访问"）。这在计算 thread-value（TV）分解时至关重要。

##### Thread-Value (TV) 分解——线程映射的统一框架

GPU 编程的核心问题之一是：**128 个线程要协作处理一个 128x32 的 tile，每个线程负责哪些元素？**

CUTLASS 2.x 为每种情况（GMEM 加载、SMEM 写入、MMA 操作）编写不同的映射逻辑。CuTe 用一个统一的框架解决：**TV Layout**。

一个 TV Layout 是一个 `(Thread, Value)` → `(M, K)` 的映射：
- **Thread 维度**：thread ID（0 ~ NumThreads-1）
- **Value 维度**：每个线程负责的多个元素的索引

```cpp
// 一个 128 线程的向量化加载策略：每个线程用 LDG.128 加载 8 个 FP16 元素
// 总共处理 128 * 8 = 1024 个元素 = 128x8 的 tile 的一行
auto copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{};

// 构造 TiledCopy：128 个线程，排列成 (32, 4) 的线程网格
// 每个线程每次处理 (1, 8) 个元素（LDG.128 = 8 个 FP16）
auto tiled_copy = make_tiled_copy(
    copy_atom,
    Layout<Shape<_32, _4>>{},    // Thread layout: 32x4 的线程排布
    Layout<Shape<_1, _8>>{}      // Value layout: 每线程 1x8 个元素
);
// 总覆盖面积: (32*1, 4*8) = (32, 32) —— 恰好是一个 32x32 的 SMEM tile
```

同样，MMA 操作也有 TV 分解：

```cpp
// Ampere mma.sync.m16n8k16 的 TV 分解
auto mma_atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{};

// 扩展成 TiledMma：4 个 warp（128 线程），覆盖 (64, 64, 16) 的 tile
auto tiled_mma = make_tiled_mma(
    mma_atom,
    Layout<Shape<_2, _2, _1>>{},    // Atom layout: 2x2x1 个 atom 排列
    Tile<_64, _64, _16>{}           // 期望的 tile 大小
);
// 每次调用计算 (64, 64) @ (64, 16)^T 的部分结果
```

TV 分解的优势是**一致性**：无论是 `cp.async` 从 GMEM→SMEM 的拷贝、`ldmatrix` 从 SMEM→Register 的加载、还是 `mma.sync` 的矩阵乘法，都用 `(Thread, Value) → (M, K/N)` 的 layout 来描述。这使得验证数据在不同阶段的映射是否匹配变得简单——只需要检查两个 TV layout 的兼容性。

##### Swizzle——用 Layout 代数消除 Bank Conflict

Shared memory 有 32 个 bank，bank ID = `(地址 / 4字节) % 32`。当同一 warp 的多个线程访问同一 bank 时，发生 bank conflict。

CuTe 用 **Swizzle** 函数来消除 bank conflict，其本质是一个位操作变换。CuTe 的 Swizzle 定义为三个参数 `Swizzle<B, M, S>`：

```
swizzled_offset = offset ^ ((offset >> S) & mask)
其中 mask = ((1 << B) - 1) << M
```

- `B`：参与 XOR 的位数（swizzle 的"宽度"）
- `M`：mask 的起始位位置
- `S`：右移量（哪些高位用来 XOR 低位）

```cpp
// 常见的 shared memory swizzle 配置
// 对于 FP16 (2 bytes)，32x32 的 SMEM tile：
auto smem_layout = composition(
    Swizzle<3, 3, 3>{},                         // XOR 变换
    make_layout(make_shape(Int<32>{}, Int<32>{}),
                make_stride(Int<32>{}, Int<1>{}))  // 基础行主序 layout
);
// Swizzle<3,3,3> 表示：用 row 的低 3 位 XOR col 的高 3 位
// 效果：同一列的相邻行会映射到不同的 bank → 消除列访问时的 bank conflict
```

这比 CUTLASS 2.x 手动编写 swizzle 特化要优雅得多——swizzle 只是 layout 的一次 `composition`，与其他所有 layout 操作完全兼容。

#### 3.2.4 实战：CuTe GEMM 深度解析

前面介绍了 CuTe 的 Layout 代数、TV 分解和 Swizzle 等核心概念，但它们如何组合成一个完整的高性能 GEMM kernel？本节将通过一个生产级的 CuTe GEMM 实现，从全局数据流到每一行代码的设计意图，完整呈现 CUTLASS 3.x 的工程哲学。
##### 完整代码

```cpp
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

using namespace cute;

// ==================== 配置参数 ====================
struct GemmConfig {
    using T = half_t;
    using AccT = float;

    // 三级 Tiling 参数
    static constexpr int kBlockM  = 128;   // CTA tile M
    static constexpr int kBlockN  = 128;   // CTA tile N
    static constexpr int kBlockK  = 32;    // CTA tile K（每次主循环迭代处理的 K 深度）
    static constexpr int kStages  = 3;     // 流水线深度

    // MMA atom: Ampere mma.sync.aligned.m16n8k16.f32.f16.f16.f32
    // TN 表示 A 行主序(T)、B 列主序(N) 进入 MMA
    using MMAOp = SM80_16x8x16_F32F16F16F32_TN;

    // TiledMMA: 把 MMA atom 扩展到覆盖整个 CTA tile
    //   Layout<Shape<_2, _2, _1>>: 4 个 warp，M 方向 2 个，N 方向 2 个
    //   Tile<_128, _128, _16>: 期望覆盖 128×128×16 的子问题
    //   → 每个 warp 负责 64×64×16
    using TiledMMA = decltype(
        make_tiled_mma(MMAOp{},
                       Layout<Shape<_2, _2, _1>>{},
                       Tile<_128, _128, _16>{})
    );

    // G→S: cp.async 128-bit (CACHEGLOBAL 策略，不污染 L1)
    using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>;
    // S→R: ldmatrix 指令（专为 Tensor Core 设计的 SMEM→Register 搬运）
    using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using S2RCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

    // 线程数 = TiledMMA 需要的线程数 = 4 warps × 32 = 128
    static constexpr int kThreadNum = size(TiledMMA{});
};

// ==================== 共享内存布局 ====================
static constexpr int kBlockM  = GemmConfig::kBlockM;
static constexpr int kBlockN  = GemmConfig::kBlockN;
static constexpr int kBlockK  = GemmConfig::kBlockK;
static constexpr int kStages  = GemmConfig::kStages;

// SmemLayoutAtom: 一个 128×32 的 swizzled 行主序布局
// Swizzle<3,3,3> 消除 ldmatrix 列访问时的 bank conflict
using SmemLayoutAtom = decltype(
    composition(Swizzle<3, 3, 3>{},
                make_layout(make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                            make_stride(Int<kBlockK>{}, Int<1>{})))
);
//维度 0：kBlockM
//CTA 的 A-tile 在 M 方向的大小（多少行）。

//维度 1：kBlockK
//CTA 每次 mainloop 迭代吃的 K 分块大小（多少列 / K-slice 宽度）。

//维度 2：kStages
//pipeline stage 维度：有 kStages 份互相独立的 buffer（常见是 2 或 3），每一份都存一块“swizzled 的 A tile”。 
using SmemLayoutA = decltype(
    tile_to_shape(SmemLayoutAtom{},
                  make_shape(Int<kBlockM>{}, Int<kBlockK>{}, Int<kStages>{}))
);
using SmemLayoutB = decltype(
    tile_to_shape(SmemLayoutAtom{},
                  make_shape(Int<kBlockN>{}, Int<kBlockK>{}, Int<kStages>{}))
);

// ==================== Kernel 实现 ====================
template <typename Config>
__global__ void cute_gemm_kernel(
    const void* __restrict__ Aptr,
    const void* __restrict__ Bptr,
    void* __restrict__ Cptr,
    int M, int N, int K
) {
    using T    = typename Config::T;
    using AccT = typename Config::AccT;

    constexpr int kBlockM = Config::kBlockM;
    constexpr int kBlockN = Config::kBlockN;
    constexpr int kBlockK = Config::kBlockK;
    constexpr int kStages = Config::kStages;

    int tid = threadIdx.x;

    // ---- 全局内存 Tensor ----
    // A: (M, K) row-major → stride = (K, 1)
    Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(Aptr)),
                            make_shape(M, K), make_stride(K, Int<1>{}));
    // B: (N, K) row-major → host 端做了 B.t().contiguous()，符合 TN 约定
    Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(Bptr)),
                            make_shape(N, K), make_stride(K, Int<1>{}));
    // C: (M, N) row-major → stride = (N, 1)
    Tensor mC = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(Cptr)),
                            make_shape(M, N), make_stride(N, Int<1>{}));

    // ---- CTA Tiling: 从全局矩阵中提取当前 block 负责的子矩阵 ----
    // cta_tiler 描述了三个维度的 tile 大小
    auto cta_tiler = make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
    // blockIdx.y → M 方向，blockIdx.x → N 方向，_ → K 方向（全部保留，主循环遍历）
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
    // Step<_1, X, _1>: 从 A(M,K) 中取 M 和 K 维度的 tile，跳过 N
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    // Step<X, _1, _1>: 从 B(N,K) 中取 N 和 K 维度的 tile，跳过 M
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    // Step<_1, _1, X>: 从 C(M,N) 中取 M 和 N 维度的 tile，跳过 K
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    // ---- 共享内存分配 ----
    extern __shared__ char smem_buf[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_buf)),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(
                            smem_buf + cosize(SmemLayoutA{}) * sizeof(T))),
                            SmemLayoutB{});

    // ---- G→S Copy: 用 cp.async 128-bit 从全局内存搬到共享内存 ----
    auto g2s_copy = make_tiled_copy(
        typename Config::G2SCopyAtom{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},   // 128 线程排成 32×4
        Layout<Shape< _1, _8>>{}                     // 每线程搬 1×8 个 half_t
    );
    auto g2s_thr_copy_A = g2s_copy.get_slice(tid);
    Tensor tAgA_g = g2s_thr_copy_A.partition_S(gA);  // 当前线程在 GMEM 中的视图
    Tensor tAsA_s = g2s_thr_copy_A.partition_D(sA);  // 当前线程在 SMEM 中的视图

    auto g2s_thr_copy_B = g2s_copy.get_slice(tid);
    Tensor tBgB_g = g2s_thr_copy_B.partition_S(gB);
    Tensor tBsB_s = g2s_thr_copy_B.partition_D(sB);

    // ---- TiledMMA + 寄存器 Fragment ----
    typename Config::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // partition_fragment_A/B: 在寄存器中分配 MMA 操作数空间
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));   // A 的寄存器 fragment
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));   // B 的寄存器 fragment
    Tensor tCrC = thr_mma.partition_fragment_C(gC);            // C 的 FP32 累加器
    clear(tCrC);  // 初始化为 0

    // ---- S→R Copy: 用 ldmatrix 从共享内存搬到寄存器 ----
    // make_tiled_copy_A: 自动构造与 TiledMMA 兼容的 S→R copy
    auto s2r_copy_A = make_tiled_copy_A(typename Config::S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_slice(tid);
    Tensor tCsA = s2r_thr_copy_A.partition_S(sA);       // SMEM 中的 copy 视图
    Tensor tCrA_view = s2r_thr_copy_A.retile_D(tCrA);   // 寄存器的 copy 视图
    // tCrA 和 tCrA_view 是同一块寄存器的两个"视角"

    auto s2r_copy_B = make_tiled_copy_B(typename Config::S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_slice(tid);
    Tensor tCsB = s2r_thr_copy_B.partition_S(sB);
    Tensor tCrB_view = s2r_thr_copy_B.retile_D(tCrB);

    // ==================== 软件流水线 ====================
    int num_k_tiles = ceil_div(K, kBlockK);
    constexpr int num_mma_k = size<2>(tCrA);   // kBlockK / MMA_K = 32/16 = 2

    // ---- Pipeline Fill: 预加载前 (kStages-1) 个 tile ----
    CUTE_UNROLL
    for (int stage = 0; stage < kStages - 1; ++stage) {
        if (stage < num_k_tiles) {
            copy(g2s_copy, tAgA_g(_, _, _, stage), tAsA_s(_, _, _, stage));
            copy(g2s_copy, tBgB_g(_, _, _, stage), tBsB_s(_, _, _, stage));
        }
        cp_async_fence();   // 每个 stage 作为独立的 fence group
    }

    // ---- Main Loop: 边算边加载 ----
    int smem_pipe_read  = 0;              // 当前计算的 stage
    int smem_pipe_write = kStages - 1;    // 下一次写入的 stage

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // Step 1: 等待当前 stage 数据就绪
        cp_async_wait<kStages - 2>();     // wait<1>: 允许 1 组仍在飞行
        __syncthreads();

        // Step 2: 异步加载下一个 tile（如果还有）
        int next_tile = k_tile + kStages - 1;
        if (next_tile < num_k_tiles) {
            copy(g2s_copy, tAgA_g(_, _, _, next_tile),
                           tAsA_s(_, _, _, smem_pipe_write));
            copy(g2s_copy, tBgB_g(_, _, _, next_tile),
                           tBsB_s(_, _, _, smem_pipe_write));
        }
        cp_async_fence();

        // Step 3: S→R (ldmatrix) + MMA 计算
        CUTE_UNROLL
        for (int k_inner = 0; k_inner < num_mma_k; ++k_inner) {
            copy(s2r_copy_A, tCsA(_, _, k_inner, smem_pipe_read),
                             tCrA_view(_, _, k_inner));
            copy(s2r_copy_B, tCsB(_, _, k_inner, smem_pipe_read),
                             tCrB_view(_, _, k_inner));
        }
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);   // tCrC += tCrA @ tCrB

        // Step 4: 推进环形缓冲区
        smem_pipe_read  = (smem_pipe_read  + 1) % kStages;
        smem_pipe_write = (smem_pipe_write + 1) % kStages;
        __syncthreads();
    }

    // ==================== Epilogue: FP32 → FP16, 写回 GMEM ====================
    Tensor tCgC = thr_mma.partition_C(gC);            // C 在 GMEM 中的当前线程视图
    Tensor tCrC_out = make_tensor_like<T>(tCrC);      // 分配 FP16 寄存器
    copy(tCrC, tCrC_out);                             // FP32 → FP16 截断转换
    copy(AutoVectorizingCopy{}, tCrC_out, tCgC);      // 自动向量化写回
}

// ==================== Host 端 PyTorch 接口 ====================
torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().type() == torch::kCUDA, "A must be on CUDA");
    TORCH_CHECK(B.device().type() == torch::kCUDA, "B must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // B: (K,N) row-major → Bt: (N,K) row-major，符合 TN 约定
    auto Bt = B.t().contiguous();
    auto C = torch::empty({M, N}, A.options());

    using Config = GemmConfig;

    dim3 block(Config::kThreadNum);   // 128 threads
    dim3 grid(
        (N + Config::kBlockN - 1) / Config::kBlockN,   // N 方向
        (M + Config::kBlockM - 1) / Config::kBlockM    // M 方向
    );

    // 共享内存: A(24KB) + B(24KB) = 48KB
    constexpr int smem_size =
        cosize(SmemLayoutA{}) * sizeof(half_t) +
        cosize(SmemLayoutB{}) * sizeof(half_t);

    // 超过 48KB 需要显式请求
    if constexpr (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            cute_gemm_kernel<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    cute_gemm_kernel<Config><<<grid, block, smem_size>>>(
        A.data_ptr(), Bt.data_ptr(), C.data_ptr(), M, N, K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
```

```
[Kernel-1] 构造全局视图 mA(M,K), mB(N,K), mC(M,N)（gmem_ptr + row-major layout）
[Kernel-2] CTA 切块：用 blockIdx 选当前 CTA 的 global tile 视图 gA(BM,BK,ktile), gB(BN,BK,ktile), gC(BM,BN)
[Kernel-3] 动态共享内存切分：sA(BM,BK,stages), sB(BN,BK,stages)（smem_ptr + swizzled SmemLayout）
[Kernel-4] 构造 G2S tiled_copy(cp.async 128b) 并对每线程 partition：tAgA_g/tAsA_s 与 tBgB_g/tBsB_s（thread-local GMEM/SMEM copy 视图）
[Kernel-5] 构造 tiled_mma(mma.sync) 并对每线程分配寄存器 fragment：tCrA,tCrB,tCrC；clear(tCrC)
[Kernel-6] 构造 S2R tiled_copy_A/B(ldmatrix) 并对每线程 partition：tCsA/tCsB（SMEM 读视图）与 tCrA_view/tCrB_view=retile_D(tCrA/tCrB)（REG 写视图）
[Kernel-7] Pipeline 预热：for stage in [0..kStages-2] 若 stage<num_k_tiles 则 cp.async copy(gA→sA[stage]) 与 copy(gB→sB[stage])；cp_async_fence() 提交一组
[Kernel-8] 主循环：for k_tile in [0..num_k_tiles-1] 执行 cp_async_wait(kStages-2)+__syncthreads() 等待当前读 stage ready
[Kernel-9] 主循环并行预取：计算 next_tile=k_tile+kStages-1；若存在则 cp.async copy(next_tile 的 gA/gB → 写入 sA/sB[smem_pipe_write])；cp_async_fence()
[Kernel-10] 主循环计算：for k_inner in [0..BK/16-1] 用 ldmatrix 把 sA/sB[smem_pipe_read] → tCrA_view/tCrB_view；随后 gemm(tiled_mma): tCrC += tCrA @ tCrB
[Kernel-11] 推进环形 stage：smem_pipe_read=(read+1)%kStages；smem_pipe_write=(write+1)%kStages；__syncthreads()
[Kernel-12] Epilogue：tCgC=partition_C(gC) 得到本线程 GMEM 写回视图；tCrC_out(FP16) = convert(tCrC FP32→FP16)；向量化 copy(tCrC_out→tCgC)
```

##### CuTe 核心 API 速查表

| API | 作用 | 返回 |
|-----|------|------|
| `make_tensor(ptr, layout)` | 创建 tensor | Tensor |
| `make_layout(shape, stride)` | 创建布局 | Layout |
| `local_tile(tensor, tiler, coord, step)` | 提取子 tile | Tensor |
| `make_tiled_copy(atom, thr_layout, val_layout)` | 创建协作拷贝 | TiledCopy |
| `make_tiled_mma(atom, thr_layout, val_tile)` | 创建协作 MMA | TiledMMA |
| `partition_S(tensor)` / `partition_D(tensor)` | 按 copy 分区（Source/Dest） | Tensor |
| `partition_fragment_A/B/C(tensor)` | 按 MMA 分区（寄存器 fragment） | Tensor |
| `retile_D(tensor)` | 重新分区已有寄存器（零开销视图转换） | Tensor (alias) |
| `copy(copy_op, src, dst)` | 执行拷贝 | void |
| `gemm(mma, D, A, B, C)` | 执行 MMA: D = A × B + C | void |
| `cp_async_fence()` | 标记 async 组边界 | void |
| `cp_async_wait<N>()` | 等待到 ≤N 组未完成 | void |
| `composition(swizzle, layout)` | 应用 swizzle 变换 | Layout |
| `tile_to_shape(layout, shape)` | 复制 tile 到目标 shape | Layout |
| `cosize(layout)` | 布局的 co-domain 大小（实际占用元素数） | int |

##### 全局视角：数据流与瓶颈

计算 `C = A × B`（FP16 计算、FP32 累加、FP16 输出）在 GPU 上需要同时解决三个瓶颈：

| 瓶颈 | 根因 | CUTLASS 的解法 |
|------|------|---------------|
| **全局内存带宽** | HBM 延迟 ~400 cycles | 多级 Tiling + 数据复用 |
| **共享内存 Bank Conflict** | 32 bank × 4B，同 bank 串行 | Swizzle 地址重排 |
| **指令发射 / 计算吞吐** | MMA 单元需要持续喂饱 | 软件流水线，重叠 load 与 compute |

整个 kernel 的数据流路径：

```
Global Memory (HBM)
    │  cp.async 128-bit   ← TiledCopy G→S
    ▼
Shared Memory (SMEM)
    │  ldmatrix            ← TiledCopy S→R
    ▼
Register File
    │  mma.sync            ← TiledMMA
    ▼
Accumulator (FP32 regs)
    │  convert + store     ← Epilogue
    ▼
Global Memory (HBM)
```

每一步都有专门的硬件指令和 CuTe 抽象与之对应。理解这条数据流是理解整个 kernel 的关键。

##### 三级 Tiling 架构

CUTLASS 的核心设计模式是**三级 Tiling**——把一个巨大的矩阵乘法层层分解：

```
Level 1: CTA Tile (Thread Block)
    问题：整个 M×N 的 C 矩阵太大
    解法：每个 thread block 负责 128×128 的 C 子块
    K 维度以 kBlockK=32 为步长循环

Level 2: Warp Tile
    问题：128×128 对单个 warp 还是太大
    解法：TiledMMA 自动把 block tile 划分给多个 warp
    Layout<Shape<_2, _2, _1>>  → 4 个 warp，M 方向 2 个，N 方向 2 个

Level 3: MMA Instruction
    问题：每个 warp 需要执行实际的矩阵乘
    解法：SM80_16x8x16 MMA atom，每次算 16×8 的 C 子块，消耗 16 深的 K
```

**关键参数关系**：

```
Block Tile:    128 × 128 × 32  (kBlockM × kBlockN × kBlockK)
MMA Atom:       16 ×   8 × 16  (MMA_M × MMA_N × MMA_K)
Thread Layout:   2 ×   2 ×  1  (warp 在 M, N, K 方向的复制次数)
Value Tile:     64 ×  64 × 16  (每个 warp 覆盖的 M, N, K 范围)

验证覆盖完整性：
  M 方向: 64 × 2 = 128 = kBlockM ✓
  N 方向: 64 × 2 = 128 = kBlockN ✓
  K 方向: 16 × 1 =  16 (MMA_K，内层循环需迭代 32/16=2 次) ✓
```

`make_tiled_mma` 做的事情就是把这三层关系编码成一个对象，后续的 `partition_fragment_A/B/C` 能自动算出每个线程负责的寄存器片段。

##### TN 布局约定

这是 CUTLASS 最容易让人困惑的地方。理解它需要区分**存储布局**和 **MMA 期望布局**。

MMA 指令 `SM80_16x8x16_F32F16F16F32_TN` 的含义：
- **T (Transposed)**：A 操作数以行优先（row-major）方式在寄存器中排列
- **N (Normal)**：B 操作数以列优先（column-major）方式在寄存器中排列

代码中的对应：

```cpp
// A 存储: (M, K) row-major，stride = (K, 1)
// → 直接符合 "T"，不需要额外转置

// B 存储: PyTorch 给的是 (K, N) row-major
// → kernel 需要 (N, K) row-major（每行 N 对应一个 output column）
// → 所以 host 端做了 B.t().contiguous()
// → 然后声明为 make_shape(N, K), make_stride(K, 1)
```

为什么用 TN 而不是 NN 或 TT？因为当 A 是 row-major、B 也是 row-major（经转置后变成 (N,K)）时，TN 恰好匹配。这是 CUTLASS 最常用的路径，大多数优化都针对它做过。

##### TiledCopy 与 TiledMMA 详解

**TiledCopy：数据搬运的分工方案**

```cpp
auto g2s_copy = make_tiled_copy(
    G2SCopyAtom{},                              // 硬件指令：cp.async 128-bit
    Layout<Shape<_32, _4>, Stride<_4, _1>>{},   // 线程排布：32行×4列
    Layout<Shape<_1, _8>>{}                      // 每线程搬 1×8 个元素
);
```

这个配置的物理意义：
- 128 个线程排成 32×4 的网格
- 每个线程用 `cp.async` 一次搬 128 bit = 8 个 `half_t`
- 单次调用覆盖 32×(4×8) = 32×32 = 1024 个元素
- 我们的 tile 是 128×32，所以 `partition_S` 会把它分成多个 "CPY_M" 块，每次 `copy` 调用在循环中搬完所有块

如果画图的话这种layout长下面这样
```
Shape = 32 rows × 4 cols

          c=0  c=1  c=2  c=3
r=0        0    1    2    3
r=1        4    5    6    7
r=2        8    9   10   11
r=3       12   13   14   15
r=4       16   17   18   19
r=5       20   21   22   23
r=6       24   25   26   27
r=7       28   29   30   31   ← warp0（tid 0..31）刚好覆盖 r=0..7 的整块 8×4
r=8       32   33   34   35   ← warp1 开始
...

```

作为对比的话Layout<Shape<32,4>, Stride<1,32>>长下面这样
```
          c=0  c=1  c=2   c=3
r=0        0   32   64    96
r=1        1   33   65    97
r=2        2   34   66    98
r=3        3   35   67    99
r=4        4   36   68   100
r=5        5   37   69   101
r=6        6   38   70   102
r=7        7   39   71   103
...
r=31      31   63   95   127

```

**partition_S 与 partition_D**

```cpp
Tensor tAgA_g = g2s_thr_copy_A.partition_S(gA);  // Source: 全局内存
Tensor tAsA_s = g2s_thr_copy_A.partition_D(sA);  // Dest:   共享内存
```

partition 后的 tensor 维度变为 `(CPY, CPY_M, CPY_K, ...)`：
- **CPY**：一条 copy 指令搬的元素数（8 个 half_t）
- **CPY_M, CPY_K**：这个线程需要在 M 和 K 方向上迭代的次数
- 末尾维度：K-tile 索引或 stage 索引

调用 `copy(g2s_copy, src, dst)` 时，CuTe 会自动展开 CPY_M × CPY_K 的循环。

**TiledMMA 的双重视图技巧**

这是一个精妙的设计：

```cpp
// MMA 视图（gemm() 使用）
Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));

// Copy 视图（copy() 使用）
Tensor tCrA_view = s2r_thr_copy_A.retile_D(tCrA);
```

**tCrA** 和 **tCrA_view 指向完全相同的寄存器**，但布局不同：
- `tCrA` 的布局对齐 MMA 指令的操作数格式
- `tCrA_view` 的布局对齐 `ldmatrix` 的输出格式

`retile_D` 就是做这个布局转换的——它不移动数据，只改变逻辑视图。这样 `copy(s2r, src, tCrA_view)` 把数据搬进寄存器后，`gemm(mma, tCrC, tCrA, tCrB, tCrC)` 能直接用，无需任何寄存器内的数据重排。

##### 软件流水线（Pipeline）

这是整个 kernel 性能的关键所在。

**为什么需要流水线？**

`cp.async` 从全局内存搬数据到共享内存的延迟大约 **400-800 cycles**。如果采用朴素方式：

```
load A[0], B[0] to smem     ← 等 400 cycles
__syncthreads()
compute on smem data         ← 计算 ~100 cycles
```

计算单元有 80% 的时间在空等！

**流水线的核心思想——重叠**：

```
朴素方式:
  [===LOAD===][COMPUTE][===LOAD===][COMPUTE][===LOAD===][COMPUTE]

流水线 (3 stage):
  [===LOAD 0===]
       [===LOAD 1===]
            [===LOAD 2===][===LOAD 3===][===LOAD 4===]
  [          ][COMPUTE 0 ][COMPUTE 1   ][COMPUTE 2   ][COMPUTE 3]...

当 GPU 在计算 tile i 时，tile i+2 的加载已经在异步进行！
```

**kStages = 3 的含义**——共享内存中同时存在 3 份 tile：

```
SMEM 布局:
  ┌──────────┬──────────┬──────────┐
  │ Stage 0  │ Stage 1  │ Stage 2  │
  │ 128×32   │ 128×32   │ 128×32   │
  └──────────┴──────────┴──────────┘
     ↑ read     ↑ read     ↑ write
     (正在计算)  (已加载)   (正在加载)
```

三个 stage 是延迟隐藏和资源占用之间的经典平衡点：
- **2 stage**：只能隐藏一个 tile 的延迟，不够
- **3 stage**：能隐藏两个 tile 的延迟，通常足够
- **4+ stage**：共享内存占用过大，限制 occupancy

**cp.async 的关键特性**：

`cp.async` 是 SM80 引入的异步拷贝指令，它有两个重要特性：

1. **非阻塞**：发射后线程立即继续执行，数据在后台搬运
2. **绕过寄存器**：数据直接从全局内存到共享内存，不经过寄存器文件

配合 `cp_async_fence()` 和 `cp_async_wait<N>()`：

```cpp
cp_async_fence();       // 插入一个"栅栏"，标记到此为止的所有 cp.async 为一组
cp_async_wait<N>();     // 等待直到最多还剩 N 组未完成
```

**Swizzle 与多 Stage 的配合**：

```cpp
using SmemLayoutA = decltype(
    tile_to_shape(SmemLayoutAtom{},
                  make_shape(Int<128>{}, Int<32>{}, Int<3>{}))
);
```

`tile_to_shape` 在第三维度上做 tile 复制——相当于在共享内存里开了 3 块独立的 128×32 缓冲区，每块都保持 swizzle 布局。这就是流水线的硬件基础。

##### 主循环的精密编排

这是整个 kernel 中逻辑最精密的部分。

**流水线填充（Pipeline Fill）**

```cpp
for (int stage = 0; stage < kStages - 1; ++stage) {    // stage = 0, 1
    if (stage < num_k_tiles) {
        copy(g2s_copy, tAgA_g(_, _, _, stage), tAsA_s(_, _, _, stage));
        copy(g2s_copy, tBgB_g(_, _, _, stage), tBsB_s(_, _, _, stage));
    }
    cp_async_fence();    // 每个 stage 的加载作为独立的一组
}
```

填充阶段预加载 `kStages - 1 = 2` 个 tile。每个 tile 的加载后面跟一个 fence，这样后续可以用 `cp_async_wait` 精确等待特定组。

**主循环四步曲**

每次迭代执行四个严格有序的步骤：

```cpp
int smem_pipe_read  = 0;          // 正在计算的 stage
int smem_pipe_write = kStages - 1; // 下一次写入的 stage
```

**步骤 1：等待当前 stage 数据就绪**

```cpp
cp_async_wait<kStages - 2>();   // wait<1>: 最多 1 组未完成
__syncthreads();
```

`cp_async_wait<1>` 的语义是"等到未完成的 fence group 数量 ≤ 1"。由于我们在每次迭代中都发射一个新的 group，这保证了 `smem_pipe_read` 对应的 group 已经完成。

为什么是 `kStages - 2 = 1` 而不是 0？因为我们希望**允许一组加载仍在飞行中**——这就是流水线的精髓：

```
当前状态（k_tile = 0 时）:
  group 0 (stage 0): ✅ 完成 (等待保证)
  group 1 (stage 1): 🔄 可能还在传输 (允许)
  → 我们可以安全地计算 stage 0 的数据
```

**步骤 2：异步加载下一个 tile**

```cpp
int next_tile = k_tile + kStages - 1;
if (next_tile < num_k_tiles) {
    copy(g2s_copy, tAgA_g(_, _, _, next_tile), tAsA_s(_, _, _, smem_pipe_write));
    copy(g2s_copy, tBgB_g(_, _, _, next_tile), tBsB_s(_, _, _, smem_pipe_write));
}
cp_async_fence();
```

`smem_pipe_write` 指向的 stage 是**已经被计算过的**（或还没用过的），可以安全覆写。

**步骤 3：S→R 拷贝 + MMA 计算**

```cpp
for (int k_inner = 0; k_inner < num_mma_k; ++k_inner) {   // 0, 1 (32/16=2)
    // ldmatrix: 从共享内存搬到寄存器
    copy(s2r_copy_A, tCsA(_, _, k_inner, smem_pipe_read), tCrA_view(_, _, k_inner));
    copy(s2r_copy_B, tCsB(_, _, k_inner, smem_pipe_read), tCrB_view(_, _, k_inner));
}
gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
```

`kBlockK = 32` 被 MMA 的 K 维度（16）切成 `num_mma_k = 2` 块。`ldmatrix` 和 `mma` 交替执行允许编译器做指令级流水线——当 MMA 单元在执行乘加时，`ldmatrix` 可以在 load 单元上准备下一块数据。

**步骤 4：推进环形缓冲区**

```cpp
smem_pipe_read  = (smem_pipe_read  + 1) % kStages;
smem_pipe_write = (smem_pipe_write + 1) % kStages;
__syncthreads();
```

`% kStages` 实现了环形缓冲区。`__syncthreads()` 确保所有线程完成当前 stage 的读取后才可能覆写它。

**完整时间线**（K=160, kBlockK=32, kStages=3, num_k_tiles=5）：

```
操作          │ k=0      │ k=1      │ k=2      │ k=3      │ k=4
─────────────┼──────────┼──────────┼──────────┼──────────┼─────────
Fill (预填充) │ G→S[0,1] │          │          │          │
Wait         │ wait≤1   │ wait≤1   │ wait≤1   │ wait≤1   │ wait≤1
Load (async) │ G→S[2]   │ G→S[3]   │ G→S[4]   │ (skip)   │ (skip)
Compute      │ stage 0  │ stage 1  │ stage 2  │ stage 0  │ stage 1
             │ S→R+MMA  │ S→R+MMA  │ S→R+MMA  │ S→R+MMA  │ S→R+MMA
read_idx     │ 0        │ 1        │ 2        │ 0        │ 1
write_idx    │ 2        │ 0        │ 1        │ 2        │ 0

飞行中的组:
k=0: group[0]✅ group[1]🔄 →发射→ group[2]🔄
k=1: group[1]✅ group[2]🔄 →发射→ group[3]🔄
k=2: group[2]✅ group[3]🔄 →发射→ group[4]🔄
k=3: group[3]✅ group[4]🔄  (无新发射)
k=4: group[4]✅             (无新发射)
```

##### Epilogue：累加器回写

MMA 在 FP32 寄存器中累加（精度需要），但输出矩阵 C 是 FP16（存储效率）：

```cpp
Tensor tCgC = thr_mma.partition_C(gC);         // 全局内存的 MMA 分区视图
Tensor tCrC_out = make_tensor_like<T>(tCrC);    // 分配 FP16 寄存器空间
copy(tCrC, tCrC_out);                           // FP32 → FP16 截断
copy(AutoVectorizingCopy{}, tCrC_out, tCgC);    // 写回全局内存
```

`thr_mma.partition_C(gC)` 返回的 tensor 只包含**当前线程负责的那些 C 元素**在全局内存中的位置，形状和 `tCrC` 一一对应，所以直接 copy 就行。`AutoVectorizingCopy` 让 CuTe 自动选择最宽的向量化 store 指令（如 `ST.128`）。

> **💡 Tip: 生产级 Epilogue 的额外考虑**
>
> 本代码是简化版。完整的 CUTLASS epilogue 还需要处理：
> 1. **边界条件**：当 M 或 N 不是 block tile 的整数倍时，需要 predication 避免越界写
> 2. **Epilogue Fusion**：bias 加法、activation（ReLU/GELU）、residual connection 等可以在写回时一起做，避免额外的 kernel launch（详见第 7 章）
> 3. **Split-K**：当 K 很大而 M×N 较小时，可以让多个 block 分担 K 维度，最后做 reduction

### 3.3 Triton 设计哲学与核心抽象

[Triton](https://github.com/triton-lang/triton) 是由 OpenAI 开发的基于 Python 的 GPU 编程语言和编译器。与 CUTLASS 的 C++ 模板方法截然不同，Triton 采用了"**编写 block-level 伪代码，编译器生成高效 GPU 代码**"的设计哲学。

#### 3.3.1 Block-Level 编程模型

Triton 的核心创新在于其**编程粒度**：

- 在 CUDA 中，开发者编写单个 **thread** 的行为，然后启动成千上万个 thread
- 在 Triton 中，开发者编写单个 **program instance**（对应一个 threadblock）的行为，操作的基本单元是 **block**（一个小的 2D tensor）

这意味着开发者不需要关心：
- 线程如何在 warp 内协作（编译器决定）
- Shared memory 何时分配、数据如何放置（编译器决定）
- Register 如何分配给各个线程（编译器决定）
- Bank conflict 如何避免（编译器的 swizzle pass 处理）
- 指令如何调度以隐藏延迟（编译器的 scheduling pass 处理）

开发者只需要描述 block-level 的算法逻辑，Triton 编译器负责将其高效地映射到硬件。

> **💡 Tip: Triton Block vs CUDA Block——名字相似，含义不同**
>
> 两者名字相似，但指代的东西完全不同：
>
> **CUDA Block (Thread Block)**：是一组线程的集合（如 256 个线程）。你需要手动管理每个线程做什么：线程索引、shared memory 分配、同步（`__syncthreads`）、内存合并访问等。层级结构：Grid → Block → Thread。一个 block 内的线程共享 shared memory，可以同步。
>
> **Triton Block (Tile / Block of Data)**：指的是一块数据（如一个 `(128, 32)` 的矩阵切片）。Triton 中没有"线程"的概念暴露给用户——你操作的是整个 tile。`BLOCK_SIZE_M`、`BLOCK_SIZE_K` 等都是描述数据分块的大小，不是线程数量。底层的线程调度、shared memory、合并访问全部由 Triton 编译器自动处理。
> **本质映射**：Triton 的一个 program（`tl.program_id(0)` 对应的一个实例）在底层会被编译成一个 CUDA thread block。Triton 编译器根据你声明的 `BLOCK_SIZE_*` 自动决定这个 CUDA block 里需要多少线程、怎么分配 shared memory、怎么排布内存访问。
>
> 简单来说：**CUDA block = 你管理线程，自己搬数据；Triton block = 你声明要处理的数据形状，编译器帮你管线程和搬数据。** 这也是为什么模板里我们只写 `tl.load` / `tl.dot` / `tl.store` 这种 tile 级别的操作，而不需要关心 `threadIdx` 或 shared memory。

#### 3.3.2 Triton Kernel 编写蓝图

任何 Triton kernel 的编写都遵循同一套固定流程。掌握这个蓝图后，无论是写 vector add、softmax 还是 Flash Attention，你只需要往蓝图的每一步填入具体的计算逻辑即可。

```
Step 1: 确定 pid → tile 的映射（"我负责输出的哪一块？"）
        │
Step 2: 根据 tile 坐标构造输入指针（tl.arange + 指针算术）
        │
Step 3: 加载数据（tl.load + mask 处理边界）
        │
Step 4: 计算（标量运算 / tl.dot / tl.sum / ...）
        │
Step 5: 存储结果（tl.store + mask）
```

**Step 1 是最关键的一步**——它决定了整个 kernel 的并行策略和数据访问模式。其余步骤都是围绕 Step 1 的结果展开的机械操作。下面用两个例子说明 Step 1 如何随问题复杂度升级。

---

**例 1：Vector Add——最简单的一维映射**

输出是一维向量，长度为 N。每个 program 处理 `BLOCK_SIZE` 个连续元素：

```
输出向量: [ ---- BLOCK_SIZE ---- | ---- BLOCK_SIZE ---- | ... | -- 剩余 -- ]
             pid=0                  pid=1                       pid=G-1

Grid 大小: G = cdiv(N, BLOCK_SIZE)
```

映射代码只需一行：

```python
pid = tl.program_id(0)                              # 我是第几个 program
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 我负责的元素下标
mask = offsets < N                                     # 最后一个 block 可能越界
```

这里没有 `pid_m` / `pid_n` 的概念——因为数据是一维的，一个 `pid` 就够了。

---

**例 2：GEMM——二维映射 + Grouped Ordering（Swizzle）**

输出是二维矩阵 C (M×N)。每个 program 负责一个 `(BLOCK_M, BLOCK_N)` 的 tile。现在需要把一维的 `pid` 映射到二维的 `(pid_m, pid_n)`：

```
输出矩阵 C:
         N 方向 →
        ┌──────────┬──────────┬──────────┬──────────┐
        │(0,0)     │(0,1)     │(0,2)     │(0,3)     │  ← 每格是一个
   M    │ BLOCK_M  │          │          │          │    (BLOCK_M × BLOCK_N)
   方   │ × BLOCK_N│          │          │          │    的 tile
   向   ├──────────┼──────────┼──────────┼──────────┤
   ↓    │(1,0)     │(1,1)     │(1,2)     │(1,3)     │
        ├──────────┼──────────┼──────────┼──────────┤
        │(2,0)     │(2,1)     │(2,2)     │(2,3)     │
        └──────────┴──────────┴──────────┴──────────┘

Grid 大小: G = cdiv(M, BLOCK_M) × cdiv(N, BLOCK_N)
```

**朴素映射（行优先）**：

```python
pid = tl.program_id(0)       # 一维 pid: 0, 1, 2, ..., G-1
pid_m = pid // num_pid_n     # 行号
pid_n = pid  % num_pid_n     # 列号
```

pid 分配结果：

```
        col0  col1  col2  col3
row0  [  0     1     2     3  ]  ← pid 0~3 同时运行，需要 A 的同一行
row1  [  4     5     6     7  ]    但 B 的 4 个不同列 → L2 cache 利用率差
row2  [  8     9    10    11  ]
```

**Grouped ordering（swizzle）**：把相邻 pid 分到一个小矩形组内，组内列优先排列：

```python
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n     # 每组的 tile 总数
group_id = pid // num_pid_in_group              # 属于哪个组
first_pid_m = group_id * GROUP_SIZE_M           # 组的起始行
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  # 组内列优先
pid_n = (pid % num_pid_in_group) // group_size_m
```

`GROUP_SIZE_M=2` 时的 pid 分配：

```
        col0  col1  col2  col3
row0  [  0     2     4     6  ]  ← pid 0,1 共享 B 的 col0
row1  [  1     3     5     7  ]    pid 0,2 共享 A 的 row0
row2  [  8    10    12    14  ]    → 同时运行的 program 共享更多数据
row3  [  9    11    13    15  ]    → L2 cache 命中率显著提高
```

---

**蓝图的价值**：一旦 Step 1 确定了 `pid_m, pid_n`（或者更简单的 `pid + offsets`），后面的步骤就是模式化的——用 `pid_m, pid_n` 算出输入指针、加载、计算、写回。**不同 kernel 之间的区别几乎只在 Step 1（怎么分 tile）和 Step 4（怎么算）**。后面的所有代码示例都会遵循这个蓝图。

---

#### 3.3.3 核心抽象与代码示例

**1. Program ID 与 Grid——工作分配**

Triton kernel 的第一步是确定当前 program instance 负责哪部分数据：

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,  # 编译期常量
):
    # 获取当前 program instance 的 ID（类似 blockIdx.x）
    pid = tl.program_id(axis=0)

    # 计算当前 block 负责的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # shape: (BLOCK_SIZE,)

    # 生成 mask 处理边界情况（最后一个 block 可能越界）
    mask = offsets < N

    # Block-level 加载：一次加载整个 block 的数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Block-level 计算
    result = x + y

    # Block-level 存储
    tl.store(out_ptr + offsets, result, mask=mask)
```

启动 kernel：

```python
import torch

N = 100000
x = torch.randn(N, device='cuda')
y = torch.randn(N, device='cuda')
out = torch.empty(N, device='cuda')

# 计算 grid 大小
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

# 启动 kernel
vector_add_kernel[grid](x, y, out, N, BLOCK_SIZE=1024)
```

**2. 指针算术——Block 级指针构造**

Triton 的指针模型是其区别于 NumPy 的关键所在。在 NumPy 中我们操作数组切片，在 Triton 中我们构造指针 block：

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    # 每个 program instance 处理一行
    row_idx = tl.program_id(0)

    # 构造列偏移向量
    col_offsets = tl.arange(0, BLOCK_SIZE)  # [0, 1, 2, ..., BLOCK_SIZE-1]

    # 构造指向当前行各元素的指针 block
    input_ptrs = input_ptr + row_idx * n_cols + col_offsets
    mask = col_offsets < n_cols

    # 加载整行数据
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Block-level softmax 计算
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # 写回
    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, result, mask=mask)
```

**3. `tl.dot()`——Block 级矩阵乘法**

`tl.dot()` 是 Triton 进行矩阵乘法的核心操作。它自动利用 Tensor Core，将两个 2D block 相乘并累加：

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 计算当前 block 负责的输出 tile 坐标
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 构造行/列偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    offs_k = tl.arange(0, BLOCK_K)                     # (BLOCK_K,)

    # 构造 A 和 B 的指针 block
    # A: (BLOCK_M, BLOCK_K), B: (BLOCK_K, BLOCK_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # 初始化 FP32 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿 K 维度迭代
    for k in range(0, K, BLOCK_K):
        # 加载 A 和 B 的当前 tile
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Block-level 矩阵乘法——自动使用 Tensor Core
        # a: (BLOCK_M, BLOCK_K), b: (BLOCK_K, BLOCK_N) → acc += a @ b
        acc = tl.dot(a, b, acc)

        # 移动指针到下一个 K-tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    # 将 FP32 累加结果转回 FP16 并写回
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
```

注意这段代码的简洁性：不到 40 行 Python 代码就完成了一个 Tensor Core GEMM kernel，而等价的 CUDA 代码通常需要 200+ 行。

**4. `tl.constexpr`——编译期常量**

所有标记为 `tl.constexpr` 的参数在编译期确定，这使得编译器可以：

- 完全展开循环（`for k in range(0, K, BLOCK_K)` 中 `BLOCK_K` 已知）
- 静态分配 shared memory 和 register
- 优化指令选择（根据 block shape 选择最佳的 Tensor Core 指令）

```python
BLOCK_SIZE: tl.constexpr  # 编译器在 JIT 编译时确定具体值
# 每个不同的 BLOCK_SIZE 值会生成一个不同的 kernel 二进制文件
```

**5. `@triton.autotune`——自动调优**

手动搜索最优的 tile size 配置既繁琐又容易遗漏好的选择。Triton 提供了内置的 autotune 装饰器：

```python
@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=3, num_warps=8
        ),
    ],
    key=['M', 'N', 'K'],  # 当这些值变化时重新搜索最优配置
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    # ... strides ...
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ... kernel 实现 ...
    pass
```

Autotune 的工作流程：
1. 首次调用时，对所有候选配置运行 benchmark
2. 选择耗时最短的配置
3. 后续调用直接使用最优配置（缓存结果按 `key` 参数索引）

#### 3.3.4 Triton 编译器流水线

理解 Triton 编译器的工作流程有助于理解其性能特征和限制：

```
Python 源代码 (@triton.jit 装饰的函数)
        │
        ▼
    Python AST 解析
        │
        ▼
    Triton IR (MLIR Dialect)
    ├─ 类型推断
    ├─ Block-level 操作语义
    └─ 尚未涉及线程映射
        │
        ▼
    Triton GPU IR
    ├─ 自动插入 shared memory 操作
    ├─ 确定线程/warp 到数据的映射
    ├─ 生成 software pipelining (根据 num_stages)
    ├─ 应用 memory coalescing 优化
    └─ Bank conflict avoidance (swizzle)
        │
        ▼
    LLVM IR (NVPTX backend)
    ├─ Register allocation
    ├─ Instruction scheduling
    └─ Peephole optimizations
        │
        ▼
    PTX Assembly
        │
        ▼
    SASS (通过 ptxas 编译)——最终的机器码
```

编译器自动完成的关键优化包括：

- **Shared memory allocation**：分析 `tl.load` / `tl.store` 模式，自动决定哪些数据缓存到 shared memory
- **Software pipelining**：根据 `num_stages` 参数，将数据加载和计算重叠执行。例如 `num_stages=3` 意味着同时有 3 个 K-tile 的数据在流水线中
- **Register tiling**：将 block-level 操作分解到每个线程的寄存器操作
- **Vectorized loads**：当检测到连续访问模式时，自动生成 128-bit 向量加载指令
- **Tensor Core lowering**：将 `tl.dot` 自动降低为适当的 `mma` 指令序列

#### 3.3.5 实战：Triton 高性能 GEMM

有了蓝图和核心抽象的知识，我们现在来看一个完整的、生产级的 Triton GEMM 实现。对比第 2 章手写 CUDA（2.3-2.6）需要手动管理共享内存、寄存器分块、bank conflict、向量化访问等底层细节（~200 行），Triton 只需要 ~60 行核心逻辑，编译器自动处理其余一切。

```python
import triton
import triton.language as tl
import torch

@triton.autotune(
    # 自动搜索最优配置（类似 CUTLASS 的 tile size 选择）
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],  # 根据矩阵尺寸重新搜索最优配置
)
@triton.jit
def gemm_kernel(
    # 矩阵指针
    A_ptr, B_ptr, C_ptr,
    # 矩阵维度
    M, N, K,
    # 矩阵 leading dimension（支持非连续布局）
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 缩放因子
    alpha,
    # 编译期常量（由 autotune 确定）
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    计算 C = alpha * A @ B
    A: (M, K), B: (K, N), C: (M, N)

    每个 Triton program instance 计算 C 中一个 (BLOCK_M, BLOCK_N) 的 tile。
    """
    # ==================== Step 1: pid → tile 映射 ====================
    # pid 是当前 program 的一维索引，需要映射到 C 矩阵的二维 tile 坐标
    pid = tl.program_id(axis=0)

    # Grouped ordering（L2 cache 友好的 tile 访问顺序）
    # 参见 3.3.2 蓝图中的 GEMM swizzle 示例
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ==================== Step 2: 构造输入指针 ====================
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # 二维指针数组：广播 (BLOCK_M,1) + (1,BLOCK_K) → (BLOCK_M, BLOCK_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ==================== Step 3+4: 加载 + 计算（沿 K 维分块累加）====================
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # FP32 累加

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k_start * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        # tl.dot 自动使用 Tensor Core、自动管理 SMEM 和寄存器 tiling
        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = (accumulator * alpha).to(tl.float16)

    # ==================== Step 5: 写回结果 ====================
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def gemm_triton(a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Triton GEMM wrapper: a (M,K) @ b (K,N) → c (M,N)"""
    assert a.shape[1] == b.shape[0], "矩阵维度不匹配"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 1D launch grid：总共 ceil(M/BLOCK_M) * ceil(N/BLOCK_N) 个 program
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        alpha,
    )
    return c
```

注意代码中的注释标注了蓝图的 5 个步骤。对比第 2 章的手写 CUDA：

| 维度               | CUDA（第 2 章手写）               | Triton（本节）               |
| ---------------- | --------------------------- | ------------------------ |
| 代码量              | ~200 行                      | ~60 行核心逻辑                |
| 抽象层级             | 线程级（`threadIdx`, `warp_id`） | Block 级（`tl.program_id`） |
| 共享内存             | 手动声明、加载、padding             | 编译器自动管理                  |
| 寄存器分块            | 手动 `reg_a/reg_b/reg_c`      | `tl.dot()` 内部自动处理        |
| Bank conflict    | 手动 +4 padding               | 编译器自动消除                  |
| Double buffering | 手动两个 buffer 交替              | `num_stages` 参数，编译器生成流水线 |
| Tensor Core      | 手动 WMMA API                 | `tl.dot()` 自动使用          |
| Tile size 调优     | 手动尝试                        | `@triton.autotune` 自动搜索  |
| 性能               | 手动优化可达 cuBLAS 90%+          | 通常达到 cuBLAS 80-90%       |

---

## 课后练习题

### 练习 1：GEMM tiling 三层结构

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">block tile、warp tile、mma tile 各自解决什么问题？</span></summary>

block tile 决定一个 thread block 负责的 C 矩阵区域，并把 A/B tile 搬到 shared memory 做复用。warp tile 把 block tile 分给多个 warp。mma tile 对应 Tensor Core 指令形状，决定寄存器 fragment 和指令发射。三层 tile 对齐后，才能同时提高数据复用和 Tensor Core 利用率。

</details>

### 练习 2：Triton tl.dot 的检查点

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">怎么确认 Triton matmul 用上 Tensor Core？</span></summary>

看输入 dtype 和 block shape 是否支持 Tensor Core，检查编译后的 PTX/SASS 是否出现 mma 或 wgmma 指令，profiling 里看 tensor pipe utilization。只写 tl.dot 不代表一定高效；shape、alignment、num_warps、num_stages 和 shared memory pipeline 都会影响最终吞吐。

</details>
