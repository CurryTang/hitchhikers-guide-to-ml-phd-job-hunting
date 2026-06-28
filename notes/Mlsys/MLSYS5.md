# CUDA Parallel Primitives: Histogram & Scan

本讲承接上一讲的Reduce Kernel，介绍两个重要的并行原语：Histogram和Scan（Prefix Sum）。

## Part 1: Histogram Kernel

https://leetgpu.com/challenges/histogramming

### 1.1 从Reduce到Histogram

上一讲我们学习了Reduce：将N个元素归约为1个值。Histogram可以看作是**多目标Reduce**：

```
Reduce:     N elements → 1 value      (所有元素归约到同一目标)
Histogram:  N elements → K bins       (元素按条件归约到K个不同目标)
```

| 对比维度 | Reduce | Histogram |
|---------|--------|-----------|
| 输出大小 | 1 | K (bins数量) |
| 写入目标 | 固定 | 数据依赖 |
| 并行难点 | 归约树设计 | **原子操作竞争** |
| Roofline | Memory-bound | Memory-bound + Atomic-bound |

核心区别：Reduce的写入目标是确定的，而Histogram的写入目标取决于输入数据值，这导致**多个线程可能同时更新同一个bin**。

### 1.2 Histogram的核心挑战：原子竞争

当数据分布集中（如大部分元素落入少数几个bin）时，原子操作严重串行化：

```
线程0 → bin[3] ─┐
线程1 → bin[3] ─┼─→ 串行执行！
线程2 → bin[3] ─┤
线程3 → bin[5] ─┘
```

### 1.3 解决方案：层次化私有化

核心思想：**减少竞争范围**，从global竞争→block内竞争→warp内竞争

```
┌─────────────────────────────────────────────────────────┐
│  Version 1: Global Atomic                               │
│  所有线程 → Global Memory (竞争最严重)                    │
├─────────────────────────────────────────────────────────┤
│  Version 2: Shared Memory Privatization                 │
│  Block内线程 → Shared Memory → Global Memory            │
│  竞争范围从全GPU缩小到单个Block (256线程)                  │
├─────────────────────────────────────────────────────────┤
│  Version 3: Warp-level + Local Accumulation             │
│  进一步减少atomic次数                                    │
└─────────────────────────────────────────────────────────┘
```

### 1.4 实现版本

#### Version 1: Naive (Baseline)

```cpp
// __global__ 修饰符：声明这是一个 GPU kernel 函数
//   - 由 CPU (host) 调用，在 GPU (device) 上执行
//   - 返回类型必须是 void
__global__ void histogram_v1_naive(
    // __restrict__ 关键字：告诉编译器这个指针是访问该内存的唯一方式
    //   - 保证 data 和 hist 指向的内存区域不重叠（no pointer aliasing）
    //   - 允许编译器进行更激进的优化（如循环展开、指令重排）
    //   - 类似于 C99 的 restrict，但在 CUDA 中使用双下划线
    const int* __restrict__ data,  // 输入数据数组（只读）
    int* __restrict__ hist,         // 输出直方图数组（读写）
    int n,                          // 输入数据的元素个数
    int num_bins                    // 直方图的 bin 数量
) {
    // 计算当前线程的全局索引
    // blockIdx.x: 当前 block 在 grid 中的索引
    // blockDim.x: 每个 block 中的线程数
    // threadIdx.x: 当前线程在 block 中的索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop 模式的步长
    // gridDim.x: grid 中 block 的总数
    // stride = 所有线程的总数，用于处理数据量大于线程数的情况
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop：每个线程处理多个元素
    // 这种模式的优点：
    //   1. 可以处理任意大小的输入数据
    //   2. 线程数可以独立于数据大小进行调优
    //   3. 保持良好的内存访问模式（相邻线程访问相邻内存）
    for (int i = idx; i < n; i += stride) {
        int bin = data[i];
        // 边界检查：确保 bin 值在有效范围内
        if (bin >= 0 && bin < num_bins) {
            // atomicAdd：原子加操作
            //   - 保证多个线程同时更新同一位置时的正确性
            //   - 缺点：当多个线程竞争同一个 bin 时会产生串行化
            //   - 这是 naive 版本的主要性能瓶颈
            atomicAdd(&hist[bin], 1);
        }
    }
}
```

**内存流动与延迟分析**

让我们逐步分析这个 kernel 中数据的内存流动过程：

1. **索引计算阶段**：`idx` 和 `stride` 这两个变量的计算完全在寄存器中完成。`blockIdx.x`、`blockDim.x`、`threadIdx.x` 和 `gridDim.x` 都是 CUDA 提供的内置变量，存储在特殊寄存器中，访问延迟极低（约 1 个时钟周期）。

2. **数据读取阶段**：`int bin = data[i]` 是整个 kernel 的第一个内存瓶颈。`data` 数组位于全局内存（Global Memory）中，访问延迟高达 400-800 个时钟周期。不过，由于我们使用了 grid-stride loop，相邻线程访问相邻内存地址，这形成了合并访问（Coalesced Access）模式。当一个 warp（32 个线程）同时访问连续的 32 个 int 时，这些请求会被合并成一次 128 字节的内存事务，大大提高了带宽利用率。读取的 `bin` 值会被存储在每个线程的寄存器中。

3. **边界检查阶段**：`if (bin >= 0 && bin < num_bins)` 的比较操作在寄存器中完成，延迟可忽略。这里可能有分支发散（Branch Divergence），但目前先忽略

4. **原子更新阶段**：`atomicAdd(&hist[bin], 1)` 是性能的核心瓶颈。这个操作涉及：
   - 首先，根据 `bin` 的值计算 `hist` 数组中对应元素的地址（寄存器操作）
   - 然后，发起一次全局内存的原子读-改-写操作
   - 原子操作本身的延迟与普通全局内存访问相当（400-800 周期），但问题在于串行化：当多个线程同时更新同一个 bin 时，这些操作必须排队执行
   - 如果数据分布不均匀（某些 bin 特别热门），竞争会更加严重，延迟可能累积到数千个周期

**延迟隐藏与性能特点**

GPU 通过大量并行线程来隐藏内存延迟。当一个 warp 等待内存访问完成时，调度器会切换到其他就绪的 warp 执行。然而，这个 naive 版本的问题在于：

- 全局内存的原子操作无法被有效隐藏，因为同一 bin 的更新必须串行
- 所有线程都在竞争同一个 `hist` 数组，形成严重的内存争用
- 当 bin 数量较少或数据分布集中时，性能会急剧下降

这就是为什么后续版本使用 Shared Memory 进行私有化：Shared Memory 的访问延迟只有约 20-30 个时钟周期，比全局内存快一个数量级，可以大幅减少原子操作的开销。

#### Version 2: Shared Memory Privatization ⭐

这是最实用的优化版本：

```cpp
__global__ void histogram_v2_shared(
    const int* __restrict__ data,
    int* __restrict__ hist,
    int n, int num_bins
) {
    extern __shared__ int s_hist[];  // 每个block的私有histogram
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Step 1: 初始化shared memory
    for (int i = tid; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Step 2: 在shared memory中累积 (block内竞争，比global快~10x)
    for (int i = idx; i < n; i += stride) {
        int bin = data[i];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();
    
    // Step 3: 归约到global memory (每个bin只需1次global atomic)
    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}
```

**为什么有效**：
- Shared memory atomic比global快约10倍
- 竞争从百万线程降到256线程/block
- 最终global atomic次数 = num_bins × num_blocks（而非n次）

**限制**：bins数量受shared memory限制（48KB → ~12K int bins）

#### Version 3: Local Accumulation (处理数据局部性)

当数据有局部性（连续元素倾向于落入相同bin）时：

```cpp
// 核心思想：利用数据的时间局部性，用寄存器累积连续相同的 bin
// 适用场景：当输入数据具有局部性（如图像像素、排序后的数据），连续元素往往落入相同 bin
// 优化原理：将多次原子操作合并为一次，减少原子操作的总次数
__global__ void histogram_v3_local(
    const int* __restrict__ data,
    int* __restrict__ hist,
    int n, int num_bins
) {
    extern __shared__ int s_hist[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_bins; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // ========== 核心优化：本地累积器 ==========
    // last_bin: 记录上一次处理的 bin 索引（存储在寄存器中）
    // count: 累积计数器，记录连续相同 bin 的出现次数（存储在寄存器中）
    // 关键洞察：寄存器访问是免费的（1 周期），而 atomicAdd 代价高昂
    int last_bin = -1;
    int count = 0;

    for (int i = idx; i < n; i += stride) {
        int bin = data[i];
        if (bin == last_bin) {
            // 连续遇到相同 bin：只增加寄存器中的计数器
            // 这是纯寄存器操作，零内存访问开销
            count++;
        } else {
            // 遇到不同 bin：需要"刷新"之前的累积值到 shared memory
            // 只有在 bin 切换时才执行一次 atomicAdd，而不是每个元素都执行
            if (count > 0) atomicAdd(&s_hist[last_bin], count);
            last_bin = bin;  // 更新追踪的 bin
            count = 1;       // 重置计数器
        }
    }
    // 循环结束后，最后一批累积的计数还在寄存器中，需要最终刷新
    if (count > 0) atomicAdd(&s_hist[last_bin], count);

    __syncthreads();

    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}
// 性能分析：
// - 最好情况：数据完全有序，每个线程只需 1 次 atomicAdd（所有元素都在同一 bin）
// - 最坏情况：数据完全随机，退化为 Version 2 的性能（每个元素都触发 atomicAdd）
// - 额外开销：每次循环多了一次比较和条件分支，但这比 atomicAdd 便宜得多
```

### 1.5 性能特征与选择指南

| 场景 | 推荐版本 | 原因 |
|------|---------|------|
| bins < 12K | V2 Shared | 通用最优 |
| 数据有局部性 | V3 Local | 减少atomic次数 |
| bins > 12K | Multi-pass或CUB | 超出shared memory |
| 生产环境 | CUB库 | 高度优化 |

**Roofline 理论分析**

让我们用 Roofline 模型分析 Histogram kernel 的性能特征。

**符号定义**：
- $\pi$：GPU 峰值算力（FLOP/s）
- $\beta$：显存带宽（Byte/s）
- $N$：输入数据元素个数
- $B$：bin 的数量
- $s$：单个数据元素的字节数（如 int32 则 $s=4$）


**Histogram 的算术强度分析**：

对于 naive 版本，处理 $N$ 个元素：

- **内存访问量**：
  - 读取输入数据：$N \cdot s$ 字节
  - 原子更新直方图（读-改-写）：$N \cdot 2s$ 字节（最坏情况，每次都访问不同 bin）
  - 总计：$M = 3Ns$ 字节

- **计算量**：
  - 每个元素：边界检查 + 加法 ≈ $\alpha$ FLOPs（$\alpha \approx 2$）
  - 总计：$F = \alpha N$ FLOPs

- **算术强度**：
$$I_{hist} = \frac{F}{M} = \frac{\alpha N}{3Ns} = \frac{\alpha}{3s}$$

对于 int32（$s=4$），$I_{hist} = \frac{\alpha}{12} \ll 1$ FLOP/Byte

**Roofline 结论**：

由于现代 GPU 的 $I_{ridge} \gg 1$（通常 $I_{ridge} > 100$），而 $I_{hist} < 1$，因此：
$$I_{hist} \ll I_{ridge}$$

Histogram 处于严重的 **Memory-Bound** 区域，可达性能为：
$$P_{attainable} = I_{hist} \cdot \beta = \frac{\alpha \beta}{3s}$$

峰值算力利用率：
$$\eta = \frac{P_{attainable}}{\pi} = \frac{\alpha \beta}{3s\pi} = \frac{\alpha}{3s \cdot I_{ridge}} \ll 1$$

**考虑原子操作竞争**：

设 $\gamma \in (0, 1]$ 为原子操作的有效带宽系数（竞争越激烈，$\gamma$ 越小），实际性能为：
$$P_{real} = \gamma \cdot I_{hist} \cdot \beta = \frac{\gamma \alpha \beta}{3s}$$

**优化版本的 Roofline 视角**：

| 版本 | 优化效果 | Roofline 影响 |
|------|---------|--------------|
| V2 Shared | 用 Shared Memory 替代 Global Memory | 等效将 $\beta$ 提升 $\kappa$ 倍（$\kappa \approx 10\text{-}20$） |
| V3 Local | 寄存器累积，减少原子操作次数 | 将 $\gamma$ 提升至接近 1 |

**结论**：Histogram 的 $I \ll I_{ridge}$，始终是 memory-bound。优化策略：
1. **提升有效带宽**：使用更快的存储层级（Shared Memory），等效增大 $\beta$
2. **减少竞争**：私有化直方图，提升 $\gamma$
3. **减少访存次数**：本地累积，减小 $M$

---

## Part 2: Scan (Prefix Sum) Kernel

https://leetgpu.com/challenges/prefix-sum

### 2.1 什么是Scan

Scan（前缀和）是另一个核心并行原语，计算数组的累积操作：

```
输入:  [3, 1, 7, 0, 4, 1, 6, 3]

Exclusive Scan (不含当前元素):
输出:  [0, 3, 4, 11, 11, 15, 16, 22]
       ↑  ↑
       0  0+3

Inclusive Scan (含当前元素):  
输出:  [3, 4, 11, 11, 15, 16, 22, 25]
       ↑  ↑
       3  3+1
```

### 2.2 Scan的重要性

Scan是构建其他并行算法的基础：

| 应用 | 如何使用Scan |
|------|-------------|
| Stream Compaction | 标记→Scan→Scatter |
| Radix Sort | 计数→Scan→分配位置 |
| 稀疏矩阵 | CSR格式的row_ptr |
| 并行分配 | 计算每个线程的输出偏移 |

### 2.3 朴素并行Scan的问题

直观想法：每个元素独立计算前缀和

```cpp
// 错误！O(n²)复杂度
__global__ void scan_naive(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i = 0; i <= idx; i++) {  // 每个线程遍历[0, idx]
        sum += data[i];
    }
    data[idx] = sum;
}
```

问题：第i个线程做i次加法，总工作量O(n²)，完全没有利用并行性。

### 2.4 高效并行Scan：Blelloch算法

Blelloch算法分两个阶段，总工作量O(n)，跨度O(log n)：

```
Phase 1: Up-sweep (Reduce)
建立归约树，计算部分和

Phase 2: Down-sweep
从根向下传播，计算前缀和
```

#### 伪代码

```python
def blelloch_scan(x):
    """
    Blelloch 并行前缀和算法
    输入: x[0..n-1]，长度 n 必须是 2 的幂
    输出: exclusive prefix sum
    """
    n = len(x)

    # ========== Phase 1: Up-sweep (Reduce) ==========
    # 从叶子到根，构建归约树
    # 每一步将相邻元素对的和存储到右边元素的位置
    for d in range(log2(n)):           # d = 0, 1, ..., log2(n)-1
        stride = 2^(d+1)               # stride = 2, 4, 8, ...
        for i in parallel(0, n, stride):  # i = 0, stride, 2*stride, ...
            x[i + stride - 1] += x[i + stride/2 - 1]

    # 此时 x[n-1] 包含所有元素的总和

    # ========== Phase 2: Down-sweep ==========
    # 从根到叶子，利用归约树计算前缀和
    x[n-1] = 0                         # 将根设为 0（identity element）

    for d in range(log2(n)-1, -1, -1): # d = log2(n)-1, ..., 1, 0
        stride = 2^(d+1)               # stride = n, n/2, ..., 4, 2
        for i in parallel(0, n, stride):
            left = i + stride/2 - 1
            right = i + stride - 1

            temp = x[left]             # 保存左子节点的值
            x[left] = x[right]         # 左子节点 = 父节点的值（来自上方）
            x[right] += temp           # 右子节点 = 父节点值 + 原左子节点值

    return x  # 现在 x 包含 exclusive prefix sum
```

**复杂度分析**：
- 时间复杂度：$O(\log N)$ 步（每步内部并行）
- 工作复杂度：$O(N)$ 次加法操作
- 空间复杂度：$O(1)$ 额外空间（in-place 算法）

#### 图解（8个元素）

```
输入: [3, 1, 7, 0, 4, 1, 6, 3]

=== Up-sweep (Reduce) ===
Step 1 (stride=1): 
[3, 4, 7, 7, 4, 5, 6, 9]
     ↑     ↑     ↑     ↑
    3+1   7+0   4+1   6+3

Step 2 (stride=2):
[3, 4, 7, 11, 4, 5, 6, 14]
           ↑            ↑
         4+7          5+9

Step 3 (stride=4):
[3, 4, 7, 11, 4, 5, 6, 25]
                        ↑
                     11+14

=== Down-sweep ===
设置根为0: [3, 4, 7, 11, 4, 5, 6, 0]
                                  ↑

Step 1 (stride=4):
[3, 4, 7, 0, 4, 5, 6, 11]
           ↑            ↑
        交换并累加

Step 2 (stride=2):
[3, 4, 7, 0, 4, 5, 6, 11]
     ↓     ↓     ↓      ↓
[3, 0, 7, 4, 4, 11, 6, 16]

Step 3 (stride=1):
[0, 3, 4, 11, 11, 15, 16, 22]

输出 (Exclusive): [0, 3, 4, 11, 11, 15, 16, 22] ✓
```


```
new_left  = right           // 左子继承父节点传来的值
new_right = left + right    // 右子 = 继承值 + 左兄弟的子树和
树形视角

Up-sweep 结果（子树和）：        Down-sweep 传递（左边的和）：
        25                              0
       /  \                           /    \
     11    14                        0       11
    / \   / \                       / \     /   \
   4   7  5  9                     0   4   11    16
  /\  /\ /\  /\                   /\  /\   /\    /\
 3 1 7 0 4 1 6 3                 0 3 4 11 11 15 16 22


**规则**：
- **左子节点**：继承父节点的值（我左边 = 父亲左边）
- **右子节点**：父节点值 + 左兄弟的和（我左边 = 父亲左边 + 左兄弟）


temp[n-1] = 0;  // 在down-sweep开始前
因为是 **exclusive** scan——第一个元素的前缀和是 0（它左边没有任何元素）。这个 0 会在 down-sweep 过程中传播到位置 0。


输入:     [3, 1, 7, 0, 4, 1, 6, 3]

Up-sweep后: [3, 4, 7, 11, 4, 5, 6, 25]
                                   ↓ 设为0
           [3, 4, 7, 11, 4, 5, 6, 0]

Down-sweep:
  stride=4: [3, 4, 7, 0,  4, 5,  6, 11]   // 根层
  stride=2: [3, 0, 7, 4,  4, 11, 6, 16]   // 第二层  
  stride=1: [0, 3, 4, 11, 11, 15, 16, 22] // 叶子层 ✓
```

### 2.5 CUDA实现

#### Version 1: 单Block Scan (Blelloch)

```cpp
__global__ void scan_blelloch_single_block(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    
    // 加载到shared memory
    temp[2*tid] = data[2*tid];
    temp[2*tid+1] = data[2*tid+1];
    
    int offset = 1;
    
    // === Up-sweep (Reduce) ===
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // 清除最后一个元素（为down-sweep准备）
    if (tid == 0) temp[n-1] = 0;
    
    // === Down-sweep ===
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // 写回
    data[2*tid] = temp[2*tid];
    data[2*tid+1] = temp[2*tid+1];
}
```

**限制**：只能处理单个block大小的数据（通常≤2048元素）

#### Version 2: 多Block Scan (三阶段)

处理任意大小数组需要三个阶段：

```
阶段1: Block-level Scan
每个block独立scan自己的部分，保存block总和

阶段2: Scan Block Sums  
对所有block的总和做scan

阶段3: Add Block Offsets
每个block加上前面所有block的总和
```

```cpp
// 阶段1: 每个block scan并保存总和
__global__ void scan_blocks(int* data, int* block_sums, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * blockDim.x * 2;
    
    // 加载数据
    int ai = tid;
    int bi = tid + blockDim.x;
    temp[ai] = (block_offset + ai < n) ? data[block_offset + ai] : 0;
    temp[bi] = (block_offset + bi < n) ? data[block_offset + bi] : 0;
    
    // Blelloch scan (同上)
    // ... up-sweep ...
    // ... down-sweep ...
    
    __syncthreads();
    
    // 保存block总和
    if (tid == 0) {
        block_sums[bid] = temp[blockDim.x * 2 - 1];
    }
    
    // 写回scan结果
    if (block_offset + ai < n) data[block_offset + ai] = temp[ai];
    if (block_offset + bi < n) data[block_offset + bi] = temp[bi];
}

// 阶段3: 加上block偏移
__global__ void add_block_sums(int* data, int* block_sums, int n) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (blockIdx.x > 0 && idx < n) {
        data[idx] += block_sums[blockIdx.x];
    }
    if (blockIdx.x > 0 && idx + blockDim.x < n) {
        data[idx + blockDim.x] += block_sums[blockIdx.x];
    }
}
```

#### Version 3: Work-Efficient with Bank Conflict Avoidance

Shared memory有bank conflict问题。添加padding避免：

```cpp
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

__global__ void scan_optimized(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    
    // 带padding的索引，避免bank conflict
    int ai = tid;
    int bi = tid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
    temp[ai + bankOffsetA] = data[ai];
    temp[bi + bankOffsetB] = data[bi];
    
    int offset = 1;
    
    // Up-sweep with conflict-free addressing
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // ... 类似处理down-sweep ...
}
```


### 2.6 Scan的Roofline特征

```
内存访问: 读n + 写n = 2n
计算量: O(n) 加法
Arithmetic Intensity: ~0.25 ops/byte (int32)

结论: Memory-bound，但比histogram好（没有原子操作）
```


## Part 3: Scan在Mamba 1.0中的应用

### 3.1 背景：State Space Model (SSM)

SSM是一种序列建模方法，可以看作连续时间系统的离散化：

```
连续形式:
  h'(t) = A·h(t) + B·x(t)     (状态更新)
  y(t)  = C·h(t) + D·x(t)     (输出)

离散化后:
  h_t = Ā·h_{t-1} + B̄·x_t    (线性递推！)
  y_t = C·h_t + D·x_t
```

**这是一个线性递推，正是Scan可以并行化的形式！**

### 3.2 SSM的两种计算模式

**模式1: Recurrent（顺序计算）**

```python
# O(L) 时间，O(1) 空间，但完全顺序
h = zeros(N)  # hidden state
for t in range(L):
    h = A @ h + B @ x[t]   # 必须等上一步完成
    y[t] = C @ h
```

- 推理时很高效：O(1) per token
- 训练时很慢：无法并行，GPU利用率低

**模式2: Convolution（并行计算）**

对于**时不变**SSM（A, B, C固定），可以展开成卷积：

```
y = x * K，其中 K = (CB̄, CĀB̄, CĀ²B̄, ...)
```

- 训练时高效：FFT卷积 O(L log L)
- 但要求A, B, C是常数（时不变）

### 3.3 Mamba的困境：Selective但不能用卷积

Mamba的核心创新是**Selective SSM**：让B, C, Δ依赖于输入

```python
# Selective SSM: 参数随输入变化
Δ_t = Linear(x_t)  # 离散化步长
B_t = Linear(x_t)  # 输入矩阵
C_t = Linear(x_t)  # 输出矩阵

h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B_t·x_t
y_t = C_t·h_t
```

**问题**：参数随时间变化 → 不再是时不变系统 → 卷积模式失效！

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mamba 的困境                                                               │
│                                                                             │
│  想要 Selective（性能好）→ B, C, Δ 必须依赖输入 → 时变系统                    │
│  想要 Fast Training     → 需要并行化 → 卷积要求时不变                        │
│                                                                             │
│  矛盾！传统方法只能二选一                                                    │
│                                                                             │
│  解决方案: Parallel Associative Scan                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Associative Scan：并行化线性递推

**关键洞察**：虽然参数时变，但递推仍然是**结合律**的！

SSM递推可以写成：
```
h_t = A_t · h_{t-1} + b_t

定义二元组: (A_t, b_t)
定义结合运算 ⊗: (A₂, b₂) ⊗ (A₁, b₁) = (A₂·A₁, A₂·b₁ + b₂)
```

**验证结合律**：

```
(A₃, b₃) ⊗ [(A₂, b₂) ⊗ (A₁, b₁)]
= (A₃, b₃) ⊗ (A₂·A₁, A₂·b₁ + b₂)
= (A₃·A₂·A₁, A₃·A₂·b₁ + A₃·b₂ + b₃)

[(A₃, b₃) ⊗ (A₂, b₂)] ⊗ (A₁, b₁)
= (A₃·A₂, A₃·b₂ + b₃) ⊗ (A₁, b₁)
= (A₃·A₂·A₁, A₃·A₂·b₁ + A₃·b₂ + b₃)

两者相等！满足结合律 ✓
```

### 3.5 Parallel Scan应用于SSM

有了结合律，就可以用Blelloch scan并行计算：

```
输入: [(A₁,b₁), (A₂,b₂), (A₃,b₃), (A₄,b₄), ...]

目标: 计算所有前缀积
  h₁ = (A₁,b₁)
  h₂ = (A₂,b₂) ⊗ (A₁,b₁)
  h₃ = (A₃,b₃) ⊗ (A₂,b₂) ⊗ (A₁,b₁)
  ...

使用 Blelloch Scan:
  Step 1 (Up-sweep): 构建部分积
  Step 2 (Down-sweep): 传播前缀积
  
  复杂度: O(L) work, O(log L) span
  可以在 O(log L) 步内完成！
```

**图示**：

```
Sequential (O(L) steps):
  h₁ → h₂ → h₃ → h₄ → h₅ → h₆ → h₇ → h₈
  
Parallel Scan (O(log L) steps):
  Step 1:  [1-2]   [3-4]   [5-6]   [7-8]     (4 pairs)
  Step 2:  [1-4]           [5-8]             (2 pairs)  
  Step 3:  [1-8]                             (1 pair)
  Down-sweep: 分发前缀积到每个位置
  
  总共 2·log₂(8) = 6 步，而非 8 步
```


```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mamba 的三大优化策略                                                        │
│                                                                             │
│  1. Kernel Fusion（算子融合）                                                │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ 传统: HBM → 离散化 → HBM → Scan → HBM → 输出 → HBM               │    │
│     │       (多次HBM读写，I/O瓶颈)                                      │    │
│     │                                                                 │    │
│     │ Fused: HBM → SRAM [离散化 + Scan + 输出] → HBM                   │    │
│     │        (一次读入，一次写出)                                       │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. Parallel Scan in SRAM                                                   │
│     • 不materialize中间状态到HBM                                            │
│     • 所有scan操作在SRAM中完成                                               │
│     • 只写最终输出到HBM                                                      │
│                                                                             │
│  3. Recomputation（重计算）                                                  │
│     • Forward: 不保存中间状态                                                │
│     • Backward: 重新计算需要的状态                                           │
│     • 用计算换内存                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.6 Mamba1 Kernel 核心代码解析

> Example source: https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan/

#### 核心文件结构

```
csrc/selective_scan/
├── selective_scan_common.h      # Associative scan 算子定义
├── selective_scan_fwd_kernel.cuh  # Forward pass kernel
├── selective_scan_bwd_kernel.cuh  # Backward pass kernel
└── reverse_scan.cuh             # 反向scan（用于梯度传播）
```

#### 1. Associative Scan 算子（核心！）

SSM递推公式: `h[t] = A * h[t-1] + B * x[t]`

**关键洞察**：这个递推可以表示为二元组 `(a, b)` 的结合运算：
- 状态表示为 `(decay, value)` = `(A, B*x)`
- 组合两个状态: `(a0, b0) ⊕ (a1, b1) = (a1*a0, a1*b0 + b1)`

```cuda
// selective_scan_common.h - 核心算子
// 这是Mamba能够并行化的数学基础！

template<>
struct SSMScanOp<float> {
    __device__ __forceinline__ float2 operator()(
        const float2 &ab0,  // (a0, b0) = 前一个状态
        const float2 &ab1   // (a1, b1) = 当前状态
    ) const {
        // 结合律: (a1*a0, a1*b0 + b1)
        // ab.x = decay factor (累积衰减)
        // ab.y = value contribution (累积输入)
        return make_float2(
            ab1.x * ab0.x,           // 累积decay: a1 * a0
            ab1.x * ab0.y + ab1.y    // 累积value: a1 * b0 + b1
        );
    }
};

// 复数版本（用于某些SSM变体）
template<>
struct SSMScanOp<complex_t> {
    __device__ __forceinline__ float4 operator()(
        const float4 &ab0, const float4 &ab1
    ) const {
        complex_t a0(ab0.x, ab0.y), b0(ab0.z, ab0.w);
        complex_t a1(ab1.x, ab1.y), b1(ab1.z, ab1.w);
        complex_t out_a = a1 * a0;
        complex_t out_b = a1 * b0 + b1;
        return make_float4(out_a.real_, out_a.imag_,
                          out_b.real_, out_b.imag_);
    }
};

// 跨chunk边界的状态传递
template <typename scalar_t>
struct SSMScanPrefixCallbackOp {
    using scan_t = std::conditional_t<
        std::is_same_v<scalar_t, float>, float2, float4>;
    scan_t running_prefix;  // 上一个chunk的最终状态

    __device__ scan_t operator()(scan_t block_aggregate) {
        scan_t old_prefix = running_prefix;
        // 将当前block的聚合结果与running prefix组合
        running_prefix = SSMScanOp<scalar_t>()(running_prefix, block_aggregate);
        return old_prefix;  // 返回给当前block使用
    }
};
```

#### 2. Forward Kernel 核心逻辑

```cuda
// selective_scan_fwd_kernel.cuh (简化版)

template<typename Ktraits>
__global__ void selective_scan_fwd_kernel(SSMParamsBase params) {
    // ========== 常量和配置 ==========
    constexpr int kNThreads = Ktraits::kNThreads;     // 线程数
    constexpr int kNItems = Ktraits::kNItems;         // 每线程处理的元素数
    constexpr int kChunkSize = kNThreads * kNItems;   // 2048 (典型值)

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;   // 每个block处理一个(batch, dim)

    // ========== Step 1: 加载参数 ==========
    // A: [D, N] - 状态转移矩阵（通常是负数，表示衰减）
    // delta: [B, L, D] - 时间步长（input-dependent）
    // B: [B, L, N] 或 [B, N] - 输入矩阵
    // C: [B, L, N] 或 [B, N] - 输出矩阵
    // u: [B, L, D] - 输入

    float A_val = A[dim_id * N + state_idx];  // 对每个state维度
    A_val *= LOG2E;  // 预乘log2(e)，使用exp2f更快

    // ========== Step 2: 按chunk处理序列 ==========
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        int chunk_offset = chunk * kChunkSize;

        // 2a. 加载这个chunk的数据到寄存器
        float delta_vals[kNItems], u_vals[kNItems];
        float B_vals[kNItems], C_vals[kNItems];

        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            int seq_idx = chunk_offset + threadIdx.x * kNItems + i;
            delta_vals[i] = delta[batch_id][seq_idx][dim_id];
            u_vals[i] = u[batch_id][seq_idx][dim_id];

            // Delta softplus (可选): delta = log(1 + exp(delta))
            if (kDeltaSoftplus) {
                delta_vals[i] = delta_vals[i] <= 20.f
                    ? log1pf(expf(delta_vals[i]))
                    : delta_vals[i];
            }
        }

        // 2b. 计算scan的输入: (decay, value) pairs
        float2 thread_data[kNItems];

        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            // decay = exp(delta * A) = exp2(delta * A * log2(e))
            float decay = exp2f(delta_vals[i] * A_val);

            // value = delta * B * u
            float delta_u = delta_vals[i] * u_vals[i];
            float value = delta_u * B_vals[i];

            thread_data[i] = make_float2(decay, value);
        }

        // ========== Step 3: Parallel Associative Scan ==========
        // 使用CUB的BlockScan，配合自定义的SSMScanOp

        using BlockScanT = cub::BlockScan<float2, kNThreads,
                                          cub::BLOCK_SCAN_WARP_SCANS>;

        SSMScanPrefixCallbackOp<float> prefix_op(running_prefix);

        BlockScanT(smem_scan).InclusiveScan(
            thread_data,        // 输入: (decay, value) pairs
            thread_data,        // 输出: scan后的结果
            SSMScanOp<float>(), // 结合运算符
            prefix_op           // 处理跨chunk的状态传递
        );

        // 更新running_prefix用于下一个chunk
        running_prefix = prefix_op.running_prefix;

        // ========== Step 4: 计算输出 ==========
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            // thread_data[i].y 现在是 h[t]（隐藏状态）
            // output = C * h
            out_vals[i] += thread_data[i].y * C_vals[i];
        }

        // 写回HBM（只在所有state维度累加完后写）
    }
}
```

#### 3. 数学原理图解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Associative Scan 的数学基础                                                 │
│                                                                             │
│  递推公式: h[t] = a[t] * h[t-1] + b[t]                                      │
│                                                                             │
│  表示为二元组: (a, b) 其中 a=decay, b=input                                  │
│                                                                             │
│  组合运算 ⊕:                                                                │
│    (a0, b0) ⊕ (a1, b1) = (a1*a0, a1*b0 + b1)                               │
│                                                                             │
│  验证结合律:                                                                │
│    [(a0,b0) ⊕ (a1,b1)] ⊕ (a2,b2)                                           │
│    = (a1*a0, a1*b0+b1) ⊕ (a2,b2)                                           │
│    = (a2*a1*a0, a2*(a1*b0+b1)+b2)                                          │
│    = (a2*a1*a0, a2*a1*b0 + a2*b1 + b2)                                     │
│                                                                             │
│    (a0,b0) ⊕ [(a1,b1) ⊕ (a2,b2)]                                           │
│    = (a0,b0) ⊕ (a2*a1, a2*b1+b2)                                           │
│    = (a2*a1*a0, a2*a1*b0 + a2*b1 + b2)   ✓ 相等！                           │
│                                                                             │
│  因此可以用parallel prefix sum!                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Parallel Scan 执行过程 (8个元素示例)                                        │
│                                                                             │
│  输入: (a0,b0) (a1,b1) (a2,b2) (a3,b3) (a4,b4) (a5,b5) (a6,b6) (a7,b7)     │
│                                                                             │
│  Up-sweep (reduce):                                                         │
│  Level 0:  [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]            │
│              \    /        \    /        \    /        \    /              │
│  Level 1:   [0:1]         [2:3]         [4:5]         [6:7]               │
│                 \          /                \          /                   │
│  Level 2:       [0:3]                       [4:7]                          │
│                      \                    /                                │
│  Level 3:            [0:7] (全局聚合)                                       │
│                                                                             │
│  Down-sweep (distribute):                                                   │
│  把部分和传播回去，得到每个位置的inclusive scan结果                            │
│                                                                             │
│  输出: h[0]  h[1]  h[2]  h[3]  h[4]  h[5]  h[6]  h[7]                      │
│                                                                             │
│  复杂度: O(log L) 深度，O(L) 总工作量                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```


### 3.7 Mamba-2：从Scan到矩阵乘法

Mamba-1的parallel scan有个问题：**无法利用Tensor Core**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mamba-1 vs Mamba-2                                                         │
│                                                                             │
│  Mamba-1:                                                                   │
│  • 使用 parallel associative scan                                           │
│  • Scan操作是element-wise，无法用Tensor Core                                 │
│  • State dimension 限制为 N=16（更大会变慢）                                 │
│  • A100: 只用到 19 TFLOPS (FP32 arithmetic)                                 │
│                                                                             │
│  Mamba-2:                                                                   │
│  • 发现SSM可以写成structured matrix乘法                                      │
│  • 用矩阵乘法替代scan（可以用Tensor Core！）                                  │
│  • State dimension 可以扩展到 N=64, 128                                     │
│  • A100: 可用 312 TFLOPS (BF16 matmul) - 16x 提升！                         │
│                                                                             │
│                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 课后练习题

### 练习 1：histogram 为什么比 reduce 难？

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">histogram 的并行瓶颈是什么？</span></summary>

reduce 的写入目标固定，histogram 的写入目标由数据值决定。多个线程可能同时更新同一个 bin，导致 atomic serialize。数据分布越偏，热点 bin 越严重。常见优化是先做 warp-local 或 block-local histogram，再合并到 global histogram。

</details>

### 练习 2：scan 的核心不变量

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">parallel scan 为什么要分 upsweep / downsweep？</span></summary>

upsweep 建立局部区间和，downsweep 把前缀偏移传播回每个位置。核心不变量是每个内部节点保存一个区间 aggregate，最后每个元素拿到它左侧所有元素的和。GPU 上通常会做 block 内 scan，再 scan block sums，最后把 block prefix 加回每个 block。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">histogram 和 reduce 都是聚合，为什么 histogram 更难？</span></summary>

reduce 的写入位置通常可控，block 内可以先局部归约再写少量结果；histogram 的 bin 由数据决定，多个 thread 可能同时写同一个 bin，需要 atomic 或 privatization。难点在冲突、skew、atomic 热点和局部 histogram 合并。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">scan 能并行的前提是什么？</span></summary>

scan 依赖一个 associative operator。普通加法满足结合律，所以可以先局部 prefix，再合并 block summary，再把前缀偏移加回去。很多 recurrence 也能改写成 associative pair operation；如果算子不满足结合律，就不能直接套 parallel prefix。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">exclusive scan 和 inclusive scan 有什么差别？</span></summary>

inclusive scan 的第 `i` 个输出包含 `x_i`；exclusive scan 的第 `i` 个输出只包含 `x_0 ... x_{i-1}`。很多 compaction 和 prefix offset 场景需要 exclusive scan，因为它直接给每个元素写入目标位置。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">什么时候用 privatized histogram？代价是什么？</span></summary>

当全局 atomic 冲突严重、bin 数不大时，可以让每个 block/warp 维护私有 histogram，再做合并。它减少全局 atomic 热点，但增加 shared memory 或临时 buffer 使用，并多出一次 merge。如果 bin 很多或分布均匀，privatization 不一定划算。

</details>
