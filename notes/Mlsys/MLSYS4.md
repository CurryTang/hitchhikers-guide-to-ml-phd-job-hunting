# CUDA Reduce Kernel 完全指南：从原理到优化

## 目录
1. [什么是 Reduce Kernel](#1-什么是-reduce-kernel)
2. [算法原理与并行化思想](#2-算法原理与并行化思想)
3. [Reduce Kernel 的演进：7 个版本](#3-reduce-kernel-的演进7-个版本)
4. [Roofline 分析与性能建模](#4-roofline-分析与性能建模)
5. [Profiling](#5-profiling)

---

## 1. 什么是 Reduce Kernel

### 1.1 定义

**Reduce（归约）** 是一种将一组数据通过某种二元运算（如加法、最大值、最小值）聚合成单个结果的操作。

```
输入:  [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇]
操作:  sum (加法)
输出:  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇
```

### 1.2 为什么 Reduce 在 MLSys 中重要？

Reduce 操作在深度学习中无处不在：

| 场景 | Reduce 类型 | 示例 |
|------|-------------|------|
| Loss 计算 | Sum/Mean | CrossEntropyLoss 对 batch 求平均 |
| Softmax | Max + Sum | 数值稳定性需要先求 max |
| LayerNorm/BatchNorm | Mean + Variance | 统计量计算 |
| Attention | Sum | Softmax 后的加权求和 |
| 梯度聚合 | Sum | 分布式训练 AllReduce |


## 2. 算法原理与并行化思想

### 2.1 树形归约（Tree Reduction）

并行 Reduce 的核心思想是**树形归约**：

```
Step 0:  [a₀] [a₁] [a₂] [a₃] [a₄] [a₅] [a₆] [a₇]
              ↘↙      ↘↙      ↘↙      ↘↙
Step 1:    [a₀+a₁]  [a₂+a₃]  [a₄+a₅]  [a₆+a₇]
                 ↘  ↙            ↘  ↙
Step 2:      [a₀+a₁+a₂+a₃]  [a₄+a₅+a₆+a₇]
                      ↘    ↙
Step 3:        [a₀+a₁+a₂+a₃+a₄+a₅+a₆+a₇]
```

- **每一步**：活跃线程数减半
- **总步数**：log₂(n)
- **Work（总操作数）**：n-1（与串行相同）
- **Span（关键路径）**：log₂(n)

### 2.2 树形归约的两种索引方式

在 GPU 上实现树形归约有两种常见的索引方式，选择不同会直接影响性能：

#### 方式一：Interleaved Addressing（步长递增）

```
数组: [0] [1] [2] [3] [4] [5] [6] [7]    (8个元素)

Step s=1: 步长=1，线程 0,2,4,6 工作
  Thread 0: arr[0] += arr[1]    →  [0+1] [ ] [2] [3] [4] [5] [6] [7]
  Thread 2: arr[2] += arr[3]    →  [0+1] [ ] [2+3] [ ] [4] [5] [6] [7]
  Thread 4: arr[4] += arr[5]
  Thread 6: arr[6] += arr[7]
  结果: [0+1] [ ] [2+3] [ ] [4+5] [ ] [6+7] [ ]

Step s=2: 步长=2，线程 0,4 工作
  Thread 0: arr[0] += arr[2]
  Thread 4: arr[4] += arr[6]
  结果: [0..3] [ ] [ ] [ ] [4..7] [ ] [ ] [ ]

Step s=4: 步长=4，线程 0 工作
  Thread 0: arr[0] += arr[4]
  结果: [0..7] ...

索引公式: if (tid % (2*s) == 0) arr[tid] += arr[tid + s]
```

**问题**：
- 活跃线程不连续（0,2,4,6 → 0,4 → 0），导致 **warp divergence**
- 后期访问步长大，导致 **bank conflict**

#### 方式二：Sequential Addressing（步长递减）✓ 推荐

```
数组: [0] [1] [2] [3] [4] [5] [6] [7]    (8个元素)

Step s=4: 步长=4，线程 0,1,2,3 工作（前半部分）
  Thread 0: arr[0] += arr[4]    →  [0+4] [1] [2] [3] | [4] [5] [6] [7]
  Thread 1: arr[1] += arr[5]    →  [0+4] [1+5] [2] [3] | ...
  Thread 2: arr[2] += arr[6]
  Thread 3: arr[3] += arr[7]
  结果: [0+4] [1+5] [2+6] [3+7] | (不再需要)

Step s=2: 步长=2，线程 0,1 工作
  Thread 0: arr[0] += arr[2]
  Thread 1: arr[1] += arr[3]
  结果: [0..3+4..7的一部分] [另一部分] | ...

Step s=1: 步长=1，线程 0 工作
  Thread 0: arr[0] += arr[1]
  结果: [最终和] ...

索引公式: if (tid < s) arr[tid] += arr[tid + s]
```

**优势**：
- 活跃线程始终连续（0,1,2,3 → 0,1 → 0），**无 warp divergence**
- 连续线程访问连续内存，**无 bank conflict**


## 3. Reduce Kernel 的演进：7 个版本

我们将实现一个对 `N = 2^24 = 16M` 个 float 求和的 kernel，逐步优化。

### Version 0: Interleaved Addressing with Divergent Branching

**最朴素的实现**

```cpp
__global__ void reduce_v0(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    // 每个线程从 global memory 加载一个元素到 shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // 树形归约
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // ❌ 问题：线程发散！
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 只有 thread 0 写回结果
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

理解变量的内存结构

![[assets/Pasted image 20251229150638.png]]

理解变量的数据流向

![[assets/Pasted image 20251229151159.png]]

![[assets/Pasted image 20251229151249.png]]


什么情况下需要syncthreads?
![[assets/Pasted image 20251229151721.png]]

**问题分析：**

```
Step s=1:  线程 0,2,4,6... 活跃，1,3,5,7... 空闲
           → 一个 warp (32线程) 中只有 16 个活跃
           → 50% 效率损失 + 分支发散

Step s=2:  线程 0,4,8,12... 活跃
           → 25% 效率

...以此类推
```

**性能瓶颈：**
- Warp divergence（同一 warp 内线程走不同分支）
- 大量线程空闲
- 条件判断 `tid % (2*s) == 0` 开销大

---

### Version 1: Interleaved Addressing with Bank Conflicts

**消除分支发散，但引入 Bank Conflict**

```cpp
__global__ void reduce_v1(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // 改进：连续的线程执行相同操作
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // 计算配对的索引
        int index = 2 * s * tid;
        
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**改进：**
- 前 N/2 个线程连续执行，消除了 warp divergence
- 但...引入了新问题：**Shared Memory Bank Conflict**

```
初始数据: sdata[0..7] = [a, b, c, d, e, f, g, h]

═══════════════════════════════════════════════════════════════════════
                         V0: tid % (2*s) == 0
═══════════════════════════════════════════════════════════════════════

s=1: 活跃线程是 tid % 2 == 0，即 tid = 0, 2, 4, 6
     
     tid:    0     1     2     3     4     5     6     7
           活跃   空闲  活跃   空闲  活跃   空闲  活跃   空闲
             │           │           │           │
             ▼           ▼           ▼           ▼
           [0]+[1]     [2]+[3]     [4]+[5]     [6]+[7]

     问题: 一个 warp 内，奇数线程空闲 → Warp Divergence!

s=2: 活跃线程是 tid % 4 == 0，即 tid = 0, 4
     
     tid:    0     1     2     3     4     5     6     7
           活跃   空闲  空闲  空闲  活跃   空闲  空闲  空闲
             │                       │
             ▼                       ▼
           [0]+[2]                 [4]+[6]

     问题: 更多线程空闲，divergence 更严重!

═══════════════════════════════════════════════════════════════════════
                      V1: index = 2 * s * tid  
═══════════════════════════════════════════════════════════════════════

s=1: index = 2 * 1 * tid = 2*tid
     
     tid:    0     1     2     3     4     5     6     7
           活跃   活跃  活跃   活跃  空闲   空闲  空闲   空闲
             │     │     │     │
             ▼     ▼     ▼     ▼
     index:  0     2     4     6
             │     │     │     │
             ▼     ▼     ▼     ▼
           [0]+[1] [2]+[3] [4]+[5] [6]+[7]

     改进: 前 4 个线程连续执行，后 4 个连续空闲 → 无 Divergence!

s=2: index = 2 * 2 * tid = 4*tid
     
     tid:    0     1     2     3     4     5     6     7
           活跃   活跃  空闲  空闲  空闲   空闲  空闲  空闲
             │     │
             ▼     ▼
     index:  0     4
             │     │
             ▼     ▼
           [0]+[2] [4]+[6]

     改进: 前 2 个线程连续执行 → 无 Divergence!
```

## 核心思想
```
V0 思路: 每个线程判断"我该不该工作"
         tid=0 工作，tid=1 不工作，tid=2 工作，tid=3 不工作...
         → 交错的活跃/空闲 → Divergence

V1 思路: 每个线程计算"我要操作哪个位置"
         tid=0 操作 index=0，tid=1 操作 index=2，tid=2 操作 index=4...
         → 前 N/2 个线程连续活跃 → 无 Divergence
```

## 为什么 V1 仍有问题？

V1 消除了 divergence，但引入了 **Bank Conflict**：
```
s=1 时:
  Thread 0 访问 sdata[0] 和 sdata[1]
  Thread 1 访问 sdata[2] 和 sdata[3]
  → 没问题

s=16 时: index = 32 * tid
  Thread 0 访问 sdata[0]  和 sdata[16]   → Bank 0, Bank 16
  Thread 1 访问 sdata[32] 和 sdata[48]   → Bank 0, Bank 16  ← 冲突!
  
  sdata[0]  在 Bank 0
  sdata[32] 在 Bank 0  (32 % 32 = 0)
  → 两个线程访问同一个 Bank 的不同地址 → 串行化!
```

**Bank Conflict 解释：**

Shared Memory 分成 32 个 bank（每 4 字节一个 bank）。当同一 warp 内的多个线程访问同一 bank 的不同地址时，访问会**串行化**。

```
Step s=1:
Thread 0 访问 sdata[0] 和 sdata[1]  → Bank 0, Bank 1
Thread 1 访问 sdata[2] 和 sdata[3]  → Bank 2, Bank 3
...没问题

Step s=16:
Thread 0 访问 sdata[0] 和 sdata[16]  → Bank 0, Bank 16 ✓
Thread 1 访问 sdata[32] 和 sdata[48] → Bank 0, Bank 16 ✗ 冲突!
...32-way bank conflict!
```

---

### Version 2: Sequential Addressing (消除 Bank Conflict)

**关键改进：改变归约方向**

```cpp
__global__ void reduce_v2(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // 改进：从大步长开始，逐步减半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**为什么消除了 Bank Conflict？**

```
blockDim.x = 256, s = 128:
Thread 0 访问 sdata[0] 和 sdata[128]   → Bank 0, Bank 0 (同一bank同一地址=广播)
Thread 1 访问 sdata[1] 和 sdata[129]   → Bank 1, Bank 1
...

s = 64:
Thread 0 访问 sdata[0] 和 sdata[64]    → Bank 0, Bank 0
...

连续线程访问连续内存，无冲突！
```

**访存模式对比：**

```
Version 1 (Interleaved):         Version 2 (Sequential):
Step 1: [0,1] [2,3] [4,5]...     Step 1: [0,128] [1,129] [2,130]...
Step 2: [0,2] [4,6] [8,10]...    Step 2: [0,64] [1,65] [2,66]...
→ 步长越来越大，冲突加剧          → 连续访问，无冲突
```

---

### Version 3: First Add During Load (减少 Global Memory 访问)

```cpp
__global__ void reduce_v3(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 改进：每个线程在加载时就做一次加法
    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();
    
    // 后续归约同 v2
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**效果分析：**

```
原来：N 个元素需要 N/blockDim.x 个 block
现在：N 个元素只需要 N/(blockDim.x*2) 个 block

→ Block 数量减半
→ 每个线程做更多工作
→ 更好地隐藏内存延迟
```

**扩展：可以让每个线程加载更多元素**

```cpp
// 每个线程加载 4 个元素
unsigned int i = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
float mySum = 0;
if (i < n) mySum += g_idata[i];
if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
if (i + 2*blockDim.x < n) mySum += g_idata[i + 2*blockDim.x];
if (i + 3*blockDim.x < n) mySum += g_idata[i + 3*blockDim.x];
```

![[assets/Pasted image 20251229161226.png]]



如何找最优呢？后面会介绍grid-strided loop

### Version 4: Unroll Last Warp (利用 Warp 内隐式同步)

**关键洞察**：当 s <= 32 时，所有活跃线程都在同一个 warp 内

在 CUDA 中，**同一 warp 内的线程天然同步执行**（SIMT），不需要 `__syncthreads()`！

```cpp
// Warp 内归约辅助函数（使用 volatile 防止编译器优化）
__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v4(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();
    
    // 只需要归约到 s > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 最后一个 warp 内的归约，无需同步
    if (tid < 32) warpReduce(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**为什么需要 `volatile`？**

没有 `volatile`，编译器可能会：
1. 将 `sdata[tid]` 缓存到寄存器
2. 多次操作后才写回 shared memory
3. 导致其他线程读到旧值

`volatile` 强制每次操作都真正访问 shared memory。

**现代替代方案：使用 `__shfl_down_sync`**（见 Version 6）

---

### Version 5: Complete Unroll (完全展开循环)

**当 blockDim.x 在编译时已知，可以完全展开循环**

```cpp
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce_v5(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    
    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockSize < n) mySum += g_idata[i + blockSize];
    sdata[tid] = mySum;
    __syncthreads();
    
    // 完全展开的归约循环
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// 调用方式：
// reduce_v5<256><<<gridSize, 256, 256*sizeof(float)>>>(d_in, d_out, n);
```

**编译器优化：**

由于 `blockSize` 是编译时常量，编译器会：
1. 消除所有不满足条件的 `if` 分支
2. 完全展开循环
3. 生成最精简的指令序列

---

### Version 6: Warp Shuffle (现代 GPU 最佳实践)

**使用 Warp Shuffle 指令：零延迟，无需 shared memory**

从 Kepler 架构（CC 3.0）开始，CUDA 提供了 **warp shuffle** 指令：

```cpp
// T __shfl_down_sync(unsigned mask, T var, unsigned int delta);
// 让 lane i 获取 lane i+delta 的 var 值
```

```cpp
__device__ float warpReduceSum(float val) {
    // 0xffffffff 表示所有 32 个 lane 都参与
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    // 每个 warp 先内部归约
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    // Warp 0 的前几个线程收集各 warp 的结果
    __shared__ float shared[32];  // 最多 32 个 warp
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // 只有 warp 0 做最后归约
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__global__ void reduce_v6(float *g_idata, float *g_odata, int n) {
    float sum = 0;
    
    // Grid-stride loop：每个线程处理多个元素
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        sum += g_idata[i];
    }
    
    // Block 内归约
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) g_odata[blockIdx.x] = sum;
}
```

**Warp Shuffle 优势：**

| 特性 | Shared Memory | Warp Shuffle |
|------|--------------|--------------|
| 延迟 | ~5 cycles | ~1 cycle |
| 是否需要同步 | 是 | 否（warp内） |
| Bank conflict | 可能 | 不存在 |
| 资源消耗 | 占用 shared memory | 无 |
![[assets/Pasted image 20260102223044.png]]

```
T __shfl_down_sync(unsigned mask, T var, unsigned int delta);

// mask: 哪些 lane 参与 (0xffffffff = 全部 32 个)
// var:  要交换的值 (在寄存器中)
// delta: 从 lane+delta 获取值

// 返回值: lane i 获得 lane i+delta 的 var 值
//         如果 i+delta >= 32，返回自己的 var

### 图解 `__shfl_down_sync`
__shfl_down_sync(0xffffffff, val, 4):

Before:
Lane:    0    1    2    3    4    5    6    7   ...   28   29   30   31
val:    [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7] ... [a28][a29][a30][a31]

After (返回值):
Lane:    0    1    2    3    4    5    6    7   ...   28   29   30   31
result: [a4] [a5] [a6] [a7] [a8] [a9][a10][a11] ... [a28][a29][a30][a31]
                                                      ↑    ↑    ↑    ↑
                                                    超出范围，返回自己的值

Lane 0 得到了 Lane 4 的值
Lane 1 得到了 Lane 5 的值
...
Lane 27 得到了 Lane 31 的值
Lane 28-31 得到自己的值（因为 28+4=32 >= 32）
```


blockreducesum的实现
```
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // 最多 32 个 warp 的结果
    
    int lane = threadIdx.x % 32;  // warp 内位置
    int wid = threadIdx.x / 32;   // warp 编号
    
    // 第一层: 每个 warp 内部归约
    val = warpReduceSum(val);
    
    // 每个 warp 的 lane 0 写入 shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // 第二层: warp 0 归约所有 warp 的结果
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}
```

### 两层归约结构
```
假设 blockDim.x = 256 (8 个 warp)

┌─────────────────────────────────────────────────────────────────────────┐
│                        第一层: Warp 内归约                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Warp 0 (Thread 0-31):    32 个值 ──warpReduce──► sum_0 (在 Lane 0)    │
│  Warp 1 (Thread 32-63):   32 个值 ──warpReduce──► sum_1 (在 Lane 0)    │
│  Warp 2 (Thread 64-95):   32 个值 ──warpReduce──► sum_2 (在 Lane 0)    │
│  ...                                                                    │
│  Warp 7 (Thread 224-255): 32 个值 ──warpReduce──► sum_7 (在 Lane 0)    │
│                                                                         │
│  使用: warp shuffle (无 shared memory，无同步)                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      中间: 写入 Shared Memory                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  if (lane == 0) shared[wid] = val;                                     │
│                                                                         │
│  shared[0] = sum_0  (Thread 0 写入)                                    │
│  shared[1] = sum_1  (Thread 32 写入)                                   │
│  shared[2] = sum_2  (Thread 64 写入)                                   │
│  ...                                                                    │
│  shared[7] = sum_7  (Thread 224 写入)                                  │
│                                                                         │
│  __syncthreads();  // 确保所有 warp 都写完                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      第二层: Warp 0 归约 Warp 结果                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  // 只有 Warp 0 的前 8 个线程参与                                       │
│  val = (threadIdx.x < 8) ? shared[lane] : 0;                           │
│                                                                         │
│  Warp 0, Lane 0: val = shared[0] = sum_0                               │
│  Warp 0, Lane 1: val = shared[1] = sum_1                               │
│  ...                                                                    │
│  Warp 0, Lane 7: val = shared[7] = sum_7                               │
│  Warp 0, Lane 8-31: val = 0  (padding)                                 │
│                                                                         │
│  if (wid == 0) val = warpReduceSum(val);                               │
│                                                                         │
│  → Warp 0 的 Lane 0 持有最终结果！                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 为什么只需要 32 个 shared memory？
```
最大 block 大小 = 1024 线程
1024 / 32 = 32 个 warp
所以最多只有 32 个 warp 结果需要存储

对比 V2-V5:
  需要 shared[blockDim.x] = 256 或 1024 个 float

V6:
  只需要 shared[32] = 32 个 float！
  
Shared memory 使用量: 1024 bytes → 128 bytes (8倍减少！)
```

**Grid-Stride Loop 解释：**

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
     i < n; 
     i += blockDim.x * gridDim.x)
```

- 每个线程不只处理一个元素，而是间隔 `gridSize * blockSize` 处理
- 优势：
  1. 同一份代码适用于任意大小的输入
  2. 可以调整 grid size 优化 occupancy
  3. 更好地利用内存带宽

```
// 核心三要素
for (int i = blockIdx.x * blockDim.x + threadIdx.x;  // 1. 起始位置
     i < n;                                          // 2. 边界
     i += blockDim.x * gridDim.x)                    // 3. 步长
gridDim.x = 2, blockDim.x = 4 (简化示例), n = 20

总线程数 = 2 * 4 = 8
步长 (stride) = 8

线程编号和起始 i:
  Block 0: Thread 0 → i=0, Thread 1 → i=1, Thread 2 → i=2, Thread 3 → i=3
  Block 1: Thread 0 → i=4, Thread 1 → i=5, Thread 2 → i=6, Thread 3 → i=7

Global Memory 索引:
  [ 0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 | 16 17 18 19 ]
    ─────────────────────   ───────────────────────   ───────────
           第1轮                    第2轮                第3轮
           (i)                   (i + 8)             (i + 16)

Thread 0 (Block 0): i = 0, 8, 16     → 处理 3 个元素
Thread 1 (Block 0): i = 1, 9, 17     → 处理 3 个元素  
Thread 2 (Block 0): i = 2, 10, 18    → 处理 3 个元素
Thread 3 (Block 0): i = 3, 11, 19    → 处理 3 个元素
Thread 0 (Block 1): i = 4, 12        → 处理 2 个元素 (20 > 20 停止)
Thread 1 (Block 1): i = 5, 13        → 处理 2 个元素
Thread 2 (Block 1): i = 6, 14        → 处理 2 个元素
Thread 3 (Block 1): i = 7, 15        → 处理 2 个元素

总计: 4*3 + 4*2 = 20 个元素 ✓
```


### Version 7: Cooperative Groups + atomicAdd

**最简洁的实现（CUDA 9.0+）**

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void reduce_v7(float *g_idata, float *g_odata, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    float sum = 0;
    
    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        sum += g_idata[i];
    }
    
    // Warp reduce using cooperative groups
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum += warp.shfl_down(sum, offset);
    }
    
    // 每个 warp 的 lane 0 原子加到结果
    if (warp.thread_rank() == 0) {
        atomicAdd(g_odata, sum);
    }
}
```

**关于 atomicAdd 的性能：**

在旧架构上，全局内存的 atomicAdd 很慢（串行化）。但现代 GPU 上：
- 硬件优化显著改善了原子操作性能
- 对于只有少量原子操作的场景（每个 warp 一次），开销可接受
- 代码极其简洁，易于维护

```
// ═══════════════════════════════════════════════════════════════════════
//                           传统方式
// ═══════════════════════════════════════════════════════════════════════

// Warp 内位置计算
int lane = threadIdx.x % 32;           // 手动计算
int wid = threadIdx.x / 32;            // 手动计算

// Warp shuffle
val += __shfl_down_sync(0xffffffff, val, offset);  // 手动指定 mask

// Block 同步
__syncthreads();                        // 全局函数


// ═══════════════════════════════════════════════════════════════════════
//                      Cooperative Groups 方式
// ═══════════════════════════════════════════════════════════════════════

// 获取线程组
cg::thread_block block = cg::this_thread_block();
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

// Warp 内位置
int lane = warp.thread_rank();          // 更清晰！
int wid = warp.meta_group_rank();       // 更清晰！

// Warp shuffle
val += warp.shfl_down(val, offset);     // 无需手动指定 mask！

// Block 同步
block.sync();                           // 面向对象风格
```

## 4. Roofline 分析与性能建模

### 4.1 Reduce 的 Roofline 分析

**计算 Operational Intensity：**

对于 N 个元素的 sum reduce：
- **FLOPs**: N-1 次加法 ≈ N 次
- **Bytes**: 读取 N 个 float = 4N bytes，写入 1 个 float ≈ 4N bytes
- **OI = N / 4N = 0.25 FLOP/Byte**

这是一个**极低的 operational intensity**！

**与硬件对比（以 A100 为例）：**

```
A100 规格：
- Peak FP32 Performance: 19.5 TFLOPS
- Memory Bandwidth: 2039 GB/s

Ridge Point（斜率 = 带宽，平台 = 峰值算力）:
OI_ridge = 19.5 TFLOPS / 2.039 TB/s = 9.56 FLOP/Byte

Reduce 的 OI = 0.25 << 9.56

→ Reduce 是严重的 memory-bound！
```

**Roofline 图解：**

```
Performance (TFLOPS)
    ^
19.5├─────────────────────────┬─────────
    │                        /│
    │                       / │
    │                      /  │
    │                     /   │
    │                    /    │
    │                   /     │
 0.5├──────────*───────/      │  ← Reduce (OI=0.25)
    │         /       /       │
    │        /       /        │
    │       /       /         │
    │      /       /          │
    │     /       /           │
    └─────┴───────┴───────────┴──────────→ OI
         0.25    9.56
         
* 在 OI=0.25 时，理论最大性能：
  0.25 × 2039 GB/s = 509.75 GFLOPS ≈ 0.5 TFLOPS
```

Profiling (A6000)

可以看到v3的first add优化非常重要

| Kernel | Time(ms) | BW(GB/s) | BW Eff% | GFLOPS | vs V0 |
|--------|----------|----------|---------|--------|-------|
| V0-Naive | 0.3326 | 201.79 | 26.3 | 50.45 | 1.00x |
| V1-NoDiverg | 0.2422 | 277.08 | 36.1 | 69.27 | 1.37x |
| V2-Sequential | 0.2331 | 287.94 | 37.5 | 71.99 | 1.43x |
| V3-FirstAdd | 0.1253 | 535.56 | 69.7 | 133.89 | 2.65x |
| V4-UnrollWarp | 0.1081 | 620.63 | 80.8 | 155.16 | 3.08x |
| V5-FullUnroll | 0.1081 | 621.03 | 80.9 | 155.26 | 3.08x |
| V6-Shuffle | 0.1141 | 588.15 | 76.6 | 147.04 | 2.91x |
| V7-CoopGroups | 0.1110 | 604.48 | 78.7 | 151.12 | 3.00x |
| PyTorch-sum | 0.1016 | 660.73 | 86.0 | 165.18 | 3.27x |
### 4.3  Memory-Bound Kernel 优化策略

既然 Reduce 是 memory-bound，优化目标就是**最大化内存带宽利用率**：

| 策略 | 作用 |
|------|------|
| 合并访存（Coalesced Access） | 32个线程访问连续128字节 |
| 减少读取次数 | First add during load |
| 减少 shared memory 依赖 | Warp shuffle |
| 增加 ILP | 循环展开 |
| Grid-stride loop | 更好的 occupancy |



## Ref

1. [NVIDIA Parallel Reduction (Mark Harris)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
2. [CUB Library Documentation](https://nvlabs.github.io/cub/)

---

## 课后练习题

### 练习 1：reduce kernel 为什么慢？

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">naive reduce 的主要问题是什么？</span></summary>

常见问题包括 global memory 访问不合并、每轮分支导致 warp divergence、跨 block 聚合依赖 atomic 或多 kernel launch、shared memory bank conflict，以及没有利用 warp-level primitive。优化路线通常是 block 内树形归约，再减少 divergence，最后用 warp shuffle 消除最后一个 warp 的 shared memory 同步。

</details>

### 练习 2：warp shuffle 的作用

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么最后 32 个元素适合用 shuffle reduce？</span></summary>

同一个 warp 内线程 lockstep 执行，可以用 shuffle 在寄存器之间直接交换数据，不必写 shared memory，也不需要 block-level barrier。这减少了 shared memory traffic 和同步开销。前提是参与线程 mask 正确，inactive lane 的值不能污染结果。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">reduce 为什么天然 arithmetic intensity 很低？</span></summary>

对 FP32 sum 来说，每个元素通常读 4 bytes，只贡献一次加法，AI 大约是 `1 FLOP / 4 bytes = 0.25 FLOP/Byte`。即使 kernel 写得很好，上界也主要由 HBM bandwidth 决定，而不是 FP32 或 Tensor Core peak。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">parallel reduction 的优化主线是什么？</span></summary>

先保证 global load 合并访存；再让每个 thread 在加载阶段多做一点累加，减少后续归约元素数；block 内用 shared memory 或 warp shuffle 做树形归约；最后把跨 block 汇总交给第二个 kernel 或 atomic。每一步都在减少同步、分支和内存流量。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">为什么 reduce 的最后阶段常用 warp-level primitive？</span></summary>

最后 32 个 lane 在同一个 warp 内 lockstep 执行，可以用 `shuffle` 在寄存器之间交换数据。这样不需要 shared memory round-trip，也不需要 `__syncthreads()`。但必须传正确 active mask，尤其是处理非 32 整数倍长度时。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">reduce kernel 优化后仍比 PyTorch/CUB 慢，通常差在哪里？</span></summary>

库实现会针对不同 dtype、长度、对齐、SM 架构选择不同策略，并处理向量化 load、多元素 per thread、warp/block 级归约、跨 block 汇总和 edge cases。手写教学 kernel 通常只覆盖一条路径，性能接近但不一定超过高度调参的库。

</details>
