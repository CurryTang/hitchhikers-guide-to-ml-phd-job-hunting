# MLSYS1 · GPU 体系结构入门

## Q1 基础环境搭建与 hello world kernel

```quiz
title: 练习题
question: 在 PyTorch JIT 编译 CUDA 扩展时，包含 `<<<>>>` kernel launch 语法的代码应该放在哪里？
answer: B
A. cpp_sources
B. cuda_sources
C. setup.py
explanation: `<<<>>>` 只能由 nvcc 编译，因此需要放在 CUDA 源码部分。
```

### 项目文件架构

```
project/
├── csrc/
│   ├── kernels.cpp     # C++ 声明 + pybind11 绑定
│   └── kernels.cu      # CUDA 内核实现
├── setup.py            # 使用 setuptools 编译扩展
└── main.py             # import 编译好的模块
```

> [!tip]
> - 快速实验用 `load_inline()`，代码嵌入 Python 字符串中
> - 正式项目用 `setup.py` + 分离文件，便于版本控制和调试
> - `.cu` 文件由 nvcc 编译，`.cpp` 文件由系统 C++ 编译器编译

### 内联编译方式

使用 PyTorch JIT 编译 CUDA 内核时，需要将代码分为两部分：

**C++ 声明（cpp_sources）** - 只包含函数声明，由普通 C++ 编译器编译：

```cpp
#include <torch/extension.h>

// 函数声明
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
```

**CUDA 源码（cuda_sources）** - 包含内核定义和调用内核的包装函数：

```cuda
// 内核定义
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 包装函数（必须在 .cu 文件中，因为使用了 <<<>>> 语法）
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), n
    );
    return c;
}
```

> [!important]
> `<<<>>>` 内核启动语法只能被 nvcc 编译，必须放在 `cuda_sources` 中，不能放在 `cpp_sources` 中。

### 编译与加载

```python
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name='cuda_kernels',
    cpp_sources=cpp_source,        # 只有声明
    cuda_sources=full_cuda_source,  # 内核 + 包装函数
    functions=['vector_add'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

# 使用编译好的内核
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = module.vector_add(a, b)
```

## Q2 理解 hello world kernel

### GPU 硬件架构 Intro

NVIDIA GPU 采用层次化的并行架构：

```
GPU
├── SM (Streaming Multiprocessor) × N    # 多个流式多处理器
│   ├── CUDA Cores × M                   # 每个 SM 有多个 CUDA 核心
│   ├── Shared Memory                    # 片上共享内存（快）
│   ├── L1 Cache                         # 一级缓存
│   └── Warp Scheduler                   # 线程束调度器
└── Global Memory (HBM/GDDR)             # 全局显存（慢）
```

**关键概念**：
- **SM**：独立的计算单元，可以同时运行多个线程块
- **Warp**：32 个线程组成一个 warp，是实际执行的最小单位，warp 内的线程同步执行相同指令
- **Shared Memory**：同一 block 内的线程共享，速度接近寄存器
- **Global Memory**：所有线程可访问，但延迟高（~400 cycles）

### CUDA 编程模型

CUDA 将线程组织为三层结构，与硬件对应：

```
Grid (网格)
├── Block 0                    # 线程块，映射到 SM
│   ├── Thread 0..31  (Warp 0) # 线程，映射到 CUDA Core
│   ├── Thread 32..63 (Warp 1)
│   └── ...
├── Block 1
└── ...
```

### 解析

```cuda
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**`__global__`**：声明这是一个 kernel 函数，从 CPU 调用，在 GPU 上执行

**内置变量**：

| 变量 | 含义 | 示例值 |
|------|------|--------|
| `blockIdx.x` | 当前 block 在 grid 中的索引 | 0, 1, 2, ... |
| `blockDim.x` | 每个 block 中的线程数 | 256 |
| `threadIdx.x` | 当前线程在 block 中的索引 | 0, 1, ..., 255 |

**全局索引计算**：
```
idx = blockIdx.x * blockDim.x + threadIdx.x

例如：blocks=4, threads=256, 总共 1024 个线程
Block 0: idx = 0*256 + 0..255  = 0..255
Block 1: idx = 1*256 + 0..255  = 256..511
Block 2: idx = 2*256 + 0..255  = 512..767
Block 3: idx = 3*256 + 0..255  = 768..1023
```

**边界检查** `if (idx < n)`：因为线程总数可能大于数据量，需要防止越界访问

### kernel 启动

```cuda
int threads = 256;
int blocks = (n + threads - 1) / threads;  // 向上取整
vector_add_kernel<<<blocks, threads>>>(a, b, c, n);
```

**`<<<blocks, threads>>>`**：CUDA 特有的 kernel 启动语法
- 第一个参数：grid 中的 block 数量
- 第二个参数：每个 block 中的线程数

**向上取整公式**：`(n + threads - 1) / threads` 确保有足够的线程覆盖所有数据
```
n=1000, threads=256
blocks = (1000 + 255) / 256 = 4
总线程数 = 4 * 256 = 1024 >= 1000 ✓
```

### 执行流程

```
CPU                          GPU
 │                            │
 ├─ 分配 GPU 内存 ────────────►│
 ├─ 拷贝数据到 GPU ───────────►│
 ├─ 启动 kernel ──────────────►├─ 调度 blocks 到 SMs
 │                            ├─ 每个 SM 执行 warps
 │                            ├─ 线程并行计算
 ├─ 等待完成 ◄────────────────┤
 ├─ 拷贝结果回 CPU ◄──────────┤
 │                            │
```

> [!tip]
> 选择 `threads=256` 是因为：
> 1. 是 32（warp size）的倍数，避免资源浪费
> 2. 足够大以隐藏内存延迟
> 3. 不超过硬件限制（通常 1024）





Ref: 
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.

---

## 课后练习题

### 练习 1：CUDA extension 文件边界

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么 kernel launch 必须放在 .cu 文件？</span></summary>

CUDA kernel launch 语法只能由 nvcc 解析。普通 C++ compiler 只能处理 binding、函数声明和 CPU 侧 wrapper。真正包含 global kernel 和 launch syntax 的代码应该放进 cuda source 或 .cu 文件。

</details>

### 练习 2：block / thread / warp 的映射

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">一个 block 有 256 threads，会被拆成多少个 warp？</span></summary>

一个 warp 是 32 threads，所以 256 threads 对应 8 个 warps。一个 block 会整体调度到某个 SM 上，block 内 warps 再由 warp scheduler 发射。优化时要同时看 block 数是否提供足够并行度，以及每个 block 是否占用太多 register 或 shared memory。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">CPU 代码、CUDA kernel、PyTorch binding 三层各自负责什么？</span></summary>

CPU 侧负责分配张量、检查 shape/dtype/device、调用 launcher；`.cu` 里的 kernel 负责 GPU 上每个 thread 实际做什么；PyTorch binding 负责把 Python 调用接到 C++/CUDA 实现上。面试里不要把三层混在一起：`pybind` 不执行并行计算，kernel 也不负责 Python API。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">为什么 kernel 里几乎总要写边界检查？</span></summary>

CUDA launch 通常按 block size 向上取整，实际线程数会大于数据元素数。如果没有 `if (idx < n)`，最后一个 block 的多余线程可能越界读写。这个 bug 在小样例里可能不报错，但会污染显存或造成 nondeterministic failure。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">给 `n=10000, threads=256`，block 数怎么算？为什么不是整除？</span></summary>

用 `(n + threads - 1) // threads` 向上取整，得到 `(10000 + 255) // 256 = 40`。如果只用 `n // threads = 39`，只能覆盖 9984 个元素，最后 16 个元素没有线程处理。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">一个 CUDA hello-world kernel 慢，先不要怀疑什么？应该先查什么？</span></summary>

先不要怀疑算法复杂度。入门 kernel 更常见的问题是 launch 配置太小、数据在 CPU/GPU 之间反复拷贝、没有同步导致计时错误、dtype/device 不一致、边界检查错误。先确认正确性、计时方式和数据流，再谈优化。

</details>
