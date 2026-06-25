## Q1 Basic environment construction and hello world kernel

```quiz
title: Practice
question: When compiling a CUDA extension with PyTorch JIT, where should code containing the `<<<>>>` kernel launch syntax live?
answer: B
A. cpp_sources
B. cuda_sources
C. setup.py
explanation: `<<<>>>` is compiled by nvcc, so it belongs in the CUDA source portion.
```

### Project file structure

```
project/
├── csrc/
│   ├── kernels.cpp     # C++ declarations + pybind11 bindings
│   └── kernels.cu      # CUDA kernel implementation
├── setup.py            # Build the extension with setuptools
└── main.py             # import the compiled module
```

> [!tip]
> - Quick experiment with `load_inline()`, code embedded in a Python string
> - Formal projects use `setup.py` + separated files to facilitate version control and debugging
> - `.cu` files are compiled by nvcc and `.cpp` files are compiled by the system C++ compiler

### Inline compilation method

When compiling a CUDA kernel using PyTorch JIT, you need to divide the code into two parts:

**C++ declarations (cpp_sources)** - Contains only function declarations, compiled by a normal C++ compiler:

```cpp
#include <torch/extension.h>

// Function declaration
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
```

**CUDA source code (cuda_sources)** - Contains kernel definitions and wrapper functions that call the kernel:

```cuda
// Kernel definition
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Wrapper function (must be in the .cu file because it uses the <<<>>> syntax)
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
> `<<<>>>` Kernel startup syntax can only be compiled by nvcc and must be placed in `cuda_sources`, not in `cpp_sources`.

### Compile and load

```python
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name='cuda_kernels',
    cpp_sources=cpp_source,        # declarations only
    cuda_sources=full_cuda_source,  # kernel + wrapper function
    functions=['vector_add'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

# Use the compiled kernel
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = module.vector_add(a, b)
```

## Q2 Understand hello world kernel

### GPU Hardware Architecture Intro

NVIDIA GPUs use a hierarchical parallel architecture:

```
GPU
├── SM (Streaming Multiprocessor) × N    # Multiple streaming multiprocessors
│   ├── CUDA Cores × M                   # Each SM has multiple CUDA cores
│   ├── Shared Memory                    # On-chip shared memory (fast)
│   ├── L1 Cache                         # Level-1 cache
│   └── Warp Scheduler                   # Warp scheduler
└── Global Memory (HBM/GDDR)             # Global device memory (slow)
```

**Key concepts**:
- **SM**: independent computing unit that can run multiple thread blocks at the same time
- **Warp**: 32 threads form a warp, which is the smallest unit of actual execution. The threads in the warp execute the same instructions synchronously
- **Shared Memory**: Threads in the same block share, the speed is close to the register
- **Global Memory**: accessible to all threads, but high latency (~400 cycles)

### CUDA Programming Model

CUDA organizes threads into a three-layer structure, corresponding to the hardware:

```
Grid
├── Block 0                    # Thread block, mapped to an SM
│   ├── Thread 0..31  (Warp 0) # Threads, mapped to CUDA cores
│   ├── Thread 32..63 (Warp 1)
│   └── ...
├── Block 1
└── ...
```

### Analysis

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

**`__global__`**: declares that this is a kernel function, called from the CPU and executed on the GPU

**Built-in variables**:

| Variables | Meaning | Example values ​​|
|------|------|--------|
| `blockIdx.x` | The index of the current block in the grid | 0, 1, 2, ... |
| `blockDim.x` | Number of threads in each block | 256 |
| `threadIdx.x` | The index of the current thread in block | 0, 1, ..., 255 |

**Global Index Calculation**:
```
idx = blockIdx.x * blockDim.x + threadIdx.x

Example: blocks=4, threads=256, for a total of 1024 threads
Block 0: idx = 0*256 + 0..255  = 0..255
Block 1: idx = 1*256 + 0..255  = 256..511
Block 2: idx = 2*256 + 0..255  = 512..767
Block 3: idx = 3*256 + 0..255  = 768..1023
```

**Boundary Check** `if (idx < n)`: Because the total number of threads may be greater than the amount of data, out-of-bounds access needs to be prevented

### kernel startup

```cuda
int threads = 256;
int blocks = (n + threads - 1) / threads;  // Round up
vector_add_kernel<<<blocks, threads>>>(a, b, c, n);
```

**`<<<blocks, threads>>>`**: CUDA-specific kernel startup syntax
- The first parameter: the number of blocks in the grid
- The second parameter: the number of threads in each block

**Round up formula**: `(n + threads - 1) / threads` ensures there are enough threads to cover all data
```
n=1000, threads=256
blocks = (1000 + 255) / 256 = 4
Total threads = 4 * 256 = 1024 >= 1000 ✓
```

### Execution process

```
CPU                          GPU
 │                            │
 ├─ Allocate GPU memory ──────►│
 ├─ Copy data to GPU ─────────►│
 ├─ Launch kernel ────────────►├─ Schedule blocks onto SMs
 │                            ├─ Each SM executes warps
 │                            ├─ Threads compute in parallel
 ├─ Wait for completion ◄─────┤
 ├─ Copy results to CPU ◄─────┤
 │                            │
```

> [!tip]
> `threads=256` was chosen because:
> 1. Is a multiple of 32 (warp size) to avoid waste of resources
> 2. Large enough to hide memory latency
> 3. Does not exceed hardware limit (usually 1024)





Ref:
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
