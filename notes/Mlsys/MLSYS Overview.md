# MLSYS Overview：学习路径与目录

这页是 MLSYS 板块的入口。目标不是替代每篇笔记，而是帮你快速决定：

```text
今天应该读哪一篇？
这篇笔记在系统面试里解决什么问题？
前后依赖关系是什么？
```

## 目录

1. [学习主线](#学习主线)
2. [GPU 与 CUDA 基础](#gpu-与-cuda-基础)
3. [Kernel 优化](#kernel-优化)
4. [训练与推理系统](#训练与推理系统)
5. [精度与量化](#精度与量化)
6. [LLM八股 去哪里了](#llm八股-去哪里了)

## 学习主线

MLSYS 面试题通常按这条链路展开：

```text
GPU architecture
-> CUDA programming model
-> memory / compute bound analysis
-> kernel optimization
-> distributed training
-> inference serving
-> precision / quantization
```

建议顺序：

```text
MLSYS1 -> MLSYS2 -> MLSYS3 -> MLSYS4/5/6 -> MLSYS7/8/9
       -> MLSYS10 -> MLSYS11/12 -> MLSYS13
```

如果你时间很紧，优先读：

| 目标 | 推荐笔记 |
|---|---|
| GPU / CUDA 基础 | [[MLSYS1]], [[MLSYS2]] |
| 性能分析框架 | [[MLSYS3]] |
| Kernel 优化套路 | [[MLSYS4]], [[MLSYS5]], [[MLSYS6]] |
| GEMM / compute-bound | [[MLSYS7 Compute-Bound Kernel (1)]], [[MLSYS8 Compute-Bound Kernel (2)]], [[MLSYS9 Compute-bound kernel (3)]] |
| 分布式训练 | [[MLSYS10 parallelism]] |
| 推理系统 | [[MLSYS11 nano-vllm-1]], [[MLSYS12 nano-vllm-2]] |
| 量化与精度 | [[MLSYS13 Quantization and precision]] |

## GPU 与 CUDA 基础

### [[MLSYS1|MLSYS1 · GPU 体系结构入门]]

先建立 GPU 的执行模型：

- SM / warp / thread block 的关系
- memory hierarchy 的层级
- 为什么 GPU 适合高并行吞吐

### [[MLSYS2|MLSYS2 · CUDA 编程模型]]

把硬件模型落到 CUDA 代码：

- grid / block / thread indexing
- shared memory
- synchronization
- memory coalescing 的基本直觉

## Kernel 优化

### [[MLSYS3|MLSYS3 · Roofline Analysis]]

Roofline 是后面所有 kernel 优化的判断工具。

你要会先问：

```text
这个算子是 memory-bound 还是 compute-bound？
瓶颈是带宽、访存模式、还是算力利用率？
```

### [[MLSYS4|MLSYS4 · CUDA Reduce Kernel]]

Reduce 是最适合练习 CUDA 优化基本功的题：

- tree reduction
- warp divergence
- bank conflict
- unroll
- warp shuffle

### [[MLSYS5|MLSYS5 · Histogram & Scan]]

Parallel primitives 是很多高阶算子的组成块。

重点看：

- histogram 的 atomic / privatization
- scan 的 prefix-sum 思想
- work-efficient vs step-efficient

### [[MLSYS6|MLSYS6 · Memory-Bound Kernel 优化]]

围绕 memory-bound 算子的优化：

- coalescing
- vectorized load/store
- shared memory tiling
- 减少 global memory traffic

## Compute-Bound Kernel

### [[MLSYS7 Compute-Bound Kernel (1)|MLSYS7 · Compute-Bound Kernel (1)]]

Compute-bound 的入口，通常从 GEMM / matmul 思路开始。

### [[MLSYS8 Compute-Bound Kernel (2)|MLSYS8 · Compute-Bound Kernel (2)]]

继续拆解 tiling、register blocking、shared memory reuse 等优化。

### [[MLSYS9 Compute-bound kernel (3)|MLSYS9 · Compute-Bound Kernel (3)]]

更接近面试里会追问的性能细节：

- occupancy
- arithmetic intensity
- tensor core / MMA
- pipeline thinking

## 训练与推理系统

### [[MLSYS10 parallelism|MLSYS10 · 分布式训练并行范式]]

训练系统核心是并行策略：

- data parallel
- tensor parallel
- pipeline parallel
- ZeRO / FSDP
- communication vs computation overlap

### [[MLSYS11 nano-vllm-1|MLSYS11 · nano-vllm 精读 (1)]]

推理系统第一部分，重点理解：

- prefill / decode
- KV cache
- attention FLOPs / memory traffic
- serving 系统为什么和 training 系统不同

### [[MLSYS12 nano-vllm-2|MLSYS12 · nano-vllm 精读 (2)]]

推理系统第二部分，重点理解：

- paged KV cache
- continuous batching
- prefix cache
- block table
- CUDA graph / flash attention integration

## 精度与量化

### [[MLSYS13 Quantization and precision|MLSYS13 · 量化与精度]]

这篇负责把 precision / quantization 串起来：

- FP32 / FP16 / BF16 / FP8
- weight-only quantization
- KV cache quantization
- low precision training 的稳定性问题

## LLM八股 去哪里了

Post-training、RL Infra、推理加速、MoE 系统已经从 MLSYS 主线里拆出去，放到与 `MLSYS`、`LeetCode` 并行的新板块：

```text
LLM八股
```

入口：

- [打开 LLM八股板块](#llm)
- [[MLSYS14 Post-Training Infra|LLM八股 14 · Post-Training Infra]]
- [[MLSYS15 LLM Inference Speculative Decoding DFlash|LLM八股 15 · Inference：Speculative Decoding 到 DFlash]]
- [[MLSYS16 Modern MoE SonicMoE|LLM八股 16 · Modern MoE：SonicMoE]]
- [[MLSYS15 RL Infra 自测 35 问|LLM八股 附录 · RL Infra 自测 35 问]]

这样拆分后：

```text
MLSYS = GPU / kernel / training / inference / precision
LLM八股 = post-training / RL infra / inference acceleration / MoE systems
LeetCode = data structure & algorithm patterns
```
