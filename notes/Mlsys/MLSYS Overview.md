# MLSYS Overview：学习路径与目录

MLSYS 板块按系统层次组织，从 GPU/CUDA 到 kernel、分布式训练、推理服务、post-training infra 和长上下文模型结构。入口页用于快速定位阅读顺序和前后依赖：

```text
今天应该读哪一篇？
每篇笔记解决哪类系统问题？
前后依赖关系是什么？
```

## 目录

1. [学习主线](#学习主线)
2. [GPU 与 CUDA 基础](#gpu-与-cuda-基础)
3. [Kernel 优化](#kernel-优化)
4. [训练与推理系统](#训练与推理系统)
5. [精度与量化](#精度与量化)
6. [Post-training / Efficient Attention / KV Cache / Inference / MoE](#post-training--efficient-attention--kv-cache--inference--moe)
7. [LLM八股 去哪里了](#llm八股-去哪里了)

## 学习主线

MLSYS 系统问题通常按这条链路展开：

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
       -> MLSYS10 -> MLSYS11/12 -> MLSYS13 -> MLSYS14/15/16/17/18
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
| Post-training / RL Infra | [[MLSYS14 Post-Training Infra]] |
| Efficient attention / 长上下文架构 | [[MLSYS15 Efficient Attention Modern Architectures|MLSYS15 Efficient Attention]] |
| KV cache / 长上下文推理 | [[MLSYS15 KV Cache Prefix Caching IndexShare|MLSYS16 KV Cache]] |
| 推理加速 | [[MLSYS15 LLM Inference Speculative Decoding DFlash|MLSYS17 Inference]] |
| MoE 系统 | [[MLSYS16 Modern MoE SonicMoE|MLSYS18 MoE Systems]] |

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

更接近真实性能排查会碰到的细节：

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

精度与量化主线：

- FP32 / FP16 / BF16 / FP8
- weight-only quantization
- KV cache quantization
- low precision training 的稳定性问题

## Post-training / Efficient Attention / KV Cache / Inference / MoE

### [[MLSYS14 Post-Training Infra|MLSYS14 · Post-Training Infra]]

Post-training 属于 MLSYS 主线，因为它讨论的是训练、推理和环境服务共同组成的系统形态：

- rollout / training / reward / weight sync
- veRL、slime、SkyRL、AReaL 等 RL infra 框架
- SearchR1、terminal agent、sandbox 等 agentic RL 负载

### [[MLSYS15 Efficient Attention Modern Architectures|MLSYS15 · Efficient Attention：现代长上下文架构]]

Efficient attention 主线：

- associative memory 视角下的 dense、linear、sparse、hybrid attention
- DeepSeek DSA、DMA、DHSA 的 dynamic sparse routing
- Qwen3-Next、Kimi Linear、MiniMax-M1 的 hybrid / recurrent attention 实现
- prefill、decode、cache manager、spec decode 下的系统约束

### [[MLSYS15 KV Cache Prefix Caching IndexShare|MLSYS16 · KV Cache：内存管理、前缀复用与 IndexShare]]

KV cache 课补上推理系统的 cache 层：

- KV cache 到底存什么
- PagedAttention、prefix cache、RadixAttention 的边界
- KV cache 容量、量化、淘汰、传输
- GLM-5.2 IndexShare / IndexCache 在 Transformers、ATOM/vLLM 里的实现路径

### [[MLSYS15 LLM Inference Speculative Decoding DFlash|MLSYS17 · Inference：并行解码与草稿验证]]

推理系统中的 decode 加速：

- speculative decoding 的 exact sampling
- Medusa、EAGLE、DFlash 的 drafter 设计
- vLLM / SGLang 中如何打开和评估 spec decode

### [[MLSYS16 Modern MoE SonicMoE|MLSYS18 · MoE Systems：路由、通信与 Kernel]]

训练与推理系统中的 MoE 专题：

- router、top-k、capacity、load balance
- expert parallel、all-to-all、grouped GEMM
- SonicMoE 如何优化 fine-grained sparse MoE 的 kernel 和 activation memory

## LLM八股 去哪里了

`LLM八股` 板块现在只放自测/面试题型内容。系统性教程仍然放在 `MLSYS` 主线里。

```text
LLM八股
```

入口：

- [打开 LLM八股板块](#llm)
- [[MLSYS15 RL Infra 自测 35 问|强化学习练习]]

这样拆分后：

```text
MLSYS = GPU / kernel / training / inference / precision / post-training / attention / MoE systems
LLM八股 = RL infra self-check / interview drills
LeetCode = data structure & algorithm patterns
```
