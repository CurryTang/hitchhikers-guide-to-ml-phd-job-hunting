# MLSYS12 · nano-vllm 精读 (2)

## nano-vllm Part 2：推理引擎系统设计

> 本章聚焦 nano-vllm 的**推理引擎层**设计——调度、内存管理、模型执行与注意力计算。
> 前置知识：已了解 Part 1 中的模型结构（Qwen3、Tensor Parallelism 等）。

![[assets/Pasted image 20260304162418.png]]

## 1 从用户视角出发：LLM 入口

```python
# nanovllm/llm.py
class LLM(LLMEngine):
    pass
```

`LLM` 直接继承 `LLMEngine`，对外暴露的核心 API 是 `generate()`：

```python
llm = LLM("path/to/qwen3")
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.7, max_tokens=128))
```

这一行背后，nano-vllm 依次完成：tokenize → 调度 → 模型前向 → 采样 → 检查终止条件 → 循环直到全部完成。

---

## 2 全局配置

```python
@dataclass
class Config:
    model: str                          # 模型路径（本地目录）
    max_num_batched_tokens: int = 16384 # 单次 prefill 最大 token 数
    max_num_seqs: int = 512             # 最大并发序列数
    max_model_len: int = 4096           # 单序列最大长度
    gpu_memory_utilization: float = 0.9 # GPU 显存使用率上限
    tensor_parallel_size: int = 1       # TP 并行度
    enforce_eager: bool = False         # 强制 eager 模式（禁用 CUDA Graph）
    kvcache_block_size: int = 256       # KV Cache 块大小（token 数）
```

关键约束：
- `kvcache_block_size` 必须是 256 的倍数——对齐 Triton kernel 的向量化
- `max_num_batched_tokens >= max_model_len`——保证至少能 prefill 一条完整序列
- `__post_init__` 中自动从 HuggingFace config 读取模型参数

---

## 3 Prefill 与 Decode：LLM 推理的两个阶段

理解这两个阶段是理解整个引擎架构的钥匙。

### 3.1 Autoregressive 生成的本质

LLM 的生成是**自回归**的：每次只生成一个 token，拼接到序列再继续前向。关键观察：**之前 token 的 K、V 向量不会随新 token 改变**（causal mask 保证），因此可以缓存——这就是 **KV Cache**。

有了 KV Cache，生成过程自然分成两个特征截然不同的阶段。

### 3.2 Prefill：Compute-Bound

一次性处理完整 prompt（长度 $L$），核心开销是 attention 的 $O(L^2 d)$ 矩阵乘法。

```
Prefill Attention 算术强度:
  FLOPs  = 2 · L² · d
  Bytes  = 2 · L · d · sizeof
  AI     = L / sizeof   （随序列长度线性增长）

  L=2048, bf16 → AI = 1024 FLOP/Byte
  A100 的 compute/bandwidth ≈ 156 FLOP/Byte
  → AI >> 156 → Compute-Bound ✓
```

优化策略：Flash Attention（减少 HBM 访问）、变长 Batching。

### 3.3 Decode：Memory-Bound

每步只处理 **1 个 token**，但要读取整个 KV Cache。

```
Decode Attention 算术强度:
  FLOPs  = 2 · S · d
  Bytes  = 2 · S · d · sizeof
  AI     = 1 / sizeof = 0.5   （与序列长度无关！）

  A100 的 156 FLOP/Byte >> 0.5 → 纯 Memory-Bound ✓
  GPU 计算能力被浪费 99.7%，瓶颈在 HBM 读取速度
```

优化策略：大 batch size（读一次权重服务多个 token）、CUDA Graph（消除 kernel launch overhead）、Paged KV Cache（最大化并发序列数）。

### 3.4 为什么必须分离两个阶段

| 维度 | Prefill | Decode |
|------|---------|--------|
| Bound 类型 | Compute-Bound | Memory-Bound |
| 每步 token 数 | 数百~数千 | 1 |
| Attention 计算 | GEMM ($L \times L$) | GEMV ($1 \times S$) |
| Flash Attention API | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| CUDA Graph | 不适用（输入形状动态） | 非常适用（形状固定） |

nano-vllm 的选择：每轮 `step()` 要么全做 prefill，要么全做 decode，不混合。

> [!note] 关于混合调度（Chunked Prefill）
> vLLM 支持将长 prompt 切片与 decode token 混合同一 batch，提升整体 GPU 利用率。nano-vllm 不实现此特性，保持代码简洁。

### 3.5 KV Cache 的显存消耗

```
每个 token 的 KV Cache = 2 × num_layers × num_kv_heads × head_dim × sizeof(dtype)

例: Qwen3-8B (32 layers, 8 KV heads, head_dim=128, bf16):
  = 2 × 32 × 8 × 128 × 2 = 128 KB / token

512 并发序列 × 2048 context = 128 GB ← 远超单卡！
```

这就是为什么需要 **Paged Attention**。

### 3.6 Paged Attention：从操作系统到 GPU 显存管理

#### 传统方案的问题

为每个序列预分配 `max_model_len` 的连续空间，三个致命问题：

1. **内部碎片**：按最大长度分配，实际使用率通常 < 50%
2. **外部碎片**：序列完成后释放空间，留下大小不一的空洞
3. **无法共享**：相同 prompt prefix 的 KV Cache 各自独立存储

#### 核心思想：虚拟内存分页

| 操作系统概念 | Paged Attention 对应 |
|-------------|---------------------|
| 虚拟地址空间 | 序列的逻辑 KV Cache（token 0, 1, 2, ...） |
| 物理页帧 | GPU 上预分配的固定大小 KV Cache Block |
| 页表 | `block_table`：逻辑 block index → 物理 block ID |
| 页大小 | `block_size`（nano-vllm 中为 256 token） |
| 按需分配 | 序列增长时才分配新 block，非预留 max_model_len |
| 共享页 | Prefix Caching：多序列共享相同 prefix 的物理 block |

#### 分页后的内存布局

```
GPU KV Cache 物理空间（预分配的 block 池）:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ Blk0 │ Blk1 │ Blk2 │ Blk3 │ Blk4 │ Blk5 │ Blk6 │ Blk7 │
│(空闲)│ SeqA  │ SeqB │(空闲)│ SeqA │ SeqB │ (空闲)│共享   │
│      │0-255 │0-255 │      │256-  │256-  │      │prefix│
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

SeqA (len=500): block_table=[1,4]  ← 2 blocks，利用率 500/512=98%
SeqB (len=456): block_table=[2,5]
SeqC (共享 prefix): block_table=[7,3]  ← Block7 与他序列共享
```

对比传统：SeqA 按 max=4096 分配，利用率 500/4096=12%。

#### 三个关键机制

**前提定义**：每个序列维护一个 `block_table: list[int]`，将**逻辑 block 索引**（0, 1, 2, ...）映射到 GPU 上**物理 block ID**。所有物理 block 在引擎启动时一次性预分配，存于全局 KV Cache tensor `[2, layers, num_blocks, block_size, heads, d]`，物理 block ID 就是这个 tensor 第 3 维的索引。`block_size=256` 即每个 block 存 256 个 token 的 K/V 向量。

---

**机制一：按需分配**

序列**不预留**最大长度的空间，只在需要时分配新 block：

```
设 block_size=256，一个序列从 0 开始增长：

len=1   → 逻辑 block 0 刚创建，分配物理 block #5 → block_table=[5]
len=256 → block #5 恰好写满（存了 token 0..255）
len=257 → 跨入逻辑 block 1，分配物理 block #3 → block_table=[5,3]
len=512 → block #3 写满（存了 token 256..511）
len=513 → 跨入逻辑 block 2，再分配新 block ...
```

由 `BlockManager.can_append()` 和 `may_append()` 实现，条件是 `len(seq) % block_size == 1`（刚跨入新 block 的第一个 token）。

---

**机制二：block_table 间接寻址**

序列的 KV Cache 在物理上**不连续**，通过 `block_table` 做一层间接映射：

```
序列总长 500 token，block_size=256，block_table=[5, 3]

逻辑视图（连续）:   token 0..255       | token 256..499
                    逻辑 block 0        逻辑 block 1
                         ↓ block_table[0]=5    ↓ block_table[1]=3

物理视图（不连续）:  Block #5 存 K/V[0..255]   Block #3 存 K/V[256..499]
```

Flash Attention 的 `block_table` 参数直接接收此映射，在 GPU kernel 内完成间接寻址，**无需将分散的 block 拷贝为连续内存**。

---

**机制三：引用计数与前缀共享**

两个序列如果有相同的 prompt prefix，可以共享同一组物理 block 而非各自存一份：

```
SeqA: "系统提示词（256 token）+ 问题A"  block_table=[5, 2]
SeqB: "系统提示词（256 token）+ 问题B"  block_table=[5, 8]
                                              ↑ 共享 Block #5
                                              ref_count = 2
```

`block.ref_count` 记录有多少序列引用了该 block，只有归零时才真正回收。共享通过**内容哈希**匹配：每个 block 的哈希由 `(前一个 block 的哈希, 当前 block token_ids)` 链式计算（xxhash64），相同前缀的序列在 allocate 时会命中同一个哈希，自动复用物理 block。

#### slot_mapping：Token 到物理位置的精确映射

```
物理 slot = block_table[逻辑 block] × block_size + block 内偏移

例: block_table=[5,3], block_size=256
  token 0   → slot = 5×256+0   = 1280
  token 256 → slot = 3×256+0   = 768    ← 注意: slot 不是单调递增的！
  token 400 → slot = 3×256+144 = 912
```

Triton kernel `store_kvcache` 接收 `slot_mapping` 数组，scatter 写入各 token 的 K/V。

#### nano-vllm 的三层实现架构

```
┌──────────────────────────────────────────────────────┐
│ BlockManager (CPU, Python)                           │
│   管理 block 分配/释放/共享/哈希                      │
│   产出: block_table, slot_mapping                    │
├──────────────────────────────────────────────────────┤
│ ModelRunner.prepare_prefill / prepare_decode         │
│   转换 block_table → GPU tensor，异步传输             │
├──────────────────────────────────────────────────────┤
│ Attention Layer (GPU)                                │
│   store_kvcache (Triton): scatter 写入 KV Cache      │
│   flash_attn (CUDA): block_table 间接寻址读取         │
│   物理存储: [2, layers, blocks, block_size, heads, d] │
└──────────────────────────────────────────────────────┘
```

> [!important] 为什么 block_size = 256？
> 太小（如 16）：block_table 变长，管理开销大；太大（如 4096）：内部碎片增加，退化为连续分配。256 对 Flash Attention 的 GPU 内存访问模式友好，平均浪费 128 token ≈ 16 KB，远小于传统方案的 MB 级浪费。

---

## 4 LLMEngine：推理主循环

### 4.1 初始化：多进程与组件创建

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config = Config(model, **config_kwargs)

        # ── 1. 多进程启动（Tensor Parallelism） ──
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()

        # ── 2. 主进程的 ModelRunner（rank 0） ──
        self.model_runner = ModelRunner(config, 0, self.events)

        # ── 3. Tokenizer 和 Scheduler ──
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.scheduler = Scheduler(config)
```

**为什么 TP 要用多进程而非多线程？** 两个约束叠加：

1. **GIL**：Python 同一时刻只有一个线程能执行字节码。多线程做 TP，两块 GPU 的矩阵乘法在 Python 层面串行，无法真正并行。
2. **CUDA context**：每块 GPU 需要一个独立的 CUDA context 管理显存和 kernel 队列。多线程共享同一进程的 context 会引发竞争；多进程天然隔离，每个进程独占一块卡的 context。
3. **NCCL**：`all_reduce` 等集合通信是为进程间设计的，不支持线程间模式。

因此 TP 的正确姿势是：**N 张 GPU = N 个进程，每进程独占一张卡**，进程间通过 NCCL 做张量同步。

nano-vllm 在此基础上做了轻量化：**控制指令**（调度哪些序列、执行什么操作）走 SharedMemory + Event，避免 socket/pipe 开销；只有真正需要合并激活的地方才用 NCCL。

**`ctx = mp.get_context("spawn")`**：Python multiprocessing 有三种启动模式：

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `fork` | 子进程继承父进程完整内存副本（COW） | Linux 默认，但 CUDA 不兼容 |
| `spawn` | 子进程从零启动新 Python 解释器 | **CUDA 安全**，nano-vllm 使用 |
| `forkserver` | 由 server 进程 fork | 介于两者之间 |

**`ctx.Event()`**：跨进程同步原语，类似布尔标志：

```python
event.wait()    # Worker: 阻塞等待
event.set()     # Rank 0: 唤醒 worker
event.clear()   # Worker: 重置，准备下一轮
```

**Worker 的生命周期**：`ModelRunner.__init__` 末尾调用 `self.loop()`（rank > 0 时），进入无限循环——worker 进程**永远不从 `__init__` 返回**。

整个系统启动后分为两个阶段：**初始化**（单次）和**推理循环**（反复）。

**初始化阶段**：Rank 0 先 `process.start()` 把 worker 踢起来，然后自己也做 `ModelRunner(rank=0)` 的初始化。两侧并行加载模型权重、分配 KV Cache、捕获 CUDA Graph，完成后 worker 进入 `loop()` 阻塞等待。

**推理循环阶段**：每次 `step()` 时，Rank 0 把指令写入 SharedMemory，`event.set()` 唤醒 worker；双方同时执行相同的模型前向，遇到 `RowParallelLinear` 等需要合并结果的地方，通过 NCCL `all_reduce` 同步张量；前向结束后 worker 重新阻塞，Rank 0 拿到 logits 继续调度。

```
【初始化阶段（单次）】

  Rank 0 (主进程)              Rank 1 (Worker)
  ─────────────────────        ─────────────────────
  process.start()      ──→     __init__() 开始
  ModelRunner(rank=0)  开始      加载模型权重
    加载模型权重                  分配 KV Cache
    分配 KV Cache                捕获 CUDA Graph
    捕获 CUDA Graph              loop() → event.wait() ← 阻塞在此
  返回，进入 step() 循环


【推理循环阶段（每次 step()）】

  Rank 0 (主进程)              Rank 1 (Worker)
  ─────────────────────        ─────────────────────
  write_shm("run", seqs)
  event.set()          ──→     event.wait() 返回
                                read_shm() 读取指令
  model.forward() 开始          model.forward() 开始
      │                             │
      └──── NCCL all_reduce ────────┘   ← 在 RowParallel 处同步
      │                             │
  model.forward() 结束          model.forward() 结束
  拿到 logits，继续调度          event.wait() ← 重新阻塞
```

### 4.2 Step 循环：调度 → 执行 → 后处理

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

LLMEngine 是**单线程事件循环**，每次 `step()` 完成一轮完整的「调度 → 执行 → 后处理」。`generate()` 在 `while not self.is_finished()` 中反复调用 `step()`。

> [!important] Prefill 优先
> `schedule()` 总是优先处理 prefill。只有 waiting 队列为空时才进入 decode。每轮 step 要么全是 prefill，要么全是 decode——不存在混合调度。

---

## 5 Sequence：序列的生命周期

**Sequence 在整个系统中的位置**：它是数据流的载体。用户的每一条请求进来都会被封装成一个 Sequence 对象，从 `generate()` 一直流转到 `postprocess()` 结束。Scheduler 操作的是 Sequence 的队列，BlockManager 为 Sequence 分配物理 block，ModelRunner 把 Sequence 转换成 GPU 张量，采样结果又写回 Sequence——**所有组件都围绕 Sequence 转**。

Sequence 里存了什么：

| 属性 | 说明 |
|------|------|
| `token_ids` | 完整 token 序列（prompt + 已生成的 completion） |
| `num_prompt_tokens` | prompt 长度（不变，用于区分 prompt 和 completion） |
| `num_cached_tokens` | Prefix Cache 命中的 token 数，prefill 时跳过这部分计算 |
| `block_table` | 该序列占用的物理 block ID 列表，由 BlockManager 维护 |
| `status` | 当前所处阶段（见状态机） |
| `last_token` | 最后一个 token，decode 时模型输入只需要这一个 |

### 状态机

每个 Sequence 在系统中经历一个明确的状态转换：

```
              用户请求到达
                  ↓
              WAITING         ← 在 waiting 队列中等待被调度
                  │
     (scheduler.schedule() 选中，BlockManager 分配 block)
                  ↓
              RUNNING         ← 在 running 队列中，每轮 decode 都会处理
                  │
     (生成 EOS token，或达到 max_tokens)
                  ↓
              FINISHED        ← 从 running 队列移除，block 被释放

              [特殊情况]
              RUNNING ──(KV Cache 空间不足，被抢占)──→ WAITING
```

RUNNING → WAITING 的抢占路径是 nano-vllm 应对显存压力的唯一手段：被抢占的序列已生成的内容**不会丢失**（token_ids 还在），但 KV Cache 会被清空，下次重新进入 RUNNING 时需要重新 prefill 整个序列。

### 序列化优化

Sequence 需要通过 SharedMemory 传给 TP worker，传输的内容越少越好：

```python
def __getstate__(self):
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)
```

关键设计：**prefill 时传完整 `token_ids`**（worker 需要知道所有 token 来构建输入），**decode 时只传 `last_token`**（只有上一步新生成的这个 token 会作为模型输入，worker 不需要历史）。对于长序列，这能把传输量从数千 token 压缩到 1 个 token。

---

## 6 Scheduler：调度的核心逻辑

**Scheduler 在整个系统中的位置**：它是 `step()` 的第一步，每次推理循环的入口。Scheduler 的任务是回答一个问题：**这一轮，应该处理哪些序列，做 prefill 还是 decode？** 它不接触 GPU，纯 CPU 逻辑，产出一个序列列表和一个 `is_prefill` 标志，交给 ModelRunner 去执行。

Scheduler 维护两个队列，对应序列的两种等待状态：

```python
self.waiting: deque[Sequence]  # 新到的请求，还没有做过 prefill
self.running: deque[Sequence]  # 已经 prefill 完，正在逐 token decode
```

### 6.1 Schedule：每轮的决策逻辑

`schedule()` 的决策有严格的优先级：**优先处理 waiting 队列（prefill），只有 waiting 为空时才处理 running 队列（decode）**。两者永远不会混在同一个 batch 里。

原因在于计算特性不同（§3.4 已分析）：prefill 是 compute-bound、decode 是 memory-bound，混合调度会让两边都达不到最优。

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # ── Phase 1: 尽量多地装入 waiting 序列做 prefill ──
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens \
           or not self.block_manager.can_allocate(seq):
            break   # token 总数超限，或 KV Cache block 不够，停止装入
        self.block_manager.allocate(seq)      # 为这个序列分配物理 block
        num_batched_tokens += len(seq) - seq.num_cached_tokens  # prefix cache 命中的不算
        seq.status = RUNNING
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True   # 有 prefill 任务，直接返回

    # ── Phase 2: waiting 为空，处理 running 队列做 decode ──
    while self.running:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            # decode 需要写入新 token 的 KV，但空闲 block 不够了
            # 解决方案：抢占 running 队列最后加入的序列（LIFO），腾出 block
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq); break      # 连自己都要抢占，这轮 decode 跳过
        else:
            self.block_manager.may_append(seq)   # 确认/分配这个 token 需要的 block
            scheduled_seqs.append(seq)
    return scheduled_seqs, False
```

`num_batched_tokens += len(seq) - seq.num_cached_tokens` 这行是 prefix cache 的关键：命中 cache 的 token 不需要经过模型计算，所以不占 batch 的 token 配额，这让有共同系统提示词的请求可以装入更多序列。

### 6.2 Preemption 抢占机制

**抢占只发生在 decode 阶段**。Prefill 和 decode 遇到空间不足时的处理方式完全不同：

| 阶段 | 空间不足时的行为 | 已有 RUNNING 序列是否受影响 |
|------|----------------|--------------------------|
| Prefill（Phase 1） | `break`，停止装入更多序列，本轮 prefill 少一些 | **不受影响** |
| Decode（Phase 2） | 主动 `preempt()`，抢占 running 队列尾部的序列 | **受影响，被踢回 WAITING** |

原因也不同：prefill 阶段空间不足说明当前并发序列数已经足够，拒绝新请求即可；decode 阶段空间不足是因为现有序列在持续生长，必须腾出空间才能继续推进。

**被抢占的永远是 RUNNING 状态的序列**。WAITING 中的序列不会被抢占——它们最多是"本轮没被调度到"。

```python
def preempt(self, seq: Sequence):
    seq.status = WAITING
    self.block_manager.deallocate(seq)   # 释放这个序列占用的所有物理 block
    self.waiting.appendleft(seq)         # 放回 waiting 队列的头部（不是尾部！）
```

两个设计细节：

- **LIFO 选择**：抢占 running 队列的**尾部**（最近加入的），因为它 decode 的 token 最少，重新 prefill 的代价最小。
- **放回头部**：`appendleft` 而非 `append`，确保被抢占的序列下一轮优先恢复，不被新请求插队。

> [!note] 与 vLLM 的区别
> vLLM 支持两种抢占策略：**recompute**（清空 KV Cache，恢复时重新 prefill）和 **swap**（把 KV Cache 换出到 CPU 内存，恢复时换回 GPU）。swap 避免了重新 prefill 的计算开销，但实现复杂度高很多。nano-vllm 只实现 recompute，以简洁换取工程代价。

### 6.3 Postprocess：模型执行后的收尾

`postprocess()` 在每次 `step()` 的最后调用，接收模型采样出的 token，把结果写回序列，并判断是否结束。

它做三件事：

**① 追加新 token**：把采样结果 append 到 `seq.token_ids`，序列长度 +1。这个新 token 会成为下一轮 decode 的 `last_token`。

**② 判断终止**：两个条件任满足其一就结束——生成了 EOS token（且没有设置 `ignore_eos`），或者生成长度达到 `max_tokens` 上限。

**③ 清理资源**：终止的序列立即 `deallocate`，把物理 block 归还给 BlockManager，这些 block 马上就能被新请求使用。

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)                          # ① 追加
        if (not seq.ignore_eos and token_id == self.eos) \
           or seq.num_completion_tokens == seq.max_tokens:  # ② 判断
            seq.status = FINISHED
            self.block_manager.deallocate(seq)              # ③ 清理
            self.running.remove(seq)
```

---

## 7 BlockManager：Paged KV Cache 的精髓

**BlockManager 在整个系统中的位置**：它是 KV Cache 显存的"操作系统内核"。GPU 上的物理 KV Cache 空间由 ModelRunner 在启动时一次性分配，BlockManager 负责这块空间的**逻辑管理**——决定哪些 token 占用哪些物理 block，哪些 block 可以共享，什么时候释放。

BlockManager 和其他组件的交互关系：

```
Scheduler.schedule()
  └── can_allocate(seq) → allocate(seq)     # prefill 前：为新序列分配 block
  └── can_append(seq)  → may_append(seq)    # decode 前：为新 token 确认/分配 block
  └── deallocate(seq)                       # 序列结束或被抢占时：释放 block

ModelRunner.prepare_prefill/decode()
  └── 读取 seq.block_table                  # 计算 slot_mapping 和 block_tables tensor
                                            # 传给 GPU kernel 使用
```

BlockManager 本身是**纯 CPU 逻辑**，不碰 GPU。它只管理逻辑映射（block_table、引用计数、哈希表），真正的 KV 向量写入由 Triton kernel 完成，读取由 Flash Attention 完成。

### 7.1 Block 数据结构

```python
class Block:
    block_id: int
    ref_count: int = 0    # 引用计数（支持共享）
    hash: int = -1        # 内容哈希（用于 prefix caching）
    token_ids: list = []  # 该 block 存储的 token 内容
```

BlockManager 的核心数据结构：

```python
self.blocks: list[Block]
self.free_block_ids: deque[int]
self.used_block_ids: set[int]
self.hash_to_block_id: dict[int, int]    # 哈希 → block ID（prefix cache）
```

> [!important] 物理对应关系
> KV Cache 实际存储形状：`[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`。Block 的 `block_id` 直接对应第 3 维的索引，一个 block 存储 `block_size` 个 token 的 K 和 V 向量。

### 7.2 Allocate：首次分配与前缀缓存

```python
def allocate(self, seq: Sequence):
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # 仅对完整 block 计算哈希（最后一个不满的 block 哈希为 -1）
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)

        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True   # 一旦 miss，后续所有 block 都是 miss

        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size   # prefix cache 命中！
            if block_id in self.used_block_ids:
                self.blocks[block_id].ref_count += 1   # 共享已有 block
            else:
                block = self._allocate_block(block_id)

        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

逐步解析：
1. **链式哈希**：每个 block 的哈希由 `(prefix_hash, token_ids)` 计算，保证相同前缀 + 相同内容 → 相同哈希
2. **Cache Miss 传播**：一旦某个 block miss，后续必然 miss（哈希链断裂）
3. **引用计数**：命中的 block 若已被其他序列使用，ref_count +1 实现共享
4. **最后一个不完整 block**：哈希为 -1，不参与 prefix cache（内容还会变化）

### 7.3 Deallocate 与 May_Append：动态扩缩

**Deallocate**（逆序释放，配合 free_block_ids 顺序，尽量保留较早的 prefix cache）：

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)   # 引用计数归零才真正释放
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**Can_Append / May_Append**（decode 时扩展，核心条件极其精巧）：

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
    # len % block_size == 1 表示刚跨入新 block，需要恰好 1 个新 block
    # 否则不需要新 block（在现有 block 剩余空间写入即可）

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        # 分配新 block
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        seq.block_table.append(block_id)
    elif len(seq) % self.block_size == 0:
        # block 刚好填满 → 计算哈希、注册到 prefix cache
        token_ids = seq.block(seq.num_blocks - 1)
        prefix_hash = self.blocks[seq.block_table[-2]].hash if len(seq.block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix_hash)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
```

> [!important] Prefix Cache 的延迟注册
> block 被填满时才计算哈希并注册。**Prefix cache 在 decode 过程中逐步建立**，而非 allocate 时一次性完成。后续序列若有相同 prefix，allocate 时即可命中这些 block。

### 7.4 Prefix Caching 完整流程示例

```
请求 A: "系统提示词 + 用户问题 A"
请求 B: "系统提示词 + 用户问题 B"  ← 共享前 256 个 token
```

1. **请求 A allocate**：Block0 存 "系统提示词"（256 token），`hash_to_block_id[hash_0]=5`
2. **请求 B allocate**：Block0 命中 → `seq.num_cached_tokens += 256`，`blocks[5].ref_count=2`
3. **prefill 请求 B**：`input_ids` 跳过前 256 token，`cu_seqlens_q < cu_seqlens_k`，flash_attn 通过 `block_table` 读取缓存 KV

---

## 8 ModelRunner：从调度到执行

**ModelRunner 在整个系统中的位置**：它是 CPU 调度逻辑和 GPU 计算之间的**桥梁**。Scheduler 产出的是"应该处理哪些序列"的决定，ModelRunner 把这个决定翻译成 GPU 能执行的张量，驱动模型前向，再把采样结果交回给 Scheduler。

一次 `step()` 中 ModelRunner 的职责链：

```
接收 seqs + is_prefill
  │
  ├── prepare_prefill(seqs)  或  prepare_decode(seqs)
  │     ├── 拼接 input_ids、positions
  │     ├── 计算 cu_seqlens_q/k、slot_mapping、block_tables
  │     └── set_context(...)   ← 把这些张量塞入全局 Context
  │
  ├── call("run", ...)  →  通过 SharedMemory + Event 通知 TP worker
  │
  ├── run_model(input_ids, positions, is_prefill)
  │     ├── Prefill：eager 模式，flash_attn_varlen_func
  │     └── Decode： CUDA Graph 模式，flash_attn_with_kvcache
  │
  └── 返回 token_ids（采样结果）给 Scheduler.postprocess()
```

ModelRunner 还负责两件一次性的初始化工作：**KV Cache 分配**（根据剩余显存计算能分配多少 block，一次性 `torch.empty`）和 **CUDA Graph 捕获**（预录制各 batch size 下的 decode 计算图）。

### 8.1 Tensor Parallelism 的进程间通信

```
Rank 0 (主进程)                         Rank 1..N (Worker)
  ├── write_shm(method, args)
  └── event.set() ────────────────────→ event.wait() → read_shm()
                                          getattr(self, method)(*args)
  ←──────────── NCCL all_reduce ──────────────────────────────────┘
```

- **SharedMemory**：2MB 共享内存，pickle 序列化方法名和参数
- **Event**：multiprocessing.Event 作同步信号
- **NCCL**：实际张量同步（all_reduce）发生在 `RowParallelLinear` 内部

### 8.2 KV Cache 分配

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

    # 每个 block 的字节数
    block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
    #             ↑K+V

    # 一次性分配所有 KV Cache
    num_blocks = (total * gpu_utilization - used) // block_bytes
    self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)

    # 将每层的 k_cache/v_cache 指针绑定到 Attention 模块
    for module in self.model.modules():
        if hasattr(module, "k_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
```

> [!important] 预分配策略
> KV Cache 启动时一次性分配完毕，后续 BlockManager 只做逻辑上的 block 分配（修改 block_table），不涉及 GPU 内存的 malloc/free。避免 CUDA 内存碎片化。

### 8.3 Prepare Prefill：变长拼接与 slot_mapping

```python
def prepare_prefill(self, seqs):
    for seq in seqs:
        # 跳过 prefix cache 命中的 token
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(range(seq.num_cached_tokens, len(seq)))

        seqlen_q = len(seq) - seq.num_cached_tokens   # 实际要计算的 Q 长度
        seqlen_k = len(seq)                            # K 的完整长度（含 cache）
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

        # slot_mapping: 只映射需要写入 cache 的 token
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * block_size
            end = start + (block_size if not last_block else seq.last_block_num_tokens)
            slot_mapping.extend(range(start, end))

    # 存在 prefix cache（K 长度 > Q 长度）时需要 block_tables
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)

    set_context(is_prefill=True, cu_seqlens_q, cu_seqlens_k, slot_mapping, block_tables)
```

| 概念 | 说明 |
|------|------|
| `cu_seqlens_q` | Q 的累积序列长度，prefix cache 时只含新计算的 token |
| `cu_seqlens_k` | K 的累积序列长度，始终是完整序列（含 cache） |
| `slot_mapping` | token 在 batch 中的位置 → KV Cache 物理 slot |
| `block_tables` | 仅有 prefix cache 时需要，告诉 flash_attn 从哪些 block 读 KV |

### 8.4 Prepare Decode：逐 token 解码的输入构造

```python
def prepare_decode(self, seqs):
    for seq in seqs:
        input_ids.append(seq.last_token)          # 只需最后一个 token
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))             # 完整上下文长度
        slot_mapping.append(
            seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
        )
    block_tables = self.prepare_block_tables(seqs)   # decode 总是需要 block_tables
    set_context(is_prefill=False, slot_mapping, context_lens, block_tables)
```

与 prefill 对比：

| 维度 | Prefill | Decode |
|------|---------|--------|
| input_ids 长度 | 变长（数百~数千） | 固定 1 per seq |
| slot_mapping | 连续范围 | 单个值 per seq |
| block_tables | 仅 prefix cache 时需要 | 总是需要 |
| context_lens | 不使用 | 告诉 flash_attn 读取多长的 KV |

### 8.5 Run Model：Eager vs CUDA Graph

#### 什么是 CUDA Graph？

正常情况下（Eager 模式），PyTorch 每调用一个算子（矩阵乘法、softmax、norm……），CPU 就要向 GPU 发射一条 kernel launch 指令。这个发射过程本身有开销：CPU 需要准备参数、调用驱动、写入 GPU 的命令队列。

```
Eager 模式（每轮 decode 都重复这个过程）：

CPU                         GPU
 ├─ launch matmul ────────→  执行 matmul
 ├─ launch softmax ───────→  执行 softmax
 ├─ launch norm ──────────→  执行 norm
 ├─ ... (数百个 kernel) ──→  ...
 └─ launch final_proj ────→  执行 final_proj

每次 step() 都要重复上面的发射过程
```

对于 prefill，这不是问题——每个 kernel 要处理大量 token，计算时间远大于发射开销。但 **decode 每个序列只处理 1 个 token**，每个 kernel 的实际计算时间极短，此时 kernel launch 的开销占总时间的比例就很显著了。

**CUDA Graph 的解法**：提前把整个前向计算"录制"成一张图，之后每次 replay 只需 CPU 发射一条指令，GPU 内部按图执行所有 kernel：

```
CUDA Graph 模式：

【捕获阶段（一次性）】
CPU: with torch.cuda.graph(g): model.forward(...)  ← 录制所有 kernel 调用序列

【执行阶段（每次 decode）】
CPU                         GPU
 └─ graph.replay() ──────→  按录制顺序执行所有 kernel（CPU 不再参与）
```

**限制**：CUDA Graph 录制的是固定的计算图，要求**输入输出的形状在捕获和回放时完全一致**。这就是为什么只有 decode 可以用——decode 阶段每次输入都是 `[bs, 1]`（固定形状），而 prefill 的输入长度随 prompt 变化。

nano-vllm 的做法是预捕获一组离散的 batch size（`[1, 2, 4, 8, ..., max_bs]`），实际 decode 时选 ≥ 实际 bs 的最小值，不够的位置用 padding（`slot_mapping=-1`）填充。

---

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # CUDA Graph 路径
        bs = input_ids.size(0)
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["slot_mapping"].fill_(-1)       # -1 表示 padding，Triton kernel 会跳过
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

CUDA Graph 的适用条件：仅用于 decode 阶段（prefill 输入形状不固定）、batch size ≤ 512、非 enforce_eager 模式。

> [!important] CUDA Graph 的性能意义
> Decode 每个序列只产生 1 个 token，计算量极小，此时 **kernel launch overhead 成为瓶颈**。CUDA Graph 消除逐 kernel launch 开销，小 batch size 下可带来 2-3 倍速度提升。

**CUDA Graph 捕获**（从大到小捕获，共享 `graph_pool`，大 graph 先分配内存池，小 graph 复用）：

```python
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
    self.graphs[bs] = graph
```

---

## 9 Attention Layer：KV Cache 写入与双路径计算

**Attention Layer 在整个系统中的位置**：它是 KV Cache 读写的实际执行者，也是 Prefill 与 Decode 两条路径的交汇点。整个推理引擎的每一步调度、分配决策，最终都在这一层落地为 GPU 上的 scatter 写入和 Flash Attention 计算。

理解 Attention Layer 需要先回答两个问题：

1. **它需要哪些"外部"信息**：除了 Q/K/V 张量本身，还需要 `slot_mapping`（写 KV 的位置）、`cu_seqlens_q/k`（序列边界）、`block_tables`（KV 的物理分布）——这些都来自调度层，不是模型计算本身产生的。

2. **这些信息怎么到达这里**：从 ModelRunner 产生，到 Attention Layer 消费，中间要穿过 model.forward 和 N 个 DecoderLayer.forward。如何优雅地传递？


```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()                     # ← 从全局 Context 读取调度元数据

        # Step 1: 将新计算的 K、V 写入 KV Cache（每次前向都要做）
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        # Step 2: 执行 Attention 计算，Prefill 和 Decode 走不同路径
        if context.is_prefill:
            if context.block_tables is not None:    # 有 prefix cache，K/V 需从 cache 读
                k, v = self.k_cache, self.v_cache
            o = flash_attn_varlen_func(q, k, v, ...)
        else:
            o = flash_attn_with_kvcache(q.unsqueeze(1), self.k_cache, self.v_cache, ...)
        return o
```

### 9.1 Context：推理元数据的隐式通道

#### 背景：谁需要这些元数据，谁不需要

Attention Layer 需要的调度元数据，来自 ModelRunner 在 `prepare_prefill/decode` 中的计算：

```
ModelRunner.prepare_prefill()
    ├── 计算 cu_seqlens_q / cu_seqlens_k   （序列边界）
    ├── 计算 slot_mapping                   （KV 写入位置）
    └── 计算 block_tables                   （KV 物理分布）
```

这些数据需要最终到达 `Attention.forward()`，但调用路径是这样的：

```
ModelRunner.run_model(input_ids, positions)
  └→ model.forward(input_ids, positions)
       └→ DecoderLayer.forward(hidden_states, positions)   × 32 层
            └→ Attention.forward(q, k, v)                 ← 在这里消费
```

中间的 `model.forward` 和 32 个 `DecoderLayer.forward` **根本不需要这些元数据**，它们只处理 `hidden_states` 和 `positions`。

#### 问题：参数穿透（Prop Drilling）

如果用函数参数传递,
```python
# DecoderLayer 自己用不到这些参数，但必须接收并往下传
def forward(self, hidden_states, positions,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            slot_mapping, context_lens, block_tables):
    ...
    # 这些参数 DecoderLayer 自己一个都不用，只是为了传给 Attention
    attn_out = self.attn(q, k, v, cu_seqlens_q, cu_seqlens_k,
                         max_seqlen_q, max_seqlen_k, slot_mapping, ...)
```

32 层都要这样写，且每次 nano-vllm 增减一个元数据字段，所有层的签名都要跟着改——这是典型的接口污染。
#### 解决方案：全局 Context（隐式通道）

```
ModelRunner                                  Attention
     │                                           │
     │  set_context(cu_seqlens_q, ...)           │  context = get_context()
     │         ↓                                 │
     │    [全局 Context 单例]   ─────────────────→│
     │
     中间层完全无感知，接口保持干净：
     DecoderLayer.forward(hidden_states, positions)  ← 和 HuggingFace 原版一样
```

代价是引入了隐式状态（全局变量），但收益是模型代码保持干净、便于移植。新模型只需修改 `Attention.forward` 的几行，不需要改动所有中间层接口。

> [!note] vLLM 中的相同设计
> vLLM 用的是 `forward_context` 全局变量，原理完全相同。这是推理引擎领域的通行做法，在干净接口和参数穿透之间的权衡选择了前者。

#### Context 数据结构

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None   # Prefill：Q 的序列边界
    cu_seqlens_k: torch.Tensor | None = None   # Prefill：K 的序列边界（含 prefix cache）
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None   # Prefill + Decode：KV 写入物理 slot
    context_lens: torch.Tensor | None = None   # Decode：每条序列的 KV 上下文长度
    block_tables: torch.Tensor | None = None   # Prefill(prefix cache) + Decode：物理 block 布局
```

生命周期与调用关系：

```
prepare_prefill() / prepare_decode()
  └── set_context(...)       ← 设置：每次 step() 的准备阶段

Attention.forward()          ← 读取：模型前向的每一层
  └── get_context()

run() 结束后
  └── reset_context()        ← 清理：释放 tensor 引用，防止显存泄漏
```

### 9.2 store_kvcache：Triton Kernel 写入分页缓存

#### 为什么要先写再读？

`Attention.forward` 的第一步总是 `store_kvcache`，把当前新计算的 K/V 写入 KV Cache，**然后**再调用 Flash Attention 做计算。

原因：当前 token 的 K/V 是 attention 计算所需 KV 序列的一部分（对于 decode，当前 token 的 K/V 也要参与 attention 计算，因为 `cache_seqlens` 包含了当前位置）。先写入，Flash Attention 直接从 cache 读完整序列，无需额外拼接。

#### 为什么用 Triton 而非 PyTorch 索引？

问题：KV Cache 是分页的（非连续），新计算的 K/V 需要按照 `slot_mapping` 散布到不同的物理位置——这是一次 scatter 写入：

```
新计算的 K (batch 中第 i 个 token)  →  KV Cache 的物理 slot[i]
```

PyTorch 的 `index_put_` 可以做 scatter，但每次调用本身就是一个 kernel launch，开销较大，且无法利用向量化。Triton kernel 将所有 token 的写入合并成一次 kernel launch，并利用 `tl.arange` 做向量化加载/存储：

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                          k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)                           # 每个 thread block 处理一个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return                             # padding token（CUDA Graph 填充），跳过
    key_offsets   = idx * key_stride + tl.arange(0, D)   # 源：batch 第 idx 个 token
    cache_offsets = slot * D + tl.arange(0, D)           # 目标：KV Cache 的 slot 位置
    tl.store(k_cache_ptr + cache_offsets, tl.load(key_ptr + key_offsets))
    # value 同理
```

`D = num_heads * head_dim`：将每个 token 的 `(num_heads, head_dim)` 展平，在内存中连续，可以向量化。`slot == -1` 的跳过逻辑对应 CUDA Graph 的 padding：decode 时 batch size 会被 pad 到离散值，多余的位置 slot=-1，kernel 跳过不写入。

#### 为什么 Triton 比 PyTorch 快？

这本质上是一种 **scatter write 合并优化**，PyTorch 原生实现需要拆成多步：

```python
# PyTorch 等价实现：每步都是一次独立的 CUDA kernel launch
mask = slot_mapping != -1                           # kernel 1: 比较
valid_slots = slot_mapping[mask]                    # kernel 2: compact
k_cache[valid_slots] = key[mask].reshape(-1, D)     # kernel 3+4: gather + scatter
v_cache[valid_slots] = value[mask].reshape(-1, D)   # kernel 5+6: gather + scatter
```

Triton 版本的优势在三个层面：

| 层面 | PyTorch | Triton |
|------|---------|--------|
| kernel launch 次数 | ~6 次（mask/select/reshape/put × 2） | **1 次** |
| slot_mapping 读取 | K 和 V 各读一遍 | **共享读取**，一个线程块同时写 K 和 V |
| 中间张量 | 需要分配 mask、valid_slots 等临时 tensor | **全在寄存器**，零额外分配 |

每次 kernel launch 有约 5–10 μs 的固定开销。对于 prefill（每个 token 做大量计算），这点开销可以忽略；但 **decode 阶段每个序列只写 1 个 token 的 KV**，实际写入时间极短，此时 launch 开销的占比就很显著了——这和 CUDA Graph 消除 launch 开销的动机完全相同。

#### 实测 Benchmark（A6000）

在 A6000 上对两种实现做了完整推理对比：

| 实现 | 整体吞吐 | Decode 实时速度 | 总耗时 |
|------|---------|---------------|--------|
| Triton（nano-vllm 实现） | 3902 tok/s | ~310 tok/s | 34.33s |
| PyTorch（index_put_ 等价实现） | 3867 tok/s | ~284 tok/s | 34.64s |

整体吞吐差距只有 ~0.9%，但 decode 阶段实时速度差了 **~8.5%**（310 vs 284 tok/s）。差距集中在 decode 是预期中的：decode 时 batch 很小，每个序列只有 1 个 token，此时每次 kernel launch 的固定开销占实际计算时间的比例更高。

benchmark 过程中还暴露了 PyTorch 实现的一个重要隐藏限制：**boolean indexing 产生动态形状，无法被 CUDA Graph 捕获**。这意味着 PyTorch 版本和 CUDA Graph 路径不兼容，decode 阶段只能走 eager 模式，损失进一步叠加。Triton kernel 内部的条件分支（`if slot == -1: return`）不影响 grid size，因此天然兼容 CUDA Graph——这是 Triton 实现的另一个隐藏优势。

**`slot_mapping` 的来源**：由 `ModelRunner.prepare_prefill/decode()` 计算，公式是 `block_table[逻辑 block] × block_size + 块内偏移`（§3.6 有详细推导）。`slot_mapping` 不是单调的——不同序列的 KV 散布在物理空间各处，这正是 Triton scatter 写入的意义所在。

### 9.3 Prefill Attention：flash_attn_varlen_func

#### 背景：变长 batch 的 attention 计算

Prefill 时，一个 batch 里有多条不同长度的序列，不能像 decode 那样用统一的张量形状。`flash_attn_varlen_func` 专为变长场景设计，接收拼接后的 Q/K/V（所有序列 token 拼在一起）和 `cu_seqlens` 数组来区分序列边界：

```
cu_seqlens_q = [0, 5, 13]  →  第 0 条序列 Q 范围：[0,5)，第 1 条：[5,13)
```

#### 两种 Prefill 模式

**无 prefix cache**（常规情况）：

- Q/K/V 全部来自当前 prompt 的计算结果（`ModelRunner` 传入）
- `cu_seqlens_q == cu_seqlens_k`，序列长度相同
- `block_table=None`，Flash Attention 直接在连续内存上计算

**有 prefix cache**：

- Q 只包含**未缓存**的 token（跳过了命中 prefix 的部分），`seqlen_q < seqlen_k`
- K/V 需要包含**完整**上下文（含 prefix 的缓存 KV），所以从 KV Cache 读取
- `block_table` 非 None，告诉 Flash Attention 完整 KV 在哪些物理 block

```python
o = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=context.cu_seqlens_q,    # 每个序列 Q 的边界（prefix cache 时比 K 短）
    cu_seqlens_k=context.cu_seqlens_k,    # 每个序列 K 的边界（始终是完整序列长度）
    max_seqlen_q=context.max_seqlen_q,
    max_seqlen_k=context.max_seqlen_k,
    softmax_scale=self.scale,
    causal=True,
    block_table=context.block_tables      # 仅 prefix cache 时非 None；None 时直接用 k,v 参数
)
```

| 条件 | Q 来源 | K/V 来源 | block_table |
|------|--------|----------|-------------|
| 无 prefix cache | 新计算的 Q（完整 prompt） | 新计算的 K, V | `None` |
| 有 prefix cache | 新计算的 Q（跳过 prefix 部分） | 整个 KV Cache（含已缓存） | 非 `None` |

`cu_seqlens_q` 与 `cu_seqlens_k` 的差值就是 prefix cache 节省的计算量——命中 cache 的 token 不参与 Q 侧的计算，直接从 cache 读取 K/V 做 attention。

### 9.4 Decode Attention：flash_attn_with_kvcache

#### 背景：Memory-Bound + 分页 KV

Decode 每个序列只处理 **1 个新 token**，但要 attend 到历史上所有 token 的 KV。历史 KV 分布在物理上不连续的 block 中，必须通过 `block_table` 间接寻址。

`flash_attn_with_kvcache` 是专为 paged decode 设计的 API，内部直接处理 block_table 寻址，不需要提前把 KV 拼接成连续内存：

```python
o = flash_attn_with_kvcache(
    q.unsqueeze(1),           # [batch, 1, num_heads, head_dim]  ← seqlen=1
    k_cache, v_cache,         # [num_blocks, block_size, num_kv_heads, head_dim]
    cache_seqlens=context.context_lens,   # 每个序列有多长的有效 KV（避免读到其他序列数据）
    block_table=context.block_tables,     # [batch, max_blocks_per_seq]，每行是一条序列的 block 地址
    softmax_scale=self.scale,
    causal=True,
)
```

**`q.unsqueeze(1)` 的原因**：Flash Attention 期望 Q 的形状是 `[batch, seqlen_q, num_heads, head_dim]`，decode 阶段每个序列只有 1 个新 token，所以在 seqlen 维度插入 1。

**`cache_seqlens`（即 `context_lens`）的作用**：每条序列的历史 KV 长度不同（生成了不同数量的 token），这个数组告诉 Flash Attention 对每条序列只读取前 `cache_seqlens[i]` 个 slot 的 KV，避免越界读到其他序列的数据。

**为什么 decode 总是需要 `block_table`**：decode 的 KV 完全存在 KV Cache 中（分页的），Flash Attention 必须通过 block_table 做间接寻址才能找到每个 token 的 KV 位置。而 prefill（无 prefix cache 时）的 K/V 直接在函数参数里传入，无需 block_table。

> [!tip] Paged 间接寻址的性能代价
> Decode 本身是 memory-bound，瓶颈在 HBM 读取速度。block_table 间接寻址相当于一层指针跳转，对 memory-bound 计算几乎没有性能损失——因为 GPU 的访存延迟本来就已经是瓶颈，多一次地址计算不是瓶颈所在。

---

## 10 端到端流程串联

以 2 个 prompt（各 5 token），`max_tokens=3` 为例：

### Step 1: Prefill

```
scheduler.schedule():
  → allocate(A): block_table=[0]
  → allocate(B): block_table=[1]
  → return ([A, B], is_prefill=True)

prepare_prefill():
  input_ids    = [A0..A4, B0..B4]       (10 tokens)
  positions    = [0,1,2,3,4, 0,1,2,3,4]
  cu_seqlens_q = [0, 5, 10]
  cu_seqlens_k = [0, 5, 10]
  slot_mapping = [0,1,2,3,4, 256,257,258,259,260]

run_model() [eager, is_prefill]:
  每层 Attention:
    1. store_kvcache → 写入 10 个 token 的 KV
    2. flash_attn_varlen_func → 计算 attention output
  Sampler → [tokenA5, tokenB5]
```

### Step 2-4: Decode（3 轮）

```
scheduler.schedule():
  waiting 为空 → decode
  can_append(A): 6%256=6≠1, 不需要新 block → True

prepare_decode():
  input_ids    = [tokenA5, tokenB5]
  context_lens = [6, 6]
  slot_mapping = [5, 261]       (block0*256+5, block1*256+5)
  block_tables = [[0], [1]]

run_model() [CUDA Graph, bs=2]:
  graph.replay() → 每层 Attention:
    1. store_kvcache → 写入 2 个 token 的 KV
    2. flash_attn_with_kvcache → 从 cache 读完整 KV
  Sampler → [tokenA6, tokenB6]
```

### Step 5: 完成

```
所有序列达到 max_tokens=3 → postprocess: status=FINISHED, deallocate blocks
is_finished() → True → 退出循环 → 返回结果
```

---

## 11 关键设计总结

### Paged Attention 的三层抽象

```
逻辑层 (Sequence)          管理层 (BlockManager)        物理层 (KV Cache Tensor)
seq.block_table=[5,12,3]   blocks[5].ref_count=2        kv_cache[0, layer, 5, :, :, :]
                            hash_to_block_id[h]=5
```

### 与 vLLM 的简化对比

| 特性 | vLLM | nano-vllm |
|------|------|----------|
| Preemption | Recompute + Swap | 仅 Recompute |
| Scheduling | Prefill-Decode 混合（Chunked Prefill）| Prefill 优先，不混合 |
| Prefix Cache | Radix Tree | 简单 Hash Chain |
| CUDA Graph | 支持 | 支持 |
| Tensor Parallelism | Ray / 多进程 | SharedMemory + Event |
| Beam Search | 支持 | 不支持 |

### 核心设计原则

1. **预分配 + 分页管理**：GPU 显存一次性分配，block_table 虚拟化管理，避免碎片化
2. **Prefill/Decode 分离**：两个阶段计算特征不同，分别用最优策略（Flash Attention vs CUDA Graph）
3. **全局 Context**：避免层层传递元数据，简化模型代码
4. **最小化通信**：Sequence 序列化只传必要信息，SharedMemory 避免 socket 开销
5. **延迟 Prefix Cache 注册**：block 填满时才计算哈希，decode 过程中自然建立缓存

## 参考资料

- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [nano-vllm GitHub](https://github.com/GeeeekExplorer/nano-vllm)
- [Flash Attention](https://arxiv.org/abs/2205.14135)

---

## 课后练习题

### 练习 1：scheduler 的核心状态

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">一个 LLM engine scheduler 至少要维护什么？</span></summary>

至少要维护 waiting/running/finished request 队列、每个 request 的 prompt length、generated length、KV block table、sampling params、停止条件、当前 batch token budget 和显存水位。好的 scheduler 还要决定 prefill/decode 优先级、何时抢占、何时释放 KV block。

</details>

### 练习 2：如果生成文本互相串了，优先查什么？

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">block manager bug 排查。</span></summary>

优先查 KV block table 和 block 生命周期：block 是否被过早 free、是否被两个 active sequence 复用、copy-on-write 是否正确、request finish 后是否清理、prefill 到 decode 的 position offset 是否一致。文本串扰通常是 KV cache 映射或 position 管理错误。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">PagedAttention 的三层对象分别是什么？</span></summary>

逻辑层是 sequence 的 token 位置；管理层是 block table、free list、ref count、prefix hash；物理层是预分配的 KV cache tensor。block table 把逻辑 token range 映射到物理 block，让不同长度请求共享同一块显存池。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">prefix cache 为什么通常等 block 填满后再注册？</span></summary>

填满的 block 内容稳定，hash 才适合作为可复用前缀 key。未填满 block 还会继续追加 token，提前注册会导致 hash 变化、复用粒度混乱或错误命中。延迟注册也让 decode 过程中自然形成可复用前缀。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">CUDA Graph 在 decode 阶段解决什么问题？有什么限制？</span></summary>

decode shape 相对固定，CUDA Graph 可以减少 Python/CPU launch overhead，让多层小 kernel 的调度更稳定。限制是 graph 对 batch size、地址和 shape 更敏感，runtime 常要按 batch size 准备多份 graph；动态 shape 或频繁变化的 metadata 会降低收益。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">scheduler 什么时候应该 preempt 一个 request？</span></summary>

当 KV blocks 不够、新请求优先级更高、长请求拖慢整体 tail latency，或需要给 prefill 腾出 token budget 时，可以抢占。抢占策略要决定 recompute 还是 swap/offload。nano-vLLM 简化实现偏 recompute，生产系统会更精细。

</details>
