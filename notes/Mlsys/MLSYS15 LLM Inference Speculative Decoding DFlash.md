# MLSYS15 · Inference：Speculative Decoding 到 DFlash

这篇讲一个推理系统里非常高频、也很容易被问深的主题：

```text
为什么 LLM decode 很慢？
speculative decoding 为什么能无损加速？
Medusa / EAGLE / DFlash 到底差在哪里？
真实 serving 系统里什么时候该开，什么时候不该开？
```

核心结论先放前面：

> Speculative decoding 不是“让小模型替大模型回答”。它是让便宜的 drafter 先提出若干 token，再让昂贵的 target model 一次 forward 并行验证这些 token。只要 rejection sampling 写对，最终输出分布可以和直接从 target model 采样完全一致。

---

## 目录

1. [[#一、decode 为什么是推理系统瓶颈]]
2. [[#二、最朴素的 speculative decoding]]
3. [[#三、为什么它可以保持 target model 分布]]
4. [[#四、加速来自哪里：一个可背的性能模型]]
5. [[#五、从 Medusa 到 EAGLE-3：drafter 怎么变强]]
6. [[#六、DFlash：Block Diffusion Speculative Decoding]]
7. [[#七、Kernel 与系统调度层：推理不是一个 forward]]
8. [[#八、Spec Decode 在 serving runtime 里怎么落地]]
9. [[#九、SGLang / vLLM 里怎么开 DFlash]]
10. [[#十、工程判断：什么时候开，什么时候别开]]
11. [[#十一、面试问答与常见坑]]
12. [[#参考资料]]

---

## 一、decode 为什么是推理系统瓶颈

LLM inference 可以拆成两个阶段：

| 阶段 | 输入 | 输出 | 系统特征 |
|---|---|---|---|
| Prefill | prompt 的所有 token | 第一步 KV cache 和 logits | 大矩阵乘，batch/sequence 维度大，比较 compute-heavy |
| Decode | 每次只新增 1 个 token | 下一个 token | 强串行依赖，KV cache 越来越长，容易 memory-bandwidth-bound |

decode 慢的根因不是“模型不会并行”，而是 autoregressive dependency：

```text
token_t depends on token_0 ... token_{t-1}
token_{t+1} depends on token_t
```

所以普通 decode loop 是这样：

```python
tokens = prompt_ids
kv_cache = None

for step in range(max_new_tokens):
    logits, kv_cache = target_model.forward(tokens[-1:], kv_cache)
    next_token = sample(logits)
    tokens.append(next_token)
```

每生成一个 token，都要跑一次 target model forward。即使每次输入只有一个 token，模型仍然要：

- 读大规模权重
- 读写 KV cache
- 做所有层的 attention / MLP
- 等这个 token 采样出来，才能继续下一步

因此推理系统里常看的指标是：

| 指标 | 含义 | 谁影响最大 |
|---|---|---|
| TTFT | time to first token | prefill、排队、调度 |
| ITL | inter-token latency | decode 每步延迟 |
| TPOT | time per output token | decode 平均成本 |
| Throughput | tokens/s 或 requests/s | batch、KV、调度、spec decode |

Speculative decoding 主要打的是 decode 的 ITL/TPOT。

---

## 二、最朴素的 speculative decoding

设有两个模型：

```text
target model q: 大、慢、最终要保持它的分布
draft model p: 小、快、先猜后续 token
```

一次 speculative step 做三件事：

```text
1. draft model 自回归生成 k 个候选 token
2. target model 一次 forward 并行计算这 k 个位置的 logits
3. 从左到右验证，接受最长合法前缀；遇到拒绝就按修正分布重采样
```

可视化：

```text
context:  A B C

draft proposes:
          d1 d2 d3 d4

target verifies in one pass:
          q1 q2 q3 q4 q5

accepted prefix:
          d1 d2 | reject at d3

next context:
          A B C d1 d2 x
```

关键是 target model 的一次 forward 可以同时得到多个位置的 logits：

```text
输入:  [A, B, C, d1, d2, d3, d4]
输出:        q(d1), q(d2), q(d3), q(d4), q(next)
```

虽然 draft 生成候选仍然是串行的，但 draft 很小，成本低。昂贵的 target 从“每 token 一次 forward”变成“每多个 token 一次 forward”。

一个简化版代码：

```python
def speculative_decode(prompt, target, draft, k):
    tokens = list(prompt)

    while not finished(tokens):
        proposed = []
        draft_ctx = tokens[:]

        for _ in range(k):
            p_logits = draft.forward_next(draft_ctx)
            d = sample(p_logits)
            proposed.append(d)
            draft_ctx.append(d)

        q_logits = target.forward_verify(tokens, proposed)

        accepted, replacement = verify_prefix(proposed, q_logits, draft)
        tokens.extend(accepted)

        if replacement is not None:
            tokens.append(replacement)

    return tokens
```

真实系统里不会这么写，因为要处理 KV cache、batch、tree attention、router、streaming、CUDA graph、logprob 返回等细节。但思路就是：**cheap propose, expensive verify**。

---

## 三、为什么它可以保持 target model 分布

最容易误解的一点：

> 如果 draft model 猜错了，直接丢掉重来，是不是会改变采样分布？

会。所以经典 speculative sampling 不是简单“猜对就收、猜错就 target 采样”，而是带修正的 rejection sampling。

对某个 draft token `x`：

```text
p(x) = draft model probability
q(x) = target model probability
```

接受概率：

$$
\alpha(x) = \min\left(1, \frac{q(x)}{p(x)}\right)
$$

如果接受，就把 `x` 放进输出。如果拒绝，要从修正分布里采样：

$$
q'(x) = \frac{\max(q(x) - p(x), 0)}{\sum_y \max(q(y) - p(y), 0)}
$$

直觉：

```text
draft 给某个 token 的概率太高：
  p(x) > q(x)
  -> 不能全部接受，只按 q/p 的比例接受

draft 给某个 token 的概率偏低：
  p(x) < q(x)
  -> draft 提到它时都接受，不够的概率质量由 q' 补回来
```

伪代码：

```python
def accept_or_resample(draft_token, p_probs, q_probs, rng):
    p = p_probs[draft_token]
    q = q_probs[draft_token]

    accept_prob = min(1.0, q / max(p, 1e-12))
    if rng.random() < accept_prob:
        return draft_token, True

    residual = (q_probs - p_probs).clip(min=0)
    residual = residual / residual.sum()
    return sample_from_probs(residual, rng), False
```

这就是“lossless acceleration”的含义：不是输出文本一定相同，而是采样分布和 target-only decode 相同。对 greedy / temperature=0 的情况，验证逻辑会退化成更简单的 token match。

---

## 四、加速来自哪里：一个可背的性能模型

设：

```text
k = 每轮 draft 的候选 token 数
a = 平均接受 token 数
C_t = target model 一次 decode forward 成本
C_d = draft 生成 k 个 token 的成本
C_v = target verify 一次成本，通常接近一次较长 decode/prefill-like forward
```

普通 decode 生成 `a + 1` 个 token 的成本近似：

```text
(a + 1) * C_t
```

spec decode 一轮的成本近似：

```text
C_d + C_v
```

所以粗略 speedup：

$$
\text{speedup} \approx \frac{(a + 1) C_t}{C_d + C_v}
$$

这解释了所有工程现象：

| 现象 | 原因 |
|---|---|
| draft 越准越快 | `a` 变大，target 每次 verify 能推进更多 token |
| draft 太大反而慢 | `C_d` 变大，抵消 verify 收益 |
| batch 很大时收益下降 | target verify 更容易被 batch 填满，额外 draft 成本更明显 |
| 高温采样收益下降 | draft/target 分布差异变大，接受率下降 |
| 长输出任务更受益 | decode 占比更高，节省空间更大 |

面试里可以把 speculative decoding 说成：

```text
用 draft 的额外小成本，换 target model 的多 token 并行验证。
它的收益由 acceptance length、draft cost、target verify efficiency 三者共同决定。
```

---

## 五、从 Medusa 到 EAGLE-3：drafter 怎么变强

最早的 draft model 可以是一个独立小模型。但独立小模型有两个问题：

```text
1. 训练/部署/显存都要多一个模型
2. 和 target 分布不够贴，接受率不稳定
```

后来很多方法都在回答同一个问题：

> 怎样构造一个便宜但更准的 drafter？

### 5.1 Medusa：给 target 加多个预测头

Medusa 在 target model 后面加多个 decoding heads：

```text
hidden state h_t
  -> head_1 predicts token t+1
  -> head_2 predicts token t+2
  -> head_3 predicts token t+3
```

它的优势是：

- 不需要单独跑一个完整 draft model
- heads 共享 target backbone 的 hidden state
- 可以用 tree attention 一次验证多条候选路径

代价是：

- 需要改 target model 或额外训练 heads
- head 越往后预测越难，候选质量会下降
- tree attention / candidate packing 增加 serving 复杂度

### 5.2 EAGLE：在 feature 级别做 autoregressive draft

EAGLE 系列的直觉是：直接预测 token 太硬，可以先预测下一个 hidden feature，再用 target 的 LM head 得到 token 分布。

简化图：

```text
target hidden feature h_t
  -> lightweight autoregressive drafter predicts h_{t+1}
  -> LM head maps feature to token logits
  -> target verifies
```

EAGLE-3 进一步用 multi-layer feature fusion，让 drafter 看到 target 多层信息，而不只是一层 hidden state。这样候选更准，接受率更高。

EAGLE 这一类方法的核心 tradeoff：

```text
比独立小模型更贴近 target
比直接 target decode 更便宜
但 draft 过程仍然是 autoregressive 的
```

这句话很重要，因为它正好引出 DFlash。

---

## 六、DFlash：Block Diffusion Speculative Decoding

DFlash 的突破点是：

> 传统 EAGLE / MTP 风格方法虽然让 drafter 变小了，但 draft token 仍然一个一个生成。DFlash 用 block diffusion drafter 一次并行生成一整块候选 token。

### 6.1 从 autoregressive draft 到 block draft

传统 draft：

```text
d1 = f(context)
d2 = f(context, d1)
d3 = f(context, d1, d2)
d4 = f(context, d1, d2, d3)
```

DFlash draft：

```text
[d1, d2, d3, d4, ..., dB] = block_denoise(context, mask_tokens)
```

也就是说，draft 阶段本身从串行变成更并行。这个变化对 latency 很关键，因为 spec decode 的总成本里有一项 `C_d`。如果 draft 本身串行，`k` 开大时 `C_d` 会涨；block diffusion 可以把多个 token 放到一个 denoising pass 里。

### 6.2 KV injection：为什么 DFlash 不只是“另一个小模型”

DFlash 不是盲猜。它会把 target model 的 hidden states 注入到 draft model 的 KV cache，让 drafter 条件化在 target feature 上。

可以这样理解：

```text
target prefix hidden states
        │
        ▼
draft KV cache receives target features
        │
        ▼
block diffusion drafter predicts token block
```

这个设计同时解决两个问题：

| 问题 | DFlash 的处理 |
|---|---|
| draft 太弱，接受率低 | 用 target hidden features 约束 drafter |
| draft 太慢，成本高 | 用 block diffusion 并行生成 token block |

### 6.3 Non-causal attention mask

普通 autoregressive decoder 的 attention mask 是 causal：

```text
token i 只能看 token <= i
```

DFlash block drafter 需要在一个 block 内做 denoising，因此 mask 更像：

```text
prefix tokens:     causal / already known
draft block tokens: can attend to prefix and mask-token embeddings
target features:   injected as conditioning context
```

这就是为什么 DFlash 属于“speculative decoding algorithm”，但实现上已经接近一个专门的 diffusion-style draft module。

### 6.4 Anchor 机制

vLLM speculators 文档里把 DFlash 的验证过程解释成 anchor-based speculative decoding。可以把它想成：

```text
1. 从上下文里选 anchor positions
2. DFlash 从 anchors 并行预测多个候选 block
3. target model 验证这些 block
4. 接受最长合法前缀
```

图示：

```text
context:  [........ A ........ B ........ C]
anchors:            A          B          C

DFlash proposes blocks:
          A -> a1 a2 a3 a4
          B -> b1 b2 b3 b4
          C -> c1 c2 c3 c4

target verifies:
          accept longest valid prefix, reject/resample when mismatch
```

系统收益来自两个层次的并行：

```text
block 内 token 并行 draft
多个 anchor / block 并行验证
```

### 6.5 DFlash 怎么放进前面的性能模型

回到公式：

$$
\text{speedup} \approx \frac{(a + 1) C_t}{C_d + C_v}
$$

DFlash 主要做两件事：

```text
降低 C_d:
  block diffusion 一次生成多个 candidate token

提高 a:
  KV injection 让 draft distribution 更接近 target
```

NVIDIA 2026 年 Blackwell 博客给出的公开结果显示，DFlash 在 gpt-oss-120b、Llama 3.1 8B、Qwen3、Gemma 等模型上相比传统 speculative decoding 有更高同并发吞吐加速；DFlash 论文和 vLLM/SGLang 文档也把它定位为 EAGLE-3 之后更强的 lossless speculative decoding 路线。

---

## 七、Kernel 与系统调度层：推理不是一个 forward

前面讲的是 speculative decoding 的算法。真正的 inference 系统里，一次 decode step 至少有两层调度：

```text
request scheduler:
  哪些请求进入本轮 batch？
  谁做 prefill，谁做 decode，谁被抢占？
  KV cache slot 从哪里来？

kernel scheduler:
  这一轮 attention/GEMM/MoE kernel 怎么切 tile？
  variable-length KV 怎么映射到 blocks/pages？
  CUDAGraph 需要静态 shape，但请求长度是动态的，怎么兼容？
```

这就是为什么同一个 speculative algorithm，在 notebook 里能加速，放进 vLLM/SGLang 里却可能需要一堆 runtime 工作。

### 7.1 Decode step 的真实数据流

普通 decode 不是简单调用 `model(input_id)`。一个 serving runtime 通常做下面这些事：

```python
def decode_iteration(waiting, running, kv_allocator):
    # 1. admission control: 选本轮能跑的请求
    batch = scheduler.pick(
        waiting=waiting,
        running=running,
        free_kv_pages=kv_allocator.free_pages(),
        max_num_batched_tokens=MAX_TOKENS,
        max_num_seqs=MAX_SEQS,
    )

    # 2. 为新增 token 分配 KV page
    for req in batch:
        kv_allocator.append_slot(req.request_id)

    # 3. 构造 attention metadata
    metadata = build_decode_metadata(batch)

    # 4. launch kernels
    logits = model.decode_one_token(
        input_ids=batch.last_tokens,
        positions=batch.positions,
        kv_cache=kv_allocator.kv_cache,
        metadata=metadata,
    )

    # 5. sample / stop / stream / release finished KV
    next_tokens = sampler(logits, batch.sampling_params)
    scheduler.commit(batch, next_tokens)
```

面试里要能指出：**scheduler 决定 batch shape，batch shape 决定 kernel 效率**。Speculative decoding 改变了 step 粒度：一次 target verify 可能推进多个 token，因此 scheduler 不能再假设“每个 running request 每轮只追加 1 个 KV slot”。

### 7.2 Paged KV Cache：为什么 decode 需要内存管理器

KV cache 大小近似：

```text
num_layers * 2(K,V) * num_tokens * num_kv_heads * head_dim * bytes
```

长上下文、多并发时，KV cache 比权重更容易成为服务瓶颈。PagedAttention / paged KV 的核心是把每条序列的 KV 切成固定大小的 page：

```text
request A logical tokens:
  [0..15] [16..31] [32..47]

physical KV pages:
  page 7 -> A[0..15]
  page 2 -> A[16..31]
  page 9 -> A[32..47]
```

好处：

- 变长序列不需要一整块连续显存
- 完成的请求可以立刻释放 page
- scheduler 可以根据 free pages 做 admission control

代价：

- attention kernel 需要通过 page table 间接寻址
- page size 影响碎片和访存连续性
- prefix cache / speculative accept / reject 都会改变 page 生命周期

Speculative decoding 的额外问题是：draft token 先被提出来，但只有 accepted prefix 才能成为正式上下文。因此 runtime 常见策略是：

```text
verify 前：
  draft token 可以放在临时 buffer 或 speculative slots

verify 后：
  accepted tokens commit 到正式 KV cache
  rejected suffix 对应的临时 KV / metadata 丢弃
```

如果直接把所有 draft token 写进正式 KV，再回滚，会让 allocator、streaming、prefix cache 全部复杂化。

### 7.3 CUDA-level：decode attention kernel 在做什么

下面是高度简化的 CUDA 视角。真实实现会有 tensor core、warp-level reduction、vectorized load、FlashAttention-style online softmax，但 mental model 是：

```cuda
// one query token attends to paged KV blocks
__global__ void paged_decode_attention(
    half* q, half* k_cache, half* v_cache,
    int* block_table, int* seq_lens,
    half* out
) {
    int seq = blockIdx.x;       // request in batch
    int head = blockIdx.y;      // attention head
    int lane = threadIdx.x;

    float m = -INFINITY;        // online softmax max
    float l = 0.0f;             // online softmax normalizer
    float acc[HEAD_DIM];        // output accumulator

    for (int logical_block = 0; logical_block < num_blocks(seq); ++logical_block) {
        int physical_block = block_table[seq, logical_block];

        // load K/V tile from physical KV page
        half* k_tile = k_cache + physical_block_offset(physical_block, head);
        half* v_tile = v_cache + physical_block_offset(physical_block, head);

        // q @ k, update online softmax, accumulate p * v
        update_attention_tile(q, k_tile, v_tile, &m, &l, acc);
    }

    store_normalized(out, seq, head, acc, l);
}
```

这里有三个 kernel 层面的难点：

| 难点 | 为什么影响 latency |
|---|---|
| 变长序列 | 不同 request 的 `seq_len` 不同，block 数不同，SM 容易负载不均 |
| page table 间接寻址 | KV 不连续，内存 coalescing 更难 |
| 小 batch decode | 每个 token query 很少，算力不一定打满，HBM 访问更像瓶颈 |

FlashInfer 这类 kernel library 的价值就在这里：它不是只给一个 attention kernel，而是给一套能适配不同 KV layout、不同 batch shape、不同 attention variant 的模板和 runtime 调度。

### 7.4 FlashInfer 的 plan/run 模式

FlashInfer 论文里一个很重要的系统点是：请求长度动态变化，但 CUDAGraph 喜欢静态 launch 配置。解决思路可以理解成两阶段：

```text
plan():
  读取本轮 batch 的 seq_lens / page table / qo_indptr
  生成 load-balanced 的调度 metadata
  尽量让 SM 工作量均匀

run():
  所有层复用 plan metadata
  launch attention kernel
  与 CUDAGraph / JIT 模板兼容
```

这比“每层重新根据 shape 动态分配工作”更稳定：

```python
plan = flashinfer.plan(
    qo_indptr=batch.qo_indptr,
    paged_kv_indptr=batch.kv_indptr,
    paged_kv_indices=batch.kv_indices,
    num_heads=num_heads,
    head_dim=head_dim,
)

for layer in layers:
    hidden = layer.attn.run(hidden, kv_cache[layer], plan)
    hidden = layer.mlp(hidden)
```

这类设计解释了一个常见现象：LLM inference 的优化不只是“写一个更快的 CUDA kernel”，而是 **kernel API、metadata format、scheduler、CUDAGraph replay** 一起设计。

### 7.5 Prefill / Decode disaggregation

Prefill 和 decode 的硬件行为不同：

| 阶段 | Kernel 形态 | 系统目标 |
|---|---|---|
| Prefill | 大矩阵、大 sequence，compute-heavy | 尽快得到 first token，吞掉长 prompt |
| Decode | 每步 1 或少量 token，memory-heavy | 稳定 ITL，持续高并发 |

把 prefill 和 decode 混在同一批 GPU 上，会互相干扰：

```text
long prefill 占满 batch token budget
  -> decode request 等待
  -> p95/p99 ITL 变差
```

所以现代 serving 系统会做 chunked prefill、prefill/decode 分离、或者让不同 instance 专门处理不同 phase。Speculative decoding 加剧这个问题：target verify 有点像“短 prefill”，一次输入 `context + draft block`，会把 decode step 变得更 prefill-like。调度器必须决定：

```text
这一轮是让更多普通 decode 进来？
还是给某些请求做 speculative verify？
verify block 太长会不会拖慢别人的 ITL？
```

这就是 DFlash / EAGLE 在 serving 里真正难的地方：算法只说 acceptance length，系统还要看 batch interference。

---

## 八、Spec Decode 在 serving runtime 里怎么落地

### 8.1 Runtime 状态机

一个支持 speculative decoding 的 request 通常不是单一 `RUNNING` 状态，而更像：

```text
PREFILL
  -> DRAFTING
  -> VERIFYING
  -> COMMIT_ACCEPTED
  -> STREAM_OUTPUT
  -> DRAFTING ...
```

如果写成伪代码：

```python
while not req.done:
    if req.state == "DRAFTING":
        req.draft_ids = drafter.propose(req.context, k=req.k)
        req.state = "VERIFYING"

    elif req.state == "VERIFYING":
        q_logits = target.verify(req.context, req.draft_ids)
        accepted, replacement = rejection_sample(req.draft_ids, q_logits)
        req.pending_commit = accepted + maybe(replacement)
        req.state = "COMMIT_ACCEPTED"

    elif req.state == "COMMIT_ACCEPTED":
        kv_cache.commit(req.id, req.pending_commit)
        streamer.emit(req.pending_commit)
        req.state = "DRAFTING"
```

这个状态机需要和 continuous batching 共存。也就是说，同一个 batch 里可能同时有：

```text
普通 decode requests
DFlash draft requests
target verify requests
prefill requests
刚完成、要释放 KV 的 requests
```

调度器的本质是把这些不同形态的工作塞进同一批 GPU，让吞吐、TTFT、ITL、显存都不爆。

### 8.2 Verify kernel 为什么接近 prefill

验证 `k` 个 draft token 时，target 不是逐个 token 跑，而是输入：

```text
[context, d1, d2, ..., dk]
```

对 draft suffix 的每个位置并行算 logits：

```text
logits for d1: target(context)
logits for d2: target(context, d1)
...
logits for dk: target(context, d1, ..., d{k-1})
```

在 kernel 层面，这更像一个短序列 prefill：

```text
Q length = k + 1
KV length = context_len + k
attention mask = causal
```

所以它会消耗更多 `max_num_batched_tokens`，也会引入更多 activation/logit 临时 buffer。线上如果只看 tokens/s，很容易忽略 verify 对其他请求 ITL 的挤压。

### 8.3 DFlash 特别需要 hidden-state plumbing

DFlash 不只是“多一个 draft model”。它需要 target hidden features 作为条件：

```text
target model:
  context -> hidden features

DFlash drafter:
  hidden features + masked block -> draft block
```

因此 serving runtime 要额外支持：

| 需求 | 系统影响 |
|---|---|
| 暴露 target hidden states | target forward API 不能只返回 logits |
| hidden feature cache | 避免重复计算，但要控制显存 |
| drafter KV injection | draft model 的 KV cache 和 target feature 对齐 |
| online speculator training | policy / target 变动时 drafter 也要跟上 |

这解释了为什么 DFlash/EAGLE 的生产化通常依赖 vLLM/SGLang 这类 runtime，而不是在 Transformers loop 上加几行 Python。

### 8.4 正确性边界：lossless 不等于实现简单

Speculative decoding 的理论正确性依赖三个条件：

```text
1. draft token 的 proposal probability p 能被正确拿到
2. target distribution q 是对同一上下文计算的
3. reject 后从 residual distribution q' 采样
```

系统实现里常见破坏点：

| 破坏点 | 后果 |
|---|---|
| tokenizer / chat template 不一致 | draft token 和 target token 对不上 |
| target verify 的 context 少了某个 accepted token | q 不是同一条件分布 |
| top-p/top-k mask 没同步 | residual distribution 错 |
| FP8/INT8 quantized draft 没校准 logprob | acceptance ratio 偏 |
| streaming 先发出未验证 token | 用户看到后来被 reject 的内容 |

所以生产系统通常只 streaming 已经 commit 的 accepted tokens，而不是 draft tokens。

---

## 九、SGLang / vLLM 里怎么开 DFlash

真实部署时你不会手写 `verify_prefix`。你会在 serving runtime 里打开对应 spec decode backend。

### 9.1 SGLang 示例

以公开的 Qwen3-8B DFlash checkpoint 为例，SGLang 启动方式大致是：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --tp-size 1 \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --mem-fraction-static 0.75 \
  --trust-remote-code
```

要关注的不是命令怎么背，而是这些参数对应什么：

| 参数 | 含义 |
|---|---|
| `--model-path` | target model |
| `--speculative-algorithm DFLASH` | 打开 DFlash verifier/drafter 逻辑 |
| `--speculative-draft-model-path` | DFlash drafter checkpoint |
| `--attention-backend` | 是否用 FA3 等高效 attention backend |
| `--mem-fraction-static` | 预留 KV / graph / runtime 内存 |

### 9.2 vLLM 示例

vLLM 的 speculator 配置通常放在 JSON 参数里：

```bash
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3-8B-DFlash-b16", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

`num_speculative_tokens` 不是越大越好：

```text
太小：
  target verify 推进 token 少，收益有限

适中：
  接受率高，draft 成本可控

太大：
  后半段接受率下降，draft/verify 额外成本增加
```

线上一般会按模型、采样参数、请求长度、batch 状态做 profiling，而不是固定一个全局最优值。

---

---

## 十、工程判断：什么时候开，什么时候别开

### 10.1 适合开 speculative decoding 的场景

| 场景 | 原因 |
|---|---|
| 长输出 | decode token 多，节省空间大 |
| 低温 / greedy / deterministic workload | draft acceptance 高 |
| target model 很大 | target forward 贵，多 token verify 更值 |
| batch 不总是打满 | spec decode 能把单请求 latency 拉低 |
| drafter 很贴 target | acceptance length 稳定 |

### 10.2 不一定适合的场景

| 场景 | 风险 |
|---|---|
| 高温 creative writing | draft/target 分布差，接受率低 |
| 极短输出 | draft 初始化和 verify 开销不划算 |
| serving batch 已经极大 | target 吞吐被充分摊薄，spec decode 收益下降 |
| drafter 占显存太多 | KV cache 可用空间下降，反而降低并发 |
| target/draft tokenizer 不一致 | correctness 和实现复杂度都危险 |

### 10.3 线上必须看哪些指标

```text
spec/acceptance_length_p50
spec/acceptance_length_p90
spec/rejection_rate
spec/draft_latency_ms
spec/verify_latency_ms
serving/itl_ms
serving/tpot_ms
serving/gpu_memory_used
serving/max_running_requests
```

一个常见故障模式：

```text
benchmark tokens/s 变高
但是 p95 latency 变差
```

原因可能是 drafter 占了显存，KV cache 容量下降，scheduler 更频繁排队。Speculative decoding 是系统优化，不是单 kernel 优化，必须和 scheduler / memory manager 一起看。

---

### 10.4 读论文时要带着系统问题读

最近几条线可以这样连起来：

| 论文/实现 | 你应该关注什么 |
|---|---|
| FlashInfer | 动态请求形状如何和 CUDAGraph、JIT kernel、load-balanced scheduling 共存 |
| vLLM / SGLang | request scheduler、paged KV、prefix cache、chunked prefill、spec decode 如何组合 |
| EAGLE-3 / EAGLE 3.1 | drafter 质量提高后，serving bottleneck 会从 target 转到 draft / verify plumbing |
| DFlash | 用 block diffusion 降低 autoregressive draft cost，但要求 runtime 支持 hidden-state extraction 与 KV injection |
| MiniMax-M2 系列 | agent RL serving 里 prefill/decode disaggregation、MTP spec decode、global KV cache pool 会一起出现 |

面试时不要只说“DFlash 更快”。更高质量的回答是：

```text
DFlash 降低 draft 串行成本，提高 acceptance length；
但 serving runtime 必须处理 hidden-state plumbing、verify batch scheduling、
KV commit/rollback、streaming correctness 和 p95 ITL 保护。
```

---

## 十一、面试问答与常见坑

### Q1：Speculative decoding 为什么不是 distillation？

Distillation 是让小模型学大模型，最终可能直接用小模型输出。Speculative decoding 的最终分布仍由 target model 决定，draft 只提出候选。候选会被 target 验证，必要时用 rejection sampling 修正。

### Q2：为什么 target 一次 forward 可以验证多个 token？

因为给 target 输入 `context + draft_tokens` 后，transformer 会并行计算每个位置的 logits。第 `i` 个 draft token 是否合理，可以用它前一个位置的 target logits 来验证。

### Q3：acceptance rate 和 acceptance length 有什么区别？

Acceptance rate 通常看 token 级接受比例；acceptance length 看一轮 verify 平均推进多少 token。系统加速更直接依赖 acceptance length，因为它决定 target forward 被 amortize 到多少输出 token 上。

### Q4：DFlash 和 EAGLE 的一句话区别？

EAGLE-3 仍是轻量 autoregressive drafter，只是用 target 多层 feature 提高候选质量。DFlash 用 block diffusion drafter 并行生成 token block，并通过 KV injection 条件化在 target hidden states 上，同时降低 draft latency 和提高接受率。

### Q5：为什么说 DFlash 是 lossless？

只要验证和 rejection sampling 遵循 target distribution，输出分布和 target-only decode 一致。DFlash 改的是候选生成方式，不改最终由 target 验证和修正的原则。

### Q6：线上开 spec decode 的第一步是什么？

先在真实 workload 上记录 baseline：

```text
prompt length distribution
output length distribution
temperature / top_p
batch occupancy
ITL / TTFT / throughput
GPU memory headroom
```

然后逐步打开 spec decode，比较 acceptance length、latency、memory、p95/p99，而不是只看单请求 demo。

---

## 参考资料

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)
- [vLLM: Inside vLLM, Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [vLLM EAGLE 3.1 Blog](https://vllm.ai/blog/2026-05-26-eagle-3-1)
- [NVIDIA: Boost Inference Performance up to 15x on Blackwell Using DFlash Speculative Decoding](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/)
- [LMSYS: Next-Generation Speculative Decoding with DFlash and Spec V2](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- [vLLM Speculators DFlash Documentation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)
- [z-lab/dflash GitHub](https://github.com/z-lab/dflash)
- [Qwen3-8B-DFlash-b16 Hugging Face model card](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16)
- [MiniMax-M2 Series Technical Report](https://arxiv.org/abs/2605.26494)
