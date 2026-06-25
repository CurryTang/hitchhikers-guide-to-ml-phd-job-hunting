# LLM八股 1 · Post-Training Infra：从 TRL 到 Forge

> [!info] 概述
> 本教程从**系统工程视角**解析 LLM post-training 中的强化学习基础设施，覆盖 TRL、veRL、slime、AReaL、ROLL、Forge 等主流框架。算法部分只讲 PPO 与 GRPO 的关键设计决策，重点在「它们如何决定系统形态」。配套练习见 [[MLSYS15 RL Infra 自测 35 问]]。

---

## 目录

1. [[#一、引言：为什么 RL Infra 是独立的系统问题]]
2. [[#二、全景图：先看地图，再进森林]]
3. [[#三、最小算法背景：PPO 与 GRPO]]
4. [[#四、解剖 RL 训练系统：通用组件与设计轴]]
5. [[#五、框架巡礼：两次范式转移]]
6. [[#六、专题深入]]
7. [[#七、选型决策树与展望]]
8. [[#八、代码级框架精读：slime、SkyRL 与 Sandbox]]
9. [[#九、可以继续做的方向]]

---

## 一、引言：为什么 RL Infra 是独立的系统问题

### 1.1 Post-training 全景

一个 LLM 从预训练到上线，通常经历如下阶段：

```
Pre-training → SFT → RM 训练 → RLHF/RLVR → (Agentic RL)
```

每个阶段的**系统负载形态截然不同**。预训练是静态数据 + 大批次前向/反向传播，工程问题收敛于「吞吐量最大化」。RL 训练则引入了一个根本性的新约束：

> **训练数据由当前策略（policy）在线生产，而不是预先存储在磁盘上。**

这意味着每一个训练步，都必须先用当前模型做一批推理（rollout），把生成的结果当作训练样本，再做反向传播更新模型权重，然后同步更新推理侧的权重，再进行下一轮 rollout。这个「生成 → 训练 → 同步」的环，是 RL Infra 所有复杂性的根源。

### 1.2 RL 训练的独特负载形态

**生成是主要瓶颈。** 来自 16 个开源框架的实测数据表明，80–90% 的训练墙钟时间消耗在 rollout 生成上，而非反向传播。一个直观的数字：

| 配置 | 生成时间（每批 512 rollouts） |
|------|-------------------------------|
| 7B 模型 @ 6300 tok/s，输出 2K token | ~3 分钟 |
| 32B 模型 @ 1200 tok/s，输出 8K token | ~56 分钟 |
| **32B 模型 @ 1200 tok/s，输出 32K token（长推理）** | **~3.7 小时** |

这个数字直接说明：对于 GRPO 训练 DeepSeek-R1 这类长推理模型，**同步等待生成完成再做训练是不可接受的**。异步化不是优化，是必需。

**生成和训练的计算特征完全相反。** 推理引擎（vLLM/SGLang）围绕 decode 优化：paged KV cache、continuous batching、speculative decoding；训练引擎（Megatron/FSDP）围绕大批次前向反向：算子融合、gradient checkpointing、ZeRO 分片。两套系统不能用同一套内核和内存管理方式。这是「为什么不能用训练引擎直接做 rollout」的根本原因。

### 1.3 三难困境（贯穿全文的主线）

RL Infra 的设计本质上是在三个维度之间权衡：

```
       吞吐量 (Throughput)
            ▲
           / \
          /   \
         /     \
On-policyness ─── 灵活性 (Agentic/Env)
```

- **吞吐量**：让 GPU 尽可能忙，rollout 和 train 不互相等待
- **On-policyness**：训练样本来自当前策略，staleness（过时度）低，算法收敛好
- **灵活性**：支持复杂的 agentic 场景（工具调用、多轮对话、自定义 reward 函数）

没有框架能三者全得。这三维的取舍决定了每个框架的基本架构选择。

---

## 二、全景图：先看地图，再进森林

在进入细节之前，先建立整体坐标系。

### 2.1 框架版图（2025 年）

```
─────────────────────────────────────────────────────────
                    同步 (Synchronous)
─────────────────────────────────────────────────────────
   TRL (HuggingFace)  ──  OpenRLHF  ──  veRL (同步模式)
─────────────────────────────────────────────────────────
                       ↓ 异步化
─────────────────────────────────────────────────────────
  veRL (异步)  ──  slime (sync/async 双模式)  ──  ROLL
─────────────────────────────────────────────────────────
                       ↓ 完全异步
─────────────────────────────────────────────────────────
         AReaL (fully async)   ──   Forge (agent-native)
─────────────────────────────────────────────────────────
```

### 2.2 六条设计轴（分析任何框架的坐标系）

| 轴 | 两极 | 核心 Trade-off |
|----|------|----------------|
| **控制流** | Single-controller | Multi-controller | 灵活性 vs 效率 |
| **资源放置** | Colocated（时分复用） | Disaggregated（空间分离） | 显存 vs 利用率 |
| **权重同步** | NCCL broadcast | Filesystem/RDMA | 速度 vs 复杂性 |
| **同步性** | Strictly on-policy | Fully async | 算法正确性 vs 吞吐量 |
| **训练后端** | Megatron-Core | FSDP2 / DeepSpeed ZeRO | MoE/PP 支持 vs 易用性 |
| **Rollout 引擎** | vLLM | SGLang | 各有优劣（详见 §4.3） |

### 2.3 重要数字（请记住这些）

- 权重广播延迟：Qwen3-235B 在 8xH800 上约 **6.75 秒**；Kimi-K2（~1T 参数）在 256xH20 上约 **21.5 秒**
- slime 对 Qwen3-30B-A3B 在 8xH100 上权重传输约 **7 秒**（分桶 NCCL）
- veRL 分桶传输（packed=True）可将广播时间从 ~500ms 压缩到 **~20ms**（适用于较小模型）
- AReaL 相比同步系统在相同 GPU 数量下实现 **2.77× 吞吐提升**

---

## 三、最小算法背景：PPO 与 GRPO

> 本章只讲够用的算法，重心在「算法选择如何决定系统形态」。

### 3.1 PPO 一页纸

PPO 的训练循环有四个模型角色：

| 角色 | 功能 | 显存占用 |
|------|------|----------|
| **Actor** | 当前被训练的策略 | 参数 + 梯度 + 优化器状态 |
| **Reference** | 初始策略（frozen），用于计算 KL 惩罚 | 参数（推理模式） |
| **Critic（Value Model）** | 估计状态价值 $V(s)$，用于计算 advantage | 参数 + 梯度 + 优化器状态 |
| **Reward Model** | 给生成结果打分 | 参数（推理模式） |

PPO 目标函数（裁剪版）：

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] - \beta \cdot \text{KL}[\pi_\theta || \pi_{\text{ref}}]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比率，$\hat{A}_t$ 是 GAE 估计的 advantage，$\epsilon$ 是裁剪阈值（通常 0.2）。

**Advantage 计算（GAE）：** $\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$，其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。这需要 critic 模型对每个状态做推理，是 PPO 系统复杂性的来源之一。

### 3.2 GRPO 一页纸

GRPO（Group Relative Policy Optimization）的核心创新是**去掉 critic 模型**，用 group 内相对比较代替绝对价值估计。

对同一个 prompt $q$，采样 $G$ 个 completion $\{o_1, o_2, ..., o_G\}$，advantage 计算变为：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$

其中 $\mathbf{r} = [r_1, ..., r_G]$ 是同组的 reward。这是简单的 z-score 归一化，**不需要任何神经网络**来估计价值。

> [!important] GRPO 对系统形态的关键影响
> 1. **砍掉 critic**：从 4 个模型变成 3 个（actor + ref + reward），节省约 25% 显存，但失去了精细的步骤级 advantage 估计
> 2. **Group sampling**：同一 prompt 生成 G 个输出（通常 G=4~32），天然形成 prefix sharing 机会——G 个序列共享同一个 prefill，KV cache 可复用
> 3. **KL 的位置**：GRPO 的 KL 可以放在 reward 内（$r' = r - \beta \text{KL}$）或 loss 外（显式正则项），两种位置有不同的数值特性

### 3.3 算法选择如何决定系统形态

| 算法特性 | 系统影响 |
|----------|----------|
| PPO 需要 4 份模型 | 显存压力大，更倾向 colocate + ZeRO / 更复杂的 placement 策略 |
| GRPO 砍掉 critic | 多余显存可用于更大 batch size 或更长 context |
| GRPO 的 group sampling | Prefix-sharing → SGLang 的 RadixAttention 直接受益 |
| clip + IS ratio 的存在 | **这是异步化的算法基础**：$r_t(\theta) = \pi_\theta / \pi_{\text{old}}$ 可以修正 off-policy 误差，允许适度 staleness |
| GRPO 大 group size | 更快的 policy drift → 更频繁需要权重同步 |

**变体一览表**（不展开，按需查阅）：

| 方法 | 相比 GRPO 改了什么 | 系统侧影响 |
|------|---------------------|------------|
| DAPO | 去掉 KL 约束，改 clip lower bound | 可能更激进漂移 |
| GSPO | KL 基于序列级而非 token 级 | reward 计算变化 |
| Dr.GRPO | 对 degenerate group 做特殊处理 | 额外 filter 逻辑 |
| CISPO | IS 修正以容忍更大 staleness | **直接服务于异步架构** |

---

## 四、解剖 RL 训练系统：通用组件与设计轴

### 4.1 三大组件

任何 RL 训练系统都由三个核心组件构成：

```
┌─────────────────────────────────────────────────────┐
│                    Orchestrator                     │
│      (Ray / asyncio / 单控制器 / HTTP gateway)       │
└──────┬─────────────────────────────────────┬────────┘
       │ rollout 数据                         │ 权重更新
       ▼                                     ▼
┌─────────────┐                    ┌──────────────────┐
│   Rollout   │◄── weight sync ────│   Training       │
│   Engine    │                    │   Engine         │
│ vLLM/SGLang │                    │ Megatron / FSDP  │
└─────────────┘                    └──────────────────┘
```

**Rollout Engine** 的核心能力：
- **Paged KV Cache**：将 KV cache 分页管理，避免内存碎片，支持可变长度序列
- **Continuous Batching**：不等待整批完成，新请求随时插入，提升 GPU 利用率
- **Prefix Sharing**（SGLang RadixAttention）：相同前缀的请求共享 KV cache，GRPO 的 group sampling 直接受益

**Training Engine** 的核心能力：
- Megatron-Core：TP × PP × EP × CP 全套并行；pipeline bubble 优化；MoE EP 正确实现
- FSDP2（PyTorch 原生）：ZeRO-3 风格参数分片，易用但缺少 PP 和 EP 支持
- DeepSpeed ZeRO-3：ZeRO offload，适合资源受限场景，MoE EP 支持弱

### 4.2 设计轴 1：控制流（Single vs Multi-controller）

**Single-controller**：一个中央进程编排所有数据流，把数据发到各 worker 执行。
- 优势：数据流逻辑集中在一处，易于理解、调试
- 劣势：控制器本身成为瓶颈；控制器到 worker 的 RPC 开销

**Multi-controller**：每个 worker 组（rollout workers / training workers）有自己的控制器，组间通过消息队列或 RPC 协作。
- 优势：更好的扩展性，控制器不成为瓶颈
- 劣势：数据流逻辑分散，难以全局优化

veRL 的 **HybridFlow** 是这两种模式的混合：用 single-controller 表达高层数据流（「先 rollout，再 compute advantages，再 train」），用 multi-controller（每个 worker 组的 SPMD 进程）执行实际的算子计算。

### 4.3 设计轴 2：资源放置（Colocated vs Disaggregated）

**Colocated（共置）**：rollout 和 training 使用同一批 GPU，时分复用。

```
GPU 0-7:  [Rollout phase] → reshard → [Training phase] → reshard → [Rollout phase]...
```

- 优势：节省硬件，rollout 数据直接在本地用
- 劣势：两个阶段不能重叠；reshard（从训练并行布局切换到推理并行布局）有开销；推理用的 KV cache 和训练用的激活值争抢显存

**Disaggregated（分离）**：rollout 和 training 在不同 GPU 上同时运行。

```
GPU 0-3:  [Rollout]  [Rollout]  [Rollout]...  (持续生成)
GPU 4-7:  [Training] [Training] [Training]... (持续训练)
          ←──── weight update（异步） ────→
```

- 优势：两侧 GPU 都不闲置；rollout 侧可以独立扩展
- 劣势：两侧各自等待对方；硬件成本翻倍；权重同步需要跨机传输

### 4.4 设计轴 3：权重同步

训练更新完 actor 权重后，必须把新权重同步给 rollout 引擎，否则 rollout 用的是旧权重。

**挑战**：训练和推理的并行布局往往不同。训练可能用 TP=8, PP=4，推理可能用 TP=8, PP=1。权重从训练布局 reshard 到推理布局，需要做 AllGather + 切分。

**主流同步方案**：

| 方案 | 延迟 | 代表框架 | 说明 |
|------|------|----------|------|
| NCCL Broadcast（朴素） | 100–500ms | OpenRLHF | 每层分别广播 |
| NCCL + Bucketing | ~20ms | veRL | 把多层参数打包成 1GB 块一次广播 |
| CUDA IPC | <1ms | NeMo-RL, MILES | 同机 GPU 共享内存，无需网络 |
| Filesystem + reload | 秒级 | PRIME-RL, AReaL | 写磁盘，推理侧 reload；适合跨机异步 |
| RDMA P2P（Mooncake） | 超大模型下优于 NCCL | 部分框架 | 1T 参数下 ~16-17s |

**MoE 的额外挑战**：Expert Parallelism 下，每个 GPU 只持有部分 expert。广播前需要先 AllGather 所有 expert 的参数到一处，再广播给推理侧。这个 $O(N_{\text{experts}} \times E_{\text{size}})$ 的开销在密集模型里不存在。

### 4.5 设计轴 4：同步性谱系

```
严格 On-policy                                   完全 Async
     │                                               │
 Batch 0 生成              Buffer 队列             永不停止的
 → 等待完成                存 1-K 步旧数据           rollout workers
 → Training               → Training               → Training（随时）
 → 同步权重               → 同步权重                → 权重异步推送
     │                                               │
  TRL/基础 veRL           veRL async/ROLL/slime      AReaL/Forge
```

**Staleness**（过时度）的定义：生成样本时使用的 policy 版本，距离当前训练的 policy 版本有多少步差距。

- 纯同步：staleness = 0（严格 on-policy）
- Buffer depth = K：staleness ∈ [0, K]
- 完全异步（AReaL）：staleness 通常控制在 8 步以内（实验显示 ≤8 步不影响最终性能）

**CISPO / IS 修正**：$r_t(\theta) = \pi_{\theta}(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$ 本质上是 importance sampling 比率，可以修正 off-policy 误差。这是允许适度 staleness 的算法保证。

---

## 五、框架巡礼：两次范式转移

> **第一次转移**：「训练脚本」→「混合推理+训练引擎」（TRL → veRL）
> **第二次转移**：「同步批次」→「持续异步流」（veRL → AReaL/Forge）

### 5.1 TRL（HuggingFace）—— RLHF 的 Hello World

**定位**：最易上手的 RLHF 框架，研究原型首选。

**架构**：`accelerate` + 单控制器；`PPOTrainer` / `GRPOTrainer` 直接调用。支持 vLLM 作为 colocated 引擎（同进程）或独立 server。

**优势**：集成 HuggingFace 生态，10 行代码跑通 GRPO；适合单机实验；LoRA 支持完善。

**天花板**：
- 单机思维：多节点扩展困难
- Rollout 和 Training 串行：80-90% 时间浪费在等待生成
- 缺乏 Megatron 后端：无 PP/EP，难以支持超大模型

**适用场景**：≤70B 模型，单节点，快速验证 idea。

### 5.2 OpenRLHF —— Ray 分离架构的先驱

**定位**：第一个系统性地用 Ray 把 rollout 和 training 解耦的框架。

**架构**：vLLM 作为独立 rollout service，training 用 DeepSpeed ZeRO。通过 Ray Actor 协调各角色（actor/critic/ref/reward）。

**历史意义**：证明了 rollout service 化的可行性；启发了后来所有框架的分离式设计。

**局限**：缺乏 Megatron 后端（无 TP/PP），大模型支持不足；生态相对 veRL 小。

### 5.3 veRL（ByteDance）—— 混合引擎时代的标志

**定位**：当前最完整的 RL 训练框架（19.6K GitHub stars），工业界最广泛采用。

**核心创新：HybridFlow**

```
Single-controller（Python 端）:
  rollout_data = rollout_engine.generate(batch)
  advantages  = compute_advantages(rollout_data)
  model.train(advantages)           ← 高层数据流，清晰可读

Multi-controller（各 worker 组）:
  TP=4, PP=2 Megatron training workers   ← 实际执行
  TP=8, PP=1 vLLM/SGLang rollout workers ← 实际执行
```

**3D-HybridEngine**：colocate 下 rollout→training 的权重 reshard 流程：
1. 训练结束 → AllGather 参数（从 ZeRO/TP 分片还原）
2. 按推理的 TP 切分 → 广播给推理侧进程
3. 推理引擎加载新权重（reload 或 in-place update）

**双后端**：同时支持 Megatron 和 FSDP，可选 vLLM 或 SGLang 做 rollout。

**异步模式**：veRL 也支持 disaggregated + buffer 异步，但社区使用更多是 colocated 同步模式。

**生态**：DAPO、PRIME、SkyRL、AceMath 等众多工作都基于 veRL fork 实现。

### 5.4 slime（THUDM/智谱）—— 做减法的哲学

**定位**：「我们只做 SGLang 和 Megatron 之间的数据流胶水」。最终实际生产 GLM-4.5/4.6/4.7/GLM-5/GLM-5.1 的 RL 训练后端。

**架构（三服务分离）**：

```
┌───────────────┐   prompt/completion   ┌─────────────────┐
│  Data Buffer  │ ◄──────────────────── │  Rollout Engine │
│ (中间层，协调) │ ──────────────────── ►│  (SGLang 集群)  │
└───────┬───────┘   training data       └─────────────────┘
        │ packing + dispatch                     ▲
        ▼                                        │ weight update
┌───────────────┐                                │ (分桶 NCCL)
│    Training   │ ────────────────────────────── ┘
│   (Megatron)  │
└───────────────┘
```

**数据流详解**（对应 Q34）：
1. Rollout Engine（SGLang）生成 completion，附带 logprobs
2. Data Buffer 收集 rollout 数据，做 reward 计算（可并行调用 reward server）
3. Data Buffer 把数据 pack 成 Megatron 接受的格式，dispatch 给 training workers
4. Megatron 做 forward/backward，loss = PPO/GRPO clip loss，含 IS 修正权重
5. Megatron 更新权重后，通过分桶 NCCL 广播给 SGLang 集群
6. SGLang 集群更新权重（`update_weights` API），继续下一批 rollout

**同步 / 异步双模式**：
- **同步**：等 rollout 全部完成再 train，再同步权重
- **异步**：Data Buffer 是个持续运转的队列；training 和 rollout 解耦，用 IS 修正 staleness

**关键设计选择**：
- 不自己实现并行策略，完全使用 Megatron 的 TP/PP/EP/CP
- 不包装 SGLang，直接暴露 SGLang 的所有参数（`--sglang-xxx` 前缀）
- `OPSM masking`（Optimal Policy Sampling Mask）：只对最优行为的 token 计算梯度，对应 GRPO 里 group 内最高 reward 的序列

### 5.5 AReaL（蚂蚁 & 清华 IIIS）—— 完全异步的代表

**论文**：*AReaL: A Large-Scale Asynchronous RL System for Language Reasoning*（arxiv 2505.24298）

**核心思想**：rollout workers 永远不应该停。一旦 training 需要权重同步，naive 做法会让所有 rollout worker 空等。AReaL 的解法是：**让 rollout 继续跑，收集到的旧数据用 staleness-aware PPO 来修正**。

**关键设计**：

1. **Interruptible Rollout**：rollout worker 的每个序列都可以被中途打断（cancel），不需要等到 EOS。当 training 需要新权重时，可以中断进行中的序列，用新权重重新推理（re-prefill）。

2. **Decoupled PPO clip**：将 PPO 的 clip 范围根据 staleness 动态调整：
   $$\epsilon(s) = \epsilon_0 \cdot e^{-\lambda s}$$
   staleness $s$ 越大，clip 越紧，防止过时数据带来的大梯度更新。

3. **Staleness 控制**：实验表明 staleness ≤ 8 步时，最终性能不受影响；实践中 AReaL 控制 staleness 均值在 2–4 步。

4. **Dynamic batching**：变长输出的动态批处理，GPU 利用率持续 ≥95%。

**性能**：同等 GPU 数量下，相比同步系统实现 **2.77× 吞吐提升**。

**AReaL vs slime 对 rollout 瓶颈的理解**（对应 Q32）：
- slime：rollout 慢是因为生成时间本身长（token-by-token decode），解法是更好的推理引擎（SGLang）+ 分桶权重传输减少 dead time
- AReaL：rollout 慢还因为 **long-tail 序列**拖慢了整批（最慢的一个序列决定批次延迟），解法是 interruptible rollout，让慢序列不阻塞整体

### 5.6 ROLL（阿里巴巴）—— 平台型框架

**定位**：覆盖更多训练场景（RLHF/RLVR/Agentic）的平台型框架，支持五种角色抽象。

**五角色**：actor / critic / reference / reward / **env（环境）**。最后一个是 ROLL 的特色——把 agentic RL 的环境交互做成一等公民，而非事后补丁。

**ROLL Flash**（异步扩展）：在基础 ROLL 之上叠加异步 rollout 能力，支持 agentic 场景下的多轮长序列。

**IS 支持**：内置 TIS（Token-level IS）、TOPR（Top-P Rejection sampling）、CISPO 等多种 off-policy 修正方案。

**后端**：DeepSpeed / Megatron / FSDP2 三选一；vLLM 或 SGLang 做推理。

### 5.7 Forge（MiniMax）—— Agent-Native 时代

**论文/博客**：*Forge: Scalable Agent RL Framework and Algorithm*（Hugging Face Blog by MiniMax-AI）

**核心问题**：如何在训练框架和 agentic scaffold 完全解耦的情况下做 RL？

传统框架的问题：agentic 场景下，agent 的内部结构（记忆压缩、历史改写、multi-agent 协调、工具调用）与训练框架深度耦合，改一个 agent 就要改训练代码。

**Forge 的解法**：引入 **RL Service Gateway**，作为 agent 和训练引擎之间的抽象层：

```
Agent Scaffold                    RL Service Gateway
(任意内部结构) ──► HTTP/RPC ──► Gateway ──► Training Engine
                                   │
                                   ├── 处理 token 级 credit 归属
                                   ├── 支持 context 操纵（memory 压缩等）
                                   └── 统一 reward 归因
```

Agent 不需要知道训练框架的任何细节，只需要把生成的 trajectory 和结果通过 Gateway 上报。Gateway 负责 credit assignment（哪些 token 获得哪些 reward）。

**规模**：用于训练 MiniMax-M2.5，支持 200K token context、十万种以上 agent scaffold、日均百万级样本。

**配套算法 CISPO**（Clipped IS Policy Optimization）：在 IS 修正的基础上加 clip，专门为 Forge 的异步特性设计。

### 5.8 大对比表

| 框架 | 控制流 | 资源放置 | 训练后端 | Rollout 引擎 | 异步支持 | Agentic | 代表模型 |
|------|--------|----------|----------|-------------|----------|---------|---------|
| **TRL** | Single | Colocated | HF Trainer | vLLM/内置 | ✗ | 弱 | 各类小模型 |
| **OpenRLHF** | Ray | Disaggregated | DeepSpeed | vLLM | 弱 | ✗ | — |
| **veRL** | Hybrid | 两者均支持 | Megatron/FSDP | vLLM/SGLang | 双模式 | 弱 | DAPO/SkyRL |
| **slime** | Ray | Disaggregated | Megatron | SGLang | 双模式 | ✗ | GLM-4.5~5.1 |
| **AReaL** | asyncio+Ray | Disaggregated | FSDP2/Megatron | vLLM/SGLang | 完全异步 | 弱 | — |
| **ROLL** | Ray | 两者均支持 | DS/Megatron/FSDP2 | vLLM/SGLang | 双模式 | ✓ | — |
| **Forge** | HTTP Gateway | Disaggregated | 内部 | 内部 | 完全异步 | **原生** | MiniMax-M2.5 |

---

## 六、专题深入

### 6.1 显存账本：GRPO 训练时显存里有几份模型？

（对应 Q20）

**不做任何优化的基准**：

| 组件 | 格式 | 计算方式（7B 模型示例） |
|------|------|------------------------|
| Actor 参数（bf16） | 2B/param | 14 GB |
| Actor 梯度（fp32） | 4B/param | 28 GB |
| Adam 一阶矩（fp32） | 4B/param | 28 GB |
| Adam 二阶矩（fp32） | 4B/param | 28 GB |
| Reference 参数（bf16，frozen） | 2B/param | 14 GB |
| Rollout 引擎权重副本（bf16） | 2B/param | 14 GB |
| KV Cache（推理侧） | 视 context 长度 | 可达数十 GB |
| 激活值（梯度 checkpointing 前） | 视 batch size | 视情况 |
| **合计（训练参数部分）** | — | **~126 GB**（不含 KV/激活） |

**各优化的节省**：

| 优化方法 | 节省内容 | 代价 |
|----------|----------|------|
| ZeRO-3 / FSDP | 把参数/梯度/优化器状态分摊到 N 张卡 | AllGather 通信开销 |
| Gradient Checkpointing | 不保存激活值，反向传播时重算 | ~33% 算力开销 |
| CPU Offload（Adam 状态） | Adam 状态放 CPU，省 56 GB | PCIe 带宽瓶颈 |
| LoRA | 只训练低秩矩阵，梯度/Adam 只有少量参数 | 表达能力受限 |
| Ref 参数共享 | Actor 和 Ref 共享同一份权重，通过 lm_head 之前截断来模拟 | 精度下降 |
| Sleep mode | 不用时把模型权重卸载到 CPU | 加载延迟 |

**GRPO vs PPO 的显存差异**：GRPO 去掉了 critic（省 ~14 GB 参数 + ~56 GB 优化器状态），但 group 采样导致 batch 扩大 G 倍，KV cache 和激活值随之增长。实际上，GRPO 的显存压力未必低于 PPO。

### 6.2 长尾 Rollout 与对策

（对应 Q23）

**长尾问题**：在一批请求里，99% 的序列在 2K tokens 内完成，但 1% 的「超长序列」可能跑到 32K tokens 才结束。在同步系统里，整批数据必须等最慢的序列完成，导致大量 GPU 空转。

**量化**：设 batch = 64 prompts，平均输出 8K tokens，但最长的序列需要 32K tokens。最慢的序列会使整批时间多出 4×，等效 GPU 利用率降至 25%。

**对策**：

| 方法 | 原理 | 框架实现 |
|------|------|---------|
| **Partial Rollout（中断续采）** | 超时后中断序列，下一步继续 | AReaL（re-prefill）、SkyRL（prefix-resume）、slime |
| **Length-aware scheduling** | 按预估长度排队，避免短序列被长序列阻塞 | SGLang 内置，vLLM 部分支持 |
| **Over-sampling + 截断** | 生成 G'>G 个序列，截取最先完成的 G 个 | DAPO、部分 GRPO 实现 |
| **Rejection sampling** | 设最大长度，超出则丢弃，只用合法序列 | 最简单，但浪费算力 |

**GRPO 的特殊挑战**：同一 prompt 的 G 个 completion 需要一起计算 group advantage，意味着它们必须全部完成才能开始训练。这使得 GRPO 对长尾问题特别敏感。

### 6.3 Continuous Batching 在 RL 里的新问题

（对应 Q24/Q25）

**Continuous batching 的基本原理**：传统 batching 等一批请求都完成才出结果；continuous batching 让完成的序列立刻释放 slot，新请求马上插入。这在推理服务中效果极好。

**在 RL 里引入的新问题**：

1. **序列边界对齐**：RL training 需要完整的 episode（从 BOS 到 EOS），中间不能断。Continuous batching 可能导致同一 episode 的 token 散落在不同 batch 里，需要额外对齐逻辑。

2. **Reward 归因时序**：Reward 通常在序列完成后才能计算（terminal reward）。但 process reward（步骤奖励）需要在序列中途触发。Continuous batching 使序列中途的状态难以捕获。

3. **KV cache 压力**：RL rollout 产生的序列比推理服务的序列平均更长，paged KV cache 的页面换出（eviction）更频繁，影响 throughput。

**vLLM vs SGLang 的关键差异**（对应 Q24）：

| 特性 | vLLM | SGLang |
|------|------|--------|
| KV cache 管理 | PagedAttention，固定页大小 | RadixAttention，基于前缀树共享 |
| Prefix 共享 | 需手动启用 | **原生支持**（GRPO 直接受益） |
| Server 接口 | OpenAI 兼容 | OpenAI 兼容 + 更多扩展 |
| RL 专属优化 | update_weights API | update_weights + abort + prefix-resume |
| 生态 | 更大，文档更全 | 更新，SGLang-native 框架（slime）加持 |

**衡量利用率**（对应 Q25）：
- vLLM：`vllm_metrics`，关注 `gpu_cache_usage_perc`（KV 利用率）和 `num_running_seqs`
- SGLang：`/get_server_info` 接口，关注 `cache_hit_rate`（前缀命中率）和 queue depth
- RL 场景下，KV cache 利用率低（<50%）通常意味着 prompt 差异大，prefix 共享失效

### 6.4 异步 RL 的设计空间

（对应 Q27/Q32/Q33）

**为什么要异步？** 同步系统的时序：

```
[Rollout]────────────────┐
                         ▼
                   [Training]────┐
                                 ▼
                         [Weight Sync]──┐
                                        ▼
                                   [Rollout]...
```

每个 `[Weight Sync]` 和等待 rollout 完成的时间都是纯 idle。异步系统把这些阶段流水线化：

```
Rollout workers: [Gen]──[Gen]──[Gen]──[Gen]──[Gen]──...
Training:           [Train]──[Train]──[Train]──...
Weight sync:              [Sync]──────[Sync]──────...
```

**主流异步框架及其解法**（对应 Q27）：

| 框架 | 解决的核心瓶颈 | 机制 |
|------|---------------|------|
| AReaL | 长尾阻塞 + weight sync idle | Interruptible rollout + re-prefill + staleness-aware PPO |
| slime（async 模式） | weight sync dead time | 分桶 NCCL + abort-in-flight + buffer 队列 |
| ROLL Flash | agentic 场景的多轮等待 | 异步 reward server + episode-level buffer |
| PipelineRL | weight sync overhead | 逐 forward pass 更新权重（per-forward-pass swap） |
| PRIME-RL | 跨提供商的大 staleness | 版本跟踪 + depth bound + IS 修正三合一 |

**AReaL vs slime 的根本分歧**（对应 Q32）：

- **slime 视角**：rollout 的主要瓶颈是「权重同步期间的 dead time」和「推理引擎本身的效率」。解法是更快的权重传输（分桶 NCCL）+ 更好的推理引擎（SGLang）。不需要 interruptible rollout 这种复杂机制。

- **AReaL 视角**：rollout 的主要瓶颈是「长尾序列拖慢整批」。只提升权重传输速度解决不了本质问题——1% 的超长序列依然会让 99% 的 GPU 空等。Interruptible rollout + re-prefill 才是根本解法。

两种视角都对，针对不同的业务场景：slime 更适合平均长度适中的 RLVR 任务；AReaL 更适合长 CoT 推理和 agentic 任务。

**Staleness 实践**（对应 Q33）：
- AReaL 论文中，staleness ≤ 8 步时性能不受影响
- 实践中，大多数异步框架控制均值 staleness 在 1–4 步
- 完全不加控制的 staleness 会导致算法发散（相当于把 IS ratio clip 失效化）
- 常见控制手段：buffer depth bound + 超时丢弃 + IS weight clip（$r_t$ clip 到 [0.1, 10]）

**Partial rollout 下的 KV cache 问题**（对应 Q28）：

AReaL 选择 **re-prefill**：中断序列，用新权重重新做 prefill，重建 KV cache，再继续 decode。不保留旧 policy 的 KV cache（会引入 KV 和权重的不一致）。

这比「保留 KV cache + 继续 decode」更正确，因为旧 KV 是用旧 policy 的注意力参数算出来的，与新权重不匹配。代价是 prefill 的额外计算开销（通常可忽略，prefill 比 decode 快得多）。

### 6.5 Train–Inference Mismatch

（对应 Q11/Q31）

**什么是 mismatch？** 同一个 token 序列，训练侧（Megatron）计算的 logprob 与推理侧（vLLM/SGLang）生成时计算的 logprob 不一致。不一致会导致 IS ratio 出现虚假的大值，训练不稳定。

**来源一：算子实现差异**
- attention 实现：FlashAttention2 vs FlashAttention3 vs cuDNN，对 softmax 的精度处理略有差异
- layernorm 顺序、dropout 位置等细节

**来源二：精度差异**
- 推理侧可能用 FP8，训练侧用 BF16；量化引入的舍入误差累积

**来源三：Batch Invariance 问题**（对应 Q31 的确定性部分）

**Batch invariance** 指：给定相同的输入 token，无论 batch size 是多少，logprob 应该完全相同。这在正确实现下是成立的，但有几种情况会破坏它：

1. **Atomic Add 问题**：在 GPU 上，`atomicAdd` 不保证浮点数的加法顺序，导致结果随并发线程的调度而变化。这会导致 LayerNorm、Attention softmax 等操作在不同 batch size 下产生细微差异。

2. **MoE 路由不一致**（最严重的 mismatch 来源）：推理时（vLLM）和训练时（Megatron）各自独立实现了 MoE 的 router（top-k gating）。浮点精度差异可能导致边界情况下 expert 选择不同，等于训练的是另一个 sequence 的 logprob。

**解法**：
- 「Keep Routing」：推理侧记录每个 token 的 expert routing 决策，训练侧重放（replay），强制使用相同的 routing。目前尚无开源框架实现。
- 「Keep Sampling Mask」：推理侧记录 top-p/top-k 的截断 mask，训练侧对完整词表 logit 施加同样的 mask 再计算 logprob。同样尚无开源框架实现。

这两个问题是 DeepSeek-V3.2 规模的 MoE RL 训练的核心工程挑战。

### 6.6 精度专题：INT8 vs FP8

（对应 Q22）

| 精度 | 比特数 | 硬件支持 | 典型场景 | 精度损失 |
|------|--------|----------|----------|---------|
| FP32 | 32 | 全部 | Adam 状态，主参数（mixed precision 主参） | 无 |
| BF16 | 16 | H100/A100+ | 参数存储、KV cache、激活值 | 极小 |
| **FP8（E4M3/E5M2）** | 8 | H100+ | **推荐用于 training（matmul）** | 小，需 scaling |
| **INT8** | 8 | 全系列 | **推荐用于 inference（weight-only quant）** | 中，需 calibration |
| INT4 | 4 | 部分 | 极端内存受限推理 | 较大 |

**Training 推荐 FP8**：
- H100 的 FP8 FLOPS 是 BF16 的 2×，可直接提升矩阵乘法速度
- FP8 分两种：E4M3（更高精度，forward pass）和 E5M2（更大范围，backward pass），混合使用
- 需要 per-tensor 或 per-block scaling factor，实现复杂但收益明显

**Inference 推荐 INT8 Weight-Only**：
- 权重 INT8，激活 FP16/BF16，无需 calibration（直接量化，不损失精度）
- 省显存（权重体积减半），且 decode 阶段通常是 memory-bound，INT8 权重减少 HBM 带宽压力
- vLLM/SGLang 均内置 INT8 weight-only 量化

**RL 训练的特殊考量**：rollout 侧用 INT8 inference，training 侧用 FP8 training；两侧精度不同是 mismatch 的来源之一。有些框架（ROLL）提供了统一精度配置来减少这种差异。

参考：[FP8-RL: A Practical and Stable Low-Precision Stack for LLM RL](https://arxiv.org/abs/2601.18150)

### 6.7 MoE × RL

（对应 Q11/Q19/Q29/Q30）

**Expert Parallelism（EP）对吞吐的影响**（对应 Q29）：

EP 把不同的 expert 分布在不同 GPU 上，每个 GPU 只保留 $N_{\text{experts}} / N_{\text{EP}}$ 个 expert。Forward pass 需要做 AllToAll 通信（token 路由到持有目标 expert 的 GPU）。

吞吐影响：
- 好处：每 GPU 的 expert 参数更少，HBM 压力降低；稀疏激活意味着更少的 FLOPs
- 坏处：AllToAll 是一个 latency 很高的通信原语（所有 GPU 同步），在小 batch 下延迟显著
- 经验规则：EP 在 batch size 足够大（每个 expert 至少 8-16 个 token）时才有收益

**长上下文的 Compute-Communication Overlap**（对应 Q30）：

长序列（CP，Context Parallelism）下，Attention 跨多 GPU 切分（Sequence Parallelism）。关键是把 all-gather / reduce-scatter 通信与矩阵乘法重叠：

- **Megatron 方案**：pipeline bubble 化 + 显式 async AllReduce，可实现 ≥90% overlap
- **FSDP2 方案**：parameter prefetch 在 backward 时提前 AllGather 下一层的参数；SP 需要额外集成（不原生支持）

Megatron 在 PP + TP + CP + EP 完整组合下比 FSDP2 灵活得多，这是大规模 MoE RL 训练几乎都选 Megatron 的根本原因。

**多节点 backpropagation**（对应 Q26）：

大规模训练的反向传播跨越多个节点：
- 梯度通过 `AllReduce`（DDP）或 `ReduceScatter + AllGather`（ZeRO/FSDP）聚合
- 流水线并行（PP）下，梯度通过 P2P send/recv 在 pipeline stage 间传递
- 1F1B 调度（Megatron）：1 次 forward + 1 次 backward 交替，最小化 pipeline bubble
- Interleaved 1F1B：进一步减少 bubble，代价是更多的通信

---

## 七、选型决策树与展望

### 7.1 Q35 直答：VeRL、TRL、Unsloth、AReaL、slime 选哪个？

```
单机，≤70B，快速验证 idea
    └─► TRL（或 Unsloth，如果 LoRA 够用）

多节点，≤70B，需要标准化配置，团队熟悉 HuggingFace
    └─► veRL（colocate 模式，FSDP 后端）

多节点，100B–300B 密集模型，需要 TP+PP，有 GLM/DeepSeek 类任务
    └─► slime（Megatron + SGLang，生产验证最充分）

多节点，长 CoT 推理，rollout 输出 >8K tokens，想最大化吞吐
    └─► AReaL（fully async + interruptible rollout）

超大 MoE（100B+ Expert 数），需要 EP + TP + PP + CP 组合
    └─► veRL（Megatron 后端）或 slime，配合 Megatron 的 EP 支持

Agentic RL，工具调用 + 多轮对话 + 自定义 scaffold
    └─► ROLL（成熟的五角色抽象）或 Forge（如果你有 MiniMax 量级的规模和工程团队）
```

**不推荐 Unsloth 做大规模 RL 训练**：Unsloth 专注于单 GPU LoRA 效率，没有多节点分布式支持，rollout 引擎也不独立，适合做 SFT 不适合做 RL。

### 7.2 趋势判断

**1. 异步成为默认**：随着 CoT 推理和 agentic 任务的序列长度飙升，同步 RL 的 GPU 利用率过低，异步化不可避免。未来 1–2 年，大多数工业级框架都会默认异步模式。

**2. Training-as-a-Service**：把 RL 训练封装成一个服务（给定 trajectory + reward，返回梯度或更新后的权重），让 agent 开发者不需要理解底层训练框架。Forge 的 Gateway 设计是这个方向的先行者；Tinker（魔搭）是另一个代表。

**3. MoE mismatch 问题亟待解决**：Keep Routing + Keep Sampling Mask 目前无开源实现，但随着 MoE 成为主流架构，这是必须解决的工程问题。预计 2025 年底会有框架率先解决。

**4. 确定性推理**：Batch invariance 和训练-推理一致性问题会随着精度要求提高而受到更多关注。FP8-RL 方向值得持续跟踪。

---

## 八、代码级框架精读：slime、SkyRL 与 Sandbox

前面讲的是抽象设计轴：同步/异步、colocate/disaggregate、训练后端、rollout 后端、权重同步。真正写系统时，面试官通常会继续问：

```text
你说这个框架支持 agentic RL，那一条 trajectory 到底怎么从 sandbox 变成 loss？
你说 fully async，那 buffer、staleness、weight sync 在代码里谁负责？
你说 slime 和 SkyRL 都能做 RL infra，它们的边界有什么不同？
```

这一章按代码路径讲。目标不是背 API，而是能画出一条真实数据流。

### 8.1 slime 的设计：Megatron + SGLang + Data Buffer

slime 的路线很明确：它不是一个“大而全”的 agent 框架，而是把 Megatron 训练、SGLang rollout、Data Buffer 和自定义 generation/reward hook 串成一个高性能 RL substrate。

核心路径可以这样记：

```text
Prompt data
  -> Data Buffer
  -> SGLang rollout / custom_generate
  -> reward / verifier / env feedback
  -> Data Buffer stores Sample
  -> Megatron computes logprob / advantage / loss
  -> weight sync back to SGLang
```

slime 的关键工程判断是：

| 设计点 | slime 的选择 | 为什么 |
|---|---|---|
| 训练后端 | Megatron | 大模型、MoE、TP/PP/CP/EP 组合更成熟 |
| rollout 后端 | SGLang | 专注一个 backend，直接吃 SGLang router、prefix cache、PD disaggregation、spec decoding 等能力 |
| 扩展点 | function path hooks | agent/RAG/sandbox 不改训练核，只改生成和奖励 |
| 数据单位 | `Sample` / rollout group | 便于把一条 agent trajectory 拆成多个 trainable segments |
| 权重同步 | NCCL / disk full / disk delta | colocated、跨集群、跨硬件都有对应路径 |

最重要的不是“slime 支持 agent”，而是它把 agent 放在 `custom_generate` 这个边界外面：

```python
async def custom_generate(args, sample, sampling_params):
    # 1. Run agent loop / tool calls / sandbox
    # 2. Capture token ids, loss mask, reward
    # 3. Return one Sample or multiple Samples
    return sample_or_segments
```

这意味着 slime 的核心训练代码不需要理解 SWE-Bench、Search-R1、Tau-Bench 或 browser task。它只要求你最后交回可训练的 token 轨迹：

```text
tokens
loss_mask
rollout_logprobs
reward
rollout_id / session_id
metadata
```

这就是 slime 能支持很多 agentic 形态的原因：复杂性被放在 generation adapter / sandbox / verifier，而不是塞进 Megatron 训练 loop。

### 8.2 slime 的 Agentic RL：custom_generate 不是“文本生成函数”

slime 文档里 `--custom-generate-function-path` 的签名看起来很简单：

```python
async def custom_generate(args, sample, sampling_params):
    ...
```

但在 agentic RL 里，它实际上是一整个 rollout orchestrator。以 coding-agent RL 示例为例，真实流程是：

```text
base_sample
  -> boot sandbox
  -> prepare workspace
  -> start agent harness (Claude Code / Codex / custom CLI)
  -> agent calls model through adapter
  -> adapter records exact token ids and logprobs from SGLang
  -> agent edits repo / runs commands
  -> capture git diff
  -> evaluate diff in a fresh sandbox
  -> adapter.finish_session(...)
  -> return one or more trainable Samples
```

注意这里的两个 sandbox：

```text
Sandbox A: agent writes code, may run exploratory commands
Sandbox B: clean grading environment, applies diff and runs tests
```

这解决了一个很实际的问题：如果 agent 在同一个环境里先看测试、改测试、留下缓存，再被同一个环境评测，reward 就不可信。干净评测 sandbox 是防止 test cheating 和环境污染的基础设计。

slime 的 coding-agent 示例把这条链路拆成三层：

| 层 | 责任 |
|---|---|
| `sandbox` | `exec / write_file / read_file / close`，隐藏 E2B/Docker/VM 差异 |
| `harness` | 安装和运行 agent CLI，比如 Claude Code、Codex、OpenCode |
| `adapter` | 把 agent 的 Anthropic/OpenAI 请求转成 SGLang generate，并记录 token 级 provenance |

这三个层次非常重要，因为 agentic RL 最容易犯的错是把“字符串对话记录”当成训练目标。正确做法是：

```text
string/message history is only an interface
sampled token ids are the training target
```

也就是说，训练时不能把最终 conversation string 重新 tokenize 当作 response。必须使用 rollout 时模型实际采样出来的 `output_ids`，并且只有这些 token 的 `loss_mask=1`。

### 8.3 String-in, Token-out：agent 轨迹为什么难训

工具调用 agent 的输入输出天然是字符串：

```text
assistant: use tool
tool: stdout / file diff / browser result
assistant: next action
tool: next observation
...
```

但 RL loss 是 token 级的：

$$
\mathcal{L} = -\sum_t \hat{A}_t \log \pi_\theta(a_t | s_t)
$$

所以每个 token 必须回答两个问题：

```text
1. 这个 token 是模型采样出来的吗？
2. 这个 token 对应的 rollout logprob 是多少？
```

工具 observation、系统模板、用户消息、环境反馈都不是模型动作，不能训练：

| token 来源 | loss mask |
|---|---|
| model output / action | 1 |
| user prompt | 0 |
| tool observation | 0 |
| sandbox stdout/stderr | 0 |
| chat template token | 0 |
| compacted context / re-rendered prefix | 0，除非能证明 token provenance |

slime coding-agent 示例里有一个关键 guard：如果后续 prompt 和之前保存的 sampled output 在 token 层面对不上，就保留上下文用于继续 agent，但不对无法证明 provenance 的 token 回传梯度。这是 agentic RL 的 correctness 核心。

面试里可以这样概括：

> Agent 轨迹可以是 string-in，但训练目标必须是 token-out。任何从字符串重新 tokenize 恢复出来的“模型输出”都不可靠，因为 tokenizer、chat template、tool observation、compaction 都可能改变 token 边界。

### 8.4 SkyRL 的设计：GeneratorInterface 是系统边界

SkyRL 走的是另一条路线：它把环境抽象做得更显式。核心接口是 `GeneratorInterface`：

```python
class GeneratorInterface:
    async def generate(self, input_batch) -> GeneratorOutput:
        ...
```

`GeneratorOutput` 里不只是 response：

```python
{
    "prompt_token_ids": ...,
    "response_ids": ...,
    "rewards": ...,
    "loss_masks": ...,
    "rollout_logprobs": ...,
    "trajectory_ids": ...,
    "trajectory_generation_times": ...,
    "rollout_expert_indices": ...,
}
```

这说明 SkyRL 的训练 loop 不关心你是单轮 GSM8K、多轮 SQL、LiveCodeBench，还是一个复杂 agent。只要 generator 最后返回这些字段，trainer 就能做 PPO/GRPO/DAPO 等算法。

对应的数据流是：

```text
RayPPOTrainer
  -> prepare_generator_input(...)
  -> generator.generate(...)
  -> validate_generator_output(...)
  -> fwd_logprobs_values_reward(...)
  -> train_critic_and_policy(...)
  -> dispatch.save_weights_for_sampler()
```

其中最值得讲透的是 `SkyRLGymGenerator.agent_loop()`。

### 8.5 SkyRLGymGenerator.agent_loop：一条 trajectory 怎么生成

SkyRL-Gym 约定每个 environment 有三个基本方法：

```python
env.init(prompt) -> first_observation
env.step(action) -> {observations, reward, done, metadata}
env.close()
```

`SkyRLGymGenerator.agent_loop()` 的主循环可以简化成：

```python
while not done:
    engine_input = {
        "prompt_token_ids": [current_input_ids],
        "session_ids": [trajectory_id],
        "sampling_params": sampling_params,
    }
    engine_output = await inference_engine_client.generate(engine_input)

    action_text = engine_output["responses"][0]
    action_ids = engine_output["response_ids"][0]
    action_logprobs = engine_output.get("response_logprobs")

    step_output = env.step(action_text)
    observation_ids = tokenize_observation(step_output["observations"])

    append action_ids with loss_mask=1
    append observation_ids with loss_mask=0
    append reward at turn boundary
```

这段代码把 agentic RL 最核心的 token accounting 讲清楚了：

```text
input_ids_next = input_ids_prev + model_action_tokens + env_observation_tokens
loss_mask_next = loss_mask_prev + [1 ... 1]       + [0 ... 0]
reward_next    = reward placed at response boundary
```

如果是 step-wise training，SkyRL 会把每个 turn 作为一个 `TrajectoryOutput`，方便做更细粒度的 credit assignment。否则，它会把整条多轮 trajectory 合成一个 response 序列，reward 可以是最后一个 token 的 outcome reward，也可以是 per-step reward list。

这就是为什么 SkyRL 的 env 抽象比“写一个 reward function”强：环境不仅给 reward，还决定 observation 如何回到下一轮 prompt，最终影响 loss mask 和 credit assignment。

### 8.6 SkyRL 的 inference 架构：data plane / control plane 分离

SkyRL 新 inference path 把请求分成两类：

```text
Data plane:
  /v1/chat/completions
  /v1/completions
  generate / tokenize / detokenize
  -> 走 router / proxy_url

Control plane:
  pause / resume
  sleep / wake_up
  start_weight_update / update_weights / finish_weight_update
  -> fan-out 到每个 backend server
```

图示：

```text
                  Trainer
                     │
                     ▼
          RemoteInferenceClient
             │              │
             │ data plane   │ control plane
             ▼              ▼
          VLLMRouter     vLLM API servers
             │              ▲
             └──────────────┘
```

这个拆分背后的理由很具体：

| 请求 | 为什么这样走 |
|---|---|
| generation | 需要 load balancing，且多轮 trajectory 需要 session stickiness |
| pause/resume | 必须让所有 backend 一起停/起 |
| weight sync | 必须 fan-out 更新每个 replica |
| sleep/wake_up | colocated 模式要释放/恢复显存 |

SkyRL generator 会给每条 trajectory 一个稳定 `session_id`。vLLM router 使用 consistent hash，把同一个 session 的多轮请求固定到同一个 backend。这样做的收益是 prefix cache：

```text
turn 1: prompt + action + observation
turn 2: same prefix + new action
turn 3: same prefix + more observation
```

如果每个 turn 被随机路由到不同 backend，prefix cache 命中率会掉，agentic rollout 的吞吐会很差。这个点是 agent serving 和普通 single-turn serving 的关键区别。

### 8.7 SkyRL fully async：按源码讲五个组件

SkyRL 的 fully async trainer 是 `FullyAsyncRayPPOTrainer`，继承同步的 `RayPPOTrainer`。它不是“开一个 async flag”这么简单，而是新增了五个控制组件：

| 组件 | 源码角色 | 责任 |
|---|---|---|
| `GenerationWorker` | `asyncio.Task` | 从 dataloader 拿一个 prompt group，调用 `generator.generate()` |
| `TrainingWorker` | trainer 主线程 | 从 buffer 取 mini-batch，训练 policy，触发 weight sync |
| `GenerationOutputGroupBuffer` | `asyncio.Queue` | 存已经完成的 rollout group |
| `AsyncDataloader` | `_AsyncDataloader` | 多 worker 并发取样、记录 consumed UID、支持 checkpoint resume |
| `AsyncStalenessManager` | `_AsyncStalenessManager` | 控制 generation 不要跑得比 training 太超前 |

真实 loop 可以压缩成：

```python
for epoch in epochs:
    buffer = asyncio.Queue(maxsize=B * (S + 1))

    generator_tasks = [
        create_task(run_generate_loop(buffer))
        for _ in range(num_parallel_generation_workers)
    ]

    for step in training_steps:
        groups = await collect_B_groups(buffer)
        training_input = convert(groups)
        status = await run_training(training_input)
        mark_consumed(groups)

        await dispatch.save_weights_for_sampler()
        await staleness_manager.notify_capacity_change(global_step + 1)
```

这里 `B = policy_mini_batch_size`，`S = max_staleness_steps`。

最关键的是 staleness capacity：

$$
\text{capacity} = (S + \text{current\_global\_step}) \times B
$$

生成侧要满足：

$$
\text{accepted} + \text{running} \le \text{capacity}
$$

含义是：

- `accepted`：已经生成完、可能还没训练的 group
- `running`：正在生成的 group
- `current_global_step`：当前训练到的模型版本
- `S`：允许 generation 最多领先 training 多少步

生成 worker 在开始一个新 group 前调用 `acquire_submission_slot()`；如果 buffer 和 running 已经太多，就阻塞。训练 step 完成后，`notify_capacity_change()` 增加容量，释放被阻塞的 generation worker。

这不是严格的 per-sample staleness 保证。极端长尾 trajectory 可能跑了很久才结束，实际版本差超过 `S`。SkyRL 当前选择接受它并记录 warning，而不是直接丢弃。这是一个很现实的系统取舍：

```text
drop stale sample -> 更 on-policy，但浪费长尾 rollout 成本
keep stale sample -> 吞吐更高，但需要 IS / clip 控制 off-policy bias
```

### 8.8 SkyRL 的 in-flight weight update

fully async 的关键不是“训练和生成并行”，而是训练后如何更新还在生成中的 inference engines。

SkyRL 在每个 training step 后调用：

```python
await self.dispatch.save_weights_for_sampler()
```

底层分两种情况：

**Non-colocated：NCCL broadcast**

```text
trainer step done
  -> client.pause(mode="keep")
  -> NCCL broadcast weights to vLLM workers
  -> client.resume()
  -> frozen requests continue
```

`pause(mode="keep")` 的意义是：不 abort 正在 decode 的请求，而是把它们冻结在 scheduler 里，KV cache 保留，resume 后继续跑。这个设计吞吐好，但要承担一个细节：继续 decode 的 trajectory 前半段来自旧权重，后半段来自新权重，所以训练必须记录 policy version / rollout logprobs / staleness，否则算法上说不清。

**Colocated：CUDA IPC**

```text
trainer and inference share GPUs
  -> inference sleep / release memory
  -> training runs
  -> pack updated weights into CUDA buffers
  -> send CUDA IPC handles
  -> inference wake_up and load weights
```

CUDA IPC 适合同机/同 GPU colocate；NCCL broadcast 适合 training 和 inference 分开的 GPU group。

### 8.9 slime vs SkyRL：不是谁替代谁

把两个框架放在同一张图里：

| 维度 | slime | SkyRL |
|---|---|---|
| 核心路线 | Megatron + SGLang native RL scaling | Modular full-stack RL + Gym/Agent/Tinker/backends |
| agent 接入 | `custom_generate` / `rollout_function` | `GeneratorInterface` / `SkyRLGymGenerator` / SkyRL-Agent |
| inference backend | 深度绑定 SGLang | vLLM HTTP path 为主，也支持不同 backend |
| 权重同步 | NCCL/disk/full/delta，强调跨集群外部 engine | broadcast / CUDA IPC，强调 data/control plane 与 in-flight update |
| 环境抽象 | sandbox/harness/adapter 作为示例和库 | `skyrl-gym` 明确 Env API |
| 适合场景 | 大模型 Megatron 路线、SGLang-heavy、GLM/DeepSeek 类 scale | 研究快速迭代、多 backend、agent/env 模块化 |

面试回答可以这样说：

> slime 更像一个高性能 RL kernel：训练和 rollout 路线更固定，但能把 Megatron/SGLang 的能力吃深。SkyRL 更像一个模块化 RL OS：trainer、generator、gym env、agent dispatcher、inference client、Tinker backend 都是可替换模块。

### 8.10 Sandbox 设计：安全、可复现、可扩展

Agentic RL 的 sandbox 不是“跑代码的地方”这么简单，它同时承担四个责任：

| 责任 | 具体问题 |
|---|---|
| 隔离 | 不可信代码不能污染宿主机、其他样本、训练进程 |
| 可复现 | 同一个 sample 的初始文件系统、依赖、测试环境必须固定 |
| 可观测 | stdout/stderr、diff、工具调用、超时、资源使用必须能记录 |
| 可扩展 | 几千个 rollout 并发启动，不能因为 boot 长尾拖死训练 |

一个推荐的 sandbox 生命周期：

```text
1. boot(image, metadata)
2. prepare_workspace(problem_statement, repo, pre_commands)
3. run_agent(adapter_url, time_budget)
4. capture_artifacts(diff, logs, trajectories)
5. close_dirty_sandbox
6. boot_clean_sandbox
7. apply_diff
8. run_tests / verifier
9. return reward + diagnostics
```

为什么要分 dirty sandbox 和 clean sandbox？

```text
dirty sandbox:
  agent 可以探索、运行命令、修改文件、产生缓存

clean sandbox:
  只应用最终 diff，再跑评测
```

这能防止三类 reward hacking：

1. 修改测试文件让测试通过
2. 在 workspace 留下缓存/状态，让评测非真实通过
3. 通过网络或环境变量读取答案

真正上线时还要加这些限制：

| 控制项 | 推荐策略 |
|---|---|
| CPU / memory | cgroup / container limit / per-process timeout |
| network | 默认禁用或 allowlist |
| filesystem | per-episode writable layer，结束即丢弃 |
| secrets | sandbox 内不注入训练集答案、host token、cloud key |
| tool timeout | 每个 tool call 和整个 episode 都要有 wall-clock guard |
| artifact | 保存 patch、日志、reward breakdown，便于 debug |

slime coding-agent 示例里有一个很实用的并发控制：sandbox boot 用 semaphore 限制并发，避免同时启动太多环境导致 SSL/h2/镜像拉取长尾。SkyRL fully async 则从 trainer 侧控制 generation worker 数和 staleness。这两个控制层可以一起用：

```text
trainer capacity control
  controls how far rollout can run ahead of training

sandbox boot semaphore
  controls external environment pressure
```

这就是 agentic RL 比普通 RLVR 难的地方：瓶颈不只在 GPU，还在 sandbox、文件系统、测试器、网络、外部工具和长尾任务。

---

## 九、可以继续做的方向

这一节不是“未来展望”的空话，而是可以真正做成系统论文、开源 PR 或实习项目的方向。

### 9.1 方向一：Token Provenance Debugger

问题：agentic RL 最容易出现“字符串看起来对，token 训练目标错了”。工具 observation、sub-agent、context compaction、chat template 变化都会打断 token provenance。

可以做：

```text
给每个 token 打来源标签：
  MODEL_OUTPUT
  USER_PROMPT
  TOOL_OBSERVATION
  TEMPLATE
  COMPACTION
  RETOKENIZED_UNTRUSTED

训练前可视化一条 trajectory：
  token id / text span / loss_mask / reward / rollout_logprob / source
```

价值：

- 快速发现错误 loss mask
- Debug reward 没涨但 loss 正常下降的问题
- 给 SkyRL/slime/veRL 都能复用

### 9.2 方向二：Sandbox Pool + Warm Reset

问题：每条 trajectory 都从零 boot sandbox，长尾很严重；但复用 sandbox 又会污染状态。

可以做：

```text
warm image pool
  -> pre-boot N 个空 sandbox
  -> 每个 episode 用 overlay layer
  -> done 后 reset writable layer
  -> clean verifier sandbox 独立保留
```

关键难点：

- 文件系统 snapshot / rollback
- 网络隔离
- per-sample dependency 安装如何缓存
- 如何证明 reset 后无污染

### 9.3 方向三：Staleness-Aware Sampling

SkyRL 当前 staleness manager 是 aggregate capacity bound，不是 per-sample hard bound。长尾 group 可能实际很 stale。

可以做：

```text
if group_staleness <= S:
    use normally
elif reward is high and IS ratio stable:
    use with smaller weight
else:
    drop / resample / distill only
```

训练 loss 可以改成：

$$
w_i = \exp(-\lambda \cdot \text{staleness}_i)
$$

或者让 clip range 随 staleness 变窄：

$$
\epsilon_i = \frac{\epsilon_0}{1 + \alpha \cdot \text{staleness}_i}
$$

这和 CISPO / off-policy correction 是同一方向，但可以在真实 agent workload 上验证。

### 9.4 方向四：MoE Router Replay

MoE RL 最大的坑是 rollout 侧和 training 侧 router 不一致。SkyRL 的 `GeneratorOutput` 已经预留了 `rollout_expert_indices` 字段，这说明系统已经意识到 routing provenance 的重要性。

可以做：

```text
rollout:
  return expert_indices[token][layer][topk]

training:
  force/replay the same expert route
  compute logprob under matched routing
```

难点：

- Megatron MoE forward 是否支持 externally supplied routing
- routing replay 与 expert parallel all-to-all 的结合
- shared experts / fine-grained experts 的格式标准化

如果做出来，价值非常大：它直接解决 MoE train-inference mismatch。

### 9.5 方向五：Rollout-as-a-Service

slime external rollout engines、SkyRL RemoteInferenceClient、ProRL Agent 这类设计都指向同一个趋势：

```text
training cluster
  only owns optimizer / gradients / checkpoint

rollout service
  owns inference / agent execution / sandbox / verifier

weight sync channel
  connects them
```

可以做的系统：

- rollout service 暴露统一 API：`submit_task / get_trajectory / sync_weights / health`
- 支持不同任务：math verifier、SWE sandbox、browser task、SQL env
- 支持不同 backend：SGLang、vLLM、TensorRT-LLM
- 支持 delta weight update 和版本追踪

这会让 agent 团队和 infra 团队解耦：agent harness 不需要嵌进训练框架，训练框架也不需要理解每种环境。

### 9.6 方向六：可观测性与失败归因

RL infra 的失败经常不是 crash，而是“reward 不涨”。需要能回答：

```text
rollout 慢在哪里？
reward 为 0 是模型错、verifier 错、sandbox 错，还是 token mask 错？
staleness 分布是什么？
weight sync 时间占比多少？
哪些 env 最常 timeout？
哪些 prompt 最常产生 zero-variance group？
```

建议默认记录：

| 指标 | 用途 |
|---|---|
| `rollout/e2e_time` | 找长尾任务 |
| `rollout/tool_time` | 找 sandbox/tool 瓶颈 |
| `rollout/num_turns` | 判断是否进入无限循环 |
| `train/staleness` | 判断 off-policy 风险 |
| `train/logprob_diff` | 判断 train-inference mismatch |
| `reward/breakdown` | 判断 reward hacking 或 verifier 问题 |
| `sync/weight_update_time` | 判断同步是否吞吐瓶颈 |

这类 observability 工作看起来不“算法”，但在工业 RL 里价值很高。

---

## 参考资料

- [HybridFlow: veRL 论文](https://arxiv.org/abs/2409.19256)
- [AReaL 论文](https://arxiv.org/abs/2505.24298)
- [slime 文档](https://thudm.github.io/slime/)（LMSYS Blog: [slime: An SGLang-Native Post-Training Framework](https://www.lmsys.org/blog/2025-07-09-slime/)）
- [slime GitHub](https://github.com/THUDM/slime)
- [SkyRL GitHub](https://github.com/novasky-ai/skyrl)
- [SkyRL-Agent paper](https://arxiv.org/abs/2511.16108)
- [SkyRL fully async training docs](https://docs.skyrl.ai/docs/tutorials/fully_async)
- [SkyRL inference architecture docs](https://docs.skyrl.ai/docs/getting-started/inference_architecture)
- [SkyRL-Agent project page](https://sky.cs.berkeley.edu/project/skyrl/)
- [ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents](https://arxiv.org/html/2603.18815v1)
- [Forge: MiniMax RL 框架](https://huggingface.co/blog/MiniMax-AI/forge-scalable-agent-rl-framework-and-algorithm)
- [Keep the Tokens Flowing: 16 开源 RL 框架对比](https://huggingface.co/blog/async-rl-training-landscape)
- [Anatomy of RL Frameworks](https://www.hanifleo.com/anatomy-of-rl-frameworks/)
- [FP8-RL: 低精度 RL 训练](https://arxiv.org/abs/2601.18150)
- 配套练习：[[MLSYS15 RL Infra 自测 35 问]]
