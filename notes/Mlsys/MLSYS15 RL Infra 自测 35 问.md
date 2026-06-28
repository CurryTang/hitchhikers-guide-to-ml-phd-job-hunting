# 强化学习练习

## A · 算法基础（Q1–Q10）

<details class="exercise">
<summary><span class="q-label">Q1</span> <span class="q-text">为什么要用 Actor-Critic，而不是纯 Critic 方法？</span></summary>

**核心矛盾**：纯 Critic（如 DQN）直接学习 $Q(s,a)$，然后用 $\arg\max_a Q(s,a)$ 选动作。这在离散 action space 下可行，但 LLM 的 action space 是整个词表（32K–128K token），$\arg\max$ 不可算。

Actor 的价值在于：直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升优化期望 reward，无需枚举 action space。Critic 的价值在于：提供低方差的 advantage 估计，否则纯策略梯度的方差极大（需要大量样本才收敛）。

**对应到 LLM RL**：Actor = 语言模型本身（输出 token 分布）；Critic = 一个独立的价值头（预测当前前缀的期望 reward）。GRPO 进一步证明：当 group 足够大时，可以不用 Critic，用 group 内相对分数做 advantage，方差也足够低。

**延伸**：→ [[MLSYS14 Post-Training Infra#三、最小算法背景：PPO 与 GRPO]]

</details>

<details class="exercise">
<summary><span class="q-label">Q2</span> <span class="q-text">KL 散度、交叉熵、MLE 之间的关系是什么？</span></summary>

三者高度相关，从最基础的关系出发：

$$
\text{KL}(P \| Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)} = H(P, Q) - H(P)
$$

- $H(P,Q)$ 是交叉熵，$H(P)$ 是熵
- MLE 等价于最小化交叉熵 $H(P_{\text{data}}, Q_\theta)$，因为 $H(P_{\text{data}})$ 是常数
- **因此 MLE ≡ 最小化 KL$(P_{\text{data}} \| Q_\theta)$**

**在 RL 的作用**：
- PPO/GRPO 里的 KL 惩罚项 $\text{KL}(\pi_\theta \| \pi_{\text{ref}})$：防止 policy 偏离初始 SFT 模型太远（避免 reward hacking）
- DPO 把 RLHF 目标化为 MLE 问题，通过 preference pair 的交叉熵训练隐式对齐奖励函数
- GRPO 的 KL 可以放在 reward 里（$r' = r - \beta \text{KL}$）或 loss 里（显式正则），两者等价但数值特性不同

</details>

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">在不同 RL 场景下 reward 应该怎么设计？</span></summary>

**核心原则**：reward 应该信号强、稀疏度适中、无法被 reward hacking。

**按场景分类**：

| 场景 | Reward 类型 | 示例 |
|------|------------|------|
| 数学推理 | Outcome reward（terminal） | 最终答案对/错（0/1 或连续分） |
| 代码生成 | Execution-based | 测试用例通过率 |
| 对话/RLHF | RM score | 人类偏好模型打分 |
| Agentic 任务 | Sparse terminal + shaped | 任务完成 + 中间步骤奖励 |
| 长推理 | Process reward | 每一步推理链的质量评分 |

**设计中的常见问题**：
- **过稀疏**：只有最终答案对才有奖励，早期训练几乎没有梯度信号
- **过密集**：每步都有 reward，容易被 reward shaping 攻击（模型学会刷中间分）
- **可验证性**：数学题、代码可以用程序验证，天然抗 hacking；开放问答需要 RM，RM 本身可能被攻击

**RLVR（Verifiable Reward）的兴起**：用规则/程序做 reward verifier，代替 RM，避免 RM reward hacking。DeepSeek-R1 的成功很大程度上依赖于数学/代码任务的可验证奖励。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">重要性采样（IS）、拒绝采样（Rejection Sampling）在 RL 中如何应用？</span></summary>

**重要性采样（IS）**：当训练数据来自旧 policy $\pi_{\text{old}}$，但我们想优化新 policy $\pi_\theta$ 时，用 IS ratio 修正期望：

$$
\mathbb{E}_{\pi_\theta}[f] \approx \mathbb{E}_{\pi_{\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} f\right]
$$

这正是 PPO 目标函数中 $r_t(\theta)$ 的来源。IS 是异步 RL 的算法基础：允许用略旧的数据训练，只要 IS ratio 不爆炸（用 clip 控制）。

**拒绝采样（Rejection Sampling）**：生成多个候选序列，按接受概率保留部分：
- 最常见形式：生成 G 个候选，只保留满足质量门槛的（如 reward > 阈值）
- GRPO 里的 group sampling 是一种软拒绝采样：不丢弃样本，但给高 reward 样本更高权重（通过 advantage 归一化）
- Rejection Sampling Fine-Tuning（RFT）：每步生成 K 个候选，只用正确答案做 SFT，不做 RL

**系统侧影响**：IS ratio 需要在 training 时重新计算 logprob（需要 actor 的一次额外 forward pass），这是 PPO 比 GRPO 多一个开销的地方（GRPO 可以用 rollout 时记录的 logprob 直接做 IS）。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">PPO 和 GRPO 中 advantage 怎么算？为什么要减 baseline？标准差归一化是否必要？</span></summary>

**PPO 的 Advantage（GAE）**：
$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T}(\gamma\lambda)^l(r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l}))
$$
需要 Critic 网络对每个 token 位置估计 $V(s_t)$。参数 $\lambda$ 控制偏差-方差权衡：$\lambda=0$ 退化为 TD，$\lambda=1$ 退化为 Monte Carlo 回报。

**GRPO 的 Advantage（Group Relative）**：
$$
\hat{A}_i = \frac{r_i - \mu_g}{\sigma_g}
$$
其中 $\mu_g, \sigma_g$ 是同 group（同一 prompt 的 G 个 completion）的 reward 均值和标准差。

**为什么减 baseline**：policy gradient 的方差来自 $r_t$ 的绝对值大小。减去 baseline $b$（无论常数还是状态依赖的 $V(s_t)$）**不改变梯度期望**（因为 $\mathbb{E}[\nabla \log\pi \cdot b] = 0$），但大幅减少方差。

**标准差归一化是否必要？** 这是 GRPO 里有争议的点（Dr.GRPO 的来源）：
- **支持**：归一化使不同 prompt 的 advantage scale 统一，学习率不受 reward scale 影响
- **反对**：若某个 group 里所有答案 reward 相同（$\sigma_g = 0$），除零报错或要 clip；若所有答案都对（full-correct group），归一化后 advantage 为 0，等于丢弃了正确样本的学习信号
- Dr.GRPO 的解法：对 full-correct 和 full-wrong 的 degenerate group 做特殊处理（丢弃或恢复 variance）

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">RL 训练和测试时 scaling（test-time scaling）的探索方式有何不同？</span></summary>

**RL 训练时的探索**：通过策略的随机性（temperature > 0 的采样）来探索 action space。更高的 temperature = 更多探索 = 更多多样性，但也更多噪声。GRPO 的 group sampling 本质是在同一问题上做多次随机探索，找到高 reward 的路径。

**Test-time scaling（TTS）的探索**：模型已固定（不更新参数），通过以下方式探索解空间：
- **Best-of-N sampling**：生成 N 个候选，用 verifier/RM 选最好的
- **Beam search / Diverse beam search**：维护多条候选路径
- **Process reward model 引导**：每步用 PRM 评分，剪枝差的中间步骤
- **MCTS（树搜索）**：更系统的探索，但计算开销极大

**关键区别**：训练时的探索会更新模型参数，目标是找到好的策略；TTS 的探索不更新参数，目标是在推理时找到好的输出。TTS 的计算可以是模型参数数量的倍数级——这是 o1/R1 类模型「想得更多」的本质。

</details>

<details class="exercise">
<summary><span class="q-label">Q7</span> <span class="q-text">PPO clipping 如何工作？为什么取 min？不加 clip 会怎样？CISPO 和它有何不同？</span></summary>

**PPO clip 机制**：

$$
\mathcal{L}_{\text{clip}} = \mathbb{E}_t\left[\min\left(\underbrace{r_t \hat{A}_t}_{\text{原始目标}},\ \underbrace{\text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t}_{\text{裁剪版}}\right)\right]
$$

**为什么取 min（悲观下界）**：
- 若 $\hat{A}_t > 0$（该动作是好的）：想提高 $\pi_\theta$ 来增大 $r_t$，但 clip 不让 $r_t$ 超过 $1+\epsilon$，防止过度利用单次样本
- 若 $\hat{A}_t < 0$（该动作是坏的）：想降低 $\pi_\theta$，但 clip 不让 $r_t$ 低于 $1-\epsilon$，防止过度惩罚
- 取 min 实现了「进展有上界」：无论 advantage 正负，单次梯度更新幅度被约束

**不加 clip 的后果**：policy 可能在单次好样本上过度优化（分布崩塌），然后 reward hacking（因为 RM/verifier 只看到训练分布内的输出）。

**CISPO**（Clipped IS Policy Optimization，MiniMax Forge 配套算法）：
- 在 IS 修正的基础上，把 clip 范围从固定 $\epsilon$ 改为随 staleness 动态调整：staleness 越大 → clip 越紧
- 目的：在完全异步系统里，旧数据的 IS ratio 本身就代表了「policy 漂移」的程度，动态 clip 更合理
- 对比：标准 PPO clip 假设 $\pi_{\text{old}} \approx \pi_\theta$（接近 on-policy）；CISPO 显式处理 off-policy 情况

</details>

<details class="exercise">
<summary><span class="q-label">Q8</span> <span class="q-text">GRPO 为何要加 KL 惩罚？KL 怎么算？DAPO/GSPO 为什么要去掉它？</span></summary>

**KL 惩罚的目的**：防止 policy 在优化 reward 的过程中偏离初始 SFT 模型太远，导致：
1. 语言质量下降（生成乱码或 reward hacking 的模式）
2. 灾难性遗忘（丢失 SFT 训练的知识）

**KL 的两种计算位置**：

| 位置 | 公式 | 特点 |
|------|------|------|
| Reward 内（GRPO 原始）| $r' = r - \beta \text{KL}[\pi_\theta \|\|\pi_{\text{ref}}]$ | 每 token 计算，影响 advantage 估计 |
| Loss 外（显式正则） | $\mathcal{L} = \mathcal{L}_{\text{GRPO}} + \beta\text{KL}$ | 梯度直接作用于参数，数值更稳定 |

**DAPO 去掉 KL 的理由**：
- 对于强 verifiable reward 的任务（数学、代码），KL 惩罚会限制模型「大幅偏离 SFT 分布」的能力，而这种偏离是必要的（SFT 模型不会做长 CoT 推理）
- 实验表明，去掉 KL + 适当调大 clip（或用 clip lower bound 代替）效果更好

**GSPO 的 KL 改进**：把 token 级 KL 改为序列级 KL，减少 high-entropy token（如空格）对 KL 惩罚的不合理贡献。

</details>

<details class="exercise">
<summary><span class="q-label">Q9</span> <span class="q-text">LLM 训练中，如果 loss 被 All Reduce 了多次会怎样？</span></summary>

这是一个系统 debug 场景题，考察对分布式训练数值行为的理解。

**正常情况**：分布式数据并行（DDP/ZeRO）下，每个 rank 计算本地 batch 的 loss，然后对梯度做 AllReduce（平均），得到全局平均梯度。

**AllReduce loss 本身**（而非梯度）的问题：
- 如果代码错误地对 loss 值本身做了 AllReduce，loss 会被放大 $N$ 倍（$N$ 是 rank 数），导致梯度被放大 $N$ 倍
- 效果：实际学习率是预期的 $N$ 倍，训练迅速发散（loss NaN 或 explode）

**多次 AllReduce 梯度的问题**：
- 梯度被平均了多次，每次 AllReduce 都除以 rank 数
- 最终梯度被缩小 $N^k$ 倍（$k$ 是额外的 AllReduce 次数），学习率等效极小，训练几乎不更新

**在 RL 里的特殊场景**：GRPO 训练时，rollout 数据要分发到各 rank，如果 reward/advantage 在分发时意外做了跨 rank 的 AllReduce（本意是统计全局 mean/std），然后 loss 计算时又做了一次，就会出现双重 AllReduce 的问题。症状：训练 loss 异常低（梯度太小）或 reward 不增长。

</details>

<details class="exercise">
<summary><span class="q-label">Q10</span> <span class="q-text">DPO 中的 reward function 是什么？会有 reward hacking 吗？如何缓解？</span></summary>

**DPO 的隐式 reward**：DPO 把 RLHF 的 RM + RL 两阶段合并，其目标函数等价于用如下隐式 reward 做 RL：

$$
r_{\text{DPO}}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

即「当前 policy 相对于 reference policy 的 log-ratio」，乘以 $\beta$（KL 系数的倒数）。

**DPO 的 Reward Hacking 风险**：
- 由于隐式 reward 直接由 policy 参数决定，policy 可以通过「降低 reference 对 dispreferred 样本的概率」来虚假提高 reward，而不是真正生成更好的输出
- 具体表现：模型变得过于保守（避免输出任何 reference 模型认为 unlikely 的序列）或 length hacking（通过改变序列长度来操纵 reward）

**缓解方法**：
- **SimPO**：去掉 reference model，改用序列长度归一化的 log-prob 作为 implicit reward，更抗 length hacking
- **IPO**（Identity PO）：直接优化 preference probability，绕过奖励函数建模
- **Online DPO**：在训练过程中持续生成新的 preference pairs（而非用固定数据集），类似 RLHF 的 online 特性
- **KTO**：不需要 pair 数据，直接用 binary 反馈（good/bad），隐式 reward 设计更鲁棒

</details>

---

## B · 算法进阶（Q11–Q19）

<details class="exercise">
<summary><span class="q-label">Q11</span> <span class="q-text">MoE 模型的 train–inference mismatch 有哪些方法解决？原理是什么？</span></summary>

**根本问题**：MoE 模型的 router（top-k gating）在推理侧（vLLM/SGLang）和训练侧（Megatron）各自独立实现，浮点精度差异会导致边界情况下 expert 选择不同——等于训练的是另一个 sequence 的 logprob。

**三类主要处理方式**：

1. **Router Inconsistency（路由不一致）**：推理时 token 被路由到 Expert A，训练时 forward pass 路由到 Expert B（因为两个实现的 softmax/top-k 浮点行为略有差异）。这导致 IS ratio 出现系统性偏差。

   解法 → **Keep Routing**：推理侧记录每个 token 的 expert routing index，作为辅助数据随 trajectory 一起传给训练侧，训练时强制 replay 同样的 routing（绕过 router 网络，直接指定 expert）。

2. **Sampling Mask Mismatch**：推理时 top-p/top-k 截断了词表（只采样概率最高的一部分），但训练时对完整 32K 词表计算 logprob，导致归一化不同。

   解法 → **Keep Sampling Mask**：推理侧记录 sampling mask（哪些 token 被截断），训练时对 logit 施加同样的 mask 再 softmax，保证 logprob 归一化与推理一致。

3. **Token-level Ratio Noise**：高稀疏 MoE 的 router 抖动会让单 token logprob ratio 变噪，尤其在 GRPO/PPO 长训练中更明显。

   解法 → **GSPO / Sequence-level ratio**：用 response-level likelihood ratio 做 clipping，降低单 token route 抖动对训练目标的影响。它不是 replay router，而是把优化目标从 token-level 改到 sequence-level。

→ [[MLSYS14 Post-Training Infra#6.5 Train–Inference Mismatch]]

</details>

<details class="exercise">
<summary><span class="q-label">Q12</span> <span class="q-text">RL 训练中，group size、learning rate、PPO epochs、generation length 如何选择？</span></summary>

**Group size G**：
- 更大的 G → 更低的 advantage 方差（统计更可靠），但每批数据需要生成 G 倍序列，rollout 成本线性增加
- 实践范围：G=4（资源受限）到 G=32（长推理任务）；DAPO 推荐 G=8–16
- 若任务 difficulty 很高（大多数 completion 都错），需要更大 G 才能采到足够正确样本

**Learning rate**：
- RL 的 LR 通常比 SFT 小 5–10 倍（1e-6 到 5e-6，而 SFT 常用 1e-5）
- 原因：RL 更新不像 SFT 那样均匀，高方差的 advantage 会导致梯度波动，小 LR 防止单次爆炸

**PPO epochs**（每批数据重复训练几次）：
- 标准 PPO：2–4 epochs per batch
- 每多一个 epoch，IS ratio 偏离更大（$\pi_\theta$ 在更新，但 $\pi_{\text{old}}$ 不变），clip 越来越紧，到后期梯度接近 0
- GRPO 通常只用 1 epoch（因为 group 数据量已足够大）

**Generation length**：
- 训练时 max length 设得太短 → 截断 reward（模型学会「早停」而非解决问题）
- 太长 → 显存/时间开销爆炸，长尾问题严重
- 实践：通常设为期望输出长度的 2–3 倍；DAPO 用「动态截断」（超过阈值就截，但给截断样本 0 reward 而非丢弃）

</details>

<details class="exercise">
<summary><span class="q-label">Q13</span> <span class="q-text">Dr.GRPO、DAPO、GSPO、CISPO、SAPO、DPPO、MaxRL、SimKO 相比 GRPO 改了什么，各有什么局限？</span></summary>

| 方法 | 核心改动 | 解决的问题 | 局限 |
|------|---------|------------|------|
| **Dr.GRPO** | 对 degenerate group（全对/全错）做特殊处理 | 避免除零和虚假梯度 | 增加数据过滤复杂度 |
| **DAPO** | 去掉 KL，改 clip lower bound，token 级 loss | 让模型能大幅偏离 SFT | 没有 KL 约束，稳定性依赖 clip 设置 |
| **GSPO** | 序列级 KL 代替 token 级 | 减少 high-entropy token 的 KL 贡献 | 序列级 KL 梯度估计更复杂 |
| **CISPO** | IS clip 随 staleness 动态调整 | 异步 RL 下的 off-policy 修正 | 需要准确估计 staleness |
| **SAPO** | 自适应 clip 阈值（基于 IS ratio 分布） | 避免固定 $\epsilon$ 在不同任务上次优 | 额外的统计开销 |
| **DPPO** | Distributed PPO，把 critic 分布式部署 | 降低 critic 显存瓶颈 | 通信开销增加 |
| **MaxRL** | MiniMax 的 agentic RL 算法（配合 Forge） | 超长上下文、多 agent | 非开源 |
| **SimKO** | Simplified KL-regularized Objective | 把 KL 从 reward 移到 loss，数值更稳定 | 超参敏感 |

**关键规律**：这些方法几乎都在操弄三个旋钮：(1) KL 惩罚的位置和强度，(2) clip 的范围和形式，(3) advantage 的计算/归一化方式。没有万能方案，选哪个取决于任务和规模。

</details>

<details class="exercise">
<summary><span class="q-label">Q14</span> <span class="q-text">TRPO、DPPO、AReaL 如何对 RL 目标施加 trust region 约束？</span></summary>

**TRPO（Trust Region Policy Optimization）**：通过精确的 KL 约束做 trust region：
$$
\max_\theta \mathbb{E}\left[\frac{\pi_\theta}{\pi_{\text{old}}} A\right] \quad \text{s.t.} \quad \text{KL}[\pi_{\text{old}} \| \pi_\theta] \leq \delta
$$
用二阶方法（Fisher 信息矩阵的近似逆）求解。计算开销极大，LLM 场景不可用。

**PPO 的 trust region**：用 clip 代替精确 KL 约束，一阶方法，实践效果接近 TRPO。Clip 是一种「软性」trust region：不惩罚 ratio 在 $[1-\epsilon, 1+\epsilon]$ 内的更新。

**DPPO（Distributed PPO）**：trust region 约束层面与 PPO 相同，但解决的是 PPO 在大规模分布式场景下 critic 的瓶颈（critic 参数量 ≈ actor，显存压力大）。DPPO 把 critic 的参数分布式存储和计算，trust region 约束不变。

**AReaL 的 trust region**：Decoupled clip，staleness $s$ 越大 clip 越紧（见 Q7/CISPO 部分）。从 trust region 角度理解：staleness 越大，$\pi_\theta$ 与生成样本时的 $\pi_{\text{old}}$ 差距越大，需要更保守的更新步长。

</details>

<details class="exercise">
<summary><span class="q-label">Q15</span> <span class="q-text">RL 能根本性地扩展 LLM 的能力边界吗？</span></summary>

这是一个开放性学术讨论题，没有确定答案，但有几个关键论点：

**支持「能扩展」的视角**：
- DeepSeek-R1：通过 RL 解锁了 SFT 无法直接产生的「长推理链」能力
- AlphaGo / AlphaCode：RL 在封闭任务上的能力远超人类，说明可能可以扩展
- 理论上，RL 的探索机制可以找到 SFT 数据集中没有覆盖的解题路径

**反对/限制的视角**：
- 清华 IIIS（ICLR 2026）等研究：在严格控制的可验证环境下，RL 训练 LLM 只是在已有能力的分布内重新分配，而非产生全新能力
- RL 无法从 0 教会模型它完全不懂的知识（如从未见过的数学定理）
- Reward hacking：模型会找到「语义正确但形式满足 verifier」的取巧解

**务实结论**：RL 在「模型已有潜力，但需要激励才能可靠输出」的场景下效果显著（如数学推理）；在「真正缺乏知识」的场景下效果有限。它是一个**能力释放器**，而非**知识注入器**。

</details>

<details class="exercise">
<summary><span class="q-label">Q16</span> <span class="q-text">基于 ProRL 等工作，如何思考 RL 训练的 scaling 边界？</span></summary>

**ProRL（Prolonged RL Training）**的核心发现：RL 训练的性能随训练步数的增长遵循类似预训练的 power law，但有一个**能力涌现的阈值**：在某个 compute 量之前几乎没有改善，超过阈值后快速提升。

**Scaling 的关键变量**：
- **Rollout 数量**：更多样本 = 更好的梯度估计 = 更稳定的学习
- **任务难度分布**：太简单的任务（全对）和太难的任务（全错）都不提供学习信号；curriculum 学习很重要
- **Reward 信号质量**：reward 越可靠，scaling 越有效；RM reward hacking 是 scaling 的硬上界

**与预训练 Scaling 的异同**：
- 相同：compute 越多，性能越好；存在 power law
- 不同：RL 的 scaling 是「在给定基础模型上」的 scaling，受限于基础模型的能力边界；不像预训练可以无限累积知识

**实践建议**：在选定任务上，先用小规模 RL 验证 reward 信号和算法有效性；scale 之前确保 rollout 数据质量（特别是 reject-sample ratio 不要过高或过低）。

</details>

<details class="exercise">
<summary><span class="q-label">Q17</span> <span class="q-text">OPD（On-Policy Distillation）比传统 RL 和 SFT 有什么改进？有哪些应用场景？</span></summary>

**OPD 的核心思想**：Student 模型用自己当前的 policy 生成序列，Teacher 模型（更强的模型）给每个 token 提供 logit 指导，Student 学习 Teacher 的分布而非 one-hot label。

**相比 SFT 的优势**：
- SFT 的 teacher-forcing 不处理 distribution shift（训练时看 ground truth，推理时看自己的输出）
- OPD 用 student 自己生成的序列做 prefix，teacher 只提供 token 级软标签（KD），学生学到的是「在自己的分布下如何纠错」

**相比 RL 的优势**：
- RL 需要设计 reward function，对 verifiable 任务以外很难设计
- OPD 直接用 teacher 的 logit 作为 reward 信号，无需人工设计 reward

**应用场景**：
- 知识蒸馏：大模型（GPT-5）→ 小模型，同时保持 on-policy 特性
- 持续学习：模型自我改进，用当前版本的输出 + 更好版本的 logit 做 OPD
- RLCD（RL from Contrast Distillation）：结合 OPD 和 preference learning

</details>

<details class="exercise">
<summary><span class="q-label">Q18</span> <span class="q-text">LLM 的推理能力在训练的哪个阶段涌现？</span></summary>

**证据梳理**：

| 阶段 | 推理能力的表现 |
|------|---------------|
| 预训练 | 存在隐式推理能力（few-shot CoT 工作），但不可靠；能力由训练数据决定 |
| SFT（on CoT data） | 显著提升推理稳定性；模型学会「先想后答」的格式 |
| **RL/RLVR** | **解锁「自主探索推理路径」**：DeepSeek-R1-Zero 证明不需要 SFT，纯 RL 可以涌现长推理链 |

**关键实验（DeepSeek-R1-Zero）**：从 base model（无 SFT）直接做 GRPO，模型自主学会了：
- 「Aha moment」：中途改变思路，回溯错误推理
- 更长的 reasoning chain（自动扩展 token）
- 自我验证

**学术争议**：这个「涌现」是真正的新能力，还是预训练中已有、SFT 没能激活的能力被 RL 解锁？目前主流观点倾向于后者（RL 是能力释放器），但这不影响 RL 的实用价值。

</details>

<details class="exercise">
<summary><span class="q-label">Q19</span> <span class="q-text">从 DeepSeek R1 到 V3.2 再到未来 V4，引入了哪些 RL 相关改进？MoE 里的 RL 有什么不同？</span></summary>

**DeepSeek / MoE RL 的演进线索**：

| 版本 | RL 相关改进 |
|------|------------|
| R1-Zero | 纯 GRPO from base，无 SFT，证明 RL 可以直接涌现推理能力 |
| R1 | 加入 SFT warm-up（冷启动）+ RL；rejection sampling 补充 SFT 数据；多阶段训练 |
| V3（MoE） | RL 在 MoE 架构上，需要处理 Expert Parallelism；引入 routing replay |
| V3.2 | 进一步 RL scaling；更长 CoT；agentic 场景 |

**MoE 里的 RL 特殊挑战**：

1. **Expert Parallelism × Rollout**：rollout 时 MoE 模型需要 AllToAll 通信（token 路由），这在 decode 阶段（小 batch）效率极低。通常 rollout 用 TP 而非 EP 来避免 AllToAll 延迟。

2. **Router 漂移**：RL 训练会改变 router 的参数，可能导致 expert load imbalance（少数 expert 被过度路由）。需要 auxiliary load balancing loss。

3. **MoE Mismatch**（见 Q11）：router 不一致是 MoE RL 的核心工程问题。Routing Replay 是系统侧解法；GSPO 是算法目标侧解法。

4. **显存放大**：MoE 的 expert 参数总量远大于激活参数，EP 下每 GPU 显存不变但通信显著增加。RL 的显存压力（actor+ref+KV）与 EP 的通信压力叠加。

</details>

---

## C · 单机与显存（Q20/Q22/Q26）

<details class="exercise">
<summary><span class="q-label">Q20</span> <span class="q-text">GRPO 训练时显存压力应该怎么拆分？</span></summary>

不要用固定的“几份模型”背答案。GRPO 的显存压力取决于 colocated / disaggregated 布局、FSDP / ZeRO 分片策略、reference 是否共享、rollout engine 是否常驻，以及 rollout 的 context length、group size 和 KV cache 策略。

更稳的拆法是按四类状态回答：

| 类别 | 主要内容 | 优化方向 |
|------|----------|----------|
| 训练侧模型状态 | actor 参数、梯度、optimizer states，可能还有 reference / old policy logprob 相关状态 | FSDP / ZeRO-3 分片、optimizer state sharding、reference sharing |
| 推理侧状态 | rollout engine 权重副本、KV cache、paged cache metadata、request buffers | rollout/training 共址复用权重、prefix sharing、KV paging、sleep/offload |
| 反向传播状态 | activation、microbatch buffer、sequence packing buffer | activation checkpointing、sequence packing、microbatch 调整 |
| 瞬时通信 buffer | all-gather、reduce-scatter、weight sync bucket、reshard buffer | bucket size 调参、overlap communication、分桶权重同步 |

GRPO 相比 PPO 省掉 critic，但 group sampling 会把每个 prompt 的 completion 数放大到 G 条，KV cache、activation 和 reward 计算等待时间都会上升。所以 GRPO 的总显存压力不一定比 PPO 低，关键看 critic 节省是否大于 rollout/KV/activation 放大。

</details>

<details class="exercise">
<summary><span class="q-label">Q22</span> <span class="q-text">INT8 vs FP8，训练和推理分别推荐哪种精度？权衡是什么？</span></summary>

见 [[MLSYS14 Post-Training Infra#6.5 精度专题：INT8 vs FP8]] 的完整分析。

**快速回答**：

- **Training 推荐 FP8**：H100 的 FP8 矩阵乘法比 BF16 快 2×；需要 scaling factor；E4M3（forward）+ E5M2（backward）组合
- **Inference 推荐 INT8 weight-only**：不需要 calibration；权重减半降低 HBM 带宽压力；decode 是 memory-bound，INT8 直接提升吞吐

**RL 的额外考量**：Rollout 侧 INT8 + Training 侧 FP8 会引入 mismatch（精度不同的 logprob 差异）。有框架（ROLL）提供统一精度配置来解决这个问题。

</details>

<details class="exercise">
<summary><span class="q-label">Q26</span> <span class="q-text">大规模多节点 RL 训练中 backpropagation 是如何实现的？</span></summary>

见 [[MLSYS14 Post-Training Infra#6.6 MoE × RL]] 中的多节点 backpropagation 部分。

**快速回答**：

| 并行方式 | 梯度通信方式 | 关键技术 |
|----------|------------|---------|
| Data Parallel（DP） | AllReduce 梯度 | gradient bucketing（按 bucket 打包通信）；bucket 内 compute 与 communication 重叠 |
| ZeRO-3 / FSDP | ReduceScatter（梯度）+ AllGather（参数） | 前向 AllGather 参数，反向 ReduceScatter 梯度 |
| Tensor Parallel（TP） | AllReduce in Attention/FFN | column parallel → row parallel 的 AllReduce |
| Pipeline Parallel（PP） | P2P send/recv 梯度 | 1F1B 调度；梯度在 stage 间反向传递 |

**RL 的特殊性**：RL 的 loss 来自 advantage 加权的 log-prob，需要 rollout 时记录的 logprob 和 token-level advantage 随数据一起传到训练端。这额外增加了 data shuffling 和 packing 的 overhead，是 slime 的 Data Buffer 专门处理的问题。

</details>

---

## D · Rollout 系统（Q21/Q23/Q24/Q25/Q28）

<details class="exercise">
<summary><span class="q-label">Q21</span> <span class="q-text">分布式推理中 KV cache 迁移与多 GPU 通信策略是什么？</span></summary>

**场景**：prefill 和 decode 分离部署（PD 分离），prefill 在一批 GPU 上完成后，KV cache 需要迁移到 decode GPU 继续自回归生成。

**KV cache 迁移的挑战**：KV cache 体积 = $2 \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len} \times n_{\text{layers}}$，对 32B 模型的 8K 序列可达数 GB。迁移需要跨机网络传输，延迟不可忽视。

**主流方案**：

| 方案 | 机制 | 适用场景 |
|------|------|---------|
| NCCL P2P | GPU 直接发送（DMA） | 同集群内，NVLink/InfiniBand 高带宽 |
| RDMA（Mooncake/InfiniBand） | CPU 旁路，RDMA write 到 decode GPU | 跨机高性能，低延迟 |
| Disaggregated KV store | KV 存 CPU 内存（如 Mooncake），decode GPU 按需拉取 | 超长上下文，KV 超 GPU 显存 |
| 避免迁移 | Prefill+Decode 共址（co-serving） | 小规模，简单场景 |

**RL 里的 KV cache 通信**：AReaL 的 re-prefill 策略不迁移 KV，每次新权重下重算；slime/SGLang 的 prefix sharing 让 GRPO 的 G 个 completion 共享 prefill KV，显著减少 prefill 通信量。

</details>

<details class="exercise">
<summary><span class="q-label">Q23</span> <span class="q-text">RL rollout 的长尾问题是什么？如何解决？</span></summary>

见 [[MLSYS14 Post-Training Infra#6.1 长尾 Rollout 与对策]] 的完整分析。

**快速回答**：

长尾问题 = 批内少数超长序列拖慢整批。1% 的 32K token 序列可以让 99% 的 GPU 闲置 4× 时间。

**解法优先级**：
1. **Partial rollout / interruptible rollout**：适合 agent、多轮工具调用和长推理链，能保留慢样本的训练信号，但要维护 prefix、policy version 和恢复状态。
2. **Length-aware scheduling**：适合 prompt 长度差异明显、任务形态稳定的在线 rollout，基本不改变训练分布。
3. **Over-sampling + earliest-completion 选取**：适合要求 step time 稳定的大规模 GRPO，但要防止模型偏向短答案。
4. **截断 / rejection sampling**：适合早期实验和格式严格任务，工程最简单，但会浪费采样算力并可能引入分布偏差。

</details>

<details class="exercise">
<summary><span class="q-label">Q24</span> <span class="q-text">Continuous batching 在 RL 训练中引入了哪些新问题？vLLM 和 SGLang 有何差异？</span></summary>

见 [[MLSYS14 Post-Training Infra#6.2 Continuous Batching 在 RL 里的新问题]] 的完整分析。

**核心差异总结**：

- **Continuous batching 的 RL 新问题**：episode 边界对齐困难；process reward 触发时机复杂；KV eviction 压力大
- **vLLM vs SGLang 最关键差异**：SGLang 的 RadixAttention 对 GRPO 的 prefix sharing 天然友好（同 prompt 的 G 个 completion 共享 prefill KV），vLLM 需要手动配置 prefix caching

</details>

<details class="exercise">
<summary><span class="q-label">Q25</span> <span class="q-text">如何衡量 vLLM 和 SGLang 的利用率？RL 训练中如何评估 KV cache 利用率？</span></summary>

**vLLM 核心指标**：
- `gpu_cache_usage_perc`：KV cache 页面使用率，<50% 说明浪费或 prefix caching 命中率低
- `num_running_seqs`：并发序列数，越高 GPU 利用率越高
- `num_waiting_reqs`：队列积压，持续 >0 说明推理是瓶颈

**SGLang 核心指标**：
- `cache_hit_rate`：prefix cache 命中率，GRPO 场景下应 >80%（同 prompt G 个 completion）
- `/get_server_info` API：实时返回 queue depth、tokens per second、cache stats

**RL 场景特有关注点**：
- **abort 比率**：interruptible rollout 框架里，abort 过高说明训练和 rollout 速度严重不匹配
- **重用率**（slime 的 OPSM）：只对最优 completion 的 token 计算梯度时，有效 token 利用率
- **Reward variance within group**：variance 接近 0 说明任务太简单或太难，advantage 信号失效

</details>

<details class="exercise">
<summary><span class="q-label">Q28</span> <span class="q-text">在 AReaL 或其他 partial rollout 框架里，前一个 policy 的 KV cache 是否保留？</span></summary>

**AReaL 的选择：不保留，做 re-prefill。**

**原因**：旧 policy 计算的 KV cache 使用了旧的 Q/K/V projection 权重。新 policy 权重更新后，用旧 KV cache 继续 decode 会产生注意力不一致——新 Q 与旧 K 做 attention，物理上是无效的（但不会报错，会产生错误的 logprob 和生成质量）。

**Re-prefill 的代价**：需要对 partial sequence 的 prefix 重新做一次 prefill（前向计算）。代价通常可忽略，因为 prefill 是 compute-bound（高 FLOPS 利用率），比 decode 快得多。

**对比其他方案**：
- **Magistral（Mistral）**的 weird strategy：保留 KV cache，直接用新权重继续 decode（per-forward-pass weight swap）。物理上等于用「混合精度」的 KV（一部分由旧权重生成，一部分由新权重生成）。效率高但算法正确性存疑。
- **SkyRL / slime 的 abort + prefix-resume**：中断序列，保存 prefix token 序列（不保存 KV），用新权重从 prefix 重新做 prefill。与 AReaL 类似，更正确的选择。

</details>

---

## E · 异步框架（Q27/Q32/Q33/Q34/Q35）

<details class="exercise">
<summary><span class="q-label">Q27</span> <span class="q-text">有哪些异步 RL 框架？它们各自解决了什么同步瓶颈？</span></summary>

见 [[MLSYS14 Post-Training Infra#6.4 异步 RL 的设计空间]] 的完整分析。

**快速参考表**：

| 框架 | 解决的核心瓶颈 | 关键机制 |
|------|---------------|---------|
| **AReaL** | 长尾序列 + weight sync idle | Interruptible rollout + re-prefill + staleness-aware PPO |
| **slime（async）** | Weight sync dead time | 分桶 NCCL + abort-in-flight + buffer 队列 |
| **ROLL Flash** | Agentic 多轮等待 | 异步 reward server + episode-level buffer |
| **PipelineRL** | 极致 weight sync overhead | Per-forward-pass weight swap（逐 token 更新） |
| **PRIME-RL** | 跨地域大 staleness | 版本跟踪 + depth bound + IS 三合一 |
| **veRL（async）** | 基础的 train/rollout 解耦 | Disaggregated deployment + bounded buffer |

</details>

<details class="exercise">
<summary><span class="q-label">Q32</span> <span class="q-text">AReaL 和 slime 对 RL rollout 瓶颈的理解有何不同？</span></summary>

见 [[MLSYS14 Post-Training Infra#5.5 AReaL（蚂蚁 & 清华 IIIS）]] 和 [[MLSYS14 Post-Training Infra#5.4 slime（THUDM/智谱）]] 的详细分析。

**一句话总结**：
- **slime**：瓶颈是「权重同步期间的 dead time + 推理引擎效率」→ 更快的传输（分桶 NCCL）+ 更好的引擎（SGLang）
- **AReaL**：瓶颈是「长尾序列拖慢整批」→ Interruptible rollout，让慢序列不阻塞

两种理解都正确，针对不同场景（中等长度 RLVR vs 超长 CoT/agentic）。

</details>

<details class="exercise">
<summary><span class="q-label">Q33</span> <span class="q-text">完全异步 RL 中的 staleness 如何思考？实践中的典型取值是多少？</span></summary>

**Staleness 的数学定义**：生成样本时使用的 policy 版本 $\pi_k$，与当前训练的 policy 版本 $\pi_{k+s}$ 之间的步数差 $s$。

**Staleness 的影响**：
- 对算法：IS ratio $\pi_{k+s}/\pi_k$ 偏离 1，gradient 估计出现偏差
- 偏差大小：$s$ 越大，$\pi_{k+s}$ 与 $\pi_k$ 差越大，IS ratio 分布越宽，variance 越大
- AReaL 实验：$s \leq 8$ 时，最终性能不受影响（IS ratio 偏差在 clip 范围内）

**实践典型值**：
- 同步系统：$s = 0$（严格 on-policy）
- double-buffer：$s \in \{0, 1\}$
- bounded queue depth=K：$s \in [0, K]$，均值约 $K/2$
- **AReaL 实践**：均值 $s \approx 2–4$ 步，最大 $s \leq 8$
- PRIME-RL 跨地域设置：$s$ 可达 10+ 步，需要 IS 修正 + 版本门控

**控制手段**：
1. Buffer depth 上限（硬约束，超过则丢弃）
2. IS weight clip（$r_t$ 限制在 $[c_{\min}, c_{\max}]$，如 $[0.1, 10]$）
3. 主动平衡 rollout 和 train 速度（让 staleness 均值稳定）

</details>

<details class="exercise">
<summary><span class="q-label">Q34</span> <span class="q-text">数据在 slime 中如何流动？如何与 Megatron 集成？loss 如何计算？</span></summary>

见 [[MLSYS14 Post-Training Infra#5.4 slime（THUDM/智谱）]] 的完整数据流详解。

**快速回答**（6 步）：

1. **SGLang 生成**：Rollout Engine 接受 prompt，输出 completion + per-token logprobs + finish reason
2. **Reward 计算**：Data Buffer 调用 reward verifier（规则/RM），得到 sequence-level 或 token-level reward
3. **Advantage 计算**：Data Buffer 在组内计算 GRPO advantage（z-score 归一化）
4. **数据 packing**：Data Buffer 把 {prompt, completion, logprobs, advantage} 打包为 Megatron 可接受的格式（变长序列 packing，最大化 batch token 利用率）
5. **Megatron Forward/Backward**：
   - Forward：用当前 actor 重算 logprobs（$\pi_\theta$），同时推理 reference 模型（$\pi_{\text{ref}}$）
   - Loss = $\text{GRPO}_{\text{clip}}(\pi_\theta / \pi_{\text{old}}, \hat{A}) + \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})$（可选 OPSM mask：只对最优 completion 的 token 计算梯度）
6. **权重广播**：Megatron 完成 optimizer step 后，通过 **分桶 NCCL**（每 bucket 1GB）广播给 SGLang 集群；SGLang 调用 `update_weights` API 热更新

**OPSM Masking**：只对 group 内 reward 最高的 completion 的 token 计算 loss，其余 completion 梯度为 0。物理含义：只从「正确样本」学，不从「错误样本」学惩罚（认为惩罚信号噪声大）。

</details>

<details class="exercise">
<summary><span class="q-label">Q35</span> <span class="q-text">如果要在 VeRL、TRL、Unsloth、AReaL、slime 中选一个，如何选择？</span></summary>

见 [[MLSYS14 Post-Training Infra#7.1 Q35 直答]] 的完整决策树。

**快速决策**：

| 需求 | 推荐 |
|------|------|
| 单机，快速验证 | **TRL** |
| 单机 LoRA，极致效率 | **Unsloth**（SFT 类任务）或 TRL |
| 多节点，≤70B，标准 RLHF | **veRL**（colocate + FSDP） |
| 多节点，100B+ 密集或 MoE | **slime**（Megatron + SGLang，生产验证最充分） |
| 长 CoT，输出 >8K，要最大吞吐 | **AReaL**（fully async + interruptible） |
| Agentic，多工具，复杂 scaffold | **ROLL**（成熟的五角色 agentic 支持） |

**Unsloth 的定位**：专注单 GPU LoRA 训练效率（SFT/DPO），不支持多节点 RL，不适合做大规模 RL 训练。拿来跟 veRL/slime 比较是错误的分类。

</details>

---

## F · 进阶系统（Q29/Q30/Q31）

<details class="exercise">
<summary><span class="q-label">Q29</span> <span class="q-text">Expert Parallelism 如何影响 MoE 的吞吐？</span></summary>

见 [[MLSYS14 Post-Training Infra#6.6 MoE × RL]] 的 EP 分析。

**核心公式**：EP 的关键 overhead 是 AllToAll（token routing），其通信量为：
$$
\text{AllToAll volume} = \text{batch\_size} \times d_{\text{model}} \times 2 \times \text{top-k}
$$

**EP 的盈亏平衡点**：当每个 expert 平均处理的 token 数 $n_{\text{tokens/expert}} \geq 8$–16 时，expert 计算时间足以 overlap AllToAll 通信。低于此阈值（decode 阶段的 batch size 通常只有 1–8 token per expert），EP 的通信开销超过收益。

**实践结论**：
- Prefill（大 batch）：EP 收益显著，通信可被 overlap
- Decode（小 batch）：通常不用 EP，改用 TP；decode 的每个 expert 只处理 1–2 token，AllToAll 延迟不可接受
- RL rollout 几乎全是 decode → 推荐 rollout 侧用 TP 而非 EP

</details>

<details class="exercise">
<summary><span class="q-label">Q30</span> <span class="q-text">长上下文训练中 compute-communication overlap 如何设计？Megatron 和 FSDP 在并行策略上的差异？</span></summary>

见 [[MLSYS14 Post-Training Infra#6.6 MoE × RL]] 的长上下文和并行策略分析。

**Compute-Communication Overlap 的核心技术**：

| 技术 | 原理 | 适用场景 |
|------|------|---------|
| AllReduce 与矩阵乘 overlap | 启动 async AllReduce 后立即开始下一层计算 | DP；需要 gradient bucketing |
| AllGather prefetch（FSDP） | backward 时预先 AllGather 下一层参数 | FSDP ZeRO-3 |
| Ring attention（CP） | Sequence Parallel 下，各 GPU 轮流传递 KV，本地 Q 与收到的 KV 做 attention | 超长序列 |
| Pipeline bubble 隐藏 | 1F1B + interleaved；micro-batch 填充 bubble | PP 场景 |

**Megatron vs FSDP 的核心差异**：
- **Megatron**：原生支持 TP + PP + CP + EP 的任意组合；pipeline schedule 精细优化（1F1B、interleaved 1F1B、virtual pipeline）；MoE EP 正确实现
- **FSDP2**：仅支持 DP + TP + CP（无 PP，无 EP）；更简单易用，适合 ≤70B 密集模型；ZeRO 分片原生支持

**结论**：超大 MoE 或需要 PP 的场景必须用 Megatron；研究场景或 ≤70B 模型用 FSDP 更方便。

</details>

<details class="exercise">
<summary><span class="q-label">Q31</span> <span class="q-text">如何实现确定性执行？什么是 batch invariance？是什么导致了它？Atomic add 能解决这个问题吗？</span></summary>

**Batch Invariance 的定义**：给定相同的输入 token 序列，无论 batch size 是多少，模型输出的 logprob 应该完全相同。违反 batch invariance 会导致 RL 的 IS ratio 出现系统性误差。

**破坏 batch invariance 的来源**：

1. **Atomic Add 的非确定性**：GPU 上的并行线程使用 `atomicAdd` 对共享内存求和，浮点加法不满足结合律，线程调度顺序不同 → 结果不同。这影响 LayerNorm（求 mean/variance 时的 reduction）、softmax 等操作。

2. **不同 batch size 的 padding 方式**：padding token 的处理若不一致，attention mask 不同，输出不同。

3. **Dropout**：非确定性操作，RL 推理时应关闭（但 training forward 通常要开 dropout）。

4. **Flash Attention 的数值不确定性**：不同版本的 Flash Attention 对分块 softmax 的精度处理略有差异。

**Atomic add 能解决吗？** **部分能**：

- `atomicAdd` 可以改为排序后顺序加（确定性，但慢）；PyTorch 提供 `torch.use_deterministic_algorithms(True)` 来强制使用确定性算子
- 但强制确定性会禁用某些高效算子（如部分 cuDNN 实现），**显著降低训练速度**（通常 10–30%）
- **实践选择**：大多数框架不强制确定性；通过 IS clip 来容忍小量 mismatch，而不是追求完美确定性

**真正的解决方案**：Keep Routing + Keep Sampling Mask（见 Q11）；或用完全相同的算子实现（训练和推理用同一套 kernel）。

</details>

---

参考资料：[Sheriyuo · RL Interview Questions 2026](https://x.com/sheriyuo/status/2063295181131247674)

返回主文档：[[MLSYS14 Post-Training Infra]]
