import { Fragment, useEffect, useId, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import './App.css';

const markdownModules = import.meta.glob('../notes/**/*.md', {
  eager: true,
  import: 'default',
  query: '?url',
});

const isDraftMode = import.meta.env.DEV;

const llmDraftOverviewContent = isDraftMode
  ? String.raw`# LLM八股 Overview · JD 高频主题拆解

## Motivation

这组笔记先按岗位需求反推选题，而不是按教材目录铺开。口径是：排除 Python、PyTorch、语言、框架、infra、system、GPU、cloud，只保留算法、模型、训练、评测、安全和数据构造相关关键词。

基数是 1,673 条算法相关 JD，计数是“提到该关键词的 JD 数”。高频信号集中在 eval、安全、实验设计、agent、检索记忆、alignment、RL、SFT、pre-training、优化、数据、多模态和 personalization。

## 高频关键词

| Rank | 关键词 | JD 数 | 典型方向 |
| --- | --- | ---: | --- |
| 1 | Evaluation / Benchmarks | 482 | eval、benchmark、回归测试、上线质量 |
| 2 | LLM | 342 | 训练、适配、能力提升、应用集成 |
| 3 | Computer Vision | 319 | 图像/视频理解、VLM、机器人/自动驾驶视觉 |
| 4 | Safety | 306 | 风险控制、误用防护、安全评测、guardrail |
| 5 | Experimentation | 301 | ablation、A/B、指标分析、实验设计 |
| 6 | Autonomy / Robotics | 283 | 感知、规划、决策、控制 |
| 7 | Agents | 282 | 工具使用、多步任务、软件操作、workflow |
| 8 | RAG / Retrieval | 211 | embedding、语义搜索、知识库问答、上下文召回 |
| 9 | NLP | 176 | 文本理解、生成、分类、对话 |
| 10 | Reinforcement Learning | 160 | RLHF/RLAIF、策略优化、agent 行为优化 |
| 11 | Alignment | 155 | 人类偏好、安全规则、产品目标、政策约束 |
| 12 | Fine-tuning / SFT | 144 | SFT、LoRA、领域适配、任务适配 |
| 13 | Optimization | 137 | 训练目标、排序策略、决策策略、质量优化 |
| 14 | Reasoning | 126 | 多步推理、规划、代码/数学/工具能力 |
| 15 | Data Curation / Datasets | 109 | 数据筛选、清洗、组织、覆盖面 |
| 16 | Multimodal | 107 | VLM、语音、多模态 agent |
| 17 | Foundation Models | 102 | 基础模型训练、扩展、评测、适配 |
| 18 | Post-training | 96 | SFT、RLHF、偏好优化、agent 调优 |
| 19 | Recommendation / Ranking | 92 | 推荐、排序、搜索结果优化、召回/ranker |
| 20 | Speech / Audio | 90 | ASR、TTS、音频理解、voice agent |
| 21 | Personalization | 80 | 用户建模、个性化 assistant、feed 排序 |
| 22 | Planning | 75 | 任务规划、路径规划、多步决策 |
| 23 | Search | 75 | query understanding、召回、排序 |
| 24 | Simulation | 73 | 仿真训练、测试、验证 |
| 25 | Model Behavior | 65 | 拒答、偏差、越狱、风险行为 |
| 26 | Data Quality | 59 | 噪声过滤、一致性检查、质量指标 |
| 27 | Causal Inference | 50 | 干预效果、实验解释、策略影响 |
| 28 | Annotation / Labeling | 42 | label schema、人类反馈、标注质量 |
| 29 | Transformers | 41 | transformer 建模、训练、适配 |
| 30 | Red Teaming | 38 | 越狱、安全漏洞、滥用路径、失败案例 |

## 1. Evaluation

Placeholder：eval taxonomy、benchmark design、regression eval、online/offline eval、judge reliability、model behavior tracking。

## 2. Safety

Placeholder：policy、risk taxonomy、misuse prevention、jailbreak eval、red teaming、guardrail 设计。

## 3. 统计实验设计

Placeholder：A/B testing、ablation、power analysis、metric design、causal inference、实验解释。

## 4. Agents

Placeholder：tool use、planning、task decomposition、trajectory、environment feedback、failure recovery。

## 5. Search & Memory

Placeholder：RAG、retrieval、semantic search、memory store、query understanding、context construction。

## 6. Alignment

Placeholder：preference modeling、policy constraints、helpful/harmless/honest、model behavior shaping。

## 7. RLVR & Agentic RL

Placeholder：RLHF、RLAIF、RLVR、GRPO/PPO、verifiable reward、agent rollout、tool-use reward。

## 8. SFT

Placeholder：instruction tuning、LoRA、dataset mixture、format learning、domain adaptation、failure modes。

## 9. Pre-training

Placeholder：data mixture、scaling behavior、objective、curriculum、contamination、dedup。

## 10. 优化器

Placeholder：AdamW、learning rate schedule、weight decay、gradient clipping、stability、large-batch training。

## 11. Data Curation

Data curation 不是简单地“多收集一点数据”。它关心的是：哪些数据应该进训练集、怎么生成缺口数据、怎么过滤低质量样本、怎么让数据分布和目标能力对齐，以及怎么用 eval 反过来驱动下一轮数据构造。

一个实用的 mental model：

~~~text
target capability
  -> task / data schema
  -> raw or synthetic candidates
  -> filtering / verification / dedup
  -> train or fine-tune
  -> eval + failure mining
  -> update data recipe
~~~

这里的核心变量通常不是“数据量”，而是：

- Coverage：是否覆盖目标能力的主要 failure modes。
- Difficulty：样本是否有区分度，太简单会浪费训练预算。
- Verifiability：答案、轨迹或结果能不能自动检查。
- Diversity：是否只是模板化重复，还是覆盖不同工具、环境、领域和错误类型。
- Contamination：训练数据是否泄漏 benchmark 或 eval answer。

代表性工作：

| 工作 | 场景 | Data curation 重点 | 可以学习的点 |
| --- | --- | --- | --- |
| [SWE-smith](https://arxiv.org/abs/2504.21798) | 软件工程 agent | 从 Python repo 构建执行环境，自动合成会破坏测试的任务实例；论文报告 128 个 repo、约 50k instances。 | 把真实 repo 变成可训练环境，利用 tests 做自动验证信号。 |
| [SERA](https://arxiv.org/html/2601.20789v1) | repository coding agent | 使用 soft verification 和 synthetic coding agent trajectories；Ai2 release 提到约 200k synthetic trajectories。 | 不只收最终答案，还收 agent 轨迹，并用较便宜的验证信号控制质量。 |
| [Nemotron-Terminal](https://arxiv.org/abs/2602.21193) | terminal agent | Terminal-Task-Gen 结合 seed-based / skill-based task construction，构造 Terminal-Corpus，并研究 filtering、curriculum、long-context training。 | 面向 terminal capability 的数据工程：任务生成、过滤、课程学习和长上下文一起设计。 |
| [OpenThoughts-Agent](https://arxiv.org/abs/2606.24855) | agentic SFT / RL data recipe | 系统 ablate task source、source mixing、task augmentation、difficulty filtering、teacher trace、multi-turn rollout filtering 和 RL data source。 | agent post-training 先做 data recipe：任务分布、轨迹质量、teacher 风格和 RL 环境会直接决定模型学到的行为。 |
| [Autodata](https://www.alphaxiv.org/abs/2606.25996) | agentic data scientist | 用 agentic data scientist 做 synthetic data creation，把数据生成、诊断和更新 recipe 变成循环。 | 可以复现一个小型 autoresearch loop，看 iterative data improvement 是否超过一次性 synthetic generation。 |

这些工作共同指向一个趋势：强 agent 不是只靠更复杂的推理框架，也依赖更系统的数据构造。高质量数据通常来自“任务生成 + 可验证反馈 + 失败样本挖掘 + 迭代更新”的闭环。

## 12. 数据

Placeholder：data curation、data quality、annotation、labeling、synthetic data、filtering、coverage。

候选论文：

- [OpenThoughts-Agent: Data Recipes for Agentic Models](https://arxiv.org/abs/2606.24855)
- [Autodata: An agentic data scientist to create high quality synthetic data](https://www.alphaxiv.org/abs/2606.25996)

OpenThoughts-Agent 可以作为 “agentic SFT / RL data recipe” 的入口：先把 task source、source mixing、difficulty filtering、teacher trace、multi-turn rollout filter 和 RL data source 做成可 ablate 的变量，再讨论算法。

Autodata 可以作为 “autoresearch loop 提升数据质量” 的入口。核心问题不是单次生成 synthetic data，而是让 agent 反复做：提出数据假设、生成或改写数据、训练/评测、诊断失败 case、更新数据策略。后续可以复现一个小版本：

~~~text
seed tasks / weak dataset
  -> data scientist agent proposes data edits
  -> synthetic data generation
  -> train or fine-tune small model
  -> eval + failure mining
  -> update data recipe
  -> repeat
~~~

复现时重点看三件事：

- 数据质量指标怎么定义：accuracy gain、coverage、diversity、difficulty、contamination risk。
- agent 的 action space 是什么：改 prompt、改 schema、采样 hard cases、过滤低质量样本、生成 counterexample。
- loop 是否真的优于一次性 data generation：需要 ablation，比如 no-agent、no-failure-mining、no-iterative-update。

## 13. 多模态

Placeholder：VLM、speech/audio、video understanding、multimodal alignment、evaluation。

## 14. Personalization

Placeholder：user modeling、personalized ranking、assistant memory、preference adaptation、privacy boundary。
`
  : '';

const probabilityDraftContent = isDraftMode
  ? String.raw`# Quant 草稿 · 概率基础公式与记忆框架

> Draft：这一页先放概率面试里最常用的基础工具。内容按解题动作重排，不按截图顺序组织；后续可以继续加条件期望、Bayes、order statistics、Poisson process 和 martingale。

## 0. 先按题型选工具

很多概率题不是难在公式本身，而是难在判断应该用哪一个视角。可以先问四个问题：

~~~mermaid
flowchart TD
  A[看到随机题] --> B{问的是概率还是期望}
  B -->|概率| C{是否是多个事件的并集}
  C -->|是| D[容斥 / 补集 / union bound]
  C -->|否| E[条件概率 / Bayes / 分布计算]
  B -->|期望| F{变量是否容易拆成指示变量}
  F -->|是| G[线性期望]
  F -->|否| H{是否是非负整数等待时间}
  H -->|是| I[尾和公式]
  H -->|否| J{是否是变换后的随机变量}
  J -->|是| K[变量变换 / Jacobian]
  J -->|否| L[按定义求和或积分]
~~~

一个实用记忆：

| 题目关键词 | 第一反应 |
| --- | --- |
| 至少一个、任意一个、并集 | 容斥或补集 |
| 平均值、总次数、贡献 | 期望定义或线性期望 |
| 等多久、第一次成功 | geometric waiting time 或 first-step analysis |
| 非负整数变量 | 尾和公式 |
| $Y=g(X)$、密度变换 | 变量变换 |
| 多维坐标变换 | Jacobian determinant |

---

## 1. 期望：先把随机变量写清楚

期望是对所有可能取值做加权平均。离散随机变量：

$$
\mathbb{E}[X] = \sum_x x \cdot \mathbb{P}(X=x)
$$

连续随机变量：

$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x)\,dx
$$

面试里更常用的是线性期望：

$$
\mathbb{E}\left[\sum_i X_i\right] = \sum_i \mathbb{E}[X_i]
$$

注意：线性期望不要求 $X_i$ 独立。很多计数题会把一个复杂变量拆成很多 indicator：

$$
X = I_1 + I_2 + \cdots + I_n
$$

然后：

$$
\mathbb{E}[X] = \sum_i \mathbb{P}(I_i=1)
$$

记忆图：

~~~mermaid
flowchart LR
  A[复杂计数 X] --> B[拆成 indicator]
  B --> C["X = I1 + I2 + ... + In"]
  C --> D["E[X] = sum P(Ii = 1)"]
  D --> E[不需要独立]
~~~

### 小例子

10 个人随机入座，问坐在自己座位上的人数期望。令 $I_i$ 表示第 $i$ 个人坐对位置：

$$
\mathbb{E}[I_i] = \mathbb{P}(I_i=1)=\frac{1}{10}
$$

所以：

$$
\mathbb{E}\left[\sum_{i=1}^{10} I_i\right]
= 10\cdot \frac{1}{10}=1
$$

---

## 2. 尾和公式：不用先求完整分布

如果 $N$ 是非负整数随机变量，那么：

$$
\mathbb{E}[N] = \sum_{k=1}^{\infty}\mathbb{P}(N\ge k)
$$

这个公式的直觉是：一个取值为 $N$ 的样本，会对前 $N$ 个门槛各贡献 1 次。

~~~text
N = 4

threshold: 1  2  3  4  5  6 ...
contrib:   1  1  1  1  0  0 ...
sum = 4
~~~

对每个样本都成立，取期望后就是尾和公式。

连续非负随机变量也有对应版本：

$$
\mathbb{E}[X] = \int_0^\infty \mathbb{P}(X>x)\,dx
$$

如果 $X$ 可以为负，则可以拆成正负两边：

$$
\mathbb{E}[X]
=
\int_0^\infty \mathbb{P}(X>x)\,dx
-
\int_0^\infty \mathbb{P}(X<-x)\,dx
$$

等价地，也可以用 CDF 写成：

$$
\mathbb{E}[X]
=
\int_0^\infty (1-F_X(x))\,dx
-
\int_{-\infty}^{0} F_X(x)\,dx
$$

### 什么时候用

尾和公式适合这些题：

- 问等待时间，但直接求 $P(N=n)$ 很麻烦。
- 问最大值或覆盖时间，$P(N\ge k)$ 比 $P(N=k)$ 好写。
- 问出现次数或持续长度，事件可以按门槛分层。

---

## 3. 独立重复试验的等待时间

如果每次试验独立，事件成功概率为 $p$，等待第一次成功所需的试验次数记为 $T$，那么：

$$
\mathbb{P}(T>k)=(1-p)^k
$$

用尾和公式：

$$
\mathbb{E}[T]
=
\sum_{k=0}^{\infty}\mathbb{P}(T>k)
=
\sum_{k=0}^{\infty}(1-p)^k
=
\frac{1}{p}
$$

注意这里 $T$ 从 1 开始计数，所以尾和写成：

$$
\mathbb{E}[T]=\sum_{k=0}^{\infty}\mathbb{P}(T>k)
$$

也可以用 first-step analysis：

$$
E = p\cdot 1 + (1-p)(1+E)
$$

解得：

$$
E=\frac{1}{p}
$$

### 例题：掷硬币直到第一次正面

题目翻译：

~~~text
一枚公平硬币不断抛掷，直到第一次出现正面为止。
问：需要抛掷的次数的期望是多少？
~~~

这里“成功”定义为出现正面。公平硬币每次出现正面的概率是：

$$
p=\frac12
$$

所以等待第一次正面的次数 $T$ 服从 geometric distribution：

$$
\mathbb{P}(T=k)=\left(\frac12\right)^{k-1}\frac12,\qquad k=1,2,\ldots
$$

直接用等待时间公式：

$$
\mathbb{E}[T]=\frac{1}{p}=2
$$

也可以用 first-step analysis。设 $E$ 是从现在开始直到第一次正面需要的期望抛掷次数。第一次抛掷后：

- 以概率 $1/2$ 出现正面，过程结束，总共用了 1 次。
- 以概率 $1/2$ 出现反面，已经用掉 1 次，但问题回到原点，还需要期望 $E$ 次。

所以：

$$
E=\frac12\cdot 1+\frac12\cdot(1+E)
$$

解得：

$$
E=2
$$

答案是：平均需要抛掷 2 次。

记忆图：

~~~mermaid
stateDiagram-v2
  [*] --> Try
  Try --> Done: success p
  Try --> Try: fail 1-p, pay one more trial
  Done --> [*]
~~~

---

## 4. 容斥：并集不要重复数

容斥处理的是多个事件的并集。两个事件时：

$$
\mathbb{P}(A\cup B)
=
\mathbb{P}(A)+\mathbb{P}(B)-\mathbb{P}(A\cap B)
$$

三个事件时：

$$
\mathbb{P}(A\cup B\cup C)
=
\mathbb{P}(A)+\mathbb{P}(B)+\mathbb{P}(C)
-\mathbb{P}(A\cap B)-\mathbb{P}(A\cap C)-\mathbb{P}(B\cap C)
+\mathbb{P}(A\cap B\cap C)
$$

一般形式可以记成：

$$
\mathbb{P}\left(\bigcup_{i=1}^n A_i\right)
=
\sum_i \mathbb{P}(A_i)
-
\sum_{i<j}\mathbb{P}(A_i\cap A_j)
+
\sum_{i<j<k}\mathbb{P}(A_i\cap A_j\cap A_k)
-\cdots
$$

符号规律：

~~~text
单个事件:     加
两个交集:     减
三个交集:     加
四个交集:     减
...
~~~

可视化记忆：

~~~mermaid
flowchart LR
  A[先把每个事件都加上] --> B[重叠区域被加多了]
  B --> C[减掉两两交集]
  C --> D[三重交集被减过头]
  D --> E[加回三重交集]
  E --> F[继续交替修正]
~~~

### 常见捷径：先算补集

如果题目问“至少一个成功”，通常更容易写成：

$$
\mathbb{P}(\text{at least one success})
=
1-\mathbb{P}(\text{no success})
$$

容斥是通用方法，补集是很多“至少一个”题的简化版。

---

## 5. 一维变量变换：密度要乘伸缩因子

设 $Y=g(X)$，其中 $g$ 单调且可逆。如果 $X$ 有密度 $f_X(x)$，那么：

$$
f_Y(y)
=
f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|
$$

直觉：概率质量守恒。

$$
f_X(x)\,dx \approx f_Y(y)\,dy
$$

所以密度变换时，不只要把 $x$ 换成 $g^{-1}(y)$，还要乘上长度缩放因子：

$$
\left|\frac{dx}{dy}\right|
$$

记忆图：

~~~mermaid
flowchart LR
  X["X space: small interval dx"] --> G["y = g(x)"]
  G --> Y["Y space: interval dy"]
  Y --> J["density adjusts by |dx/dy|"]
~~~

### 小例子

如果 $Y=2X$，那么 $x=y/2$，并且：

$$
\left|\frac{dx}{dy}\right|=\frac12
$$

所以：

$$
f_Y(y)=f_X(y/2)\cdot \frac12
$$

区间被拉长 2 倍，密度就要压低一半。

---

## 6. 多维变量变换：Jacobian 是面积或体积缩放

多维情况下，设：

$$
Y=g(X),\qquad X=h(Y)
$$

如果变换可逆，那么联合密度满足：

$$
f_Y(y)
=
f_X(h(y))\cdot
\left|
\det\left(
\frac{\partial h(y)}{\partial y}
\right)
\right|
$$

这里的 determinant 是 Jacobian determinant。它表示局部面积、体积或高维体积的缩放比例。

~~~mermaid
flowchart LR
  A["x-space small rectangle"] --> B["transform y = g(x)"]
  B --> C["y-space parallelogram"]
  C --> D["area scaling = |det J|"]
  D --> E["density rescales inversely"]
~~~

常见例子是二维极坐标：

$$
x=r\cos\theta,\qquad y=r\sin\theta
$$

Jacobian determinant 是：

$$
\left|
\det
\begin{pmatrix}
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta}\\
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta}
\end{pmatrix}
\right|
= r
$$

所以：

$$
dx\,dy = r\,dr\,d\theta
$$

这就是为什么极坐标积分里会多一个 $r$。

---

## 7. 复习卡片

| 方法 | 公式 | 什么时候用 |
| --- | --- | --- |
| 期望定义 | $\mathbb{E}[X]=\sum_x xP(X=x)$ 或 $\int xf_X(x)dx$ | 分布已经清楚 |
| 线性期望 | $\mathbb{E}[\sum_i X_i]=\sum_i\mathbb{E}[X_i]$ | 计数题、indicator 拆解 |
| 尾和公式 | $\mathbb{E}[N]=\sum_{k\ge1}P(N\ge k)$ | 非负整数、等待时间、最大值 |
| 几何等待 | $\mathbb{E}[T]=1/p$ | 独立重复试验直到成功 |
| 容斥 | 加单项、减两两、加三重、交替 | 多事件并集 |
| 一维变换 | $f_Y(y)=f_X(g^{-1}(y))|(g^{-1})'(y)|$ | $Y=g(X)$ |
| 多维变换 | $f_Y(y)=f_X(h(y))|\det(\partial h/\partial y)|$ | 坐标变换、联合密度 |

## 8. 最短记忆版

~~~text
期望题:
  能拆 indicator 就拆 indicator。
  不能拆但变量非负，就试 tail sum。

等待题:
  独立重复成功概率 p -> 1/p。
  状态依赖 -> first-step analysis。

并集题:
  先想补集。
  补集不好算，再用容斥。

密度变换:
  先反解原变量。
  再乘 Jacobian。
~~~
`
  : '';

const systemDesignDbScalingContent = String.raw`# System Design 03 · 数据库扩展三件套

这篇是一个基础 system design pattern：先用 QPS / IOPS / 存储容量做粗估，再讨论主从复制、主主复制和数据分区，以及它们在 Feature Store / Embedding Store / Online KV Store 里的类比。

## 0. 基础概念：QPS、IOPS、吞吐和延迟

做数据库扩展题之前，先把几个指标说清楚。很多面试回答的问题不是“方案错了”，而是没有先估算系统到底卡在 CPU、网络、磁盘、数据库连接数，还是单机容量。

### 0.1 QPS / RPS / TPS

QPS 是 queries per second，通常表示每秒查询数。RPS 是 requests per second，通常表示服务每秒请求数。TPS 是 transactions per second，常用于数据库事务或支付交易。

它们经常接近，但不完全一样：

| 指标 | 常见含义 | 例子 |
| --- | --- | --- |
| RPS | 服务入口请求数 | API Gateway 每秒收到 10k 个 HTTP request |
| QPS | 查询请求数 | Search service 每秒处理 20k 次 query |
| TPS | 成功事务数 | Payment service 每秒完成 500 笔交易 |
| DB QPS | 数据库查询次数 | 一个 API 请求打 5 次 DB，则 DB QPS 可能是 API RPS 的 5 倍 |

一个常见坑：

~~~text
用户 QPS != 数据库 QPS

1 个 API request
  -> 读 user profile
  -> 读 feature flags
  -> 查订单列表
  -> 写 audit log

入口 RPS = 1
DB operations = 4
DB QPS 约等于 4
~~~

### 0.2 Throughput、Latency 和 Concurrency

Throughput 是单位时间完成多少工作；latency 是单个请求花多久；concurrency 是同一时刻有多少请求在系统内。

三者可以用 Little's Law 做粗估：

$$
\text{concurrency} \approx \text{QPS} \times \text{latency}
$$

注意 latency 要换成秒。

例子：

~~~text
QPS = 10,000 requests/s
平均 latency = 100 ms = 0.1 s

系统内平均并发请求数约为:
10,000 * 0.1 = 1,000
~~~

这说明即使每秒 1 万请求，如果每个请求在系统里停留 100ms，系统同时要承载大约 1000 个 in-flight requests。

记忆图：

~~~mermaid
flowchart LR
  A[QPS: 每秒进来多少] --> D[Concurrency: 同时在系统里多少]
  B[Latency: 每个请求待多久] --> D
  D --> C[线程 / 连接 / 队列 / 内存压力]
~~~

### 0.3 平均 QPS 和峰值 QPS

日活、月活、请求总量通常只能给平均 QPS。系统设计时要估峰值。

一天有：

$$
24\times 60\times 60 = 86400 \approx 10^5
$$

所以：

$$
\text{avg QPS} \approx \frac{\text{daily requests}}{10^5}
$$

峰值通常可以粗略乘一个系数：

~~~text
peak QPS = avg QPS * peak factor

普通业务: peak factor 3~5
明显潮汐业务: peak factor 5~10
秒杀/热点事件: 可能 10~100+
~~~

例子：

~~~text
每天 1 亿次请求
avg QPS ≈ 100,000,000 / 100,000 = 1,000

如果 peak factor = 5
peak QPS ≈ 5,000
~~~

面试里更重要的是说明假设，而不是死背某个倍数。

### 0.4 IOPS 和磁盘带宽

IOPS 是 input/output operations per second，表示存储系统每秒能处理多少次 I/O 操作。它主要用于估算随机读写压力。

Bandwidth / throughput 表示每秒能传多少数据，常用于大块顺序读写。

| 指标 | 关注点 | 典型瓶颈 |
| --- | --- | --- |
| IOPS | 每秒多少次读写操作 | 小块随机读写、索引 lookup、KV get |
| Bandwidth | 每秒多少 MB/GB | 扫描大文件、备份、日志传输 |
| Latency | 单次 I/O 等多久 | tail latency、同步写路径 |

一个粗略估算：

$$
\text{required IOPS}
\approx
\text{QPS} \times \text{I/O ops per request}
$$

如果每个请求需要 3 次随机读、1 次随机写：

~~~text
API peak QPS = 5,000
I/O per request = 4

required IOPS ≈ 20,000
~~~

如果每个请求还要读取 20KB 数据，那么网络或磁盘带宽约为：

$$
\text{bandwidth} \approx \text{QPS} \times \text{bytes per request}
$$

~~~text
5,000 QPS * 20 KB ≈ 100 MB/s
~~~

这两个估算回答的是不同问题：

~~~text
小对象随机读很多:
  看 IOPS

大对象连续读很多:
  看 bandwidth
~~~

### 0.5 常见容量估算模板

#### 存储容量

~~~text
daily data = daily writes * average record size
retention storage = daily data * retention days * replication factor
~~~

例子：

~~~text
每天 1 亿条 event
每条 500 bytes
保留 30 天
3 副本

raw daily data = 100,000,000 * 500B = 50GB/day
total storage ≈ 50GB * 30 * 3 = 4.5TB
~~~

#### 数据库读写拆分

~~~text
read QPS = total QPS * read ratio
write QPS = total QPS * write ratio
replica count ≈ read QPS / safe read QPS per replica
~~~

例子：

~~~text
peak QPS = 20,000
读写比 = 90% read, 10% write

read QPS = 18,000
write QPS = 2,000

如果单个 replica 安全承载 4,000 read QPS
至少需要 5 个 read replicas
~~~

#### Cache 命中后端压力

~~~text
backend QPS = total QPS * (1 - cache hit rate)
~~~

例子：

~~~text
total QPS = 100,000
cache hit rate = 95%

backend QPS = 100,000 * 5% = 5,000
~~~

这就是为什么高 QPS 系统里，cache hit rate 从 95% 掉到 90% 会很严重：后端压力直接翻倍。

#### 队列和 worker 数

如果任务平均处理时间是 $T$ 秒，每个 worker 一次处理一个任务，那么单 worker 吞吐约为：

$$
\text{worker throughput} \approx \frac{1}{T}
$$

所需 worker 数：

$$
\text{workers} \approx \text{arrival QPS} \times T
$$

例子：

~~~text
每秒进入 200 个任务
每个任务平均处理 0.5 秒

需要并发 worker ≈ 200 * 0.5 = 100
~~~

### 0.6 面试里怎么用这些数字

系统设计里，估算不是为了精确，而是为了决定架构方向。

~~~mermaid
flowchart TD
  A[估 QPS / storage / bandwidth / IOPS] --> B{单机能否承受}
  B -->|读压力大| C[replica / cache / read pool]
  B -->|写压力大| D[partition / queue / batch]
  B -->|容量大| E[sharding / cold storage / retention]
  B -->|延迟高| F[index / cache / async / locality]
  B -->|峰值高| G[autoscale / rate limit / backpressure]
~~~

一个比较稳的回答顺序：

~~~text
1. 先估入口 QPS 和峰值 QPS。
2. 再估每个请求会打多少 DB / cache / storage。
3. 把入口 QPS 转成后端 QPS、IOPS 和 bandwidth。
4. 判断读瓶颈、写瓶颈、容量瓶颈还是延迟瓶颈。
5. 再选择复制、分片、缓存、队列或异步化。
~~~

---

## 0.7 先判断压力来自哪里

数据库扩展题不要一上来就说“加缓存”或“上分片”。先判断系统瓶颈：

~~~mermaid
flowchart TD
  A[数据库压力] --> B{主要压力是什么}
  B -->|读请求太多| C[主从复制 + 读写分离]
  B -->|主库不可用风险| D[主主 / 主备 / 自动故障切换]
  B -->|写请求太多| E[数据分区 / Sharding]
  B -->|数据量太大| E
  C --> F[代价: replication lag / stale read]
  D --> G[代价: 冲突处理 / failover 复杂]
  E --> H[代价: 跨分片查询 / rebalancing / hot shard]
~~~

可以先记住一句：

| 模式 | 解决什么 | 不解决什么 |
| --- | --- | --- |
| 主从复制 | 读扩展、备份、读侧容灾 | 不扩展主库写入能力 |
| 主主复制 | 主库故障时更快切换 | 不让写入能力线性翻倍 |
| 数据分区 | 容量扩展、写入扩展、索引变小 | 增加查询路由和跨分片复杂度 |

---

## 1. 主从复制：用副本扩展读取和容灾能力

主从复制的基本结构是：

~~~text
Primary / Master  ->  Replica / Slave
~~~

主库接收写入，从库复制主库数据。所有会修改数据的操作都进入主库：

~~~text
INSERT
UPDATE
DELETE
CREATE TABLE
ALTER TABLE
~~~

从库通常不直接接收业务写入，而是跟随主库的变更日志更新本地数据。

### 1.1 复制链路怎么工作

以 MySQL 为例，主从复制的核心是 binlog。主库执行数据修改后，会把变更写进 binary log；从库持续拉取主库 binlog 的增量，写入 relay log，再在本地重放这些操作。

~~~mermaid
sequenceDiagram
  participant C as Client
  participant P as Primary
  participant B as Binlog
  participant R as Replica
  participant L as Relay Log

  C->>P: write request
  P->>P: execute mutation
  P->>B: append change event
  R->>B: pull changes after known position
  R->>L: write relay log
  R->>R: replay relay log
~~~

本质上是三步：

~~~text
主库记录变更；
从库拉取变更；
从库本地重放变更。
~~~

### 1.2 主从复制的用途

主从复制最常见的用途有四个。

第一，读写分离。写请求走主库，读请求分摊到多个从库：

~~~text
Write  -> Primary
Read   -> Replica 1 / Replica 2 / Replica 3
~~~

例如：

| 请求 | 路由 |
| --- | --- |
| 用户浏览商品 | Replica |
| 用户查看订单列表 | Replica |
| 用户修改地址 | Primary |
| 用户下单付款 | Primary |

这适合读多写少系统。需要强调：主从复制扩展的是读能力，不是写能力。

第二，查询隔离。不同从库可以承担不同类型的读任务：

~~~text
Replica 1: 线上普通查询
Replica 2: 报表查询
Replica 3: 备份任务
~~~

这样慢查询、报表、备份不直接拖垮主库。

第三，零停机备份。主库继续服务线上请求，从库执行备份任务。

第四，从库故障转移。某个从库挂了，可以从读流量池摘掉，请求转向其他副本。

### 1.3 核心代价：复制延迟

主从复制通常是异步的。主库完成写入后可以先返回成功，不必等待所有从库都重放完成。

这会带来 stale read：

~~~text
用户把昵称 Alice 改成 Bob
  -> 写入 Primary 成功
  -> 用户立刻刷新页面
  -> 读请求被路由到 Replica
  -> Replica 还没同步
  -> 页面仍显示 Alice
~~~

所以主从复制的核心 trade-off 是：

~~~text
提高读取能力和可用性；
但读副本可能短暂落后。
~~~

常见处理方式：

| 场景 | 策略 |
| --- | --- |
| 刚写完立刻读自己的数据 | read-your-writes：短时间强制读主库 |
| 可接受短暂旧数据 | 读从库 |
| 强一致关键路径 | 写主库后读主库，或使用同步复制/多数派协议 |
| 副本落后严重 | 从读池摘掉落后 replica |

---

## 2. 主主复制：高可用优先，不是写入翻倍

主从复制里，主库是唯一写入口。主库挂了以后，需要选新主库、切流量、处理旧主库恢复后的状态。主主复制把两个节点都做成可写节点：

~~~text
Primary A  <->  Primary B
~~~

A 的写入复制到 B，B 的写入也复制到 A。这通常用于双机热备。

~~~mermaid
flowchart LR
  C[Client] --> A[Primary A]
  C --> B[Primary B]
  A -- replicate changes --> B
  B -- replicate changes --> A
~~~

### 2.1 如何避免复制循环

如果 A 的写入复制到 B，B 又原样复制回 A，就会无限循环。MySQL 复制里每台服务器有 server-id，变更日志会记录事件来源。

因此：

~~~text
A 产生事件 e
  -> B 收到 e
  -> B 看到 e 来自 A
  -> B 不再把 e 当作自己的新事件复制回 A
~~~

### 2.2 为什么它不等于写扩展

主主复制看起来有两个主库，但并不意味着写入能力翻倍。原因是两个节点最终都要保存完整数据集，也都要执行对方复制来的写入。

写到 A：

~~~text
Client -> A
A 执行写入
A 写 binlog
B 拉取并重放
~~~

写到 B：

~~~text
Client -> B
B 执行写入
B 写 binlog
A 拉取并重放
~~~

最终 A 和 B 都要承担完整数据、完整索引、完整存储和复制写入。因此主主复制更适合被理解为高可用方案，而不是水平写扩展方案。

### 2.3 主主复制的代价

主主复制会引入：

~~~text
双边都保存完整数据；
双边都执行所有写入；
复制增加磁盘和网络 I/O；
两边同时写同一行可能冲突；
failover 和旧主恢复更复杂。
~~~

如果业务真的需要写扩展，通常要进入数据分区，而不是只靠两个 master。

---

## 3. 数据分区：用 Sharding 扩展容量和写入

复制保存多份相同数据；分区保存不同数据。

~~~text
Replication: 每台机器有完整副本
Sharding: 每台机器只保存一部分数据
~~~

如果单库数据太大，或者单个主库写入压力太高，就需要分片。

### 3.1 基本思想

假设用户表按 user_id 分为 4 个 shard：

~~~text
user_id % 4 = 0  ->  Shard 0
user_id % 4 = 1  ->  Shard 1
user_id % 4 = 2  ->  Shard 2
user_id % 4 = 3  ->  Shard 3
~~~

访问 user_id = 123：

~~~text
123 % 4 = 3
访问 Shard 3
~~~

这样每台机器只负责一部分用户，容量和写入压力都会被分散。

### 3.2 Sharding key 是设计核心

好的 sharding key 需要满足三件事。

第一，查询时经常带这个 key。

如果大部分查询是：

~~~sql
SELECT * FROM orders WHERE user_id = ?
~~~

那么 user_id 很自然。如果 sharding key 在查询里很少出现，就只能广播到所有 shard。

第二，分布要均匀。

按国家分片可能导致 US shard 过热；hash(user_id) 通常更均匀。

第三，减少跨分片查询。

理想查询：

~~~text
定位 shard -> 查询 -> 返回
~~~

跨分片查询：

~~~mermaid
flowchart TD
  Q[Query] --> S0[Shard 0]
  Q --> S1[Shard 1]
  Q --> S2[Shard 2]
  Q --> S3[Shard 3]
  S0 --> M[Merge / Sort / Aggregate]
  S1 --> M
  S2 --> M
  S3 --> M
  M --> R[Response]
~~~

跨分片 join、跨分片事务、全局排序都会明显变复杂。

### 3.3 分片的收益和代价

收益：

~~~text
数据量分散到多台机器；
写入压力分散到多个 shard；
每个节点维护更小索引；
单机存储和内存压力下降；
可以通过增加 shard 扩容。
~~~

代价：

~~~text
跨分片查询复杂；
跨分片事务复杂；
全局唯一 ID 需要设计；
扩容和数据迁移困难；
应用层要处理路由、重试和部分失败。
~~~

一句话：

~~~text
复制让相同数据有更多副本；
分片让不同机器承担不同数据。
~~~

---

## 4. 复制和分片通常一起用

真实系统里经常是每个 shard 内部再做主从复制：

~~~mermaid
flowchart TD
  Router[Query Router] --> S0P[Shard 0 Primary]
  Router --> S1P[Shard 1 Primary]
  Router --> S2P[Shard 2 Primary]

  S0P --> S0R1[Shard 0 Replica]
  S0P --> S0R2[Shard 0 Replica]
  S1P --> S1R1[Shard 1 Replica]
  S1P --> S1R2[Shard 1 Replica]
  S2P --> S2R1[Shard 2 Replica]
  S2P --> S2R2[Shard 2 Replica]
~~~

这样同时获得：

~~~text
分片带来的容量和写入扩展；
复制带来的读扩展、备份和高可用。
~~~

总结：

~~~text
复制解决多读、多副本、高可用；
分片解决大数据量、高写入、单机容量瓶颈。
~~~

---

## 5. Feature Store 里的对应设计

Feature Store 可以理解为给模型服务提供在线特征的分布式状态系统。

在线预测时，模型不只需要当前请求字段，还需要历史上下文，例如：

| 场景 | 需要的特征 |
| --- | --- |
| 风控 | 用户过去 5 分钟交易次数、设备关联用户数、商户拒付率 |
| 推荐 | 用户最近点击、物品曝光点击统计、用户物品交互历史 |
| 广告 | 用户兴趣、广告主预算状态、实时点击率 |

这些特征不能在请求时临时扫描日志计算，通常要提前 materialize 到 Online Feature Store。

### 5.1 Feature Store 中的复制

类比数据库：

| 数据库 | Feature Store |
| --- | --- |
| Primary 接收写入 | feature primary 接收特征更新 |
| Replica 复制数据 | feature replica 复制特征状态 |
| 应用读副本 | model serving 读副本 |

写入路径：

~~~text
feature computation / materialization -> feature primary
~~~

读取路径：

~~~text
model serving / feature service -> feature replicas
~~~

这就是特征系统里的读写分离：特征计算负责写，模型服务负责读。

### 5.2 Feature Store 中的 stale feature

数据库里有 stale read，Feature Store 里有 stale feature。

~~~text
用户刚连续支付失败 5 次；
风险特征应该升高；
feature primary 已更新；
replica 尚未同步；
模型从 replica 读到旧特征；
风险被低估。
~~~

因此需要监控 freshness：

~~~text
特征最后更新时间；
特征落后多久；
是否超过模型可接受延迟；
哪些 replica 已经落后。
~~~

不同特征 freshness 要求不同：

| 特征类型 | 常见 freshness |
| --- | --- |
| 风控短窗口特征 | 秒级到几十秒 |
| 推荐行为特征 | 分钟级 |
| 商户长期统计 | 小时级 |
| 用户画像 | 天级 |

### 5.3 Feature Store 中的数据分区

Feature Store 的 sharding key 通常是 entity key：

~~~text
user_id
item_id
merchant_id
device_id
tenant_id
session_id
~~~

例如：

| Feature group | Sharding key |
| --- | --- |
| user_features | user_id |
| item_features | item_id |
| merchant_features | merchant_id |
| device_features | device_id |
| user_item_features | hash(user_id, item_id) |

一次风控请求可能需要：

~~~text
user_features:user_id=123
merchant_features:merchant_id=888
device_features:device_id=abc
user_merchant_features:user_id=123,merchant_id=888
~~~

Feature Service 必须能根据请求里的 key 直接定位 shard，不能每次都广播所有节点。

### 5.4 Feature Store 分片的代价

第一，跨 entity 特征不适合在线临时聚合。

例如：

~~~text
某城市过去 1 小时所有用户的平均交易金额；
某品类最近 30 分钟整体点击率；
全站最近 10 分钟支付失败率。
~~~

这些通常要通过 batch 或 streaming job 提前算好，再写回 Online Feature Store。

第二，hot key 会导致负载不均衡：

~~~text
超级热门商品；
大型商户；
超大企业客户；
高活跃用户。
~~~

hot key 的本质是：分片规则可能平均，但访问流量不平均。一个热门商品、一个大商户、一个超活跃用户可能把某个 shard 或 replica 打满。

处理方式可以先分成读路径和更新路径。

读路径上，常见办法是：

~~~text
增加 replica；
加缓存；
热点 key 特殊拆分；
热门 item feature 预加载到模型服务本地；
对多次读取做 batch 和合并。
~~~

更新路径上，面试里可以讲 push 模式和 pull 模式。

| 模式 | 怎么做 | 适合场景 | 代价 |
| --- | --- | --- | --- |
| Push / active update | 上游 feature computation 产出新值后，主动把 hot feature 推到 cache / serving replica / local cache | 热点少、更新频率可控、freshness 要求高 | 写放大；需要 fanout、版本号和失败重试 |
| Pull / lazy update | serving 侧读到缺失或过期 feature 时，再去 feature store / source of truth 拉取并刷新本地缓存 | 热点变化快、长尾 key 多、允许短暂 stale | 第一次 miss 慢；需要 TTL、singleflight、防止 cache stampede |

可以这样理解：

~~~mermaid
flowchart LR
  A[Feature Update] --> B{更新模式}
  B -->|Push / active| C[主动刷新 hot cache / replica]
  B -->|Pull / lazy| D[请求 miss / TTL 过期时再刷新]
  C --> E[低读延迟 + 更高写放大]
  D --> F[低写放大 + miss 时延迟更高]
~~~

在推荐系统里，热门 item feature 往往适合 push 到 serving local cache；用户长尾特征更适合 pull + TTL，因为主动推所有用户特征会造成大量无效写入。

第三，多类特征来自不同 shard，更新时间可能不同。模型拿到的通常是近似一致的特征快照，而不是严格同一时刻的全局状态。

### 5.5 更新日志和 checkpoint

MySQL 主从复制依赖 binlog position。Feature Store 也有类似的增量同步位置：

~~~text
streaming job 处理到 Kafka offset X；
batch materialization 处理到某个时间分区；
feature group v7 同步到 checkpoint Y。
~~~

更新链路可以横向记：

~~~mermaid
flowchart LR
  A[Raw Events] --> B[Feature Computation]
  B --> C[Feature Update Log / Checkpoint]
  C --> D[Online Feature Store Primary]
  D --> E[Online Feature Store Replicas]
  E --> F[Model Serving]
~~~

这个 update log / checkpoint 在概念上类似数据库里的 binlog position：不是每次全量同步，而是记录处理进度，持续增量更新。

---

## 6. 面试回答模板

如果题目问“数据库怎么扩展”，可以按这个顺序回答：

~~~text
1. 先判断瓶颈：读多、写多、数据大、还是可用性问题。
2. 读多：主从复制 + 读写分离，但要处理 replication lag。
3. 主库故障恢复：主备或主主，重点是 failover，不是写扩展。
4. 写多或数据大：按业务访问模式选择 sharding key。
5. 分片后要讨论跨分片查询、事务、全局 ID、迁移和热点。
6. 真实系统通常是 shard 内复制，复制和分片组合使用。
~~~

展开回答时可以按这几个层次讲：

| 层次 | 要说清楚什么 | 常见追问 |
| --- | --- | --- |
| 负载估算 | 入口 QPS、读写比、峰值系数、单请求 DB/Cache 次数 | 平均 QPS 和峰值 QPS 差多少？ |
| 复制 | 主从复制扩读，主备/主主解决 failover | replication lag 怎么处理？ |
| 分片 | sharding key 贴近主查询路径，避免广播查询 | hot shard、跨分片 join、全局 ID 怎么办？ |
| 一致性 | 哪些读可以 stale，哪些必须 read-your-writes | 下单、支付、库存这类路径能不能读从库？ |
| 运维 | rebalancing、backup、schema migration、observability | 扩 shard 时怎么迁移数据？ |

如果题目是 Feature Store / Online KV / Embedding Store：

~~~text
1. 把它看成服务模型的分布式状态系统。
2. entity key 决定分片。
3. replica 承担低延迟读和高可用。
4. freshness 等价于 ML 系统里的 replication lag。
5. hot key 要拆读路径和更新路径：cache/replica/push/pull。
6. update log / checkpoint 决定增量同步和故障恢复能力。
~~~

面试里不要只说“加缓存”。更好的表述是：

~~~text
我会先估读写压力和数据规模。
如果主要是读压力，用 replica 和 cache；
如果主要是写压力或容量压力，用 sharding；
如果是可用性问题，用 failover 和复制；
如果是 Feature Store，还要额外讨论 freshness、hot key、update log 和 checkpoint。
~~~

### 6.1 选择题：读扩展应该先想到什么？

~~~quiz
title: Database Scaling Check 1
question: 一个服务读请求很多、写请求相对少，最直接的数据库扩展手段是什么？
answer: B
A. 主主复制，因为两个主库可以把所有写入吞吐翻倍
B. 主从复制加读写分离，让多个 replica 承担读请求
C. 立刻按随机 key 分片，不考虑查询路径
D. 把所有请求都改成异步队列
explanation: 读多写少时，主从复制和读写分离通常是第一步；它扩展读能力，但不扩展单主写能力。
~~~

### 6.2 选择题：Feature Store 的 hot key 怎么分析？

~~~quiz
title: Feature Store Check 1
question: 一个热门 item 的 feature 被大量请求读取，同时该 feature 更新不算频繁。哪种说法更合理？
answer: C
A. 只能把这个 item 按 item_id 重新 hash 到另一个 shard
B. 必须每次请求都从 source of truth 读取，避免 stale
C. 可以把热门 item feature push 到 serving local cache 或更多 replica，降低读路径压力
D. hot key 只影响写入，不影响读取
explanation: 热门 item 的问题主要是读流量集中。若 freshness 要求较高且热点集合较小，push/active update 到本地 cache 或 replica 很合适。
~~~

### 6.3 选择题：update log / checkpoint 解决什么？

~~~quiz
title: Feature Store Check 2
question: Feature Store 里的 update log / checkpoint 最核心的作用是什么？
answer: B
A. 让每次同步都重新扫描全量历史数据
B. 记录处理进度，支持增量更新、故障恢复和判断副本落后程度
C. 替代 sharding key，让查询不用路由
D. 保证所有 feature 在严格同一时刻更新
explanation: update log / checkpoint 类似 binlog position 或 Kafka offset，核心是记录已经处理到哪里，从而持续增量同步和恢复。
~~~

## 7. 最短记忆版

~~~text
主从复制:
  读扩展 + 备份 + 容灾
  代价是 stale read

主主复制:
  高可用 + 快速切换
  不是写入翻倍

分片:
  容量扩展 + 写扩展
  代价是跨分片复杂

Feature Store:
  传统数据库扩展思想在 ML online state 上的复用
~~~
`;

const mlsysNoteDefinitions = [
  createTutorialDefinition('MLSYS1 · GPU 体系结构入门', 'MLSYS1.md', 'MLSYS1.en.md'),
  createTutorialDefinition('MLSYS2 · CUDA 编程模型与 GPU 组件', 'MLSYS2.md', 'MLSYS2.en.md'),
  createTutorialDefinition('MLSYS3 · Roofline Analysis', 'MLSYS3.md', 'MLSYS3.en.md'),
  createTutorialDefinition('MLSYS4 · CUDA Reduce Kernel 完全指南', 'MLSYS4.md', 'MLSYS4.en.md'),
  createTutorialDefinition('MLSYS5 · CUDA Parallel Primitives: Histogram & Scan', 'MLSYS5.md', 'MLSYS5.en.md'),
  createTutorialDefinition('MLSYS6 · Memory-Bound Kernel 优化', 'MLSYS6.md', 'MLSYS6.en.md'),
  createTutorialDefinition(
    'MLSYS7 · Compute-Bound Kernel (1)',
    'MLSYS7 Compute-Bound Kernel (1).md',
    'MLSYS7 Compute-Bound Kernel (1).en.md',
  ),
  createTutorialDefinition(
    'MLSYS8 · Compute-Bound Kernel (2)',
    'MLSYS8 Compute-Bound Kernel (2).md',
    'MLSYS8 Compute-Bound Kernel (2).en.md',
  ),
  createTutorialDefinition(
    'MLSYS9 · Compute-Bound Kernel (3)',
    'MLSYS9 Compute-bound kernel (3).md',
    'MLSYS9 Compute-bound kernel (3).en.md',
  ),
  createTutorialDefinition('MLSYS10 · 分布式训练并行范式', 'MLSYS10 parallelism.md', 'MLSYS10 parallelism.en.md'),
  createTutorialDefinition('MLSYS11 · nano-vllm 精读 (1)', 'MLSYS11 nano-vllm-1.md', 'MLSYS11 nano-vllm-1.en.md'),
  createTutorialDefinition('MLSYS12 · nano-vllm 精读 (2)', 'MLSYS12 nano-vllm-2.md', 'MLSYS12 nano-vllm-2.en.md'),
  createTutorialDefinition(
    'MLSYS13 · Low-bit Quantization 核心方法详解',
    'MLSYS13 Quantization and precision.md',
    'MLSYS13 Quantization and precision.en.md',
  ),
  createTutorialDefinition(
    'MLSYS14 · Post-Training Infra：从 TRL 到 Forge',
    'MLSYS14 Post-Training Infra.md',
    'MLSYS14 Post-Training Infra.en.md',
  ),
  createTutorialDefinition(
    'MLSYS15 · Efficient Attention：现代长上下文架构',
    'MLSYS15 Efficient Attention Modern Architectures.md',
    null,
  ),
  createTutorialDefinition(
    'MLSYS16 · KV Cache：内存管理与前缀复用',
    'MLSYS15 KV Cache Prefix Caching IndexShare.md',
    null,
  ),
  createTutorialDefinition(
    'MLSYS17 · Inference：并行解码与草稿验证',
    'MLSYS15 LLM Inference Speculative Decoding DFlash.md',
    null,
  ),
  createTutorialDefinition(
    'MLSYS18 · MoE Systems：路由、通信与 Kernel',
    'MLSYS16 Modern MoE SonicMoE.md',
    null,
  ),
];

const mlsysNotes = mlsysNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const leetcodeNoteDefinitions = [
  createTutorialDefinition(
    'Core Skills 1 · Design Dynamic Array',
    'CoreSkills01 Design Dynamic Array.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 2 · Design Singly Linked List',
    'CoreSkills02 Design Singly Linked List.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 3 · Design Double-ended Queue',
    'CoreSkills03 Design Double-ended Queue.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 4 · Design Binary Search Tree',
    'CoreSkills04 Design Binary Search Tree.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 5 · Design Hash Table',
    'CoreSkills05 Design Hash Table.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 6 · Design Heap',
    'CoreSkills06 Design Heap.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 7 · Design Graph',
    'CoreSkills07 Design Graph.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 8 · Design Disjoint Set',
    'CoreSkills08 Design Disjoint Set Union Find.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 9 · Design Segment Tree',
    'CoreSkills09 Design Segment Tree.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 10 · Insertion Sort',
    'CoreSkills10 Insertion Sort.md',
    null,
    { directory: 'Leetcode', category: 'Sorting', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 11 · Merge Sort',
    'CoreSkills11 Merge Sort.md',
    null,
    { directory: 'Leetcode', category: 'Sorting', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 12 · Quick Sort',
    'CoreSkills12 Quick Sort.md',
    null,
    { directory: 'Leetcode', category: 'Sorting', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 13 · Matrix DFS',
    'CoreSkills13 Matrix Depth First Search.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 14 · Matrix BFS',
    'CoreSkills14 Matrix Breadth First Search.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    "Core Skills 15 · Shortest Path: Dijkstra & Bellman-Ford",
    'CoreSkills15 Dijkstra Algorithm.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    "Core Skills 16 · Prim's Algorithm",
    'CoreSkills16 Prim Algorithm.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    "Core Skills 17 · Kruskal's Algorithm",
    'CoreSkills17 Kruskal Algorithm.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 18 · Topological Sort / Foreign Dictionary',
    'CoreSkills18 Topological Sort.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 19 · 0 / 1 Knapsack',
    'CoreSkills19 0-1 Knapsack.md',
    null,
    { directory: 'Leetcode', category: 'Dynamic Programming', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 20 · Unbounded Knapsack',
    'CoreSkills20 Unbounded Knapsack.md',
    null,
    { directory: 'Leetcode', category: 'Dynamic Programming', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 21 · Dynamic Programming',
    'CoreSkills21 Decode Ways Dynamic Programming.md',
    null,
    { directory: 'Leetcode', category: 'Dynamic Programming', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 22 · Rejection Sampling / Rand10',
    'CoreSkills22 Rejection Sampling Rand10.md',
    null,
    { directory: 'Leetcode', category: 'Math & Probability', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 23 · Greedy Algorithms',
    'CoreSkills23 Greedy Algorithms.md',
    null,
    { directory: 'Leetcode', category: 'Greedy', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 24 · Interval Problems',
    'CoreSkills24 Interval Problems.md',
    null,
    { directory: 'Leetcode', category: 'Intervals', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 25 · Math: Fast Power',
    'CoreSkills25 Math Binary Exponentiation.md',
    null,
    { directory: 'Leetcode', category: 'Math', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 26 · Bit Manipulation: XOR',
    'CoreSkills26 Bit Manipulation XOR.md',
    null,
    { directory: 'Leetcode', category: 'Math', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 27 · String Basics',
    'CoreSkills27 String Basics Encode Decode.md',
    null,
    { directory: 'Leetcode', category: 'Strings', difficulty: 'Medium' },
  ),
];

const leetcodeNotes = leetcodeNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const llmNoteDefinitions = [
  createTutorialDefinition(
    'LLM八股 1 · World Model',
    'LLMInterview01 World Model.md',
    null,
    { directory: 'LLMInterview', category: 'Models & Agents', difficulty: 'Placeholder' },
  ),
  createTutorialDefinition(
    '强化学习练习',
    'MLSYS15 RL Infra 自测 35 问.md',
    'MLSYS15 RL Infra 自测 35 问.en.md',
  ),
];

const llmNotes = llmNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const draftNoteDefinitions = isDraftMode
  ? [
      createDraftTutorialDefinition(
        'LLM八股 Overview · JD 高频主题拆解',
        'Draft LLM Interview Overview.md',
        llmDraftOverviewContent,
      ),
      createDraftTutorialDefinition(
        'Quant 草稿 · 概率基础公式与记忆框架',
        'Draft Probability Basics.md',
        probabilityDraftContent,
      ),
    ]
  : [];

const draftNotes = draftNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createInlineVariant(definition.zhFileName, definition.content),
    en: createInlineVariant('', undefined),
  },
}));

const quantNoteDefinitions = [
  createTutorialDefinition(
    'Quant 1 · 期望与计数：Indicator / Multinomial Moment',
    'Quant01 Expectation Counting Multinomial.md',
    null,
    { directory: 'quant', category: 'Expectation', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Quant 2 · Markov Chains：状态压缩与期望时间',
    'Quant02 Markov Chains Expected Time.md',
    null,
    { directory: 'quant', category: 'Markov', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Quant 3 · 连续分布：CDF、几何区域与变量变换',
    'Quant03 Continuous Distribution Geometry Transform.md',
    null,
    { directory: 'quant', category: 'Distribution', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Quant 4 · 协方差、相关系数与相关矩阵 PSD',
    'Quant04 Correlation Matrix PSD.md',
    null,
    { directory: 'quant', category: 'Probability', difficulty: 'Medium' },
  ),
];

const quantNotes = quantNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const mlCodingNoteDefinitions = [
  createTutorialDefinition(
    'ML Coding 1 · Unicode & Pretokenization',
    'MLCoding01 Unicode Pretokenization.md',
    null,
    { directory: 'MLCoding', category: 'Tokenizer', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'ML Coding 2 · BPE Training',
    'MLCoding02 BPE Training.md',
    null,
    { directory: 'MLCoding', category: 'Tokenizer', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 3 · Tokenizer Runtime',
    'MLCoding03 Tokenizer Runtime.md',
    null,
    { directory: 'MLCoding', category: 'Tokenizer', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 4 · Tensor Modules',
    'MLCoding04 Tensor Modules.md',
    null,
    { directory: 'MLCoding', category: 'Transformer LM', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'ML Coding 5 · Attention & Transformer',
    'MLCoding05 Attention Transformer.md',
    null,
    { directory: 'MLCoding', category: 'Transformer LM', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 6 · Training Components',
    'MLCoding06 Training Components.md',
    null,
    { directory: 'MLCoding', category: 'Training', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 7 · Training Loop & Generation',
    'MLCoding07 Training Loop Generation.md',
    null,
    { directory: 'MLCoding', category: 'Training', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 8 · Experiments & Ablations',
    'MLCoding08 Experiments Ablations.md',
    null,
    { directory: 'MLCoding', category: 'Experiments', difficulty: 'Hard' },
  ),
];

const mlCodingNotes = mlCodingNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const systemDesignNoteDefinitions = [
  createTutorialDefinition(
    'System Design 0 · Overview',
    'SystemDesign00 Overview.md',
    null,
    { directory: 'SystemDesign', category: 'Overview', difficulty: 'Intro' },
  ),
  createTutorialDefinition(
    'System Design 1 · 无状态设计范式',
    'SystemDesign01 Stateless Service.md',
    null,
    { directory: 'SystemDesign', category: 'Design Pattern', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'System Design 2 · 数据库基本范式',
    'SystemDesign02 Database Paradigms.md',
    null,
    { directory: 'SystemDesign', category: 'Database', difficulty: 'Medium' },
  ),
  {
    id: 'SystemDesign03 Database Scaling.md',
    title: 'System Design 3 · 数据库扩展三件套',
    fileName: 'SystemDesign03 Database Scaling.md',
    zhFileName: 'SystemDesign03 Database Scaling.md',
    enFileName: '',
    directory: 'SystemDesign',
    category: 'Design Pattern',
    difficulty: 'Medium',
    content: systemDesignDbScalingContent,
  },
];

const systemDesignNotes = systemDesignNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: definition.content
      ? createInlineVariant(definition.zhFileName, definition.content)
      : createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const businessAlgorithmNoteDefinitions = [
  createTutorialDefinition(
    '业务算法八股 · TODO',
    'Business Algorithm TODO.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'TODO', difficulty: 'TODO' },
  ),
];

const businessAlgorithmNotes = businessAlgorithmNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const mlInterviewNoteDefinitions = [
  createTutorialDefinition(
    'ML八股 · TODO',
    'ML Interview TODO.md',
    null,
    { directory: 'MLInterview', category: 'TODO', difficulty: 'TODO' },
  ),
];

const mlInterviewNotes = mlInterviewNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const noteSections = [
  {
    id: 'mlsys',
    title: 'MLSYS',
    description: 'GPU kernels, training systems, inference systems, and performance notes',
    notes: mlsysNotes,
  },
  {
    id: 'llm',
    title: 'LLM八股',
    description: 'LLM concepts, agent training, RL practice, and interview drills',
    notes: llmNotes,
  },
  {
    id: 'quant',
    title: 'Quant',
    description: 'Probability, Markov chains, expectation, and interview math drills',
    notes: quantNotes,
  },
  {
    id: 'mlcoding',
    title: 'ML Coding',
    description: 'From-scratch machine learning implementation exercises',
    notes: mlCodingNotes,
  },
  {
    id: 'system-design',
    title: 'System Design',
    description: 'Backend system design, LLM serving, agent workflows, and infra interview drills',
    notes: systemDesignNotes,
  },
  {
    id: 'business-algorithm',
    title: '业务算法八股',
    description: 'TODO: recommendation, search, ads, ranking, and experimentation basics',
    notes: businessAlgorithmNotes,
  },
  {
    id: 'ml-interview',
    title: 'ML八股',
    description: 'TODO: machine learning fundamentals and interview drills',
    notes: mlInterviewNotes,
  },
  {
    id: 'leetcode',
    title: 'LeetCode',
    description: 'Core data structure and algorithm interview drills',
    notes: leetcodeNotes,
  },
  ...(isDraftMode
    ? [
        {
          id: 'drafts',
          title: '草稿区',
          description: 'Local-only drafts. Visible in dev, hidden from production builds.',
          notes: draftNotes,
        },
      ]
    : []),
];

const tutorials = noteSections.flatMap((section) =>
  section.notes.map((note) => ({
    ...note,
    sectionId: section.id,
    sectionTitle: section.title,
  })),
);

const noteIdByAlias = buildNoteAliasMap(tutorials);
const mediaModules = import.meta.glob('../notes/**/assets/**/*.{png,jpg,jpeg,gif,webp,svg,avif,bmp}', {
  eager: true,
  import: 'default',
  query: '?url',
});
const mediaUrlByAlias = buildMediaAliasMap(mediaModules);
const languageOptions = [
  { id: 'zh', label: '中文' },
  { id: 'en', label: 'English' },
];

const homeStats = [
  { value: noteSections.length, label: 'Sections' },
  { value: tutorials.length, label: 'Notes' },
  { value: '2', label: 'Languages' },
];

const authorLinks = [
  {
    label: 'GitHub',
    href: 'https://github.com/CurryTang',
    value: 'github.com/CurryTang',
  },
  {
    label: 'LinkedIn',
    href: 'https://www.linkedin.com/in/zhikai-chen-435252129',
    value: 'Zhikai Chen',
  },
  {
    label: 'Email',
    href: 'mailto:chenzh85@msu.edu',
    value: 'chenzh85@msu.edu',
  },
];

function createTutorialDefinition(title, zhFileName, enFileName, options = {}) {
  const directory = options.directory ?? 'Mlsys';
  return {
    id: zhFileName,
    title,
    fileName: zhFileName,
    zhFileName,
    enFileName,
    directory,
    category: options.category ?? '',
    difficulty: options.difficulty ?? '',
  };
}

function createVariant(fileName, directory) {
  if (!fileName) {
    return {
      fileName: '',
      url: null,
    };
  }

  const modulePath = `../notes/${directory}/${fileName}`;
  const url = markdownModules[modulePath];
  return {
    fileName,
    url: typeof url === 'string' ? url : null,
  };
}

function createInlineVariant(fileName, content) {
  return {
    fileName,
    url: null,
    content,
  };
}

function createDraftTutorialDefinition(title, zhFileName, content) {
  return {
    id: zhFileName,
    title,
    fileName: zhFileName,
    zhFileName,
    enFileName: '',
    directory: 'Drafts',
    category: 'Draft',
    difficulty: 'Draft',
    content,
  };
}

function variantHasContent(variant) {
  return Boolean(variant?.url || typeof variant?.content === 'string');
}

function normalizePathToken(rawValue) {
  if (!rawValue) {
    return '';
  }

  let value = rawValue.trim().replace(/\\/g, '/');
  try {
    value = decodeURIComponent(value);
  } catch {
    // Ignore malformed URI fragments and keep the original token.
  }

  value = value.replace(/^\.\//, '');
  value = value.replace(/^\//, '');
  value = value.replace(/^notes\//i, '');

  return value.toLowerCase();
}

function buildNoteAliasMap(tutorialList) {
  const map = new Map();

  const addAlias = (alias, id) => {
    const normalized = normalizePathToken(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, id);
    }
  };

  tutorialList.forEach((tutorial) => {
    const fileNames = [tutorial.variants.zh.fileName, tutorial.variants.en.fileName].filter(Boolean);

    addAlias(tutorial.id, tutorial.id);
    addAlias(tutorial.fileName, tutorial.id);
    addAlias(`${tutorial.directory}/${tutorial.fileName}`, tutorial.id);
    addAlias(`notes/${tutorial.directory}/${tutorial.fileName}`, tutorial.id);

    fileNames.forEach((fileName) => {
      const withoutMd = fileName.replace(/\.md$/i, '');
      const withoutLang = withoutMd.replace(/\.en$/i, '');
      addAlias(fileName, tutorial.id);
      addAlias(`${tutorial.directory}/${fileName}`, tutorial.id);
      addAlias(`notes/${tutorial.directory}/${fileName}`, tutorial.id);
      addAlias(withoutMd, tutorial.id);
      addAlias(withoutLang, tutorial.id);
    });
  });

  return map;
}

function buildMediaAliasMap(modules) {
  const map = new Map();

  const addAlias = (alias, url) => {
    const normalized = normalizePathToken(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, url);
    }
  };

  Object.entries(modules).forEach(([modulePath, assetUrl]) => {
    if (typeof assetUrl !== 'string') {
      return;
    }

    const relativePath = modulePath.replace('../notes/', '');
    const fileName = relativePath.split('/').at(-1) ?? relativePath;
    addAlias(relativePath, assetUrl);
    addAlias(`notes/${relativePath}`, assetUrl);
    addAlias(fileName, assetUrl);
    addAlias(`assets/${fileName}`, assetUrl);
    addAlias(`./assets/${fileName}`, assetUrl);
  });

  return map;
}

function splitObsidianTarget(rawContent) {
  const [targetPart, ...aliasParts] = rawContent.split('|');
  const target = targetPart?.trim() ?? '';
  const aliasRaw = aliasParts.join('|').trim();

  if (!aliasRaw || /^\d+$/.test(aliasRaw)) {
    return { target, alias: '' };
  }

  return { target, alias: aliasRaw };
}

function prettyLabel(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const token = withoutAnchor.split('/').at(-1) ?? withoutAnchor;
  const anchor = rawTarget.includes('#') ? cleanHeadingText(rawTarget.split('#').slice(1).join('#')) : '';
  return token.replace(/\.en\.md$/i, '').replace(/\.md$/i, '').trim() || anchor || rawTarget.trim();
}

function resolveNoteId(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const normalized = normalizePathToken(withoutAnchor);

  if (!normalized) {
    return null;
  }

  const basename = normalized.split('/').at(-1) ?? normalized;
  const candidates = [
    normalized,
    normalized.endsWith('.md') ? normalized.slice(0, -3) : `${normalized}.md`,
    basename,
    basename.endsWith('.md') ? basename.slice(0, -3) : `${basename}.md`,
  ];

  for (const candidate of candidates) {
    const match = noteIdByAlias.get(candidate);
    if (match) {
      return match;
    }
  }

  return null;
}

function resolveObsidianLink(target, alias) {
  if (target.startsWith('#')) {
    const heading = cleanHeadingText(target.slice(1));
    if (!heading) {
      return alias || '';
    }

    return `[${alias || heading}](#${slugify(heading)})`;
  }

  const noteId = resolveNoteId(target);
  if (!noteId) {
    return null;
  }

  return `[${alias || prettyLabel(target)}](#${encodeURIComponent(noteId)})`;
}

function resolveMediaUrl(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const normalized = normalizePathToken(withoutAnchor);

  if (!normalized) {
    return null;
  }

  const basename = normalized.split('/').at(-1) ?? normalized;
  const candidates = [normalized, basename, `mlsys/assets/${basename}`, `assets/${basename}`];

  for (const candidate of candidates) {
    const match = mediaUrlByAlias.get(candidate);
    if (match) {
      return match;
    }
  }

  return null;
}

function slugify(text) {
  return String(text)
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^\w一-龥-]/g, '')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

function HeadingWithAnchor({ level, children }) {
  const Tag = `h${level}`;
  const text = extractPlainText(children);
  const id = slugify(text);
  const scrollToSection = (event) => {
    event.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <Tag id={id} className="heading-anchor-host">
      {children}
      <a href={`#${id}`} className="heading-anchor" aria-label="Link to section" onClick={scrollToSection}>¶</a>
    </Tag>
  );
}

function extractPlainText(value) {
  if (typeof value === 'string' || typeof value === 'number') {
    return String(value);
  }

  if (Array.isArray(value)) {
    return value.map(extractPlainText).join('');
  }

  if (value?.props?.children) {
    return extractPlainText(value.props.children);
  }

  return '';
}

function extractMarkdownHeadings(markdownText) {
  if (!markdownText) {
    return [];
  }

  return markdownText
    .replace(/```[\s\S]*?```/g, '')
    .split('\n')
    .map((line) => {
      const match = /^(#{1,3})\s+(.+?)\s*$/.exec(line);
      if (!match) {
        return null;
      }

      const text = cleanHeadingText(match[2]);
      if (!text) {
        return null;
      }

      return {
        id: slugify(text),
        level: match[1].length,
        text,
      };
    })
    .filter(Boolean);
}

function cleanHeadingText(text) {
  return text
    .replace(/<[^>]+>/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[`*_~]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeAnswerToken(rawValue) {
  const value = String(rawValue ?? '')
    .trim()
    .replace(/^[([{\s]+/g, '')
    .replace(/[\])}\s.。:：]+$/g, '');
  if (!value) {
    return '';
  }

  if (/^\d+$/.test(value)) {
    return String(Number(value) - 1);
  }

  return value.charAt(0).toUpperCase();
}

function parseQuizSource(rawSource) {
  const lines = String(rawSource ?? '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  const quiz = {
    title: 'Practice',
    question: '',
    answer: '',
    explanation: '',
    options: [],
  };

  lines.forEach((line) => {
    const fieldMatch = line.match(/^(title|question|answer|correct|explanation|解析|答案)\s*[:：]\s*(.+)$/i);
    if (fieldMatch) {
      const [, rawKey, rawValue] = fieldMatch;
      const key = rawKey.toLowerCase();
      if (key === 'correct' || key === 'answer' || rawKey === '答案') {
        quiz.answer = normalizeAnswerToken(rawValue);
      } else if (key === 'explanation' || rawKey === '解析') {
        quiz.explanation = rawValue.trim();
      } else {
        quiz[key] = rawValue.trim();
      }
      return;
    }

    const optionMatch = line.match(/^(?:[-*]\s*)?([A-Ha-h]|\d+)[).、:：]\s+(.+)$/);
    if (optionMatch) {
      const [, rawKey, text] = optionMatch;
      quiz.options.push({
        id: normalizeAnswerToken(rawKey),
        label: /^[A-Ha-h]$/.test(rawKey) ? rawKey.toUpperCase() : String(quiz.options.length + 1),
        text: text.trim(),
      });
      return;
    }

    if (!quiz.question) {
      quiz.question = line;
    }
  });

  if (!quiz.answer && quiz.options.some((option) => /^\*/.test(option.text))) {
    const correctOption = quiz.options.find((option) => /^\*/.test(option.text));
    quiz.answer = correctOption.id;
    quiz.options = quiz.options.map((option) => ({
      ...option,
      text: option.text.replace(/^\*\s*/, ''),
    }));
  }

  return quiz;
}

function QuizBlock({ source }) {
  const quiz = useMemo(() => parseQuizSource(source), [source]);
  const [collapsed, setCollapsed] = useState(false);
  const [selectedOption, setSelectedOption] = useState('');

  const isAnswered = Boolean(selectedOption);
  const isCorrect = selectedOption === quiz.answer;

  if (!quiz.question || quiz.options.length === 0 || !quiz.answer) {
    return (
      <pre>
        <code>{source}</code>
      </pre>
    );
  }

  return (
    <section className={`practice-card ${collapsed ? 'collapsed' : ''}`}>
      <button
        className="practice-card-toggle"
        type="button"
        onClick={() => setCollapsed((current) => !current)}
        aria-expanded={!collapsed}
        aria-label={`${collapsed ? 'Show' : 'Hide'} ${quiz.title}`}
      >
        <span>{quiz.title}</span>
        <span aria-hidden="true">{collapsed ? 'Show' : 'Hide'}</span>
      </button>

      {!collapsed && (
        <div className="practice-card-body">
          <p className="practice-question">{quiz.question}</p>
          <div className="practice-options" role="group" aria-label={quiz.question}>
            {quiz.options.map((option) => {
              const optionSelected = selectedOption === option.id;
              const optionCorrect = option.id === quiz.answer;
              const stateClass = isAnswered && optionSelected
                ? isCorrect
                  ? 'correct'
                  : 'incorrect'
                : isAnswered && optionCorrect
                  ? 'correct'
                  : '';

              return (
                <button
                  key={option.id}
                  className={`practice-option ${stateClass}`}
                  type="button"
                  onClick={() => setSelectedOption(option.id)}
                  aria-pressed={optionSelected}
                >
                  <span className="practice-option-key">{option.label}</span>
                  <span>{option.text}</span>
                </button>
              );
            })}
          </div>
          {isAnswered && (
            <p className={`practice-feedback ${isCorrect ? 'correct' : 'incorrect'}`} role="status">
              {isCorrect ? 'Correct.' : 'Not quite.'}
              {quiz.explanation ? ` ${quiz.explanation}` : ''}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function ForeignDictionaryTopoVisual() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackKey, setPlaybackKey] = useState(0);
  const comparisons = [
    ['hrn', 'hrf', 'n -> f', 'pair-1'],
    ['hrf', 'er', 'h -> e', 'pair-2'],
    ['er', 'enn', 'r -> n', 'pair-3'],
    ['enn', 'rfnn', 'e -> r', 'pair-4'],
  ];

  const nodes = [
    ['h', '0', 'node-h'],
    ['e', '1', 'node-e'],
    ['r', '1', 'node-r'],
    ['n', '1', 'node-n'],
    ['f', '1', 'node-f'],
  ];

  const playAnimation = () => {
    setIsPlaying(false);
    window.requestAnimationFrame(() => {
      setPlaybackKey((current) => current + 1);
      setIsPlaying(true);
    });
  };

  return (
    <section
      className={`topo-visual ${isPlaying ? 'is-playing' : ''}`}
      aria-label="Foreign Dictionary topological sorting visualization"
    >
      <div className="topo-visual-copy">
        <div>
          <p className="topo-kicker">Animated walkthrough</p>
          <h2>从相邻单词比较，到 Kahn 拓扑序</h2>
          <p>每一组相邻单词只看第一个不同字符；这个字符对就是一条有向边。边建完后，入度为 0 的字符先进入队列。</p>
        </div>
        <button className="topo-play-button" type="button" onClick={playAnimation}>
          {isPlaying ? 'Replay' : 'Play'}
        </button>
      </div>

      <div className="topo-stage" key={playbackKey}>
        <div className="topo-words" aria-label="Adjacent word comparisons">
          {comparisons.map(([first, second, edge, pairClass]) => (
            <span className={`topo-word-pair ${pairClass}`} key={edge}>
              <b>{first}</b>
              <b>{second}</b>
              <em>{edge}</em>
            </span>
          ))}
        </div>

        <div className="topo-graph-board" aria-label="Directed graph h to e to r to n to f">
          <div className="topo-chain">
            {nodes.map(([label, indegree, nodeClass], index) => (
              <Fragment key={label}>
                <span className={`topo-node ${nodeClass}`}>
                  {label}
                  <small>{indegree}</small>
                </span>
                {index < nodes.length - 1 && (
                  <span
                    className={`topo-edge-link ${['edge-he', 'edge-er', 'edge-rn', 'edge-nf'][index]}`}
                    aria-hidden="true"
                  />
                )}
              </Fragment>
            ))}
          </div>
        </div>

        <div className="topo-output" aria-label="Topological output order">
          {['h', 'e', 'r', 'n', 'f'].map((char, index) => (
            <span className={`out-${index + 1}`} key={char}>{char}</span>
          ))}
        </div>

        <ol className="topo-timeline">
          <li className="step-1">比较 <code>hrn</code> 和 <code>hrf</code>，第一个不同字符是 <code>n/f</code>，得到 <code>n -&gt; f</code>。</li>
          <li className="step-2">比较 <code>hrf</code> 和 <code>er</code>，第一个不同字符是 <code>h/e</code>，得到 <code>h -&gt; e</code>。</li>
          <li className="step-3">比较 <code>er</code> 和 <code>enn</code>，第一个不同字符是 <code>r/n</code>，得到 <code>r -&gt; n</code>。</li>
          <li className="step-4">比较 <code>enn</code> 和 <code>rfnn</code>，第一个不同字符是 <code>e/r</code>，得到 <code>e -&gt; r</code>。</li>
          <li className="step-5">Kahn 算法从入度为 0 的 <code>h</code> 开始，依次释放 <code>e</code>、<code>r</code>、<code>n</code>、<code>f</code>，输出 <code>hernf</code>。</li>
        </ol>
      </div>
    </section>
  );
}

function CheapestFlightsBellmanVisual() {
  const [activeRound, setActiveRound] = useState(0);
  const rounds = [
    {
      label: 'init',
      title: 'Round 0 / source only',
      prices: ['0', '∞', '∞', '∞'],
      activeEdges: [],
      note: 'Only src=0 is reachable before taking any flight.',
    },
    {
      label: '1 edge',
      title: 'Round 1 / at most 1 flight',
      prices: ['0', '100', '∞', '∞'],
      activeEdges: ['flight-0-1'],
      note: 'Use the previous prices array. Flight 0 -> 1 relaxes city 1 to 100.',
    },
    {
      label: '2 edges',
      title: 'Round 2 / at most 2 flights',
      prices: ['0', '100', '200', '700'],
      activeEdges: ['flight-1-2', 'flight-1-3'],
      note: 'Copy before relaxing, so 1 -> 2 and 1 -> 3 are allowed, but 2 -> 3 cannot chain inside this same round.',
    },
  ];
  const cities = [
    ['0', 'src', 'city-0'],
    ['1', '', 'city-1'],
    ['2', '', 'city-2'],
    ['3', 'dst', 'city-3'],
  ];
  const flights = [
    ['flight-0-1', '0 -> 1', '$100'],
    ['flight-1-2', '1 -> 2', '$100'],
    ['flight-2-0', '2 -> 0', '$100'],
    ['flight-1-3', '1 -> 3', '$600'],
    ['flight-2-3', '2 -> 3', '$200'],
  ];
  const round = rounds[activeRound];

  const nextRound = () => {
    setActiveRound((current) => Math.min(current + 1, rounds.length - 1));
  };

  const previousRound = () => {
    setActiveRound((current) => Math.max(current - 1, 0));
  };

  return (
    <section className="bf-visual" aria-label="Optimized Bellman-Ford visualization for Cheapest Flights Within K Stops">
      <div className="bf-header">
        <div>
          <p className="bf-kicker">Bellman-Ford with edge budget</p>
          <h2>Cheapest Flights Within K Stops</h2>
          <p>Example: <code>n=4</code>, <code>src=0</code>, <code>dst=3</code>, <code>k=1</code>. We may use at most <code>k + 1 = 2</code> flights.</p>
        </div>
        <div className="bf-controls" aria-label="Bellman-Ford round controls">
          <button type="button" onClick={previousRound} disabled={activeRound === 0} aria-label="Previous round">Prev</button>
          <span>{round.title}</span>
          <button type="button" onClick={nextRound} disabled={activeRound === rounds.length - 1} aria-label="Next round">Next</button>
        </div>
      </div>

      <div className="bf-stage">
        <div className="bf-round-tabs" role="tablist" aria-label="Bellman-Ford rounds">
          {rounds.map((candidate, index) => (
            <button
              key={candidate.label}
              className={index === activeRound ? 'active' : ''}
              type="button"
              onClick={() => setActiveRound(index)}
              role="tab"
              aria-selected={index === activeRound}
            >
              {candidate.label}
            </button>
          ))}
        </div>

        <div className="bf-layout">
          <div className="bf-graph" aria-label="Weighted directed flights">
            {cities.map(([id, tag, className]) => (
              <div className={`bf-city ${className}`} key={id}>
                <strong>{id}</strong>
                {tag && <small>{tag}</small>}
              </div>
            ))}
            {flights.map(([className, route, price]) => (
              <div
                className={`bf-flight ${className} ${round.activeEdges.includes(className) ? 'active' : ''}`}
                key={className}
              >
                <span>{route}</span>
                <em>{price}</em>
              </div>
            ))}
          </div>

          <div className="bf-prices" aria-label="Prices array">
            <div className="bf-prices-title">
              <span>prices</span>
              <small>from previous round only</small>
            </div>
            <div className="bf-price-grid">
              {round.prices.map((price, index) => (
                <div className={`bf-price ${price !== '∞' ? 'reachable' : ''}`} key={`${round.label}-${index}`}>
                  <span>city {index}</span>
                  <strong>{price}</strong>
                </div>
              ))}
            </div>
            <p>{round.note}</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function SegmentTreeLISVisual() {
  const values = [2, 3, 5, 7, 9, 10, 18, 101];
  const steps = [
    { input: 10, rank: 5, query: '0..4', beforeBest: 0, current: 1, lis: 1, after: [0, 0, 0, 0, 0, 1, 0, 0] },
    { input: 9, rank: 4, query: '0..3', beforeBest: 0, current: 1, lis: 1, after: [0, 0, 0, 0, 1, 1, 0, 0] },
    { input: 2, rank: 0, query: 'empty', beforeBest: 0, current: 1, lis: 1, after: [1, 0, 0, 0, 1, 1, 0, 0] },
    { input: 5, rank: 2, query: '0..1', beforeBest: 1, current: 2, lis: 2, after: [1, 0, 2, 0, 1, 1, 0, 0] },
    { input: 3, rank: 1, query: '0..0', beforeBest: 1, current: 2, lis: 2, after: [1, 2, 2, 0, 1, 1, 0, 0] },
    { input: 7, rank: 3, query: '0..2', beforeBest: 2, current: 3, lis: 3, after: [1, 2, 2, 3, 1, 1, 0, 0] },
    { input: 101, rank: 7, query: '0..6', beforeBest: 3, current: 4, lis: 4, after: [1, 2, 2, 3, 1, 1, 0, 4] },
    { input: 18, rank: 6, query: '0..5', beforeBest: 3, current: 4, lis: 4, after: [1, 2, 2, 3, 1, 1, 4, 4] },
  ];
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const step = steps[activeStep];

  useEffect(() => {
    if (!isPlaying) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setActiveStep((current) => (current + 1) % steps.length);
    }, 1900);

    return () => window.clearInterval(timer);
  }, [isPlaying, steps.length]);

  const previousStep = () => {
    setIsPlaying(false);
    setActiveStep((current) => (current === 0 ? steps.length - 1 : current - 1));
  };

  const nextStep = () => {
    setIsPlaying(false);
    setActiveStep((current) => (current + 1) % steps.length);
  };

  const treeLevels = buildSegmentTreeLevels(step.after);
  const smallerValues = values.slice(0, step.rank);
  const queryDescription = smallerValues.length > 0
    ? `rank 0..${step.rank - 1}，也就是值 ${smallerValues.join(', ')}`
    : '没有更小的压缩值';

  return (
    <section className="seg-visual" aria-label="Segment tree visualization for longest increasing subsequence">
      <div className="seg-header">
        <div>
          <p className="seg-kicker">Segment tree walkthrough</p>
          <h2>LIS: 先查更小值的最好结果，再更新当前值</h2>
          <p>
            例子输入 <code>[10, 9, 2, 5, 3, 7, 101, 18]</code>。坐标压缩后，
            每个叶子存 <strong>以这个值结尾的最长递增子序列长度</strong>。
          </p>
        </div>

        <div className="seg-controls" aria-label="Segment tree animation controls">
          <button type="button" onClick={previousStep} aria-label="Previous LIS step">Prev</button>
          <button type="button" onClick={() => setIsPlaying((current) => !current)} aria-label="Play segment tree animation">
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button type="button" onClick={nextStep} aria-label="Next LIS step">Next</button>
        </div>
      </div>

      <div className="seg-stage">
        <div className="seg-explainer">
          <div>
            <span>这一帧怎么看</span>
            <p>
              现在处理输入里的第 <strong>{activeStep + 1}</strong> 个数：<strong>{step.input}</strong>。
              因为 LIS 要严格递增，它只能接在比 {step.input} 更小的值后面。
            </p>
          </div>
          <ol>
            <li>蓝色叶子是本轮查询范围：<code>{queryDescription}</code>。</li>
            <li>线段树返回这些更小值里的最大 LIS 长度：<code>{step.beforeBest}</code>。</li>
            <li>当前数自己的长度就是 <code>{step.beforeBest} + 1 = {step.current}</code>，写到绿色叶子。</li>
          </ol>
        </div>

        <div className="seg-step-summary">
          <div>
            <span>current num</span>
            <strong>{step.input}</strong>
            <small>rank {step.rank}</small>
          </div>
          <div>
            <span>query</span>
            <strong>{step.query}</strong>
            <small>best smaller = {step.beforeBest}</small>
          </div>
          <div>
            <span>update</span>
            <strong>{step.current}</strong>
            <small>tree[{step.rank}] = {step.current}</small>
          </div>
          <div>
            <span>LIS so far</span>
            <strong>{step.lis}</strong>
            <small>global answer</small>
          </div>
        </div>

        <div className="seg-board-title">
          <span>压缩后的叶子</span>
          <small>叶子里的数字 = 以该值结尾的最佳 LIS 长度</small>
        </div>
        <div className="seg-rank-board" aria-label="Compressed value leaves">
          {values.map((value, index) => {
            const inQuery = step.rank > 0 && index < step.rank;
            const isUpdated = index === step.rank;

            return (
              <div
                className={`seg-leaf ${inQuery ? 'in-query' : ''} ${isUpdated ? 'updated' : ''}`}
                key={value}
              >
                <span>rank {index}</span>
                <strong>{value}</strong>
                <em>{step.after[index]}</em>
              </div>
            );
          })}
        </div>

        <div className="seg-board-title">
          <span>线段树缓存</span>
          <small>每个内部节点保存自己区间里的最大叶子值</small>
        </div>
        <div className="seg-tree-board" aria-label="Segment tree max values">
          {treeLevels.map((level, levelIndex) => (
            <div className="seg-tree-level" key={`level-${levelIndex}`}>
              {level.map((node) => {
                const intersectsQuery = step.rank > 0 && node.left < step.rank;
                const containsUpdate = node.left <= step.rank && step.rank <= node.right;

                return (
                  <div
                    className={`seg-tree-node ${intersectsQuery ? 'touches-query' : ''} ${containsUpdate ? 'update-path' : ''}`}
                    key={`${node.left}-${node.right}`}
                  >
                    <span>[{node.left}, {node.right}]</span>
                    <strong>{node.value}</strong>
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        <div className="seg-board-title">
          <span>输入顺序</span>
          <small>点击任意一步，观察一个数如何改变整棵树</small>
        </div>
        <ol className="seg-timeline">
          {steps.map((candidate, index) => (
            <li className={index === activeStep ? 'active' : ''} key={`${candidate.input}-${index}`}>
              <button type="button" onClick={() => { setIsPlaying(false); setActiveStep(index); }}>
                <span>{index + 1}</span>
                <strong>{candidate.input}</strong>
                <em>LIS {candidate.lis}</em>
              </button>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

function buildSegmentTreeLevels(leaves) {
  const levels = [
    leaves.map((value, index) => ({ left: index, right: index, value })),
  ];

  while (levels[0].length > 1) {
    const previous = levels[0];
    const next = [];
    for (let index = 0; index < previous.length; index += 2) {
      const left = previous[index];
      const right = previous[index + 1] ?? left;
      next.push({
        left: left.left,
        right: right.right,
        value: Math.max(left.value, right.value),
      });
    }
    levels.unshift(next);
  }

  return levels;
}

const INTERVAL_VISUALS = {
  'interval-merge-demo': {
    title: 'Merge Intervals',
    subtitle: 'Sort by start, then keep extending the current merged interval.',
    domain: [0, 18],
    intervals: [
      { id: 'a', label: '[1, 3]', start: 1, end: 3 },
      { id: 'b', label: '[2, 6]', start: 2, end: 6 },
      { id: 'c', label: '[8, 10]', start: 8, end: 10 },
      { id: 'd', label: '[15, 18]', start: 15, end: 18 },
    ],
    steps: [
      {
        title: 'Step 1 · sort by start',
        note: '先按 start 排序，保证只需要和当前 merged interval 比较。',
        active: ['a'],
        result: [{ id: 'm1', label: 'current [1, 3]', start: 1, end: 3 }],
        stats: [['current', '[1, 3]'], ['output', '[]']],
      },
      {
        title: 'Step 2 · overlap, extend end',
        note: '[2, 6] 的 start <= current end 3，所以合并成 [1, 6]。',
        active: ['a', 'b'],
        muted: ['a'],
        result: [{ id: 'm1', label: 'merged [1, 6]', start: 1, end: 6 }],
        stats: [['condition', '2 <= 3'], ['current', '[1, 6]']],
      },
      {
        title: 'Step 3 · gap, flush current',
        note: '[8, 10] 的 start > current end 6，说明前一段结束，输出 [1, 6]。',
        active: ['c'],
        result: [
          { id: 'm1', label: 'output [1, 6]', start: 1, end: 6 },
          { id: 'm2', label: 'current [8, 10]', start: 8, end: 10 },
        ],
        stats: [['condition', '8 > 6'], ['output', '[[1, 6]]']],
      },
      {
        title: 'Step 4 · finish',
        note: '最后没有重叠，依次输出剩余 current。',
        active: ['d'],
        result: [
          { id: 'm1', label: '[1, 6]', start: 1, end: 6 },
          { id: 'm2', label: '[8, 10]', start: 8, end: 10 },
          { id: 'm3', label: '[15, 18]', start: 15, end: 18 },
        ],
        stats: [['answer', '[[1,6],[8,10],[15,18]]']],
      },
    ],
  },
  'interval-insert-demo': {
    title: 'Insert Interval',
    subtitle: 'Three zones: before newInterval, overlapping block, after newInterval.',
    domain: [0, 17],
    intervals: [
      { id: 'a', label: '[1, 2]', start: 1, end: 2 },
      { id: 'b', label: '[3, 5]', start: 3, end: 5 },
      { id: 'c', label: '[6, 7]', start: 6, end: 7 },
      { id: 'd', label: '[8, 10]', start: 8, end: 10 },
      { id: 'e', label: '[12, 16]', start: 12, end: 16 },
      { id: 'new', label: 'new [4, 8]', start: 4, end: 8, kind: 'new' },
    ],
    steps: [
      {
        title: 'Step 1 · append before zone',
        note: '[1, 2] 完全在 newInterval 左边，直接进 output。',
        active: ['a', 'new'],
        result: [{ id: 'o1', label: 'output [1, 2]', start: 1, end: 2 }],
        stats: [['rule', 'end < new.start'], ['output', '[[1,2]]']],
      },
      {
        title: 'Step 2 · merge overlap block',
        note: '[3,5], [6,7], [8,10] 都和 [4,8] 有交集，持续扩张 newInterval。',
        active: ['b', 'c', 'd', 'new'],
        muted: ['b', 'c', 'd'],
        result: [
          { id: 'o1', label: 'output [1, 2]', start: 1, end: 2 },
          { id: 'm1', label: 'merged [3, 10]', start: 3, end: 10 },
        ],
        stats: [['merged start', 'min(4,3)=3'], ['merged end', 'max(8,10)=10']],
      },
      {
        title: 'Step 3 · append after zone',
        note: '[12,16] 完全在合并结果右边，先放入 [3,10]，再追加剩余区间。',
        active: ['e'],
        result: [
          { id: 'o1', label: '[1, 2]', start: 1, end: 2 },
          { id: 'm1', label: '[3, 10]', start: 3, end: 10 },
          { id: 'o2', label: '[12, 16]', start: 12, end: 16 },
        ],
        stats: [['answer', '[[1,2],[3,10],[12,16]]']],
      },
    ],
  },
  'interval-rooms-demo': {
    title: 'Meeting Rooms II',
    subtitle: 'Sweep starts and ends; the answer is max active meetings.',
    domain: [0, 30],
    intervals: [
      { id: 'a', label: '[0, 30]', start: 0, end: 30 },
      { id: 'b', label: '[5, 10]', start: 5, end: 10 },
      { id: 'c', label: '[15, 20]', start: 15, end: 20 },
    ],
    events: [
      { time: 0, label: '+1' },
      { time: 5, label: '+1' },
      { time: 10, label: '-1' },
      { time: 15, label: '+1' },
      { time: 20, label: '-1' },
      { time: 30, label: '-1' },
    ],
    steps: [
      {
        title: 't = 0 · first meeting starts',
        note: 'active 从 0 变成 1，需要 1 个房间。',
        active: ['a'],
        marker: 0,
        stats: [['active', '1'], ['max rooms', '1']],
      },
      {
        title: 't = 5 · overlap appears',
        note: '[5,10] 开始时 [0,30] 还没结束，active = 2。',
        active: ['a', 'b'],
        marker: 5,
        stats: [['active', '2'], ['max rooms', '2']],
      },
      {
        title: 't = 10 · one room freed',
        note: '[5,10] 结束，active 回到 1。',
        active: ['a'],
        marker: 10,
        stats: [['active', '1'], ['max rooms', '2']],
      },
      {
        title: 't = 15 · another overlap',
        note: '[15,20] 开始时 [0,30] 仍在进行，max rooms 仍然是 2。',
        active: ['a', 'c'],
        marker: 15,
        stats: [['active', '2'], ['answer', '2']],
      },
    ],
  },
  'interval-query-demo': {
    title: 'Minimum Interval to Include Each Query',
    subtitle: 'Sort queries; push candidate intervals into a min heap by length.',
    domain: [0, 7],
    intervals: [
      { id: 'a', label: '[1, 4] len 4', start: 1, end: 4 },
      { id: 'b', label: '[2, 4] len 3', start: 2, end: 4 },
      { id: 'c', label: '[3, 6] len 4', start: 3, end: 6 },
      { id: 'd', label: '[4, 4] len 1', start: 4, end: 4 },
    ],
    queries: [2, 3, 4, 5],
    steps: [
      {
        title: 'query = 2',
        note: '加入 start <= 2 的区间：[1,4], [2,4]。最短覆盖区间是 [2,4]，长度 3。',
        active: ['a', 'b'],
        marker: 2,
        result: [{ id: 'best', label: 'best [2,4]', start: 2, end: 4 }],
        stats: [['heap top', 'len 3 [2,4]'], ['ans[2]', '3']],
      },
      {
        title: 'query = 3',
        note: '加入 [3,6]。heap 顶仍是 [2,4]，它覆盖 3。',
        active: ['b', 'c'],
        marker: 3,
        result: [{ id: 'best', label: 'best [2,4]', start: 2, end: 4 }],
        stats: [['heap top', 'len 3 [2,4]'], ['ans[3]', '3']],
      },
      {
        title: 'query = 4',
        note: '加入 [4,4]，长度 1，立刻成为最优答案。',
        active: ['a', 'b', 'c', 'd'],
        marker: 4,
        result: [{ id: 'best', label: 'best [4,4]', start: 4, end: 4 }],
        stats: [['heap top', 'len 1 [4,4]'], ['ans[4]', '1']],
      },
      {
        title: 'query = 5',
        note: '弹掉 end < 5 的区间，剩下 [3,6] 覆盖 5，长度 4。',
        active: ['c'],
        marker: 5,
        result: [{ id: 'best', label: 'best [3,6]', start: 3, end: 6 }],
        stats: [['removed', 'end < 5'], ['ans[5]', '4']],
      },
    ],
  },
};

function IntervalPatternVisual({ kind }) {
  const visual = INTERVAL_VISUALS[kind];
  const [activeStep, setActiveStep] = useState(0);
  const step = visual.steps[activeStep];
  const active = new Set(step.active ?? []);
  const muted = new Set(step.muted ?? []);
  const domain = visual.domain;
  const ticks = buildIntervalTicks(domain);

  return (
    <section className="interval-visual">
      <header className="interval-visual-header">
        <div>
          <p className="eyebrow">Interval visual</p>
          <h2>{visual.title}</h2>
          <p>{visual.subtitle}</p>
        </div>
        <div className="interval-step-counter">
          {activeStep + 1}<span>/ {visual.steps.length}</span>
        </div>
      </header>

      <div className="interval-step-note">
        <strong>{step.title}</strong>
        <span>{step.note}</span>
      </div>

      <div className="interval-axis" aria-label={`${visual.title} timeline`}>
        <div className="interval-axis-line">
          {ticks.map((tick) => (
            <span
              className="interval-tick"
              key={tick}
              style={{ left: `${intervalPercent(tick, domain)}%` }}
            >
              {tick}
            </span>
          ))}
          {step.marker !== undefined && (
            <span
              className="interval-marker"
              style={{ left: `${intervalPercent(step.marker, domain)}%` }}
            >
              q={step.marker}
            </span>
          )}
        </div>

        <div className="interval-lanes">
          {visual.intervals.map((interval) => (
            <IntervalBar
              domain={domain}
              interval={interval}
              isActive={active.has(interval.id)}
              isMuted={muted.has(interval.id)}
              key={interval.id}
            />
          ))}
        </div>

        {visual.queries && (
          <div className="interval-query-row">
            {visual.queries.map((query) => (
              <span
                className={query === step.marker ? 'active' : ''}
                key={query}
                style={{ left: `${intervalPercent(query, domain)}%` }}
              >
                {query}
              </span>
            ))}
          </div>
        )}

        {step.result && (
          <div className="interval-result-lanes">
            {step.result.map((interval) => (
              <IntervalBar
                domain={domain}
                interval={{ ...interval, kind: 'result' }}
                isActive
                key={interval.id}
              />
            ))}
          </div>
        )}

        {visual.events && (
          <div className="interval-events">
            {visual.events.map((event) => (
              <span
                className={event.time === step.marker ? 'active' : ''}
                key={`${event.time}-${event.label}`}
                style={{ left: `${intervalPercent(event.time, domain)}%` }}
              >
                {event.label}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="interval-stat-grid">
        {(step.stats ?? []).map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>

      <ol className="interval-step-list">
        {visual.steps.map((candidate, index) => (
          <li className={index === activeStep ? 'active' : ''} key={candidate.title}>
            <button type="button" onClick={() => setActiveStep(index)}>
              <span>{index + 1}</span>
              {candidate.title.replace(/^Step \d+ · /, '')}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

const POW_STEPS = [
  {
    title: 'Init',
    power: 10,
    base: 2,
    res: 1,
    bit: 0,
    action: 'power = 10, binary = 1010. Lowest bit is 0, so this round does not contribute to the answer yet.',
    next: 'Square base to 4, shift power right to 5.',
  },
  {
    title: 'Read bit 1',
    power: 5,
    base: 4,
    res: 1,
    bit: 1,
    action: 'power is odd. The current base represents x^2, so multiply it into res.',
    next: 'res = 1 * 4 = 4. Square base to 16, shift power right to 2.',
  },
  {
    title: 'Read bit 0',
    power: 2,
    base: 16,
    res: 4,
    bit: 0,
    action: 'Lowest bit is 0. x^4 is not needed for n = 10, so res stays unchanged.',
    next: 'Square base to 256, shift power right to 1.',
  },
  {
    title: 'Read bit 1',
    power: 1,
    base: 256,
    res: 4,
    bit: 1,
    action: 'power is odd again. The current base represents x^8, and n = 10 includes this bit.',
    next: 'res = 4 * 256 = 1024. Shift power to 0, stop.',
  },
  {
    title: 'Done',
    power: 0,
    base: 65536,
    res: 1024,
    bit: null,
    action: 'All bits have been consumed from right to left: 10 = 8 + 2.',
    next: 'Return 1024 for pow(2, 10).',
  },
];

function BinaryPowVisual() {
  const [activeStep, setActiveStep] = useState(0);
  const step = POW_STEPS[activeStep];
  const binaryBits = ['1', '0', '1', '0'];
  const consumedFromRight = Math.min(activeStep, binaryBits.length);

  return (
    <section className="pow-visual" aria-label="Binary exponentiation walkthrough">
      <header className="pow-header">
        <div>
          <p className="eyebrow">Math visual</p>
          <h2>Binary Exponentiation: pow(2, 10)</h2>
          <p>每一轮只看 `power` 的最低位：bit 为 1 才把当前 `base` 乘进 `res`。</p>
        </div>
        <div className="pow-counter">{activeStep + 1}<span>/ {POW_STEPS.length}</span></div>
      </header>

      <div className="pow-board">
        <div className="pow-bits" aria-label="Binary bits of exponent 10">
          {binaryBits.map((bit, index) => {
            const fromRight = binaryBits.length - 1 - index;
            const isCurrent = fromRight === consumedFromRight && activeStep < binaryBits.length;
            const isConsumed = fromRight < consumedFromRight;
            return (
              <span
                className={`${isCurrent ? 'current' : ''} ${isConsumed ? 'consumed' : ''}`}
                key={`${bit}-${index}`}
              >
                {bit}
                <small>{[8, 4, 2, 1][index]}</small>
              </span>
            );
          })}
        </div>

        <div className="pow-state-grid">
          <div>
            <span>x / base</span>
            <strong>{step.base}</strong>
          </div>
          <div>
            <span>power</span>
            <strong>{step.power}</strong>
          </div>
          <div>
            <span>power & 1</span>
            <strong>{step.bit === null ? '-' : step.bit}</strong>
          </div>
          <div>
            <span>res</span>
            <strong>{step.res}</strong>
          </div>
        </div>

        <div className="pow-explain">
          <strong>{step.title}</strong>
          <p>{step.action}</p>
          <p>{step.next}</p>
        </div>
      </div>

      <ol className="pow-timeline">
        {POW_STEPS.map((candidate, index) => (
          <li className={index === activeStep ? 'active' : ''} key={candidate.title}>
            <button type="button" onClick={() => setActiveStep(index)}>
              <span>{index + 1}</span>
              {candidate.title}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

function IntervalBar({ domain, interval, isActive = false, isMuted = false }) {
  const left = intervalPercent(interval.start, domain);
  const right = intervalPercent(interval.end, domain);
  const width = Math.max(right - left, 1.4);

  return (
    <div
      className={`interval-bar ${interval.kind ?? ''} ${isActive ? 'active' : ''} ${isMuted ? 'muted' : ''}`}
      style={{ left: `${left}%`, width: `${width}%` }}
    >
      <span>{interval.label}</span>
    </div>
  );
}

function intervalPercent(value, [min, max]) {
  if (max === min) {
    return 0;
  }

  return ((value - min) / (max - min)) * 100;
}

function buildIntervalTicks([min, max]) {
  const width = max - min;
  const step = width <= 8 ? 1 : Math.ceil(width / 6);
  const ticks = [];
  for (let value = min; value <= max; value += step) {
    ticks.push(value);
  }
  if (ticks[ticks.length - 1] !== max) {
    ticks.push(max);
  }
  return ticks;
}

function MarkdownPre({ children, ...props }) {
  const child = Array.isArray(children) ? children[0] : children;
  const className = child?.props?.className ?? '';
  const match = /language-(quiz|mcq|mermaid|topo-demo|bellman-demo|segment-tree-demo|interval-merge-demo|interval-insert-demo|interval-rooms-demo|interval-query-demo|pow-demo)/.exec(className);

  if (match?.[1] === 'mermaid') {
    return <MermaidDiagram chart={extractPlainText(child.props.children).replace(/\n$/, '')} />;
  }

  if (match?.[1] === 'topo-demo') {
    return <ForeignDictionaryTopoVisual />;
  }

  if (match?.[1] === 'bellman-demo') {
    return <CheapestFlightsBellmanVisual />;
  }

  if (match?.[1] === 'segment-tree-demo') {
    return <SegmentTreeLISVisual />;
  }

  if (match?.[1]?.startsWith('interval-')) {
    return <IntervalPatternVisual kind={match[1]} />;
  }

  if (match?.[1] === 'pow-demo') {
    return <BinaryPowVisual />;
  }

  if (match) {
    return <QuizBlock source={extractPlainText(child.props.children).replace(/\n$/, '')} />;
  }

  return <CodeBlock className={className} source={extractPlainText(child?.props?.children)} {...props} />;
}

let mermaidLoader = null;

async function getMermaid() {
  if (!mermaidLoader) {
    mermaidLoader = import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'base',
        flowchart: {
          curve: 'basis',
          htmlLabels: true,
          nodeSpacing: 70,
          rankSpacing: 85,
          padding: 24,
        },
        themeVariables: {
          background: 'transparent',
          primaryColor: '#e9f3f5',
          primaryTextColor: '#102735',
          primaryBorderColor: '#2f7b94',
          lineColor: '#315568',
          secondaryColor: '#fff7e3',
          tertiaryColor: '#f4fbf7',
          fontFamily: '"IBM Plex Mono", "Courier New", monospace',
        },
      });
      return mermaid;
    });
  }

  return mermaidLoader;
}

function MermaidDiagram({ chart }) {
  const containerRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const reactId = useId();
  const diagramId = useMemo(() => `mermaid-${reactId.replace(/[^a-zA-Z0-9_-]/g, '')}`, [reactId]);
  const zoomPercent = Math.round(zoom * 100);
  const changeZoom = (delta) => {
    setZoom((current) => Math.min(2.25, Math.max(0.75, Number((current + delta).toFixed(2)))));
  };

  useEffect(() => {
    let cancelled = false;

    async function renderDiagram() {
      if (!containerRef.current) {
        return;
      }

      const mermaid = await getMermaid();
      containerRef.current.innerHTML = '';

      try {
        const { svg } = await mermaid.render(diagramId, chart);
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      } catch {
        if (!cancelled && containerRef.current) {
          containerRef.current.textContent = 'Diagram failed to render.';
        }
      }
    }

    renderDiagram();

    return () => {
      cancelled = true;
    };
  }, [chart, diagramId]);

  return (
    <figure className="mermaid-frame">
      <figcaption className="mermaid-toolbar">
        <span className="mermaid-label">Diagram</span>
        <span className="mermaid-zoom-controls" aria-label="Diagram zoom controls">
          <button
            type="button"
            className="diagram-zoom-button"
            onClick={() => changeZoom(-0.15)}
            disabled={zoom <= 0.75}
            aria-label="Zoom out diagram"
            title="Zoom out"
          >
            -
          </button>
          <button
            type="button"
            className="diagram-zoom-reset"
            onClick={() => setZoom(1)}
            aria-label="Reset diagram zoom"
            title="Reset zoom"
          >
            {zoomPercent}%
          </button>
          <button
            type="button"
            className="diagram-zoom-button"
            onClick={() => changeZoom(0.15)}
            disabled={zoom >= 2.25}
            aria-label="Zoom in diagram"
            title="Zoom in"
          >
            +
          </button>
        </span>
      </figcaption>
      <div className="mermaid-diagram">
        <div
          className="mermaid-canvas"
          ref={containerRef}
          role="img"
          aria-label="Mermaid diagram"
          style={{ '--diagram-zoom': zoom }}
        />
      </div>
    </figure>
  );
}

function CodeBlock({ className = '', source = '' }) {
  const [copied, setCopied] = useState(false);
  const language = className.match(/language-([\w-]+)/)?.[1] ?? 'text';
  const label = formatCodeLanguage(language);
  const code = source.replace(/\n$/, '');

  const copyCode = async () => {
    if (!navigator?.clipboard) {
      return;
    }

    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <figure className="code-frame">
      <figcaption className="code-frame-header">
        <span>{label}</span>
        <button type="button" onClick={copyCode}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </figcaption>
      <pre>
        <code className={className}>
          <HighlightedCode code={code} language={language} />
        </code>
      </pre>
    </figure>
  );
}

function HighlightedCode({ code, language }) {
  const tokens = tokenizeCode(code, language);
  return tokens.map((token, index) => (
    token.type === 'text'
      ? <Fragment key={index}>{token.value}</Fragment>
      : <span className={`code-token ${token.type}`} key={index}>{token.value}</span>
  ));
}

function formatCodeLanguage(language) {
  const labels = {
    js: 'JavaScript',
    jsx: 'React JSX',
    py: 'Python',
    python: 'Python',
    text: 'Text',
  };

  return labels[language] ?? language.toUpperCase();
}

function tokenizeCode(code, language) {
  if (!['py', 'python', 'js', 'jsx'].includes(language)) {
    return [{ type: 'text', value: code }];
  }

  const keywordPattern = language === 'python'
    || language === 'py'
    ? 'False|None|True|and|as|break|class|continue|def|elif|else|for|from|if|import|in|is|not|or|return|while|with'
    : 'const|let|var|function|return|if|else|for|while|import|from|export|class|new|true|false|null|undefined|await|async';
  const builtinPattern = language === 'python' || language === 'py'
    ? 'Counter|List|abs|bool|dict|enumerate|float|heapify|heappop|int|len|list|max|min|range|set|sorted|sum'
    : 'Array|Boolean|Map|Math|Number|Object|Promise|Set|String|console';
  const tokenPattern = new RegExp(
    `(#.*|//.*|"""[\\s\\S]*?"""|'''[\\s\\S]*?'''|"(?:\\\\.|[^"\\\\])*"|'(?:\\\\.|[^'\\\\])*'|\\b(?:${keywordPattern})\\b|\\b(?:${builtinPattern})\\b|\\b\\d+(?:\\.\\d+)?\\b)`,
    'g',
  );

  const tokens = [];
  let cursor = 0;
  for (const match of code.matchAll(tokenPattern)) {
    if (match.index > cursor) {
      tokens.push({ type: 'text', value: code.slice(cursor, match.index) });
    }

    const value = match[0];
    let type = 'number';
    if (value.startsWith('#') || value.startsWith('//')) {
      type = 'comment';
    } else if (value.startsWith('"') || value.startsWith("'")) {
      type = 'string';
    } else if (new RegExp(`^(?:${keywordPattern})$`).test(value)) {
      type = 'keyword';
    } else if (new RegExp(`^(?:${builtinPattern})$`).test(value)) {
      type = 'builtin';
    }

    tokens.push({ type, value });
    cursor = match.index + value.length;
  }

  if (cursor < code.length) {
    tokens.push({ type: 'text', value: code.slice(cursor) });
  }

  return tokens;
}

function parseHashRoute(rawHash) {
  const hashValue = decodeURIComponent(String(rawHash ?? '').replace(/^#/, '')).replace(/^\/+/, '');

  if (!hashValue || hashValue === 'home') {
    return { view: 'home', noteId: null, sectionId: null };
  }

  const noteMatch = tutorials.find((tutorial) => tutorial.id === hashValue);
  if (noteMatch) {
    return { view: 'reader', noteId: noteMatch.id, sectionId: noteMatch.sectionId };
  }

  const sectionMatch = noteSections.find((section) => section.id === hashValue);
  if (sectionMatch) {
    return { view: 'reader', noteId: sectionMatch.notes[0]?.id ?? null, sectionId: sectionMatch.id };
  }

  return null;
}

function normalizeObsidianMarkdown(markdownText) {
  if (!markdownText) {
    return '';
  }

  let normalized = markdownText;

  normalized = normalized.replace(/%%[\s\S]*?%%/g, '');

  normalized = normalized.replace(/^>\s*\[!([^\]\n+-]+)(?:[+-])?\](.*)$/gim, (_, type, rawTitle) => {
    const label = type.trim();
    const title = rawTitle.trim().replace(/^[-:\s]+/, '');
    const heading = title || (label.charAt(0).toUpperCase() + label.slice(1).toLowerCase());
    return `> **${heading}:**`;
  });

  normalized = normalized.replace(/!\[\[([^\]\n]+)\]\]/g, (_, body) => {
    const { target, alias } = splitObsidianTarget(body);
    if (!target) {
      return '';
    }

    const mediaUrl = resolveMediaUrl(target);
    if (mediaUrl) {
      return `![${alias || prettyLabel(target)}](${mediaUrl})`;
    }

    const noteId = resolveNoteId(target);
    if (noteId) {
      return `[Embedded note: ${alias || prettyLabel(target)}](#${encodeURIComponent(noteId)})`;
    }

    return `*Embedded asset not found: ${alias || prettyLabel(target)}*`;
  });

  normalized = normalized.replace(/\[\[([^\]\n]+)\]\]/g, (_, body) => {
    const { target, alias } = splitObsidianTarget(body);
    if (!target) {
      return '';
    }

    const resolvedLink = resolveObsidianLink(target, alias);
    if (resolvedLink) {
      return resolvedLink;
    }

    if (/^https?:\/\//i.test(target)) {
      return `[${alias || target}](${target})`;
    }

    return alias || prettyLabel(target);
  });

  normalized = normalized.replace(/==([^=\n][^=\n]*?)==/g, '<mark>$1</mark>');

  return normalized;
}

function App() {
  const initialRoute = parseHashRoute(window.location.hash) ?? { view: 'home', noteId: null, sectionId: null };
  const initialId = initialRoute.noteId ?? tutorials[0]?.id ?? '';

  const [currentView, setCurrentView] = useState(initialRoute.view);
  const [selectedTutorialId, setSelectedTutorialId] = useState(initialId);
  const [language, setLanguage] = useState('zh');
  const [query, setQuery] = useState('');
  const [contentByKey, setContentByKey] = useState({});
  const [errorByKey, setErrorByKey] = useState({});
  const inFlightRef = useRef(new Set());
  const selectedSection = noteSections.find((section) =>
    section.notes.some((note) => note.id === selectedTutorialId),
  ) ?? noteSections[0];
  const activeSectionNotes = tutorials.filter((tutorial) => tutorial.sectionId === selectedSection?.id);

  const filteredTutorials = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return activeSectionNotes;
    }

    return activeSectionNotes.filter((tutorial) =>
      [tutorial.title, tutorial.fileName].some((field) => field.toLowerCase().includes(normalizedQuery)),
    );
  }, [activeSectionNotes, query]);

  const selectedTutorial =
    tutorials.find((tutorial) => tutorial.id === selectedTutorialId) ?? filteredTutorials[0] ?? tutorials[0] ?? null;

  const activeLanguage =
    variantHasContent(selectedTutorial?.variants[language]) ? language : 'zh';
  const selectedVariant = selectedTutorial?.variants[activeLanguage] ?? null;
  const contentKey =
    selectedTutorial && selectedVariant ? `${selectedTutorial.id}:${activeLanguage}` : '';

  useEffect(() => {
    if (!selectedVariant?.url || selectedVariant?.content !== undefined || !contentKey) {
      return;
    }

    const isLoaded = Object.prototype.hasOwnProperty.call(contentByKey, contentKey);
    if (isLoaded || errorByKey[contentKey] || inFlightRef.current.has(contentKey)) {
      return;
    }

    inFlightRef.current.add(contentKey);

    fetch(selectedVariant.url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load markdown (${response.status})`);
        }
        return response.text();
      })
      .then((content) => {
        setContentByKey((prev) => ({
          ...prev,
          [contentKey]: content,
        }));
      })
      .catch((error) => {
        setErrorByKey((prev) => ({
          ...prev,
          [contentKey]: error.message,
        }));
      })
      .finally(() => {
        inFlightRef.current.delete(contentKey);
      });
  }, [contentByKey, contentKey, errorByKey, selectedVariant]);

  const selectedInlineContent = selectedVariant?.content;
  const hasSelectedContent = selectedInlineContent !== undefined || (contentKey
    ? Object.prototype.hasOwnProperty.call(contentByKey, contentKey)
    : false);
  const selectedContent = selectedInlineContent !== undefined
    ? selectedInlineContent
    : hasSelectedContent
      ? contentByKey[contentKey]
      : '';
  const selectedError = contentKey ? errorByKey[contentKey] : '';
  const selectedIsLoading = Boolean(
    selectedTutorial && selectedVariant?.url && selectedInlineContent === undefined && !hasSelectedContent && !selectedError,
  );

  const normalizedSelectedContent = useMemo(
    () => normalizeObsidianMarkdown(selectedContent),
    [selectedContent],
  );
  const sectionHeadings = useMemo(
    () => extractMarkdownHeadings(normalizedSelectedContent).filter((heading) => heading.level <= 3),
    [normalizedSelectedContent],
  );

  const scrollToHeading = (headingId) => {
    const target = document.getElementById(headingId);
    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const navigateHome = () => {
    setCurrentView('home');
    setQuery('');
    if (window.location.hash) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    }
  };

  const navigateToSection = (sectionId) => {
    const section = noteSections.find((candidate) => candidate.id === sectionId);
    const nextId = section?.notes[0]?.id ?? tutorials[0]?.id ?? '';
    setCurrentView('reader');
    setQuery('');
    setSelectedTutorialId(nextId);
  };

  const navigateToTutorial = (tutorialId) => {
    setCurrentView('reader');
    setSelectedTutorialId(tutorialId);
  };

  const navigateToAbout = () => {
    setCurrentView('home');
    setQuery('');
    if (window.location.hash) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    }
    window.requestAnimationFrame(() => {
      document.getElementById('about')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  };

  useEffect(() => {
    if (currentView !== 'reader' || !selectedTutorial) {
      return;
    }

    const encoded = `#${encodeURIComponent(selectedTutorial.id)}`;
    if (window.location.hash !== encoded) {
      window.history.replaceState(null, '', encoded);
    }
  }, [currentView, selectedTutorial]);

  useEffect(() => {
    const handleHashChange = () => {
      const route = parseHashRoute(window.location.hash);
      if (!route) {
        return;
      }

      setCurrentView(route.view);
      if (route.noteId) {
        setSelectedTutorialId(route.noteId);
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  return (
    <div className={`site-shell ${currentView === 'home' ? 'home-view' : 'reader-view'}`}>
      <header className="top-nav">
        <button className="brand-lockup" type="button" onClick={navigateHome}>
          <span className="brand-mark">IN</span>
          <span>
            <strong>Interview Notes</strong>
            <small>systems · infra · practice</small>
          </span>
        </button>

        <nav className="top-nav-links" aria-label="Main navigation">
          <button
            className={`top-nav-link ${currentView === 'home' ? 'active' : ''}`}
            type="button"
            onClick={navigateHome}
          >
            Home
          </button>
          {noteSections.map((section) => (
            <button
              key={section.id}
              className={`top-nav-link ${currentView === 'reader' && selectedSection?.id === section.id ? 'active' : ''}`}
              type="button"
              onClick={() => navigateToSection(section.id)}
            >
              {section.title}
            </button>
          ))}
          <button
            className="top-nav-link"
            type="button"
            onClick={navigateToAbout}
          >
            About
          </button>
        </nav>
      </header>

      {currentView === 'home' ? (
        <main className="home-page">
          <section className="home-hero">
            <div className="home-hero-copy">
              <p className="eyebrow">Interview Notes</p>
              <h1>ML / LLM 技术复习笔记</h1>
              <p>
                面向 ML / LLM 领域的技术内容复习笔记，整理 MLSYS、CUDA kernel、分布式训练、
                LLM inference、ML coding、quant 和算法练习里值得反复看的知识点。
              </p>
              <div className="home-actions">
                <button className="primary-action" type="button" onClick={() => navigateToSection('mlsys')}>
                  Start MLSYS
                </button>
                <button className="secondary-action" type="button" onClick={() => navigateToTutorial('MLSYS1.md')}>
                  Try Practice
                </button>
              </div>
            </div>

            <div className="home-hero-panel" aria-label="Site summary">
              {homeStats.map((stat) => (
                <div className="home-stat" key={stat.label}>
                  <strong>{stat.value}</strong>
                  <span>{stat.label}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="home-sections" aria-label="Interview note sections">
            <div className="section-heading">
              <p className="eyebrow">Sections</p>
              <h2>当前板块</h2>
            </div>

            <div className="section-card-grid">
              {noteSections.map((section) => (
                <button
                  key={section.id}
                  className="home-section-card"
                  type="button"
                  onClick={() => navigateToSection(section.id)}
                >
                  <span className="section-card-kicker">{section.notes.length} notes</span>
                  <strong>{section.title}</strong>
                  <span>{section.description}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="home-about" id="about" aria-labelledby="about-title">
            <div className="section-heading">
              <p className="eyebrow">About</p>
              <h2 id="about-title">关于作者</h2>
            </div>

            <div className="about-panel">
              <div className="about-copy">
                <p>
                  这个网站由 <strong>Zhikai Chen</strong> 维护，用来整理 MLSYS、LLM infra、ML coding、
                  quant 和 LeetCode 面试复习笔记。
                </p>
                <p>
                  作者目前正在找工作，并有 agent memory、agentic reinforcement learning、
                  predictive foundation model 和 agentic security 方面的研究经验。如果您有合适的机会，
                  欢迎通过 GitHub、LinkedIn 或 Email 联系。
                </p>
              </div>

              <div className="about-links" aria-label="Author contact links">
                {authorLinks.map((link) => (
                  <a
                    className="about-link"
                    href={link.href}
                    key={link.label}
                    target={link.href.startsWith('http') ? '_blank' : undefined}
                    rel={link.href.startsWith('http') ? 'noreferrer' : undefined}
                  >
                    <span>{link.label}</span>
                    <strong>{link.value}</strong>
                  </a>
                ))}
              </div>
            </div>
          </section>
        </main>
      ) : (
        <div className="app-shell">
      <aside className="notes-panel">
        <header className="panel-header">
          <p className="eyebrow">Current Section</p>
          <h1>{selectedSection?.title ?? 'Notes'}</h1>
          <p className="panel-meta">{activeSectionNotes.length} notes in this section</p>
          {selectedSection?.description && (
            <p className="panel-description">{selectedSection.description}</p>
          )}
        </header>

        <label className="search">
          <span>Search {selectedSection?.title ?? 'Notes'}</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Type note title or filename"
          />
        </label>

        <div className="note-list">
          {filteredTutorials.map((tutorial) => (
            <button
              key={tutorial.id}
              className={`note-button ${selectedTutorial?.id === tutorial.id ? 'active' : ''}`}
              onClick={() => navigateToTutorial(tutorial.id)}
              type="button"
            >
              <span className="note-title">{tutorial.title}</span>
              <span className="note-subtitle">{tutorial.fileName}</span>
            </button>
          ))}
          {filteredTutorials.length === 0 && (
            <p className="list-empty">No notes matched your search.</p>
          )}
        </div>
      </aside>

      <main className="reader-panel">
        {selectedTutorial ? (
          <>
            <header className="reader-header">
              <div className="reader-header-top">
                <div>
                  <p className="reader-label">{selectedTutorial.sectionTitle} / Interview Notes</p>
                  <h2>{selectedTutorial.title}</h2>
                  <p>{selectedVariant?.fileName ?? selectedTutorial.fileName}</p>
                </div>

                <div className="reader-controls">
                  <div className="language-toggle" aria-label="Language selector" role="group">
                    {languageOptions.map((option) => (
                      <button
                        key={option.id}
                        className={`language-button ${activeLanguage === option.id ? 'active' : ''}`}
                        onClick={() => setLanguage(option.id)}
                        type="button"
                        aria-pressed={activeLanguage === option.id}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </header>

            <div className="reader-content-grid">
              <article className="markdown-body">
                {selectedError && <p className="empty-note">Load failed: {selectedError}</p>}
                {selectedIsLoading && !selectedError && <p className="empty-note">Loading markdown...</p>}
                {!selectedIsLoading && !selectedError && normalizedSelectedContent?.trim() && (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeRaw, rehypeKatex]}
                    components={{
                      a: ({ href, children, ...props }) => {
                        const external = href?.startsWith('http');
                        return (
                          <a
                            href={href}
                            target={external ? '_blank' : undefined}
                            rel={external ? 'noreferrer' : undefined}
                            {...props}
                          >
                            {children}
                          </a>
                        );
                      },
                      h1: ({ children }) => <HeadingWithAnchor level={1}>{children}</HeadingWithAnchor>,
                      h2: ({ children }) => <HeadingWithAnchor level={2}>{children}</HeadingWithAnchor>,
                      h3: ({ children }) => <HeadingWithAnchor level={3}>{children}</HeadingWithAnchor>,
                      h4: ({ children }) => <HeadingWithAnchor level={4}>{children}</HeadingWithAnchor>,
                      pre: MarkdownPre,
                      code: ({ className, children, ...props }) => (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      ),
                    }}
                  >
                    {normalizedSelectedContent}
                  </ReactMarkdown>
                )}
                {!selectedIsLoading && !selectedError && selectedContent !== undefined && !selectedContent.trim() && (
                  <p className="empty-note">This file is empty and ready for future notes.</p>
                )}
              </article>

              {sectionHeadings.length > 0 && (
                <aside className="section-toc" aria-label="Section navigation">
                  <div className="section-toc-inner">
                    <div className="section-toc-heading">
                      <p className="eyebrow">Sections</p>
                      <span>{sectionHeadings.length}</span>
                    </div>
                    <nav>
                      {sectionHeadings.map((heading, index) => (
                        <a
                          className={`toc-link level-${heading.level}`}
                          href={`#${heading.id}`}
                          key={`${heading.id}-${index}`}
                          onClick={(event) => {
                            event.preventDefault();
                            scrollToHeading(heading.id);
                          }}
                        >
                          {heading.text}
                        </a>
                      ))}
                    </nav>
                  </div>
                </aside>
              )}
            </div>
          </>
        ) : (
          <section className="reader-empty">
            <h2>No published Markdown files found</h2>
            <p>Add ready notes to an interview section and refresh.</p>
          </section>
        )}
      </main>
    </div>
      )}
    </div>
  );
}

export default App;
