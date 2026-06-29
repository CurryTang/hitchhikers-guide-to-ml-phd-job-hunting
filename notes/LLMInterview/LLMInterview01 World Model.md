# LLM八股 1 · World Model

> Placeholder：这一页先放基本概念，后续可以继续扩展到 JEPA、predictive coding、Dreamer、video world model、agent memory、model-based RL 和 foundation model planning。

## 一、什么是 World Model

World model 可以理解成模型内部对“环境如何变化”的可预测表示。它不一定是一个显式的物理模拟器，也不一定要能生成高清视频；更一般地说，它回答的是：

```text
如果当前状态是 s，我采取动作 a，
下一步可能看到什么？
哪些变量会改变？
哪些约束不会改变？
什么结果更可能发生？
```

在强化学习里，world model 通常写成环境动态：

```text
p(s_{t+1}, r_t | s_t, a_t)
```

也就是给定当前状态和动作，预测下一状态和奖励。在更宽泛的 LLM / agent 语境里，world model 可以不是一个单独模块，而是模型参数、上下文记忆、工具反馈和 latent representation 共同形成的预测能力。

## 二、为什么需要 World Model

只会做 pattern matching 的模型可以在熟悉分布上表现很好，但遇到需要规划、反事实推理、长程依赖或物理/社会约束的问题时，单步模仿往往不够。

World model 的价值在于让模型可以在行动前做内部模拟：

```text
当前状态
  -> 想象几个可能动作
  -> 预测每个动作的后果
  -> 比较风险和收益
  -> 选择更好的动作
```

这对应三个能力：

- Prediction：预测未来 observation、state、reward 或 outcome。
- Planning：在真实执行前搜索或评估多步动作序列。
- Generalization：利用环境结构，而不是只背训练样本里的表面模式。

## 三、World Model 不等于什么

World model 不是简单的“知识库”。知识库保存事实，例如“巴黎是法国首都”；world model 更关心状态转移，例如“如果我把杯子推到桌边，它可能掉下去”。

World model 也不等于普通生成模型。生成模型可以生成看起来合理的未来片段，但 world model 更强调预测对行动有用的变量：哪些状态会变、哪些约束必须保持、动作会带来什么后果。

一个简单区分：

| 概念 | 关注点 | 例子 |
| --- | --- | --- |
| Knowledge | 静态事实 | 某个 API 的参数是什么 |
| Memory | 过去经历 | 上一次用户偏好什么 |
| World Model | 状态如何演化 | 如果调用这个 API，会改变哪些系统状态 |
| Policy | 该做什么动作 | 下一步应该搜索、写代码还是运行测试 |

## 四、在 LLM / Agent 里的 World Model

LLM agent 的环境不是只有物理世界，也包括网页、代码库、shell、数据库、工具调用和多轮对话。这里的 world model 经常体现在：

- 代码修改前能预测测试为什么失败。
- 调用工具前能预测输出大概会提供什么信息。
- 多轮任务中能理解哪些状态已经被改变，哪些只是文本记录。
- 面对不确定信息时，知道应该先观察环境而不是直接下结论。

一个 agent 的 world model 越弱，它越容易出现这些问题：

- 只根据 prompt 猜答案，不检查环境。
- 不理解工具调用会改变外部状态。
- 修改代码后不运行相关测试。
- 把 observation 当成自己已经执行过的 action。
- 长任务里丢失中间状态，反复做同一件事。

## 五、为什么真实环境存在，还要 Language World Model

Language world model 不是简单为了省掉真实环境成本。更重要的是，它给 agent 训练提供了另一条 scaling axis：真实环境继续存在，但模型可以额外学习“环境会如何响应”的语言化模拟能力。

第一种用法是 decoupling：把 LWM 当成 turn-level 环境模拟器。真实环境经常有硬限制：需要 sandbox、GUI VM、真实搜索引擎、真实 app、系统权限；有些环境不可逆，有商业限制，或者根本没有公开实现。LWM 可以把这些环境的 observation / transition 变成可控模拟，让 agent 在更大规模上训练。

这里的重点不是“便宜替代品”，而是 controllability。真实 search / tool 环境通常返回正常结果，但 LWM 可以刻意模拟：

- partial search result：迫使 agent 多轮追问和交叉验证。
- tool failure：命令超时、API rate limit、权限不足。
- environment perturbation：网页结构变化、磁盘空间不足、文件被并发修改。
- rare edge case：真实系统里很少遇到，但一旦出现就会破坏任务的状态。

这类 targeted perturbation 可以暴露 agent 在真实 rollout 中不容易覆盖到的弱点。换句话说，LWM 不是替代真实环境，而是在真实环境之外补一个可控制、可放大、可定向采样的训练分布。

第二种用法是 unifying：把 world modeling 能力直接并入 agent foundation model。agent 不只是学“现在该做什么”，还要学一种面向未来的 thinking pattern：

```text
如果我执行这个 action，
环境可能返回什么 observation？
哪些状态会改变？
哪些约束仍然成立？
下一步我应该准备验证什么？
```

这和 reflection 是平行的 meta-level thinking。reflection 是回看过去：我刚才哪里错了？simulation 是预测未来：我接下来这样做会发生什么？一个强 agent 往往需要两者同时存在。

## 六、和 Model-Based RL 的关系

Model-based RL 会显式学习一个 dynamics model，然后用它做 planning 或 imagination rollout：

```text
real experience
  -> train dynamics model
  -> imagined rollout
  -> improve policy
```

Model-free RL 则更直接地学习 policy 或 value，不一定显式预测环境。两者的 trade-off 是：

| 方法 | 优点 | 代价 |
| --- | --- | --- |
| Model-free | 实现直接，不依赖模型预测准确性 | 样本效率低，规划能力弱 |
| Model-based | 可以用想象轨迹提高样本效率，适合 planning | model error 会累积，错误预测会误导 policy |

LLM agent 训练里也有类似问题：如果模型内部对环境的预测是错的，它会越想越偏；如果完全不预测后果，只靠试错，成本又很高。

## 七、面试里怎么说

一个简洁回答：

> World model 是模型对环境状态转移的内部表示。它让模型不仅知道“现在是什么”，还能够预测“如果我采取某个动作，接下来可能发生什么”。在 agent 和 RL 里，这个能力直接影响 planning、tool use、long-horizon decision making 和 sample efficiency。

再补一句系统角度：

> 对 LLM agent 来说，world model 不只包括物理世界，也包括代码库、terminal、网页、工具和用户任务状态。强 agent 需要把 observation、action、state change 和 reward 连接起来，否则它只能做文本模仿，很难稳定完成长程任务。

## 八、后续待补

- JEPA / predictive representation
- Dreamer / model-based RL
- Video world models
- Agent memory vs world model
- World model eval：prediction、planning、counterfactual、causal consistency
- World model 和 agentic security / red teaming 的关系
