# LLM Post-Training Infra 文档计划（MLSYS14 + MLSYS15）

> 状态：大纲阶段。本文件是写作计划，不进入站点渲染。
> 来源素材：@sheriyuo 的 "RL Interview Questions 2026"（35 题，x.com/sheriyuo/status/2063295181131247674），
> 以及 TRL / veRL / OpenRLHF / slime / AReaL / ROLL / MiniMax Forge 等框架的论文与技术博客。

---

## 总体结构

- **MLSYS14：LLM Post-Training 的系统视角 —— 从 TRL 到 Forge 的 RL Infra 演进**（主文档，重 infra 轻算法）
- **MLSYS15：RL Infra 自测 35 问**（练习题文档，逐题给可折叠答案，与 MLSYS14 章节互链）
- 前端改造：练习题折叠组件、标题锚点跳转、章节标题优化（见文末）

---

# MLSYS14 大纲：LLM Post-Training 的系统视角

## 一、引言：为什么 RL Infra 是一个系统问题

- 1.1 Post-training 全景图：SFT → RM → RLHF → RLVR → Agentic RL，各阶段的系统负载差异
- 1.2 RL 训练的独特负载形态：同一个 job 里同时跑「推理引擎级别的 rollout」和「预训练级别的 train」
  - 与 pretraining 的本质区别：数据不是静态的，是由当前策略**在线生产**的
  - 成本结构：rollout（generation）通常占 60–90% 的墙钟时间
- 1.3 三难困境（贯穿全文的主线）：**吞吐（throughput）× 稳定性（on-policyness / 数值一致性）× 灵活性（agentic 环境接入）**
- 1.4 本文路线图：先讲最小算法背景 → 解剖一个 RL 训练系统的通用结构 → 按时间线巡礼各框架 → 专题深入

## 二、最小算法背景：PPO 与 GRPO（只讲到够用为止）

- 2.1 PPO 一页纸：actor / critic / reference / reward 四个模型，clip objective，GAE
- 2.2 GRPO 一页纸：砍掉 critic，group 内均值做 baseline，KL 正则的两种位置（reward 内 vs loss 内）
- 2.3 算法选择如何决定系统形态（这是本节真正的重点）：
  - 模型副本数：PPO 4 份 vs GRPO 3 份（actor / ref / rollout 权重），对显存方案的影响
  - group sampling 决定 rollout 的 batch 形态（同 prompt 多 completion → prefix 共享 / KV 复用机会）
  - clip + importance sampling 的存在，是后面一切「异步化 / off-policy 容忍度」的算法基础
- 2.4 一笔带过的变体地图：DAPO / GSPO / CISPO / Dr.GRPO 各自改了哪个 term（一张表，不展开推导）

## 三、解剖一个 RL 训练系统：通用组件与设计轴

> 本章建立分析框架，后面所有框架都用这套坐标系来比。

- 3.1 三大组件：
  - **Rollout engine**：vLLM / SGLang（continuous batching、paged KV、为什么不能直接用训练引擎做生成）
  - **Training engine**：Megatron-LM / FSDP / DeepSpeed（与 pretraining 共享的并行技术栈）
  - **Orchestrator**：谁来编排数据流（Ray？单进程？gateway 服务？）
- 3.2 设计轴 1 —— 控制流：single-controller vs multi-controller（HybridFlow 的核心洞察）
- 3.3 设计轴 2 —— 资源放置：colocated（时分复用，reshard 切换）vs disaggregated（空间分离，流水）
  - 各自的失败模式：colocate 的显存挤兑与切换开销；disaggregate 的两边互相等
- 3.4 设计轴 3 —— 权重同步：NCCL broadcast / CUDA IPC / RDMA 点对点；训练并行布局 → 推理并行布局的 resharding 问题；bucket 化与流水化
- 3.5 设计轴 4 —— 同步性谱系：strictly on-policy → one-step overlap → fully async
  - staleness 的定义与控制；partial rollout（中断续采）的语义
- 3.6 横切问题先点名（专题章再展开）：长尾 rollout、train–infer mismatch、确定性、MoE、agentic 环境

## 四、框架巡礼：一条时间线，两次范式转移

> 叙事主线：**「训练脚本」时代 → 「混合引擎」时代 → 「异步原生 / agent 原生」时代**

- 4.1 TRL（HuggingFace）：RLHF 的「hello world」
  - accelerate + 单控制器；PPOTrainer/GRPOTrainer；vLLM colocate/server 模式
  - 它的天花板在哪：单机思维、generation 与 training 串行、难以扩展到多节点 agentic
- 4.2 OpenRLHF：Ray + vLLM 分离架构的先驱（简短一节，承上启下）
- 4.3 veRL（ByteDance / HybridFlow）：混合引擎时代的标志
  - hybrid controller：single-controller 表达数据流 + multi-controller 执行算子
  - 3D-HybridEngine：colocate 下的 train/rollout reshard 切换
  - FSDP & Megatron 双后端、vLLM & SGLang 双 rollout、庞大的衍生生态（DAPO、PRIME、SkyRL fork…）
- 4.4 slime（智谱/THUDM）：做减法的哲学
  - SGLang-native + Megatron-native，自己只做「数据流胶水」
  - sync/async 双模式、CPU Adam offload、GLM-4.5/4.6 的训练后端
  - 数据怎么流：rollout buffer → data packing → Megatron（对应 35 问之 Q34）
- 4.5 AReaL（蚂蚁 & 清华 IIIS）：fully async 的代表
  - interruptible rollout + 解耦的 PPO objective（decoupled clip）
  - staleness-aware 调度；为什么「rollout 永远不应该停」
  - AReaL-lite 的再设计
- 4.6 ROLL（阿里）：多后端、多场景的「平台型」框架
  - Ray-based、五种角色抽象（actor/critic/ref/reward/env）
  - ROLL Flash：异步 + agentic rollout 的扩展
- 4.7 Forge（MiniMax）：agent-native 时代
  - RL service Gateway：把「agent scaffold」从训练框架里彻底解耦出来
  - 任意 context 操纵（memory 压缩 / history 重写 / multi-agent）下的 token 级 credit 归属
  - CISPO 与系统设计的配合；M2.5 量级：20 万 token context、十万级环境、日均百万级样本
- 4.8 一张大对比表：controller 模型 / placement / 训练后端 / rollout 引擎 / 异步支持 / agentic 支持 / 代表模型
- 4.9 没展开但值得知道的：NeMo-RL、SkyRL、rLLM、prime-rl、Tinker（training-as-a-service 路线）

## 五、专题深入（每个专题对应若干道 35 问）

- 5.1 显存账本：GRPO 训练时显存里到底有几份模型？（Q20）
  - actor 参数 + 梯度 + Adam 状态 + ref + rollout 引擎权重 + KV cache + 激活值，逐项算
  - 各优化的收益：offload / ZeRO 分级 / LoRA / 参数共享 / sleep mode
- 5.2 Rollout 侧的系统问题：
  - 长尾问题与对策：partial rollout、length-aware 调度、over-sampling + 提前截断（Q23）
  - continuous batching 在 RL 里引入的新问题；vLLM vs SGLang 的差异（Q24、Q25）
  - 分布式推理的 KV cache 迁移与多卡通信（Q21）
- 5.3 异步 RL 的设计空间：
  - 现有异步框架与它们各自解决的同步瓶颈（Q27）
  - AReaL vs slime 对「rollout 瓶颈」的不同理解（Q32）
  - staleness 怎么想、实践中的典型取值（Q33）；partial rollout 时旧 policy 的 KV 是否保留（Q28）
- 5.4 Train–Inference Mismatch：
  - 来源：算子实现差异、精度差异、batch invariance（Q31 的确定性问题、atomic add）
  - MoE 放大了 mismatch：routing 漂移；routing replay 与 IS 修正（Q11、Q19 的 infra 半边）
  - 数值修正方案：token 级/序列级 importance correction（TIS/MIS 一类做法）
- 5.5 精度专题：INT8 vs FP8，训练用什么、rollout 用什么、为什么（Q22）
- 5.6 MoE × RL：Expert Parallelism 对吞吐的影响（Q29）；与长上下文的 compute–comm overlap、Megatron vs FSDP 的并行策略差异（Q30、Q26）
- 5.7 Agentic RL 的新基础设施：环境/sandbox 服务化、reward service、工具调用的失败注入

## 六、收尾：选型与展望

- 6.1 「VeRL、TRL、Unsloth、AReaL、slime 选哪个」的回答框架（Q35）：按规模 / 团队 / 场景给决策树
- 6.2 趋势判断：异步成为默认、agent 与训练框架解耦、training-as-a-service、确定性推理
- 6.3 References（论文 + 各框架 repo + 技术博客）

---

# MLSYS15 大纲：RL Infra 自测 35 问（练习题文档）

- 形式：每题一个可折叠卡片 —— **题面（始终可见）→ 提示（可选折叠）→ 详细解答（默认折叠，点击展开）**
- 组织：不按原文 1–35 顺序，而按 MLSYS14 的章节主题重排，每题标注原始题号与难度（⭐~⭐⭐⭐）
  - A 部分·算法基础（Q1–Q10，对应 MLSYS14 第二章）：Actor-Critic、KL/CE/MLE、reward 设计、IS/拒绝采样、advantage 与 baseline、clip、KL penalty、All-Reduce 翻车题、DPO
  - B 部分·算法进阶（Q11–Q19）：MoE mismatch、超参选择、GRPO 变体全家桶、trust region、RL 能力边界（ProRL/OPD/R1→V4）
  - C 部分·单机与显存（Q20、Q22、Q26）
  - D 部分·Rollout 系统（Q21、Q23、Q24、Q25、Q28）
  - E 部分·异步与框架（Q27、Q32、Q33、Q34、Q35）
  - F 部分·进阶系统（Q29、Q30、Q31）
- 每题答案末尾给「延伸阅读」链接 + 指回 MLSYS14 对应小节的 `[[MLSYS14#小节]]` 内链
- 中英双语（.md / .en.md），与现有笔记一致

---

# 前端改造计划（src/App.jsx + App.css）

1. **练习题折叠组件**
   - markdown 侧约定：用 Obsidian 折叠 callout 语法 `> [!question]- Q20 标题` + 内嵌 `> [!answer]-`，或直接写 HTML `<details class="exercise">`
   - 渲染侧：normalizeObsidianMarkdown 目前把 `[!type]±` 的折叠语义丢掉了 → 改为把折叠 callout 编译成 `<details><summary>…</summary>…</details>`（rehypeRaw 已启用，原生支持）
   - App.css 加 exercise 卡片样式：题面高亮、展开动画、"显示答案/隐藏答案" 视觉
2. **标题锚点与跳转**
   - 现状 bug：文内 `[[#一、引言]]` 这类 TOC 链接在 resolveNoteId 里查不到，会退化成纯文本，**目前目录不可点击**
   - 方案：给每个 heading 生成 slug（自定义 heading renderer 或 rehype-slug + GitHub 风格 slugger），`[[#X]]` → `[X](#slug)`；heading hover 显示 ¶/🔗 锚点，点击复制链接；URL hash 同时编码「笔记 id + 段落锚点」（现在 hash 只用来选笔记，需要兼容 `#noteId:heading` 这样的双层 hash）
   - 可选加分项：右侧浮动 TOC + scroll spy
3. **章节标题优化**
   - sidebar 现在显示 "MLSYS1"…"MLSYS6" 这类裸标题 → 在 tutorialDefinitions 里给每章人类可读标题（如 "MLSYS1 · GPU 体系结构入门"），需要先通读各章确认主题
4. 新增 MLSYS14 / MLSYS15 注册到 tutorialDefinitions，补 .en.md 变体

---

# 写作顺序建议

1. 前端改造（先把折叠/锚点能力做好，写文档时直接用新语法）
2. MLSYS14 主文档（中文 → 英文）
3. MLSYS15 练习题（中文 → 英文）
4. 回头优化全站章节标题
