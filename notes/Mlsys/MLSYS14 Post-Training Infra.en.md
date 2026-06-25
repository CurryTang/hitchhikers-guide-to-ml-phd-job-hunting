
> [!info] Note
> English version is in progress. Please switch to 中文 for the full content.

# LLM Post-Training Infrastructure: From TRL to Forge

This tutorial covers the systems engineering perspective on LLM reinforcement learning training infrastructure. Key topics include:

- Why RL training is a distinct systems problem (80–90% time on rollout generation)
- The three-way tradeoff: **Throughput × On-policyness × Agentic flexibility**
- PPO and GRPO: what matters for system design
- Four design axes: control flow, resource placement, weight sync, synchronicity
- Framework tour: TRL → OpenRLHF → veRL → slime → AReaL → ROLL → Forge
- Deep dives: memory accounting, long-tail rollouts, async design space, train-inference mismatch, FP8 vs INT8, MoE × RL
- Framework selection decision tree (answering Q35 directly)

See the Chinese version for full content. Companion exercises: [[MLSYS15 RL Infra 自测 35 问]].
