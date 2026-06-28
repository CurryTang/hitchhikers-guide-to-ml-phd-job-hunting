# ML Coding · Experiments & Ablations

对应 CS336 Assignment 1：Section 7。

使用方式：这一页都是实验型任务，不算传统 coding 题。先按“实验模板”补配置、指标和输出表；最后展开参考答案，对照实验设计和结论写法。

## Experiment 1 · Experiment Logger

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`experiment_log`

记录字段：

```text
run name
git commit / config hash
model config
optimizer config
dataset / tokenizer
train loss by step
validation loss by step
wall-clock time
tokens processed
checkpoint path
generated samples
```

输出：

```text
CSV / JSONL / wandb run
learning curve plot
experiment log markdown
```

实验模板：

```python
class ExperimentLogger:
    def __init__(self, path: str, config: dict):
        self.path = path
        self.config = config
        self.run_id = ...
        self.git_commit = ...

    def log(self, step: int, **metrics):
        row = {
            "step": step,
            "run_id": self.run_id,
            "config": self.config,
            "git_commit": self.git_commit,
            **metrics,
        }
        ...

def make_experiment_report(log_path: str) -> dict:
    return {
        "best_val_loss": ...,
        "final_train_loss": ...,
        "tokens_processed": ...,
        "wall_clock": ...,
    }
```

</details>

<details class="solution">
<summary>参考答案</summary>

experiment logger 要解决的是可复现性：以后看到一条 loss curve，要能知道它来自哪份代码、哪份配置和哪份数据。

一个简单 JSONL logger：

```python
import json, time, subprocess

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

class JSONLLogger:
    def __init__(self, path, config):
        self.f = open(path, "a", encoding="utf-8")
        self.config = config
        self.commit = git_commit()

    def log(self, step, **metrics):
        row = {
            "time": time.time(),
            "step": step,
            "git_commit": self.commit,
            "config": self.config,
            **metrics,
        }
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()
```

最低要求：

- 每个 run 有唯一名字。
- config 完整保存，不只写 “small model”。
- train/val loss、tokens processed、wall time 都记录。
- checkpoint path 和生成样例能追溯到同一个 run。

</details>

## Experiment 2 · TinyStories LR Sweep

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`learning_rate`

实验设计：

```text
fixed model config
fixed total tokens processed
vary learning rate logarithmically
include at least one divergent run
```

输出：

```text
learning curves
best validation loss
divergence threshold
edge-of-stability discussion
```

实验模板：

```python
def run_lr_sweep(base_config, learning_rates: list[float]):
    """
    Output:
        one row per LR with final loss, best val loss, divergence flag
    """
    rows = []
    for lr in learning_rates:
        config = {**base_config, "learning_rate": lr}
        result = ...  # train/eval run
        rows.append({
            "lr": lr,
            "final_train_loss": ...,
            "best_val_loss": ...,
            "diverged": ...,
            "notes": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>参考答案</summary>

LR sweep 的正确做法是只改变 learning rate，其他条件固定：

```text
same model
same tokenizer
same data order or same seed
same total tokens
same batch size
same warmup/cosine schedule shape
```

推荐 log-scale 搜索：

```text
1e-4, 3e-4, 1e-3, 3e-3, 1e-2
```

报告里不要只写“某个 LR 最好”，还要解释曲线：

| 曲线形态 | 解释 |
|---|---|
| loss 几乎不降 | LR 太小或训练时间太短 |
| loss 先降后爆 | LR 接近/超过稳定边界 |
| train loss 降、val loss 不降 | 可能过拟合或数据太小 |
| 最佳 LR 随 batch size 变化 | gradient noise scale 改了 |

最终输出应该包括一张 learning curve 图和一个表：

```text
lr | final train loss | best val loss | diverged? | tokens/sec
```

</details>

## Experiment 3 · Batch Size Sweep

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`batch_size_experiment`

实验矩阵：

```text
batch_size = 1
batch_size = 64
batch_size = 128
batch_size near memory limit
```

记录：

```text
tokens/sec
step time
validation loss
GPU memory
best LR per batch size
```

实验模板：

```python
def run_batch_size_sweep(base_config, batch_sizes: list[int]):
    rows = []
    for bs in batch_sizes:
        config = {**base_config, "batch_size": bs}
        result = ...
        rows.append({
            "batch_size": bs,
            "effective_batch_size": ...,
            "tokens_per_sec": ...,
            "step_time_ms": ...,
            "max_memory_gb": ...,
            "best_val_loss": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>参考答案</summary>

batch size sweep 要分清两个变量：micro batch 影响显存和 step time；effective batch size 还可能由 gradient accumulation 决定。

实验表：

```text
batch_size | grad_accum | effective_batch | lr | tokens/sec | max_mem | best_val_loss
```

解释思路：

- batch 太小：GPU 利用率差，tokens/sec 低，但 gradient noise 大，可能泛化还可以。
- batch 变大：吞吐通常上升，但需要调 LR。
- 接近显存上限：可能 OOM，或者 activation memory 让 step time 抖动。
- 如果 total tokens 固定，大 batch 意味着 optimizer steps 更少。

结论不能只看 step time。应该看同样 wall-clock 下的 validation loss，或同样 tokens processed 下的 validation loss。

</details>

## Experiment 4 · Generate TinyStories Samples

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`generate`

变量：

```text
temperature
top_p
prompt
checkpoint step
```

输出：

```text
sample text, at least 256 tokens or until EOS
sampling config
fluency comment
two factors affecting quality
```

实验模板：

```python
def sample_checkpoint(model, tokenizer, prompts, temperatures, top_ps):
    rows = []
    for prompt in prompts:
        for temperature in temperatures:
            for top_p in top_ps:
                text = generate(...)
                rows.append({
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "sample": text,
                    "comment": ...,
                })
    return rows
```

</details>

<details class="solution">
<summary>参考答案</summary>

generation 实验要固定 prompt，然后改变 sampling 参数，观察输出质量和多样性。

推荐记录：

```text
checkpoint_step
prompt
temperature
top_p
max_new_tokens
generated_text
stop_reason
```

解释框架：

| 现象 | 可能原因 |
|---|---|
| 重复短句 | 模型欠训练、temperature 太低、数据模式简单 |
| 语法乱 | checkpoint 太早、temperature 太高 |
| 内容流畅但单调 | top_p 太小或 prompt 太强 |
| EOS 很早 | tokenizer/EOS 学得太强，或训练数据短文本多 |

TinyStories 的样例应该重点看故事连贯性、角色一致性和句子完整性，不要用事实准确性作为主要指标。

</details>

## Experiment 5 · Remove RMSNorm

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`layer_norm_ablation`

实验：

```text
baseline pre-norm
no RMSNorm at previous best LR
no RMSNorm at lower LR
```

输出：

```text
learning curves
best stable LR
activation / gradient norm observations
normalization commentary
```

实验模板：

```python
def run_remove_rmsnorm_ablation(base_config):
    variants = [
        {"name": "baseline_prenorm", "use_rmsnorm": True, "lr": base_config["lr"]},
        {"name": "no_rmsnorm_same_lr", "use_rmsnorm": False, "lr": base_config["lr"]},
        {"name": "no_rmsnorm_lower_lr", "use_rmsnorm": False, "lr": base_config["lr"] / 3},
    ]
    rows = []
    for variant in variants:
        result = ...
        rows.append({
            **variant,
            "best_val_loss": ...,
            "diverged": ...,
            "grad_norm_p95": ...,
            "activation_norm_p95": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>参考答案</summary>

Remove RMSNorm 是一个稳定性 ablation。实验要和 baseline 对齐：

```text
baseline: pre-norm transformer + best LR
no_norm_a: remove RMSNorm + same LR
no_norm_b: remove RMSNorm + lower LR
```

需要记录：

```text
train loss
val loss
grad norm
activation norm
divergence step if any
```

预期解释：RMSNorm 控制 residual stream 的尺度，让 attention/MLP 输入分布更稳定。去掉后，同样 LR 下更容易 loss spike 或 grad norm 变大；降低 LR 可能能训练，但收敛速度和最终 loss 可能变差。

结论要写成实验观察，不要写成绝对规律；小模型、小数据上有时 no-norm 也能跑一段。

</details>

## Experiment 6 · Post-Norm Transformer

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`pre_norm_ablation`

对比：

```text
pre-norm:
  z = x + MHA(RMSNorm(x))
  y = z + FFN(RMSNorm(z))

post-norm:
  z = RMSNorm(x + MHA(x))
  y = RMSNorm(z + FFN(z))
```

输出：

```text
learning curve comparison
stability notes
best LR if retuned
```

实验模板：

```python
def run_norm_position_ablation(base_config):
    variants = ["pre_norm", "post_norm", "post_norm_retuned_lr"]
    rows = []
    for variant in variants:
        config = {**base_config, "block_type": variant}
        result = ...
        rows.append({
            "variant": variant,
            "lr": ...,
            "best_val_loss": ...,
            "stability": ...,
            "notes": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>参考答案</summary>

pre-norm 和 post-norm 的差别是 norm 放在 residual branch 前还是后。

实现 post-norm：

```python
class PostNormBlock(nn.Module):
    def forward(self, x, token_positions=None):
        x = self.ln1(x + self.attn(x, token_positions=token_positions))
        x = self.ln2(x + self.ffn(x))
        return x
```

对比时要保持参数量接近，只改 block 结构。观察点：

- post-norm 可能对 LR 更敏感。
- 深层网络里 pre-norm 梯度路径更直接，通常更容易训练。
- 如果 post-norm 表现差，要尝试更低 LR，而不是只跑一个点就下结论。

报告里建议画两张曲线：same LR 对比、retuned best LR 对比。

</details>

## Experiment 7 · NoPE vs RoPE

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`no_pos_emb`

实验：

```text
baseline RoPE
NoPE model
same training budget
same tokenizer / data
```

输出：

```text
validation loss curve
sample generation
position information discussion
```

实验模板：

```python
def run_position_ablation(base_config):
    variants = [
        {"name": "rope", "use_rope": True},
        {"name": "nope", "use_rope": False},
    ]
    rows = []
    for variant in variants:
        result = ...
        rows.append({
            "variant": variant["name"],
            "best_val_loss": ...,
            "sample_text": ...,
            "long_context_behavior": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>参考答案</summary>

NoPE vs RoPE 要回答的是：模型没有显式位置编码时，还能不能学到顺序信息。

实验设置：

```text
baseline: RoPE
ablation: remove RoPE from Q/K
same parameter count
same training tokens
same optimizer config
```

解释角度：

- causal mask 自身提供了方向性，但不提供精确相对位置表示。
- 短 context、小数据上 NoPE 可能还能下降。
- 长 context、需要位置关系的任务上 RoPE 通常更稳。
- generation 里可能出现重复、顺序混乱或长距离一致性下降。

报告不要只看 final loss；要加生成样例和 context length sensitivity。

</details>

## Experiment 8 · SwiGLU vs SiLU FFN

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`swiglu_ablation`

约束：

```text
SwiGLU: three matrices, d_ff around 8/3 d_model
SiLU baseline: two matrices, d_ff around 4 d_model
parameter counts approximately matched
```

输出：

```text
learning curves
parameter count comparison
validation loss comparison
gating discussion
```

实验模板：

```python
def run_ffn_ablation(base_config):
    variants = [
        {"name": "swiglu", "ffn_type": "swiglu", "d_ff": ...},
        {"name": "silu", "ffn_type": "silu", "d_ff": ...},
    ]
    rows = []
    for variant in variants:
        params = ...
        result = ...
        rows.append({
            "variant": variant["name"],
            "params": params,
            "tokens_per_sec": ...,
            "best_val_loss": ...,
            "notes": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>参考答案</summary>

SwiGLU 和 SiLU FFN 对比要先匹配参数量，否则结论不公平。

参数量近似：

```text
SwiGLU: 3 * d_model * d_ff_swiglu
SiLU FFN: 2 * d_model * d_ff_silu
```

所以常见设置：

```text
d_ff_silu ≈ 4 * d_model
d_ff_swiglu ≈ 8/3 * d_model
```

实验记录：

```text
variant | params | tokens/sec | best_val_loss | final_train_loss
```

解释：SwiGLU 的 gate 可以动态调制 channel，表达力更强；但多一条 projection path，kernel fusion 和显存访问也更复杂。小模型上差异可能不稳定，所以要报告多个 seed 或至少给出 run variance。

</details>

## Experiment 9 · OpenWebText Run

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`main_experiment`

输出：

```text
OpenWebText learning curve
generated sample
TinyStories vs OWT loss comparison
fluency analysis
```

分析重点：

OpenWebText 更杂、更长尾、更难压缩；同样模型和 compute budget 下，loss 和 generation quality 不能直接和 TinyStories 等价比较。

实验模板：

```python
def run_openwebtext_experiment(config):
    result = ...  # train on OWT
    sample = ...  # generate text from selected prompts
    return {
        "dataset": "OpenWebText",
        "tokenizer": ...,
        "training_tokens": ...,
        "best_val_loss": ...,
        "tokens_per_sec": ...,
        "generated_sample": sample,
        "tinystories_comparison": ...,
    }
```

</details>

<details class="solution">
<summary>参考答案</summary>

OpenWebText run 的重点是 domain shift。和 TinyStories 相比，OWT 更杂，包含网页、代码片段、论坛语言、长尾实体和噪声。

报告结构：

```text
dataset/tokenizer
model config
training tokens
best validation loss
tokens/sec
generated sample
failure examples
```

分析时要避免直接说 “OWT loss 更高所以模型更差”。不同数据分布的 entropy 不同，loss 不可直接横向比较。更合理的问题是：

- 同一模型在 OWT 上是否稳定下降？
- generation 是否有网页式碎片、重复、乱码？
- tokenizer bytes/token 是否变差？
- 同样 compute 下，OWT 学到的语言能力是否更通用？

</details>

## Experiment 10 · Leaderboard-Style Modification

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`leaderboard`

可选方向：

```text
weight tying
better initialization
LR / batch size schedule
architecture small changes
tokenizer changes within allowed data
throughput optimization
```

输出：

```text
final validation loss
wall-clock-bounded learning curve
description of modification
evidence the modification helped
```

实验模板：

```python
def run_leaderboard_modification(base_config, modification_name: str):
    baseline = ...
    modified_config = apply_modification(base_config, modification_name)
    modified = ...
    return {
        "modification": modification_name,
        "baseline_val_loss": ...,
        "modified_val_loss": ...,
        "delta": ...,
        "tokens_per_sec_delta": ...,
        "evidence": ...,
        "failure_modes": ...,
    }
```

</details>

<details class="solution">
<summary>参考答案</summary>

leaderboard-style modification 要像小型研究实验：提出假设、改一个主要变量、给证据。

几个可实现方向：

| 修改 | 假设 | 风险 |
|---|---|---|
| weight tying | embedding 和 LM head 共享，减少参数并可能正则化 | 可能限制输出层表达 |
| retune LR schedule | 原 schedule 不是最优 | 需要更多实验预算 |
| better init | 改善早期稳定性 | 可能只影响前几百步 |
| fused kernels / compile | 提高 tokens/sec | 不一定改善 final loss |
| tokenizer change | 更好压缩数据 | 会改变训练分布和 loss 可比性 |

以 weight tying 为例：

```python
model.lm_head.weight = model.token_embeddings.weight
```

报告至少包含：

```text
baseline run
modified run
same token budget
same eval protocol
validation loss delta
throughput delta
one paragraph explaining why it helped or failed
```

如果修改没有提升，也可以是好答案；关键是实验设计干净，结论和数据一致。

</details>
