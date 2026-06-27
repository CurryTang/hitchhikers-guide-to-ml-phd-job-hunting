# ML Coding · Training Components

对应 CS336 Assignment 1：Section 4。

## Exercise 1 · Cross-Entropy

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`cross_entropy`

接口：

```text
inputs: (..., vocab_size)
targets: (...)
return: scalar mean loss
```

关键约束：

- 用 log-sum-exp 形式。
- 减最大 logit 保证数值稳定。
- 支持 arbitrary leading batch dims。
- 返回 batch 平均。

测试：

```bash
uv run pytest -k test_cross_entropy
```

</details>

## Exercise 2 · SGD LR Toy Sweep

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`learning_rate_tuning`

目标：观察 learning rate 对 toy quadratic loss 的影响。

实验：

```text
lr = 1e1
lr = 1e2
lr = 1e3
steps = 10
```

记录：

```text
loss curve
final loss
converged / oscillated / diverged
```

</details>

## Exercise 3 · AdamW

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`adamw`

接口：

```text
AdamW(params, lr, betas, eps, weight_decay)
step()
```

状态：

```text
t
m: first moment
v: second moment
```

关键约束：

- decoupled weight decay。
- bias correction 用从 1 开始的 timestep。
- `p.grad is None` 跳过。
- state 按 parameter 存。

测试：

```bash
uv run pytest -k test_adamw
```

</details>

## Exercise 4 · AdamW Accounting

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`adamw_accounting`

输出：

```text
parameter memory
activation memory
gradient memory
optimizer state memory
total memory as function of batch_size
AdamW FLOPs per step
training time under MFU assumption
```

检查点：

- AdamW state 至少有 `m` 和 `v`。
- gradients 和 parameters 同量级。
- backward pass 约为 forward FLOPs 的 2 倍。

</details>

## Exercise 5 · Cosine LR with Warmup

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`learning_rate_schedule`

接口：

```text
get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
```

分段：

```text
it < warmup_iters: linear warmup
warmup_iters <= it <= cosine_cycle_iters: cosine decay
it > cosine_cycle_iters: min_lr
```

测试：

```bash
uv run pytest -k test_get_lr_cosine_schedule
```

</details>

## Exercise 6 · Gradient Clipping

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`gradient_clipping`

接口：

```text
clip_grad_norm(parameters, max_l2_norm)
```

关键约束：

- 对所有 parameter gradients 计算 global L2 norm。
- norm 小于 max 时不改变。
- norm 大于 max 时所有 gradients 同比例缩放。
- 原地修改 `.grad`。

测试：

```bash
uv run pytest -k test_gradient_clipping
```

</details>
