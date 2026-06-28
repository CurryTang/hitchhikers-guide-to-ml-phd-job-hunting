# ML Coding · Training Components

对应 CS336 Assignment 1：Section 4。

使用方式：每题先看目标和验收标准，再按“解题模板”把 TODO 补完整；最后展开参考答案，对照边界条件、sanity checks 和实现细节。

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

解题模板：

```python
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Input:
        inputs: (..., vocab_size) logits
        targets: (...) integer class ids
    Output:
        scalar mean cross entropy
    """
    logits = ...
    y = ...
    max_logits = ...
    log_sum_exp = ...
    correct_logits = ...
    return ...
```

</details>

<details class="solution">
<summary>参考答案</summary>

cross entropy 输入是 logits，不是 probabilities。稳定实现用 log-sum-exp：

```python
def cross_entropy(inputs, targets):
    # inputs: (..., vocab), targets: (...)
    logits = inputs.reshape(-1, inputs.shape[-1])
    y = targets.reshape(-1)

    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + max_logits.squeeze(-1)
    correct = logits[torch.arange(logits.shape[0], device=logits.device), y]
    loss = log_sum_exp - correct
    return loss.mean()
```

等价理解：

```text
CE = -log softmax(logits)[target]
   = logsumexp(logits) - logits[target]
```

检查点：

- targets shape 必须等于 inputs 去掉最后 vocab 维后的 shape。
- 返回 scalar。
- 不要先 softmax 再 log，数值更差。

</details>

## Lab · SGD LR Toy Sweep

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

实验模板：

```python
def run_sgd_lr(lr: float, steps: int = 10) -> list[float]:
    """
    Input:
        lr and number of update steps
    Output:
        loss values over time
    """
    x = ...
    losses = []
    for step in range(steps):
        loss = ...
        grad = ...
        x = ...
        losses.append(...)
    return losses

def sweep_lrs(lrs):
    return {lr: run_sgd_lr(lr) for lr in lrs}
```

</details>

<details class="solution">
<summary>参考答案</summary>

toy sweep 的目的不是训练模型，而是观察学习率和曲率的关系。可以用一维 quadratic：

```python
def run_sgd_lr(lr, steps=10):
    x = torch.tensor([10.0])
    values = []
    for _ in range(steps):
        loss = x.pow(2).sum()
        grad = 2 * x
        x = x - lr * grad
        values.append(float(loss))
    return values
```

对于 `f(x)=x^2`，梯度下降更新是：

```text
x_{t+1} = (1 - 2lr) x_t
```

因此：

| lr | 行为 |
|---|---|
| 很小 | 慢慢收敛 |
| 接近 0.5 | 快速接近 0 |
| 大于 1 | 振荡并发散 |

如果 assignment 指定 `1e1/1e2/1e3`，这些学习率都会明显发散；记录 loss curve 时要说明这是为了展示 instability，而不是推荐值。

</details>

## Exercise 2 · AdamW

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

解题模板：

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = ...
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                ...  # initialize step, m, v
                ...  # update biased moments
                ...  # bias correction
                ...  # decoupled weight decay
                ...  # parameter update
```

</details>

<details class="solution">
<summary>参考答案</summary>

AdamW 的关键是 decoupled weight decay：先按梯度做 Adam 更新，再单独做权重衰减，或者等价地在 update 中加入 `weight_decay * p`，但不要把 weight decay 混进梯度 moment。

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                update = m_hat / (torch.sqrt(v_hat) + eps)

                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss
```

常见错误：

- timestep 从 0 开始做 bias correction。
- `p.grad is None` 时仍更新 state。
- 把 weight decay 加进 grad，变成 Adam + L2，而不是 AdamW。

</details>

## Exercise 3 · AdamW Accounting

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

解题模板：

```python
def adamw_memory_accounting(num_params, param_bytes=2, grad_bytes=2, state_bytes=4):
    """
    Output:
        memory bytes for params, grads, Adam moments, and total
    """
    return {
        "parameters": ...,
        "gradients": ...,
        "adam_m": ...,
        "adam_v": ...,
        "total": ...,
    }

def estimate_training_time(total_tokens, flops_per_token, peak_flops, mfu):
    total_flops = ...
    return ...
```

</details>

<details class="solution">
<summary>参考答案</summary>

显存账本先按参数量 `P` 写：

```text
parameters:       P * param_bytes
gradients:        P * grad_bytes
AdamW m:          P * state_bytes
AdamW v:          P * state_bytes
master weights:   可选，混合精度训练有时额外存 FP32 copy
activations:      depends on batch_size * context_length * d_model * layers
```

BF16 参数 + FP32 AdamW 常见估算：

```text
param      2 bytes/param
grad       2 or 4 bytes/param
Adam m     4 bytes/param
Adam v     4 bytes/param
```

所以 optimizer state 往往比参数本身更大。训练 FLOPs 粗略估计：

```text
forward FLOPs = F
backward FLOPs ≈ 2F
optimizer step FLOPs ≈ O(P), 通常比 forward/backward 小
one train step ≈ 3F + optimizer
```

在 MFU 假设下：

```python
def training_time_seconds(total_flops, peak_flops, mfu):
    return total_flops / (peak_flops * mfu)
```

回答这类题时要把参数、梯度、optimizer state、activation 分开讲；不要只说“模型 7B 用 14GB”。

</details>

## Exercise 4 · Cosine LR with Warmup

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

解题模板：

```python
def get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    """
    Piecewise schedule:
        warmup -> cosine decay -> min_lr
    """
    if ...:
        return ...
    if ...:
        return ...
    progress = ...
    coeff = ...
    return ...
```

</details>

<details class="solution">
<summary>参考答案</summary>

cosine schedule 分三段：

```python
import math

def get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_lr

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)
```

检查点：

```text
it=0 -> 0 或接近 0
it=warmup_iters -> max_lr
it=cosine_cycle_iters -> min_lr
it>cosine_cycle_iters -> min_lr
```

如果 `warmup_iters == 0`，需要单独处理，避免除以 0。

</details>

## Exercise 5 · Gradient Clipping

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

解题模板：

```python
def clip_grad_norm(parameters, max_l2_norm, eps=1e-6):
    """
    Input:
        iterable of parameters with .grad
    Output:
        original global grad norm
    Side effect:
        scale gradients in-place if norm is too large
    """
    grads = ...
    total_norm = ...
    scale = ...
    for p in parameters:
        ...
    return total_norm
```

</details>

<details class="solution">
<summary>参考答案</summary>

global grad norm 是把所有参数的梯度当成一个长向量来算 L2 norm：

```python
def clip_grad_norm(parameters, max_l2_norm, eps=1e-6):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)

    total = torch.zeros((), device=params[0].grad.device)
    for p in params:
        total += torch.sum(p.grad.detach() ** 2)
    norm = torch.sqrt(total)

    scale = torch.clamp(max_l2_norm / (norm + eps), max=1.0)
    for p in params:
        p.grad.mul_(scale)
    return norm
```

重点：

- 所有梯度用同一个 scale，不是每个 tensor 单独 clip。
- norm 小于阈值时 scale clamp 到 1，不改变梯度。
- 原地修改 `.grad`，但返回 clipping 前的 norm 方便 logging。

</details>
