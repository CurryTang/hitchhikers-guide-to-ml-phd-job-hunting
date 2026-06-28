# ML Coding · Tensor Modules

对应 CS336 Assignment 1：Section 3.2-3.4.3。

使用方式：每题先看目标和验收标准，再按“解题模板”把 TODO 补完整；最后展开参考答案，对照边界条件、sanity checks 和实现细节。

## Warmup · Tensor Shape Gym

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

目标：练习 batch-like dimensions、einsum、rearrange。

练习项：

```text
batched linear projection
split d_model -> num_heads * d_head
merge heads -> d_model
broadcast scalar/vector over leading dims
pixel mixing / arbitrary tensor contraction
```

每题记录：

```text
input shape
output shape
einsum / rearrange pattern
toy sanity check
```

练习模板：

```python
def shape_gym():
    x = ...      # e.g. (batch, seq, d_model)
    W = ...      # e.g. (out_features, in_features)

    y_linear = ...       # (..., in) x (out, in) -> (..., out)
    heads = ...          # (B, T, D) -> (B, H, T, Dh)
    merged = ...         # (B, H, T, Dh) -> (B, T, D)

    assert ...
    return {
        "linear": y_linear.shape,
        "heads": heads.shape,
        "merged": merged.shape,
    }
```

</details>

<details class="solution">
<summary>参考答案</summary>

这一节的目标是把“最后一维是 feature，前面都是 batch-like dims”这个习惯练熟。几个常用 pattern：

```python
import torch
from einops import rearrange

x = torch.randn(2, 3, 12)      # batch=2, seq=3, d_model=12
W = torch.randn(16, 12)        # out, in

# batched linear: (..., in), (out, in) -> (..., out)
y = torch.einsum("...i,oi->...o", x, W)
assert y.shape == (2, 3, 16)

# split heads
h = rearrange(x, "b s (nh dh) -> b nh s dh", nh=3)
assert h.shape == (2, 3, 3, 4)

# merge heads
x2 = rearrange(h, "b nh s dh -> b s (nh dh)")
assert x2.shape == x.shape
```

写模块前建议先用这些 toy shape 测一遍。Transformer 里的大多数 bug 都是 head 维、sequence 维、batch 维放错导致的。

一个实用检查表：

| 操作 | 输入 | 输出 |
|---|---|---|
| Linear | `(..., d_in)` | `(..., d_out)` |
| MHA split | `(B, T, D)` | `(B, H, T, Dh)` |
| MHA merge | `(B, H, T, Dh)` | `(B, T, D)` |
| RMSNorm | `(..., D)` | `(..., D)` |
| RoPE | `(..., T, Dh)` | `(..., T, Dh)` |

</details>

## Exercise 1 · Linear Module

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`linear`

接口：

```text
Linear(in_features, out_features, device=None, dtype=None)
forward(x) -> (..., out_features)
```

关键约束：

- weight shape 是 `(out_features, in_features)`。
- 支持 arbitrary leading dimensions。
- 不用 `nn.Linear` 或 `nn.functional.linear`。
- 初始化用 truncated normal。

测试：

```bash
uv run pytest -k test_linear
```

解题模板：

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(...)
        ...  # truncated normal init

    def forward(self, x):
        """
        Input:
            x: (..., in_features)
        Output:
            (..., out_features)
        """
        return ...  # no nn.Linear / F.linear
```

</details>

<details class="solution">
<summary>参考答案</summary>

`Linear` 要实现的是 PyTorch `nn.Linear` 的核心行为，但不能直接调用 `nn.Linear` 或 `F.linear`。

```python
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=dtype
        ))
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return torch.einsum("...i,oi->...o", x, self.weight)
```

为什么 weight 是 `(out_features, in_features)`：这样每个输出通道对应一行权重，公式是：

```text
y[..., o] = sum_i x[..., i] * weight[o, i]
```

检查点：

```python
m = Linear(4, 7)
x = torch.randn(2, 3, 4)
assert m(x).shape == (2, 3, 7)
```

</details>

## Exercise 2 · Embedding Module

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`embedding`

接口：

```text
Embedding(num_embeddings, embedding_dim, device=None, dtype=None)
forward(token_ids) -> token_ids.shape + (embedding_dim,)
```

关键约束：

- embedding table shape 是 `(vocab_size, d_model)`。
- token ids 是 integer tensor。
- 不用 `nn.Embedding`。

测试：

```bash
uv run pytest -k test_embedding
```

解题模板：

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(...)  # (num_embeddings, embedding_dim)
        ...

    def forward(self, token_ids):
        """
        Input:
            integer tensor with arbitrary shape
        Output:
            token_ids.shape + (embedding_dim,)
        """
        return ...
```

</details>

<details class="solution">
<summary>参考答案</summary>

Embedding 是一个查表模块：token id 是整数索引，输出是对应行向量。

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        ))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return self.weight[token_ids]
```

如果 `token_ids.shape == (B, T)`，输出就是 `(B, T, D)`。PyTorch 的 advanced indexing 会保留输入 index 的 shape，并在最后追加 embedding dim。

常见错误：

- token ids 用 float tensor。
- 把 weight 初始化成 `(embedding_dim, vocab_size)`。
- 忘记支持任意 shape 的 token ids，例如 `(T,)`、`(B,T)` 都应该可用。

</details>

## Exercise 3 · RMSNorm

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`rmsnorm`

接口：

```text
RMSNorm(d_model, eps=1e-5)
forward(x) -> same shape
```

关键约束：

- normalize 最后一维。
- gain shape 是 `(d_model,)`。
- square / mean 前 upcast 到 float32。
- 输出 cast 回原 dtype。

Sanity checks：

```text
all-zero input remains finite
shape preserved
scaling input mostly cancels after normalization
```

解题模板：

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(...)

    def forward(self, x):
        """
        Normalize over the last dimension.
        """
        in_dtype = x.dtype
        x_float = ...
        rms = ...
        y = ...
        return ...
```

</details>

<details class="solution">
<summary>参考答案</summary>

RMSNorm 对最后一维做 root-mean-square normalization，不减均值：

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        y = x_float / rms
        return (y * self.weight).to(in_dtype)
```

为什么要 upcast：BF16/FP16 做平方和均值容易有精度问题，尤其是 hidden dim 大时。计算归一化统计量用 FP32 更稳，最后再 cast 回输入 dtype。

检查点：

```python
x = torch.randn(2, 3, 8, dtype=torch.float16, device="cuda")
norm = RMSNorm(8, device="cuda", dtype=torch.float16)
y = norm(x)
assert y.shape == x.shape
assert y.dtype == x.dtype
```

</details>

## Exercise 4 · SwiGLU Feed-Forward

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`positionwise_feedforward`

结构：

```text
SiLU(W1 x) * W3 x
then W2 down projection
```

关键约束：

- `d_ff` 接近 `8/3 * d_model`。
- `d_ff` 调整到 multiple of 64。
- 支持 arbitrary leading dims。

测试：

```bash
uv run pytest -k test_swiglu
```

解题模板：

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        d_ff = ...  # around 8/3 * d_model, rounded to multiple of 64
        self.w1 = ...  # gate projection
        self.w3 = ...  # up projection
        self.w2 = ...  # down projection

    def forward(self, x):
        gate = ...
        up = ...
        return ...
```

</details>

<details class="solution">
<summary>参考答案</summary>

SwiGLU FFN 做的是 gated feed-forward。常见 LLaMA 风格：

```text
FFN(x) = W_down( SiLU(W_gate x) * W_up x )
```

实现：

```python
import math

def round_up_to_multiple(x, multiple):
    return multiple * math.ceil(x / multiple)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = round_up_to_multiple(int(8 * d_model / 3), 64)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # up
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # down

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
```

为什么 `8/3 * d_model`：SwiGLU 有三套矩阵，而普通 FFN 通常是两套矩阵、hidden dim 约 `4 * d_model`。为了让参数量接近，会把 SwiGLU 的 hidden dim 调小到约 `8/3 * d_model`。

</details>

## Exercise 5 · RoPE

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`rope`

接口：

```text
RotaryPositionalEmbedding(theta, d_k, max_seq_len)
forward(x, token_positions) -> same shape
```

关键约束：

- `x` shape 是 `(..., seq_len, d_k)`。
- `token_positions` shape 是 `(..., seq_len)`。
- cos/sin 可以 precompute 为 non-persistent buffer。
- rotation 保持每对维度的 L2 norm。

测试：

```bash
uv run pytest -k test_rope
```

解题模板：

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        ...  # precompute cos/sin buffers for positions and dim pairs

    def forward(self, x, token_positions):
        """
        Input:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len)
        Output:
            same shape as x
        """
        x_even = ...
        x_odd = ...
        cos = ...
        sin = ...
        return ...
```

</details>

<details class="solution">
<summary>参考答案</summary>

RoPE 把每两个 hidden dimensions 看成一个二维平面，并按位置旋转。核心公式：

```text
[x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
```

实现骨架：

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        assert d_k % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x, token_positions):
        # x: (..., seq, d_k)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)
```

检查点：

```python
y = rope(x, positions)
assert y.shape == x.shape
assert torch.allclose(
    (x[..., 0::2] ** 2 + x[..., 1::2] ** 2),
    (y[..., 0::2] ** 2 + y[..., 1::2] ** 2),
    atol=1e-4,
)
```

RoPE 只作用在 attention 的 Q/K 上，不作用在 V 上。

</details>

## Debug Checklist

<details class="exercise">
<summary><span class="q-label">Debug</span> <span class="q-text">展开检查项</span></summary>

- 所有 module 都要支持 `device` / `dtype`。
- 明确标注 sequence dimension 和 head dimension，并用断言检查 shape。
- 用小 shape 打印中间 tensor，比直接跑完整模型更快。

</details>
