# ML Coding · Tensor Modules

对应 CS336 Assignment 1：Section 3.2-3.4.3。

## Exercise 1 · Tensor Shape Gym

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

</details>

## Exercise 2 · Linear Module

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

</details>

## Exercise 3 · Embedding Module

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

</details>

## Exercise 4 · RMSNorm

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

</details>

## Exercise 5 · SwiGLU Feed-Forward

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

</details>

## Exercise 6 · RoPE

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

</details>

## Debug Checklist

<details class="exercise">
<summary><span class="q-label">Debug</span> <span class="q-text">展开检查项</span></summary>

- 所有 module 都要支持 `device` / `dtype`。
- 明确标注 sequence dimension 和 head dimension，并用断言检查 shape。
- 用小 shape 打印中间 tensor，比直接跑完整模型更快。

</details>
