# ML Coding · Training Loop & Generation

对应 CS336 Assignment 1：Section 5-6。

## Exercise 1 · Next-Token Batch Sampler

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`data_loading`

接口：

```text
get_batch(dataset, batch_size, context_length, device) -> (x, y)
```

输出：

```text
x: (batch_size, context_length)
y: (batch_size, context_length)
```

关键约束：

- `y` 是 `x` 向右偏移一位的 target。
- sample start index 要保证 target 合法。
- 输出 tensor 放到指定 device。
- 大数据用 `np.memmap` 或 `np.load(..., mmap_mode='r')`。

测试：

```bash
uv run pytest -k test_get_batch
```

</details>

## Exercise 2 · Checkpoint Save / Load

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`checkpointing`

接口：

```text
save_checkpoint(model, optimizer, iteration, out)
load_checkpoint(src, model, optimizer) -> iteration
```

必须保存：

```text
model.state_dict()
optimizer.state_dict()
iteration
```

Sanity checks：

```text
save -> mutate -> load -> parameters restored
optimizer moments restored
iteration restored
path and file-like object both work
```

测试：

```bash
uv run pytest -k test_checkpointing
```

</details>

## Exercise 3 · Full Training Script

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`training_together`

配置项：

```text
model hyperparameters
optimizer hyperparameters
batch_size
context_length
train / val data path
checkpoint path
eval interval
log interval
device
dtype / matmul precision
```

训练 loop：

```text
sample batch
forward
compute cross entropy
backward
clip gradients
optimizer step
zero gradients
update LR
periodic eval
periodic checkpoint
periodic logging
```

Debug milestones：

| milestone | expected signal |
|---|---|
| overfit one minibatch | train loss approaches near-zero |
| tiny model on tiny data | loss decreases smoothly |
| validation eval | no gradient graph retained |
| checkpoint resume | curve continues without reset |

</details>

## Exercise 4 · Autoregressive Decoder

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`decoding`

功能：

```text
prompt -> token ids
loop:
  model forward
  take last-position logits
  apply temperature
  apply top-p filter
  sample token
  append token
  stop on EOS or max_new_tokens
decode ids -> text
```

参数：

```text
max_new_tokens
temperature
top_p
eos_token_id
context_length
```

关键约束：

- 只用最后一个 position 的 logits。
- 超出 context length 时保留最近窗口。
- top-p 后重新 normalize。

输出：

```text
generated text
token count
stop reason
sampling parameters
```

</details>
