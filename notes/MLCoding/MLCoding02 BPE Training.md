# ML Coding · BPE Training

对应 CS336 Assignment 1：Section 2.4-2.5。

## Exercise 1 · Toy BPE Merge Simulator

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`bpe_example`

目标：在小语料上实现朴素 BPE merge loop。

输入：

```text
pretoken_counts: dict[tuple[bytes, ...], int]
num_merges: int
```

输出：

```text
merges: list[tuple[bytes, bytes]]
updated pretoken representation
```

关键约束：

- pair frequency 要乘以 pre-token frequency。
- 只统计相邻 pair。
- 频率相同时选 lexicographically greater pair。
- merge 替换 non-overlapping adjacent occurrences。

PDF toy corpus：

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

目标：前几轮 merge 能复现 handout 结果。

</details>

## Exercise 2 · Full BPE Trainer

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`train_bpe`

接口：

```text
train_bpe(input_path, vocab_size, special_tokens)
```

返回：

```text
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

关键约束：

- 初始 byte vocabulary 覆盖 0..255。
- special tokens 加入 vocabulary。
- special tokens 是 merge hard boundary。
- special tokens 不参与 pair statistics。
- `vocab_size` 包括 byte vocab、merge vocab、special tokens。
- merges 按创建顺序返回。

测试：

```bash
uv run pytest tests/test_train_bpe.py
```

</details>

## Exercise 3 · BPE Performance Pass

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

目标：把 naive trainer 优化到能处理 TinyStories。

优化顺序：

```text
1. profile pre-tokenization
2. split corpus by special-token boundary
3. parallelize pre-tokenization
4. cache pair counts
5. update only pairs affected by last merge
```

记录：

```text
wall-clock time
peak memory
top bottleneck
speedup after each optimization
```

</details>

## Exercise 4 · Train TinyStories Tokenizer

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`train_bpe_tinystories`

配置：

```text
dataset: TinyStories
vocab_size: 10_000
special token: <|endoftext|>
```

输出：

```text
serialized vocab
serialized merges
training time
memory usage
longest token
profile bottleneck
```

</details>

## Exercise 5 · Train OpenWebText Tokenizer

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`train_bpe_expts_owt`

配置：

```text
dataset: OpenWebText sample
vocab_size: 32_000
```

比较：

| 维度 | TinyStories | OpenWebText |
|---|---|---|
| domain | children stories | web text |
| vocabulary diversity | lower | higher |
| long tokens | simple words / names | URLs, markup, rare strings |
| compression | easier | more variable |

</details>

## Common Failure Modes

<details class="exercise">
<summary><span class="q-label">Pitfalls</span> <span class="q-text">展开常见错误</span></summary>

- special token 被拆开参与 merge。
- merge 跨 pre-token boundary。
- pair tie-breaking 不 deterministic。
- 单个 byte 用 int 表示，导致 vocab/merge 类型对不上。
- 每轮从头扫描全 corpus，训练速度退化。

</details>
