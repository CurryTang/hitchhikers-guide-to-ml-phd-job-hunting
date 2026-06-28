# ML Coding · BPE Training

对应 CS336 Assignment 1：Section 2.4-2.5。

使用方式：每题先看目标和验收标准，再按“解题模板”把 TODO 补完整；最后展开参考答案，对照边界条件、sanity checks 和实现细节。

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

解题模板：

```python
from collections import Counter

def count_pairs(pretoken_counts: dict[tuple[bytes, ...], int]) -> Counter:
    """Output: Counter mapping adjacent byte-pair to weighted frequency."""
    pair_counts = Counter()
    for pieces, freq in pretoken_counts.items():
        ...  # add freq for every adjacent pair
    return pair_counts

def merge_one_pretoken(pieces: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Output: pieces after non-overlapping replacement of pair."""
    out = []
    i = 0
    while i < len(pieces):
        ...  # if pieces[i:i+2] == pair, append merged bytes and skip 2
    return tuple(out)

def run_bpe_merges(pretoken_counts, num_merges):
    """Output: (merges, updated_counts)."""
    merges = []
    counts = dict(pretoken_counts)
    for _ in range(num_merges):
        pair_counts = ...
        winner = ...  # max by (frequency, pair)
        merges.append(winner)
        counts = ...  # rebuild counts after merge
    return merges, counts
```

</details>

<details class="solution">
<summary>参考答案</summary>

朴素版本的核心是两步：统计 pair 频率，然后把 winner pair 在每个 pre-token 内做 non-overlapping 替换。

```python
from collections import Counter

def count_pairs(pretoken_counts):
    pair_counts = Counter()
    for pieces, freq in pretoken_counts.items():
        for a, b in zip(pieces, pieces[1:]):
            pair_counts[(a, b)] += freq
    return pair_counts

def merge_one_pretoken(pieces, pair):
    out = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
            out.append(pieces[i] + pieces[i + 1])
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return tuple(out)

def run_bpe_merges(pretoken_counts, num_merges):
    counts = dict(pretoken_counts)
    merges = []
    for _ in range(num_merges):
        pair_counts = count_pairs(counts)
        if not pair_counts:
            break
        winner = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(winner)
        next_counts = Counter()
        for pieces, freq in counts.items():
            next_counts[merge_one_pretoken(pieces, winner)] += freq
        counts = dict(next_counts)
    return merges, counts
```

tie-breaking 用 `(frequency, pair)`，因为 assignment 要求频率相同时选 lexicographically greater pair。不要用 `Counter.most_common(1)`，它的 tie 取决于插入顺序。

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

解题模板：

```python
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Input:
        raw corpus path, target vocab size, special tokens
    Output:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    vocab = {i: bytes([i]) for i in range(256)}
    ...  # append special tokens to vocab

    pretoken_counts = ...  # build counts with special-token boundaries
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = ...
        if not pair_counts:
            break
        pair = ...          # deterministic winner
        merges.append(pair)
        vocab[len(vocab)] = ...
        pretoken_counts = ...

    return vocab, merges
```

</details>

<details class="solution">
<summary>参考答案</summary>

推荐结构：

```python
def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    pretoken_counts = build_pretoken_counts(input_path, special_tokens)
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(pretoken_counts)
        if not pair_counts:
            break

        pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged = pair[0] + pair[1]
        merges.append(pair)
        vocab[len(vocab)] = merged

        updated = Counter()
        for pieces, freq in pretoken_counts.items():
            updated[merge_one_pretoken(pieces, pair)] += freq
        pretoken_counts = updated

    return vocab, merges
```

`build_pretoken_counts` 要复用上一章的规则：special token 是 hard boundary，不进入统计；普通 pre-token 内才拆成 byte pieces。`vocab_size` 包括 256 个 byte token、special tokens 和 merge 产生的新 token，所以循环条件用 `len(vocab) < vocab_size`。

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

解题模板：

```python
def profile_stage(name: str, fn, *args, **kwargs):
    """Run one stage and return result + timing/memory metadata."""
    ...

def build_pair_index(pretoken_counts):
    """
    Output:
        pair_counts: pair -> frequency
        pair_to_pretokens: pair -> affected pre-token set
    """
    ...

def update_after_merge(pretoken_counts, pair_counts, pair_to_pretokens, winner):
    """
    Only update pre-tokens containing winner.
    """
    affected = ...
    for old_pieces in affected:
        ...  # remove old pair contributions
        new_pieces = ...
        ...  # add new pair contributions
    return pretoken_counts, pair_counts, pair_to_pretokens

def benchmark_bpe_trainer(input_path):
    rows = []
    for version in ["naive", "parallel_pretok", "incremental_pairs"]:
        ...  # record time, memory, speedup
    return rows
```

</details>

<details class="solution">
<summary>参考答案</summary>

优化版本不应该每一轮完整扫描语料。比较稳的工程路线：

```text
1. 先把 raw corpus 切成 document chunks，special token 只作为边界。
2. 并行 pretokenization，得到 Counter[tuple[bytes, ...]]。
3. 建 pair -> set[pretoken] 的倒排索引。
4. 每轮选 winner pair。
5. 只更新包含 winner pair 的 pre-token。
6. 对这些 pre-token 的旧 pairs 做 decrement，新 pairs 做 increment。
```

伪代码：

```python
pair_counts = build_pair_counts(pretoken_counts)
pair_to_pretokens = build_inverted_index(pretoken_counts)

for _ in range(num_merges):
    pair = argmax_pair(pair_counts)
    affected = list(pair_to_pretokens[pair])

    for old_pieces in affected:
        freq = pretoken_counts.pop(old_pieces, 0)
        if freq == 0:
            continue
        decrement_pairs(old_pieces, freq, pair_counts, pair_to_pretokens)
        new_pieces = merge_one_pretoken(old_pieces, pair)
        pretoken_counts[new_pieces] += freq
        increment_pairs(new_pieces, freq, pair_counts, pair_to_pretokens)
```

验收时不要只报总时间，要报每一步优化后的 wall-clock 和 peak memory。BPE trainer 最常见瓶颈通常是 regex pretokenization 和每轮全量 pair recount。

</details>

## Experiment · Train TinyStories Tokenizer

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

实验模板：

```python
def train_tinystories_tokenizer(data_path: str, out_dir: str):
    """
    Output artifacts:
        vocab.json
        merges.txt
        report dict
    """
    vocab, merges = train_bpe(
        input_path=data_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    ...  # serialize vocab and merges
    report = {
        "dataset": "TinyStories",
        "vocab_size": 10_000,
        "num_merges": ...,
        "train_time_sec": ...,
        "peak_rss_mb": ...,
        "longest_tokens": ...,
    }
    return report
```

</details>

<details class="solution">
<summary>参考答案</summary>

一个可复现的输出模板：

```text
dataset: TinyStories
vocab_size: 10000
special_tokens: ["<|endoftext|>"]
num_merges: 10000 - 256 - 1 = 9743
artifacts:
  vocab.json
  merges.txt
metrics:
  train_time_sec
  peak_rss_mb
  pretoken_count
  longest_token_bytes
  top_20_longest_tokens
```

判断 tokenizer 是否合理：

- 高频英文词、空格前缀词、常见后缀会变成较长 token。
- `<|endoftext|>` 必须作为单独 token 存在。
- TinyStories domain 简单，长 token 多数应该是儿童故事里常见词和名字，而不是乱码。
- 如果 longest token 是跨文档拼出来的，很可能 special-token boundary 处理错了。

</details>

## Experiment · Train OpenWebText Tokenizer

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

实验模板：

```python
def train_owt_tokenizer(data_path: str, out_dir: str):
    """
    Train a 32K OpenWebText tokenizer and return artifacts + diagnostics.
    """
    vocab, merges = train_bpe(
        input_path=data_path,
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
    )
    return {
        "dataset": "OpenWebText",
        "vocab_size": 32_000,
        "num_merges": ...,
        "bytes_per_token": ...,
        "tokens_per_second": ...,
        "longest_tokens": ...,
        "most_frequent_tokens": ...,
    }

def compare_tokenizers(tinystories_tok, owt_tok, tinystories_sample, owt_sample):
    """
    Output:
        table comparing bytes/token across tokenizer/data pairs.
    """
    ...
```

</details>

<details class="solution">
<summary>参考答案</summary>

OpenWebText 的答案重点不是“跑出一个固定表格”，而是解释为什么结果和 TinyStories 不一样：

```text
dataset: OpenWebText sample
vocab_size: 32000
expected differences:
  more URL / markup / code-like fragments
  more rare names and non-English text
  heavier long tail in pre-token distribution
  slower pretokenization and pair update
```

报告时至少放：

```text
bytes/token on OWT validation sample
tokens/sec encode throughput
top longest tokens
top most frequent tokens
TinyStories tokenizer on OWT vs OWT tokenizer on OWT
```

典型结论：OWT tokenizer 在 web text 上 compression 更好，但 vocabulary 会被 URL、HTML、符号串和多语种内容占掉一部分；TinyStories tokenizer 更干净，但迁移到 OWT 时 tokens/byte 会变差。

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
