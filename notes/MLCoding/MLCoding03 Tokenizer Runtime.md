# ML Coding · Tokenizer Runtime

对应 CS336 Assignment 1：Section 2.6-2.7。

使用方式：每题先看目标和验收标准，再按“解题模板”把 TODO 补完整；最后展开参考答案，对照边界条件、sanity checks 和实现细节。

## Exercise 1 · Tokenizer Class

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`tokenizer`

接口：

```text
Tokenizer(vocab, merges, special_tokens=None)
Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=None)
encode(text: str) -> list[int]
encode_iterable(iterable: Iterable[str]) -> Iterator[int]
decode(ids: list[int]) -> str
```

关键约束：

- special tokens encode 时保持整体。
- 普通文本先 pre-tokenize，再在每个 pre-token 内应用 merges。
- merges 必须按训练时创建顺序应用。
- decode 是 token bytes concatenate 后 UTF-8 decode。
- malformed bytes 使用 Unicode replacement character。
- `encode_iterable` 需要 memory-efficient。

测试：

```bash
uv run pytest tests/test_tokenizer.py
```

解题模板：

```python
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = ...
        self.inverse_vocab = ...
        self.merge_rank = ...
        self.special_tokens = ...
        ...  # add special tokens to vocab if missing

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = ...     # load int -> bytes
        merges = ...    # load pair order
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids = []
        ...  # split special tokens, pretokenize ordinary text, apply BPE
        return ids

    def encode_iterable(self, iterable):
        for chunk in iterable:
            yield from ...

    def decode(self, ids: list[int]) -> str:
        raw = ...       # concatenate token bytes
        return ...      # utf-8 decode with replacement
```

</details>

<details class="solution">
<summary>参考答案</summary>

这一题要实现的是一个可用的 BPE tokenizer runtime：训练阶段已经给了 `vocab` 和 `merges`，runtime 负责把字符串稳定地映射成 token ids，并能从 ids 还原文本。

一个清晰实现可以按三层拆：

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)                  # int -> bytes
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        next_id = max(self.vocab) + 1 if self.vocab else 0
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.inverse_vocab:
                self.vocab[next_id] = b
                self.inverse_vocab[b] = next_id
                next_id += 1

        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(i): bytes(v) for i, v in raw_vocab.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                a, b = line.rstrip("\n").split(" ")
                merges.append((a.encode("latin1"), b.encode("latin1")))
        return cls(vocab, merges, special_tokens)
```

encoding 的核心是：先保护 special token 边界，再对普通片段 pre-tokenize，最后在每个 pre-token 内做 BPE merge。

```python
def apply_bpe(self, token_bytes: bytes) -> list[bytes]:
    pieces = [bytes([b]) for b in token_bytes]
    while len(pieces) >= 2:
        pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
        best = min(
            pairs,
            key=lambda p: self.merge_rank.get(p, float("inf")),
        )
        if best not in self.merge_rank:
            break

        out = []
        i = 0
        while i < len(pieces):
            if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == best:
                out.append(pieces[i] + pieces[i + 1])
                i += 2
            else:
                out.append(pieces[i])
                i += 1
        pieces = out
    return pieces
```

`decode` 不要逐 token decode。正确做法是先拼 bytes，再整体 UTF-8 decode：

```python
def decode(self, ids: list[int]) -> str:
    raw = b"".join(self.vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")
```

验收重点：

- `decode(encode(text)) == text` 对合法 UTF-8 文本成立。
- special token 不会被 regex 拆碎。
- merge 只在同一个 pre-token 内发生，不能跨 pre-token 合并。
- malformed byte sequence decode 时用 replacement character，不抛异常。

</details>

## Trace Lab · BPE Encoding Trace

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

用一个小 vocab 和 merges 手动 trace：

```text
input: "the cat ate"
pre-tokenized: ["the", " cat", " ate"]
```

练习目标：

- 打印每个 pre-token 的 byte pieces。
- 每次 merge 后打印 pieces。
- 最后映射到 token ids。

检查点：

```text
merge order changes output
merge only applies inside one pre-token
unknown text still encodable because byte vocab is complete
```

Trace 模板：

```python
def trace_bpe_token(token: str, tokenizer: Tokenizer) -> list[int]:
    """
    Input:
        one pre-token string, not a full document
    Output:
        final token ids, while printing each merge step
    """
    pieces = ...        # byte pieces
    print("start:", pieces)
    while True:
        candidate_pairs = ...
        best_pair = ... # lowest merge rank among current adjacent pairs
        if best_pair is None:
            break
        pieces = ...    # merge best_pair
        print(...)
    ids = ...
    return ids
```

</details>

<details class="solution">
<summary>参考答案</summary>

这一题要训练你看懂 tokenizer 的中间状态。debug BPE 时，最终 token ids 往往看不出问题，必须打印每轮 merge 后的 pieces。

一个 trace helper 可以这样写：

```python
def trace_bpe_token(token: str, tokenizer):
    pieces = [bytes([b]) for b in token.encode("utf-8")]
    print("start:", pieces)

    step = 0
    while len(pieces) >= 2:
        pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
        ranked = [
            (tokenizer.merge_rank[p], p)
            for p in pairs
            if p in tokenizer.merge_rank
        ]
        if not ranked:
            break

        _, pair = min(ranked)
        new_pieces = []
        i = 0
        while i < len(pieces):
            if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
                new_pieces.append(pieces[i] + pieces[i + 1])
                i += 2
            else:
                new_pieces.append(pieces[i])
                i += 1

        step += 1
        print(f"step {step}: merge {pair} -> {new_pieces}")
        pieces = new_pieces

    ids = [tokenizer.inverse_vocab[p] for p in pieces]
    print("ids:", ids)
    return ids
```

对 `"the cat ate"`，要先看到 pre-token 边界：

```text
"the"
" cat"
" ate"
```

如果某个 merge 规则能合并 `"e"` 和 `" "`，也不能跨 `"the"` 与 `" cat"` 的边界合并。BPE 的 merge scope 是 pre-token 内部，不是整段字符串。

最常见 bug：

- 把所有 bytes 拼成一整个序列再 merge，导致跨空格/跨 special token 合并。
- merge tie-break 或 merge rank 错，输出 ids 和 reference 差一点。
- vocab 里缺 byte fallback，导致陌生字符无法编码。

</details>

## Exercise 2 · Streaming Encode

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

目标：对不能一次放入内存的大文件做 lazy tokenization。

输入：

```text
Iterable[str]
```

输出：

```text
Iterator[int]
```

关键问题：

- chunk boundary 不能改变 tokenization。
- special token / document boundary 是天然 safe split。
- 如果随意按字符数切 chunk，可能改变 pre-token 和 merge 结果。

解题模板：

```python
def encode_iterable(self, iterable):
    """
    Input:
        iterable of safe-boundary text chunks
    Output:
        lazy iterator of token ids
    """
    for chunk in iterable:
        for token_id in self.encode(chunk):
            yield token_id

def encode_documents(self, docs):
    """
    Same idea, but docs are explicit tokenizer-safe boundaries.
    """
    for doc in docs:
        yield from self.encode(doc)
```

</details>

<details class="solution">
<summary>参考答案</summary>

streaming encode 的目标是处理超大文本文件时不把全文读进内存。最安全的版本是让调用者按 document 或 line 提供 iterable，并且这些边界本身就是允许切开的语义边界。

```python
def encode_iterable(self, iterable):
    for chunk in iterable:
        yield from self.encode(chunk)
```

这个版本简单但有一个前提：`chunk` 边界不会改变 tokenization。比如每个 chunk 是一篇 document，或者上游已经按 special token 分隔：

```text
doc1 <|endoftext|> doc2 <|endoftext|> doc3
```

如果你随意按固定字符数切：

```text
"intern" + "ational"
```

可能本来能 merge 成长 token 的词被切断，最终 token ids 改变。所以更严谨的 streaming 策略有两个：

1. 按 document/special token 边界切，直接 `yield from encode(doc)`。
2. 若只能按 byte chunk 读，需要保留一个 overlap buffer，并只在确认的 safe boundary 之前产出 token。

在 CS336 assignment 的实现里，通常按 iterable 文本块处理已经足够；重点是不要写成：

```python
def encode_iterable(iterable):
    return self.encode("".join(iterable))  # 错：materialize 全文件
```

验收方式：

```python
ids_stream = list(tok.encode_iterable(docs))
ids_joined = tok.encode("".join(docs))
```

只有当 `docs` 的边界是 safe boundary 时，二者才应该完全一致。这个条件要在实验说明里写清楚。

</details>

## Experiment · Compression Ratio

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`tokenizer_experiments`

指标：

```text
bytes / token
tokens / second
bytes / second
estimated time for 825GB corpus
```

实验矩阵：

| data | tokenizer | report |
|---|---|---|
| TinyStories sample | TinyStories 10K | bytes/token |
| TinyStories sample | OWT 32K | bytes/token |
| OWT sample | TinyStories 10K | bytes/token |
| OWT sample | OWT 32K | bytes/token |

实验模板：

```python
def tokenizer_report(tokenizer: Tokenizer, text: str) -> dict:
    """
    Output:
        compression and throughput metrics
    """
    raw = text.encode("utf-8")
    start = ...
    ids = tokenizer.encode(text)
    elapsed = ...
    return {
        "num_bytes": ...,
        "num_tokens": ...,
        "bytes_per_token": ...,
        "tokens_per_second": ...,
        "bytes_per_second": ...,
    }

def compare_tokenizer_matrix(samples: dict, tokenizers: dict) -> list[dict]:
    rows = []
    for sample_name, text in samples.items():
        for tok_name, tok in tokenizers.items():
            rows.append({...})
    return rows
```

</details>

<details class="solution">
<summary>参考答案</summary>

这一节不是实现新算法，而是评估 tokenizer 是否适合某个数据分布。推荐写一个统一 benchmark 函数：

```python
import time

def tokenizer_report(tokenizer, text: str) -> dict:
    raw = text.encode("utf-8")
    start = time.perf_counter()
    ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - start

    return {
        "num_bytes": len(raw),
        "num_tokens": len(ids),
        "bytes_per_token": len(raw) / max(len(ids), 1),
        "tokens_per_second": len(ids) / max(elapsed, 1e-12),
        "bytes_per_second": len(raw) / max(elapsed, 1e-12),
    }
```

解释结果时要结合语料：

| 现象 | 解释 |
|---|---|
| TinyStories tokenizer 在 TinyStories 上 bytes/token 高 | 训练域匹配，常见儿童故事词被 merge 成较长 token |
| OWT tokenizer 在 OWT 上更稳 | OWT 词汇、符号、网页噪声更多，32K vocab 覆盖更好 |
| 跨域 bytes/token 变差 | merge 学到的是训练语料的局部统计，不是通用压缩真理 |
| tokens/s 不只由 vocab 决定 | regex pretokenization、merge 数据结构和 Python overhead 都会影响速度 |

估算 825GB 语料耗时：

```python
seconds = 825 * 1024**3 / report["bytes_per_second"]
hours = seconds / 3600
```

报告里至少写三件事：

1. 压缩率：`bytes/token`。
2. 吞吐：`tokens/s` 和 `bytes/s`。
3. 迁移性：同一个 tokenizer 在不同数据域上为什么变好或变差。

</details>

## Exercise 3 · Token ID Serialization

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

目标：把 train/dev text 编码成 token id array。

输出：

```text
np.ndarray dtype uint16
```

为什么 `uint16` 合适：

```text
vocab_size <= 65,536
uint16 halves storage compared with int32
token ids are non-negative integers
```

解题模板：

```python
def encode_to_array(tokenizer: Tokenizer, texts, out_path: str, dtype="uint16") -> dict:
    """
    Input:
        iterable of text documents
    Output:
        saved numpy token-id array and metadata
    """
    ids = []
    for text in texts:
        ids.extend(...)
    arr = ...
    ...  # validate max id fits dtype
    ...  # save arr
    return {
        "path": out_path,
        "num_tokens": ...,
        "dtype": ...,
        "max_token_id": ...,
    }
```

</details>

<details class="solution">
<summary>参考答案</summary>

训练前通常会把文本预编码成 token id array，避免每个 training step 都跑 tokenizer。基本实现：

```python
import numpy as np

def encode_to_uint16(tokenizer, texts, out_path):
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text))

    arr = np.asarray(ids, dtype=np.uint16)
    np.save(out_path, arr)
    return {
        "num_tokens": int(arr.shape[0]),
        "dtype": str(arr.dtype),
        "path": str(out_path),
    }
```

如果数据很大，不要先攒 Python list，可以分 shard 写：

```python
def write_token_shards(tokenizer, docs, shard_size, prefix):
    buf = []
    shard_id = 0
    for doc in docs:
        buf.extend(tokenizer.encode(doc))
        if len(buf) >= shard_size:
            np.save(f"{prefix}-{shard_id:05d}.npy", np.asarray(buf, dtype=np.uint16))
            buf.clear()
            shard_id += 1
    if buf:
        np.save(f"{prefix}-{shard_id:05d}.npy", np.asarray(buf, dtype=np.uint16))
```

`uint16` 的前提是 vocab size 不超过 65536。若 vocab size 是 100K，就必须用 `uint32` 或更大的 dtype。训练 dataloader 读取时可以用：

```python
tokens = np.load(path, mmap_mode="r")
```

这样不会一次把整个 array 读进内存。

检查点：

- `arr.min() >= 0`
- `arr.max() < vocab_size`
- dtype 与 vocab size 匹配
- 随机抽样一段 ids decode 后能看到合理文本

</details>

## Common Failure Modes

<details class="exercise">
<summary><span class="q-label">Pitfalls</span> <span class="q-text">展开常见错误</span></summary>

- decode 时逐 token decode，而不是先拼 bytes 再 decode。
- special token 没有追加进 vocab。
- `encode_iterable` silently materializes whole file。
- streaming chunk 改变 tokenization。

</details>
