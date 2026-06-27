# ML Coding · Tokenizer Runtime

对应 CS336 Assignment 1：Section 2.6-2.7。

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

</details>

## Exercise 2 · BPE Encoding Trace

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

</details>

## Exercise 3 · Streaming Encode

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

</details>

## Exercise 4 · Compression Ratio Experiment

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

</details>

## Exercise 5 · Token ID Serialization

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

</details>

## Common Failure Modes

<details class="exercise">
<summary><span class="q-label">Pitfalls</span> <span class="q-text">展开常见错误</span></summary>

- decode 时逐 token decode，而不是先拼 bytes 再 decode。
- special token 没有追加进 vocab。
- `encode_iterable` silently materializes whole file。
- streaming chunk 改变 tokenization。

</details>
