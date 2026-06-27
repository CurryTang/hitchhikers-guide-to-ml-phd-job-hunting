# ML Coding · Unicode, UTF-8 & Pretokenization

对应 CS336 Assignment 1：Section 2.1-2.4。

## Exercise 1 · Unicode Probe

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`unicode1`

实现一个小工具观察 code point：

```text
inspect_unicode_codepoint(cp: int)
```

返回：

```text
character
repr(character)
UTF-8 bytes
visible printing behavior
```

检查点：

- `ord` / `chr` 是 code point 层面的互逆。
- `repr(x)` 和 `print(x)` 显示目的不同。
- control character 可能存在但不可见。

</details>

## Exercise 2 · UTF-8 Encoding Lab

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`unicode2`

比较 UTF-8、UTF-16、UTF-32 对不同文本的编码：

```text
English
CJK
emoji
mixed text
invalid byte sequence
```

输出：

```text
encoded bytes
byte length
round-trip result
decode error behavior
```

Sanity cases：

```text
"hello" round-trip succeeds
"こんにちは" byte length > character count
single-byte decoding a multi-byte character is wrong
invalid two-byte sequence raises or replaces
```

</details>

## Exercise 3 · GPT-2 Style Pretokenizer

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：pre-tokenization in Section 2.4。

目标：用 assignment regex 把 raw text 切成 pre-token，并转成 byte tuples。

输入：

```text
raw text
special_tokens: list[str]
```

输出：

```text
dict[tuple[bytes, ...], int]
```

关键约束：

- special token 是 hard boundary。
- special token 不贡献 merge statistics。
- 不跨 pre-token boundary 统计 byte pair。
- 用 iterator 风格处理，避免 materialize 巨大 token list。

Sanity cases：

```text
"some text that i'll pre-tokenize" matches PDF regex example
"Doc1<|endoftext|>Doc2" never merges across document boundary
repeated pre-token count accumulates frequency
```

</details>

## Debug Checklist

<details class="exercise">
<summary><span class="q-label">Debug</span> <span class="q-text">展开检查项</span></summary>

- byte-level tokenizer 的基础单位是 `bytes`，不是 Python `str`。
- 单个 byte 也表示成 `bytes` object，例如 `b'a'`。
- special token 在训练中是边界，在 encode 中是整体 token。

</details>
