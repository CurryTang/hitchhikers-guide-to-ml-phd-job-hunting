# ML Coding · Unicode, UTF-8 & Pretokenization

对应 CS336 Assignment 1：Section 2.1-2.4。

使用方式：`Lab` 用来建立背景直觉，不算正式 coding 题；`Exercise` 才是需要自己补完整实现的题。先按模板写，再展开参考答案对照边界条件。

## Lab · Unicode Probe

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

观察模板：

```python
def inspect_unicode_codepoint(cp: int) -> dict:
    """
    Input:
        cp: Unicode code point, e.g. 65 or 0x1F600
    Output:
        metadata useful for debugging display vs encoding
    """
    ch = ...       # chr(cp)
    utf8 = ...     # ch.encode("utf-8")
    return {
        "codepoint": ...,
        "character": ...,
        "repr": ...,
        "utf8_bytes": ...,
        "utf8_hex": ...,
        "is_printable": ...,
    }
```

</details>

<details class="solution">
<summary>参考答案</summary>

最小实现：

```python
def inspect_unicode_codepoint(cp: int) -> dict:
    ch = chr(cp)
    return {
        "codepoint": f"U+{cp:04X}",
        "character": ch,
        "repr": repr(ch),
        "utf8_bytes": list(ch.encode("utf-8")),
        "utf8_hex": ch.encode("utf-8").hex(" "),
        "is_printable": ch.isprintable(),
    }
```

Sanity check：

```python
assert ord(chr(65)) == 65
assert inspect_unicode_codepoint(0x41)["utf8_bytes"] == [65]
assert inspect_unicode_codepoint(0x1F600)["utf8_bytes"] == [240, 159, 152, 128]
```

`print(ch)` 是面向人看的显示效果，control character 可能不可见；`repr(ch)` 是调试表示，会把换行、零宽字符这类内容暴露出来。

</details>

## Lab · UTF-8 Encoding

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

观察模板：

```python
def compare_encodings(text: str) -> list[dict]:
    """
    Input:
        text: Python unicode string
    Output:
        rows comparing utf-8 / utf-16 / utf-32 byte length and round-trip
    """
    rows = []
    for enc in ["utf-8", "utf-16", "utf-32"]:
        raw = ...              # text.encode(enc)
        rows.append({
            "encoding": enc,
            "num_chars": ...,
            "num_bytes": ...,
            "bytes": ...,
            "roundtrip": ...,
        })
    return rows

def decode_invalid(raw: bytes, encoding="utf-8") -> dict:
    """
    Input:
        raw byte sequence, possibly malformed
    Output:
        strict / replace / ignore behavior
    """
    return {
        "strict": ...,
        "replace": ...,
        "ignore": ...,
    }
```

</details>

<details class="solution">
<summary>参考答案</summary>

实验函数：

```python
def compare_encodings(text: str):
    rows = []
    for enc in ["utf-8", "utf-16", "utf-32"]:
        raw = text.encode(enc)
        rows.append({
            "encoding": enc,
            "num_chars": len(text),
            "num_bytes": len(raw),
            "bytes": raw,
            "roundtrip": raw.decode(enc),
        })
    return rows

def decode_invalid(raw: bytes, encoding="utf-8"):
    return {
        "strict": _try_decode(raw, encoding, "strict"),
        "replace": raw.decode(encoding, errors="replace"),
        "ignore": raw.decode(encoding, errors="ignore"),
    }

def _try_decode(raw, encoding, errors):
    try:
        return raw.decode(encoding, errors=errors)
    except UnicodeDecodeError as exc:
        return type(exc).__name__
```

结论：

- ASCII 在 UTF-8 下通常是 1 char = 1 byte。
- CJK 和 emoji 在 UTF-8 下是多 byte；`len(text)` 和 `len(text.encode("utf-8"))` 不是一回事。
- UTF-16 / UTF-32 通常有 BOM 或固定宽度开销，短英文不一定更省。
- invalid byte sequence 应该显式决定用 `strict`、`replace` 还是 `ignore`，tokenizer/debug 工具里不要默默吞掉错误。

</details>

## Exercise 1 · GPT-2 Style Pretokenizer

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

解题模板：

```python
def split_by_special(text: str, special_tokens: list[str]):
    """
    Input:
        raw text and special tokens
    Output:
        iterator of (is_special, segment)
    """
    ...

def pretoken_counts(text: str, special_tokens: list[str] | None = None) -> dict:
    """
    Input:
        raw text
    Output:
        Counter mapping tuple[bytes, ...] to frequency
    """
    counts = {}
    for is_special, part in split_by_special(text, special_tokens or []):
        if is_special:
            ...                 # skip statistics
        else:
            for token in ...:    # regex matches
                pieces = ...     # tuple(bytes([b]) for b in token.encode("utf-8"))
                counts[pieces] = counts.get(pieces, 0) + 1
    return counts
```

</details>

<details class="solution">
<summary>参考答案</summary>

核心实现可以分三层：先按 special token 切边界，再对普通片段跑 GPT-2 regex，最后把每个 pre-token 转成 byte tuple 并累计频率。

```python
from collections import Counter
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_by_special(text, special_tokens):
    if not special_tokens:
        yield False, text
        return

    pattern = "(" + "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)) + ")"
    for part in re.split(pattern, text):
        if not part:
            continue
        yield part in special_tokens, part

def pretoken_counts(text, special_tokens=None):
    counts = Counter()
    for is_special, part in split_by_special(text, special_tokens or []):
        if is_special:
            continue
        for match in re.finditer(PAT, part):
            token_bytes = match.group(0).encode("utf-8")
            counts[tuple(bytes([b]) for b in token_bytes)] += 1
    return counts
```

容易错的点是 special token 不能贡献 pair statistics，也不能让左右两边的普通文本跨边界合并。这里把 token 表示成 `tuple[bytes, ...]`，后面的 BPE merge 才能直接把相邻 `bytes` 拼成更长的 `bytes`。

</details>

## Debug Checklist

<details class="exercise">
<summary><span class="q-label">Debug</span> <span class="q-text">展开检查项</span></summary>

- byte-level tokenizer 的基础单位是 `bytes`，不是 Python `str`。
- 单个 byte 也表示成 `bytes` object，例如 `b'a'`。
- special token 在训练中是边界，在 encode 中是整体 token。

</details>
