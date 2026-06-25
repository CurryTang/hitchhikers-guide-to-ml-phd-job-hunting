# Math：快速幂 / Binary Exponentiation

数学题不是靠背公式，而是把一个大问题拆成可重复的小结构。

这个页面从最常见的第一题开始：

```text
LeetCode 50: Pow(x, n)
```

题目要求实现：

```text
pow(x, n) = x^n
```

其中 `n` 可能很大，也可能是负数。

---

## Problem 1：Pow(x, n)

### 题目

给定浮点数 `x` 和整数 `n`，返回 `x^n`。

例子：

```text
Input:  x = 2.00000, n = 10
Output: 1024.00000
```

```text
Input:  x = 2.00000, n = -2
Output: 0.25000
```

因为：

```text
2^-2 = 1 / 2^2 = 1 / 4
```

## 为什么不能暴力乘 n 次

最直接的做法是：

```python
ans = 1
for _ in range(n):
    ans *= x
```

这需要 `O(n)` 次乘法。

如果 `n = 2^31 - 1`，这个做法太慢。

快速幂的目标是把复杂度降到：

```text
O(log n)
```

核心原因是：

```text
x^10 = x^(8 + 2)
```

而 `10` 的二进制是：

```text
10 = 1010₂ = 8 + 2
```

所以我们不需要一个一个乘 `x`，只需要判断哪些二进制位是 `1`。

## 核心直觉

每一轮维护两个变量：

```text
base = 当前这一位代表的幂
res  = 已经选中的幂的乘积
```

从右往左看 `n` 的二进制位。

对 `n = 10 = 1010₂`：

```text
bit 权重: 8 4 2 1
bit 值:  1 0 1 0
```

从最低位开始看：

```text
1 的位置不选
2 的位置选
4 的位置不选
8 的位置选
```

所以：

```text
x^10 = x^2 * x^8
```

这就是为什么代码里有两件事：

```python
if power & 1:
    res *= x
```

最低位是 `1`，说明当前 `x` 这份幂需要乘进答案。

然后：

```python
x *= x
power >>= 1
```

`x *= x` 表示当前幂翻倍：

```text
x^1 -> x^2 -> x^4 -> x^8 -> ...
```

`power >>= 1` 表示把二进制最低位丢掉，继续看下一位。

## 可视化：为什么是 res *= base，然后 base *= base

```pow-demo
```

把 `pow(2, 10)` 展开：

```text
10 = 1010₂
```

我们从右往左读 bit。

| 当前 power | 最低位 | 当前 base | 动作 | res |
|---|---:|---:|---|---:|
| 10 | 0 | 2 | 不乘，base 平方 | 1 |
| 5 | 1 | 4 | 乘进 res | 4 |
| 2 | 0 | 16 | 不乘，base 平方 | 4 |
| 1 | 1 | 256 | 乘进 res | 1024 |

最后：

```text
res = 4 * 256 = 2^2 * 2^8 = 2^10
```

## Iterative Binary Exponentiation

### Intuition

We want to compute `x^n` efficiently, even when `n` is very large.

The brute force approach multiplies `x` repeatedly and takes `O(n)` time.

Binary exponentiation uses the binary representation of `n`.

At each step:

- if the current lowest bit is `1`, multiply the current base into `res`
- square the base
- shift the exponent right by one bit

For negative powers:

```text
x^(-n) = 1 / x^n
```

So we compute with `abs(n)` and return the reciprocal at the end.

### Algorithm

1. If `x == 0`, return `0`.
2. If `n == 0`, return `1`.
3. Set `res = 1`.
4. Set `power = abs(n)`.
5. While `power > 0`:
   - if `power & 1`, do `res *= x`
   - do `x *= x`
   - do `power >>= 1`
6. If `n < 0`, return `1 / res`; otherwise return `res`.

### Code

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0:
            return 0
        if n == 0:
            return 1

        res = 1
        power = abs(n)

        while power:
            if power & 1:
                res *= x
            x *= x
            power >>= 1

        return res if n >= 0 else 1 / res
```

## 复杂度

每轮都会把 `power` 除以 2。

所以循环次数是：

```text
log2(|n|)
```

复杂度：

```text
Time:  O(log n)
Space: O(1)
```

## 常见坑

### 1. 忘记处理负指数

错误写法：

```python
power = n
```

如果 `n < 0`，循环逻辑就不对。

正确做法：

```python
power = abs(n)
return res if n >= 0 else 1 / res
```

### 2. 不理解 `power & 1`

`power & 1` 判断的是当前最低位是不是 `1`。

等价于：

```python
power % 2 == 1
```

但位运算更贴近这个算法的本质：一位一位读二进制。

### 3. 不理解为什么 `x *= x`

每右移一次 `power`，我们就从下一位二进制开始看。

下一位的权重会翻倍：

```text
1 -> 2 -> 4 -> 8 -> 16
```

所以当前 base 也要平方：

```text
x^1 -> x^2 -> x^4 -> x^8 -> x^16
```

### 4. `x == 0` 和负指数

LeetCode 通常不会让你处理 `0` 的负指数这种数学未定义情况。

如果面试官追问：

```text
0^-1 = 1 / 0
```

这是 undefined / division by zero，需要按题目约定处理。

## 一句话记忆

```text
快速幂 = 从右往左读 n 的二进制。
bit 是 1，就把当前 base 乘进 res；
每轮 base 平方，power 右移。
```
