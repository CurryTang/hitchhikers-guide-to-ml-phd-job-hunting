# Rejection Sampling：用 randM 实现 randN

## 面试目标

这类题的代表是 LeetCode 470：`Implement Rand10() Using Rand7()`。

题目：

```text
已知 rand7() 可以等概率返回 1..7。
实现 rand10()，要求等概率返回 1..10。
```

核心方法是 rejection sampling，中文一般叫拒绝采样。

要能讲清楚三件事：

- 多次调用 `randM()` 可以构造一个更大的均匀整数空间。
- 不能直接 `% N`，除非空间大小能被 `N` 整除。
- 截取能被 `N` 整除的最大前缀，剩下的样本丢掉重采样。

## 标准解：rand7 实现 rand10

调用两次 `rand7()`：

```python
x = (rand7() - 1) * 7 + rand7()
```

这会等概率生成 `1..49`。

为什么是均匀的？

第一次 `rand7()` 决定行，第二次 `rand7()` 决定列：

```text
7 x 7 = 49 个格子
```

每个格子的概率都是：

```text
1/7 * 1/7 = 1/49
```

所以 `x` 在 `1..49` 上均匀。

但 `49` 不能被 `10` 整除。如果直接：

```python
return x % 10
```

会导致有些结果出现 5 次，有些结果出现 4 次，概率不均匀。

因此只保留 `1..40`：

```text
40 能被 10 整除
1..40 里每个 rand10 结果刚好出现 4 次
41..49 丢掉，重新采样
```

代码：

```python
class Solution:
    def rand10(self):
        while True:
            x = (rand7() - 1) * 7 + rand7()  # uniform 1..49
            if x <= 40:
                return (x - 1) % 10 + 1
```

注意返回值写成：

```python
(x - 1) % 10 + 1
```

这样结果是 `1..10`，不是 `0..9`。

## 为什么不能直接取模

假设我们直接把 `1..49` 映射到 `1..10`。

```text
1, 11, 21, 31, 41 -> 1
2, 12, 22, 32, 42 -> 2
...
9, 19, 29, 39, 49 -> 9
10, 20, 30, 40    -> 10
```

前 9 个结果各出现 5 次，结果 `10` 只出现 4 次。

概率分别是：

```text
P(1..9) = 5/49
P(10)  = 4/49
```

这不是等概率。

拒绝采样的本质就是：只在概率能平均分配的区域里取结果。

## 通用模板：randM 实现 randN

如果有：

```text
randM() -> uniform 1..M
```

想实现：

```text
randN() -> uniform 1..N
```

套路：

1. 用 `k` 次 `randM()` 构造一个足够大的均匀空间 `1..M^k`。
2. 找到 `limit = floor(M^k / N) * N`。
3. 如果采样值 `x <= limit`，返回 `(x - 1) % N + 1`。
4. 否则拒绝，重新采样。

代码模板：

```python
def randN():
    while True:
        x = 1
        for _ in range(k):
            x = (x - 1) * M + randM()

        limit = (M ** k // N) * N
        if x <= limit:
            return (x - 1) % N + 1
```

这里的 `k` 要选到：

```text
M^k >= N
```

但不是越大越好。`k` 越大，每轮调用 `randM()` 越多；`k` 太小，拒绝概率可能很高或者根本不够覆盖 `N`。

## 怎么选择 k

最简单规则：

```text
选最小的 k，使得 M^k >= N
```

例如 `rand7 -> rand10`：

```text
7^1 = 7  < 10
7^2 = 49 >= 10
```

所以用两次 `rand7()`。

拒绝概率：

```text
usable = floor(49 / 10) * 10 = 40
reject = 9 / 49
accept = 40 / 49
```

每次成功平均需要：

```text
1 / accept = 49 / 40 rounds
```

每轮调用两次 `rand7()`，所以期望调用次数：

```text
2 * 49 / 40 = 2.45
```

## 变形一：优化 rand10，复用被拒绝的随机性

标准解丢掉 `41..49`，其实这 9 个数本身仍然是均匀的。可以把它们重新映射成 `1..9`，再乘一个新的 `rand7()`，构造 `1..63`。

代码：

```python
class Solution:
    def rand10(self):
        while True:
            x = (rand7() - 1) * 7 + rand7()  # 1..49
            if x <= 40:
                return (x - 1) % 10 + 1

            x = (x - 40 - 1) * 7 + rand7()  # 1..63
            if x <= 60:
                return (x - 1) % 10 + 1

            x = (x - 60 - 1) * 7 + rand7()  # 1..21
            if x <= 20:
                return (x - 1) % 10 + 1
```

为什么这还是正确的？

- `41..49` 一共有 9 个等概率结果。
- 把它们映射成 `1..9` 后仍然均匀。
- 再乘一次 `rand7()`，得到 `9 * 7 = 63` 个等概率结果。
- `60` 能被 `10` 整除，所以可以取 `1..60`。

这版减少了浪费，但代码更复杂。面试中先写标准解，再提这个优化即可。

## 变形二：rand5 实现 rand7

`5^1 = 5 < 7`，所以至少调用两次：

```text
5^2 = 25
limit = floor(25 / 7) * 7 = 21
```

代码：

```python
def rand7():
    while True:
        x = (rand5() - 1) * 5 + rand5()  # 1..25
        if x <= 21:
            return (x - 1) % 7 + 1
```

## 变形三：rand2 实现 rand3

两次 `rand2()` 可以生成 `1..4`：

```text
2^2 = 4
limit = floor(4 / 3) * 3 = 3
```

代码：

```python
def rand3():
    while True:
        x = (rand2() - 1) * 2 + rand2()  # 1..4
        if x <= 3:
            return x
```

这里 `x <= 3` 时直接返回 `x`，因为目标就是 `1..3`。

## 变形四：rand10 实现 rand7

如果 `M >= N`，一次调用可能就够。

`rand10()` 生成 `1..10`，要实现 `rand7()`：

```text
limit = floor(10 / 7) * 7 = 7
```

代码：

```python
def rand7():
    while True:
        x = rand10()
        if x <= 7:
            return x
```

这比 `rand7 -> rand10` 更直接，因为原始空间已经大于目标空间。

## 变形五：用 randM 实现指定范围 [a, b]

如果要生成 `[a, b]`：

```text
N = b - a + 1
```

先实现 `randN()`，再平移：

```python
return randN() + a - 1
```

例如想生成 `5..14`，就是生成 `1..10` 后加 `4`。

## 变形六：概率不是均匀的怎么办

这类题的前提通常是：

```text
randM() 本身等概率
```

如果给的是偏置硬币，不能直接套上面的公式。需要先用 Von Neumann trick 构造公平随机位：

```text
连续抛两次偏置硬币：
HT -> 0
TH -> 1
HH / TT -> 丢弃重来
```

因为：

```text
P(HT) = p(1-p)
P(TH) = (1-p)p
```

二者概率相等。

这属于另一个常见变形：用 biased coin 实现 fair coin。

## 正确性证明模板

面试里可以这样证明：

1. `k` 次 `randM()` 生成 `M^k` 个组合，每个组合概率相同，所以 `x` 在 `1..M^k` 上均匀。
2. 取 `limit = floor(M^k / N) * N`，因此 `1..limit` 可以被平均分成 `N` 组。
3. 对 `x <= limit`，使用 `(x - 1) % N + 1` 映射，每个结果出现 `limit / N` 次。
4. 对 `x > limit` 拒绝重采样，不会偏向任何结果，只会增加运行时间。

所以输出是 `1..N` 的均匀分布。

## 复杂度

对 `rand7 -> rand10` 标准解：

```text
accept probability = 40 / 49
expected rounds = 49 / 40
expected rand7 calls = 2 * 49 / 40 = 2.45
```

一般情况下：

```text
space size = M^k
limit = floor(M^k / N) * N
accept probability = limit / M^k
expected rounds = M^k / limit
expected randM calls = k * M^k / limit
```

空间复杂度是 `O(1)`。

## 常见坑

- 直接 `% N`，没有检查原始空间是否能被 `N` 整除。
- 忘记 `randM()` 返回的是 `1..M`，构造空间时没有先减 1。
- 返回 `(x % N) + 1`，导致边界分布错；推荐统一写 `(x - 1) % N + 1`。
- 只采样一次，不处理被拒绝的区域。
- 以为 rejection sampling 会死循环；只要接受概率大于 0，期望运行时间是有限的。
- 优化版复用剩余空间时，没有确认剩余空间仍然是均匀的。

## 面试回答模板

<details class="solution">
<summary>展开模板</summary>

这题不能直接对 `rand7()` 的结果取模，因为 `7` 或 `49` 不一定能被目标范围整除，取模会让某些结果出现次数更多。

我会先调用两次 `rand7()` 构造一个均匀的 `1..49` 空间：

```python
x = (rand7() - 1) * 7 + rand7()
```

然后只接受 `1..40`，因为 `40` 可以被 `10` 整除。对于接受的值，用：

```python
(x - 1) % 10 + 1
```

映射到 `1..10`。`41..49` 拒绝并重新采样。这样每个输出值在 `1..40` 中刚好出现 4 次，所以结果均匀。

</details>

```quiz
title: 练习 1
question: rand7 实现 rand10 时，为什么只接受 1..40？
answer: B
A. 因为 40 比 49 小
B. 因为 40 是不超过 49 且能被 10 整除的最大数
C. 因为 rand7 最多只能生成 40
explanation: 只有接受区间大小能被 10 整除，取模映射后每个结果出现次数才相同。
```

```quiz
title: 练习 2
question: 用 rand5 实现 rand7，至少需要几次 rand5？
answer: B
A. 1 次
B. 2 次
C. 7 次
explanation: 5 小于 7，一次不够；两次可以构造 25 个均匀结果，再取 1..21。
```

```quiz
title: 练习 3
question: rejection sampling 中被拒绝的样本会造成输出概率偏差吗？
answer: B
A. 会，因为丢掉了数据
B. 不会，只要接受区间内部能被均匀映射，拒绝只影响运行时间
C. 一定会死循环
explanation: 拒绝区域直接重采样，不参与输出分布，因此不改变接受区域内的均匀性。
```
