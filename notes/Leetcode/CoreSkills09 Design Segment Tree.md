# Design Segment Tree

## 面试目标

线段树解决一类非常固定的问题：数组会被反复修改，同时还要反复查询某个连续区间的聚合值。这里的聚合值可以是 `sum`、`min`、`max`、`gcd`、计数、最大子段信息等，只要两个相邻区间的答案能合并成大区间的答案，就可以考虑线段树。

要能在面试里讲清楚三件事：

- 每个节点表示一个连续区间，根节点表示整个数组。
- 查询区间时，把目标区间拆成 `O(log n)` 个互不重叠的树节点。
- 单点更新时，只改一个叶子，然后沿着叶子到根的路径重算父节点。

## 先把下面这张图翻译成人话

这张图不是在画普通 LIS 的 `tails` 数组，而是在画一个 DP 表：

```text
leaf[rank] = 目前为止，以这个数值结尾的最长递增子序列长度
```

例如值域压缩后：

```text
rank 0  1  2  3  4  5  6   7
值   2  3  5  7  9  10 18  101
```

如果当前处理的是 `7`，它的 rank 是 `3`。严格递增要求前一个数必须比 `7` 小，所以只能看 rank `0..2`，也就是值 `2, 3, 5`。线段树做的事情就是快速回答：

```text
在所有比 7 小的结尾值里，最长长度是多少？
```

如果查出来是 `2`，说明前面已经有长度为 2 的递增序列可以接上 `7`，所以：

```text
以 7 结尾的长度 = 2 + 1 = 3
```

然后把 rank `3` 的叶子更新成 `3`。整张图每一帧都只重复这三步：

1. 找当前数的 rank。
2. 查询所有更小 rank 的最大 DP 值。
3. 把 `最大 DP 值 + 1` 写回当前 rank。

```segment-tree-demo
```

## 线段树到底存什么

假设数组是 `arr[0..n-1]`，线段树的节点不是存单个元素，而是存一段区间的汇总结果。

例如做区间最大值：

- 叶子节点 `[i, i]` 存 `arr[i]`。
- 父节点 `[l, r]` 存 `max(left_child, right_child)`。
- 根节点 `[0, n-1]` 存整个数组的最大值。

所以线段树的核心不是“树”，而是一个可以快速重算的分治缓存。它把数组切成很多标准区间，查询时直接拿缓存，更新时只修正受影响的缓存。

## 数组实现：为什么常用 `2 * n`

有两种常见写法。

第一种是递归线段树，通常开 `4 * n` 空间，节点 `idx` 的左儿子是 `2 * idx`，右儿子是 `2 * idx + 1`。

第二种是迭代线段树，先把叶子放在数组后半段：

```text
tree[n + i] = arr[i]
tree[i] = merge(tree[2*i], tree[2*i + 1])
```

如果把 `n` 补到 2 的幂，整棵树会更规整：

```text
叶子: tree[n], tree[n+1], ..., tree[2n-1]
父亲: i // 2
左儿子: 2 * i
右儿子: 2 * i + 1
```

你贴的实现就是这种迭代写法，并且 merge 操作是 `max`：

```python
class SegmentTree:
    def __init__(self, N):
        self.n = N
        while (self.n & (self.n - 1)) != 0:
            self.n += 1
        self.tree = [0] * (2 * self.n)
```

`self.n` 被补到 2 的幂后，原始第 `i` 个位置对应叶子 `self.n + i`。多出来的叶子默认是 `0`，因为这道 LIS 题里线段树存的是长度，空状态长度正好是 `0`。

## 单点更新

单点更新只影响从某个叶子到根的一条路径。

```python
def update(self, i, val):
    self.tree[self.n + i] = val
    j = (self.n + i) >> 1
    while j >= 1:
        self.tree[j] = max(self.tree[j << 1], self.tree[j << 1 | 1])
        j >>= 1
```

这段代码的逻辑是：

1. 先把叶子 `i` 改成 `val`。
2. 令 `j` 变成父节点。
3. 每次用左右儿子的最大值重算父节点。
4. 一直更新到根节点 `1`。

复杂度是树高，也就是 `O(log n)`。

面试里可以补一句：如果同一个位置可能被多次更新，并且你要保留最大值，可以写成 `self.tree[self.n + i] = max(self.tree[self.n + i], val)`。这道 LIS 从左到右处理时，同一个 rank 的最佳长度不会变小，所以直接赋值也能工作。

## 区间查询

迭代查询最容易看晕，因为它不是递归地判断三种重叠关系，而是在叶子层用两个指针向上收缩。

```python
def query(self, l, r):
    if l > r:
        return 0
    res = float('-inf')
    l += self.n
    r += self.n + 1
    while l < r:
        if l & 1:
            res = max(res, self.tree[l])
            l += 1
        if r & 1:
            r -= 1
            res = max(res, self.tree[r])
        l >>= 1
        r >>= 1
    return res
```

关键点：

- 这里把查询区间变成半开区间 `[l, r)`，所以原来的右端点要写成 `r += self.n + 1`。
- 如果 `l` 是右儿子，说明它不能再和左兄弟一起向上合并，否则会多算左边不属于查询范围的部分，所以先把 `tree[l]` 放进答案，再 `l += 1`。
- 如果 `r` 是右边界的右儿子位置，先 `r -= 1`，把这个合法节点放进答案。
- 每轮处理完边界节点后，`l >>= 1`、`r >>= 1`，进入父层。

这个过程最多在每一层拿两个边界节点，所以也是 `O(log n)`。

## 单位元要和 merge 操作匹配

线段树查询有一个容易错的点：无贡献区间返回什么？

- 区间和：返回 `0`。
- 区间最大值：如果数组可能有负数，严格单位元应该是 `-inf`。
- 区间最小值：返回 `+inf`。
- gcd：返回 `0`，因为 `gcd(0, x) = x`。

这道 LIS 题里线段树保存的是“长度”，所有合法长度都不小于 `0`，所以空查询返回 `0` 是合理的。

## 例题：Longest Increasing Subsequence

题目：给定 `nums`，求最长严格递增子序列长度。

传统 `O(n log n)` patience sorting 写法只求长度很方便；线段树写法更像动态规划，适合扩展到“带约束的 LIS”、“值域范围查询”、“带修改的 DP”等题。

定义：

```text
dp[x] = 以值 x 结尾的最长递增子序列长度
```

当处理当前数 `num` 时，只能接在比它小的值后面：

```text
cur = max(dp[value] for value < num) + 1
dp[num] = cur
answer = max(answer, cur)
```

如果直接按真实数值开数组，数值可能很大，甚至有负数。因此先做坐标压缩，把值映射成 rank：

```python
def compress(arr):
    sortedArr = sorted(set(arr))
    order = []
    for num in arr:
        order.append(bisect_left(sortedArr, num))
    return order
```

压缩后，`rank` 越小，原始值越小。严格递增意味着当前 rank 只能查询 `[0, rank - 1]`，不能包含自己。

完整思路：

```python
from bisect import bisect_left
from typing import List

class SegmentTree:
    def __init__(self, N):
        self.n = N
        while (self.n & (self.n - 1)) != 0:
            self.n += 1
        self.tree = [0] * (2 * self.n)

    def update(self, i, val):
        self.tree[self.n + i] = val
        j = (self.n + i) >> 1
        while j >= 1:
            self.tree[j] = max(self.tree[j << 1], self.tree[j << 1 | 1])
            j >>= 1

    def query(self, l, r):
        if l > r:
            return 0
        res = float('-inf')
        l += self.n
        r += self.n + 1
        while l < r:
            if l & 1:
                res = max(res, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = max(res, self.tree[r])
            l >>= 1
            r >>= 1
        return res

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        def compress(arr):
            sortedArr = sorted(set(arr))
            order = []
            for num in arr:
                order.append(bisect_left(sortedArr, num))
            return order

        nums = compress(nums)
        n = len(nums)
        segTree = SegmentTree(n)

        LIS = 0
        for num in nums:
            curLIS = segTree.query(0, num - 1) + 1
            segTree.update(num, curLIS)
            LIS = max(LIS, curLIS)
        return LIS
```

以 `[10, 9, 2, 5, 3, 7, 101, 18]` 为例：

```text
sorted values = [2, 3, 5, 7, 9, 10, 18, 101]
compressed    = [5, 4, 0, 2, 1, 3, 7, 6]
```

处理 `7` 时，它的 rank 是 `3`，所以查询 `[0, 2]`，也就是只看值 `{2, 3, 5}` 的最佳结尾长度。此时最大值是 `2`，所以 `curLIS = 3`，代表可以形成类似 `[2, 5, 7]` 或 `[2, 3, 7]` 的长度。

处理 `18` 时，rank 是 `6`，查询 `[0, 5]`，最大值是 `3`，所以 `curLIS = 4`。最后答案是 `4`。

## 为什么不用查询 `[0, rank]`

LIS 是严格递增，不能把相等的值接在一起。如果查询 `[0, rank]`，相同值可能贡献给自己，结果会变成非严格递增。

例如 `[2, 2, 2]`：

- 正确答案是 `1`。
- 每次都应该查询比 `2` 小的值，也就是空区间。
- 如果查询包含当前 rank，就可能把前一个 `2` 的长度接到后一个 `2` 后面，语义就错了。

## 复杂度

设数组长度为 `n`，不同数值个数为 `m`。

- 坐标压缩：`O(n log n)`。
- 每个数做一次区间最大查询和一次单点更新：`O(n log m)`。
- 线段树空间：`O(m)`，实现里补到 2 的幂后仍然是线性空间。

## Lazy propagation 什么时候需要

本题只做单点更新，不需要 lazy propagation。

Lazy propagation 用在区间更新，比如：

- 给 `[l, r]` 每个元素都加上 `x`。
- 把 `[l, r]` 每个元素都赋值为 `x`。
- 频繁执行区间更新加区间查询。

如果每次区间更新都递归到所有叶子，最坏会退化到 `O(n)`。Lazy 的思路是：当一个节点区间完全被更新覆盖时，先改这个节点的聚合值，并记录一个 lazy tag；只有未来访问到它的子节点时，再把 tag 下传。

## 常见坑

- 忘记坐标压缩，直接用值当数组下标。
- 严格递增题查询了 `[0, rank]`，把相等元素也算进去了。
- 最大值线段树的无交集返回值和数据范围不匹配。
- `r += self.n + 1` 写错，导致右端点没有被包含。
- 单点更新后只改叶子，没有一路重算到根。
- 区间更新题漏掉 lazy tag 的下传时机。

## 面试回答模板

<details class="solution">
<summary>展开模板</summary>

线段树把数组分成一棵二叉区间树。每个节点维护一个区间的聚合值，比如最大值。查询 `[l, r]` 时，我们把它拆成若干个已经缓存好的节点区间并合并答案；单点更新时，只会影响叶子到根的一条路径，所以更新复杂度是 `O(log n)`。

在 LIS 这题里，我把数值坐标压缩成 rank。线段树第 `rank` 个叶子维护“以这个 rank 结尾的最长递增子序列长度”。处理当前 `rank` 时，先查询所有更小 rank 的最大值，也就是 `query(0, rank - 1)`，再加一得到当前长度，然后更新当前 rank。这样每个元素 `O(log n)`，总复杂度 `O(n log n)`。

</details>

```quiz
title: 练习 1
question: 线段树最适合优化哪类操作？
answer: A
A. 动态数组上的区间查询和更新
B. 单次全数组扫描
C. 固定字符串的前缀比较
explanation: 线段树的价值在于数组会被反复查询、反复更新，并且每次只需要区间聚合值。
```

```quiz
title: 练习 2
question: 在严格递增 LIS 的线段树写法中，当前 rank 应该查询哪个区间？
answer: B
A. [0, rank]
B. [0, rank - 1]
C. [rank, n - 1]
explanation: 严格递增只能接在更小的值后面，不能包含当前 rank 对应的相等值。
```

```quiz
title: 练习 3
question: 单点更新后为什么要一路重算到根？
answer: C
A. 为了释放内存
B. 为了重新排序所有叶子
C. 因为祖先区间的聚合值都可能被这个叶子影响
explanation: 叶子变化后，包含它的所有父区间缓存都可能变化，所以要沿父指针回溯更新。
```
