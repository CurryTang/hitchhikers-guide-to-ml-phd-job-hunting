# Greedy Algorithms：从强制选择到不变量

## Greedy 的核心判断

Greedy 不是“每一步看起来最爽就这么做”。真正可靠的贪心通常来自两类信号：

1. **强制选择**：当前最小、最早、最紧急的元素没有别的合法去处。
2. **支配关系**：做出某个局部选择后，不会让未来更差，甚至只会让未来更容易。

面试里讲 Greedy，最好不要只说“直觉上应该这样”。更好的说法是：

```text
我先找一个必须被处理的元素。
这个元素在任何合法解里都只能以某种方式出现。
所以我现在处理它不会丢掉任何可行解。
处理完后，剩下的问题和原问题同形。
```

这就是 exchange argument / forced move 的思路。

## Hand of Straights：一手顺子

题目：

```text
给定 hand 和 groupSize。
能不能把所有牌分成若干组，每组长度都是 groupSize，
并且每组都是连续整数？
```

例子：

```text
hand = [1, 2, 3, 6, 2, 3, 4, 7, 8]
groupSize = 3

可以分成：
[1, 2, 3]
[2, 3, 4]
[6, 7, 8]

answer = true
```

反例：

```text
hand = [1, 2, 3, 4, 5]
groupSize = 4

总牌数 5 不能被 4 整除。
answer = false
```

## 突破口：最小的牌没得选

如果当前剩下的最小牌是 `x`，它能放在哪里？

它不可能是某个顺子的第二张：

```text
x - 1, x, x + 1, ...
```

因为 `x - 1` 比 `x` 更小，但当前已经没有更小的牌了。

它也不可能是第三张、第四张。原因一样：那需要更小的前驱牌。

所以：

> 当前剩下的最小牌 `x`，必须作为某个顺子的开头。

这就是这题的贪心选择。

如果 `x` 有 `count[x]` 张，那么这 `count[x]` 张 `x` 都必须开头。于是后面必须同时有：

```text
count[x] 张 x + 1
count[x] 张 x + 2
...
count[x] 张 x + groupSize - 1
```

只要其中某张牌不够，答案立刻是 `False`。

## 为什么可以一次扣掉 count[x] 张

假设当前最小牌是 `x`，并且：

```text
count[x] = 3
groupSize = 4
```

这 3 张 `x` 都必须开 3 个不同的顺子：

```text
x, x+1, x+2, x+3
x, x+1, x+2, x+3
x, x+1, x+2, x+3
```

所以与其一组一组扣，不如批量扣：

```text
count[x] -= 3
count[x + 1] -= 3
count[x + 2] -= 3
count[x + 3] -= 3
```

这不是优化技巧，而是贪心正确性的直接结果：

```text
最小牌没有选择空间，所以它的所有副本都被迫开新顺子。
```

## 标准解法：Counter + Min Heap

算法流程：

1. 如果 `len(hand) % groupSize != 0`，直接返回 `False`。
2. 用 `Counter` 统计每张牌的数量。
3. 把所有不同牌面放进小顶堆，用来找到当前最小牌。
4. 每次取堆顶 `first`：
   - 如果 `count[first] == 0`，说明它已经被之前的顺子消耗完，弹出。
   - 否则它是当前还没处理完的最小牌，必须开新顺子。
5. 设 `need = count[first]`，对 `first` 到 `first + groupSize - 1` 每张牌都扣掉 `need`。
6. 如果某张牌数量不够，返回 `False`。

代码：

```python
from collections import Counter
from heapq import heapify, heappop
from typing import List

class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0:
            return False

        count = Counter(hand)
        min_heap = list(count)
        heapify(min_heap)

        while min_heap:
            first = min_heap[0]

            if count[first] == 0:
                heappop(min_heap)
                continue

            need = count[first]
            for card in range(first, first + groupSize):
                if count[card] < need:
                    return False
                count[card] -= need

        return True
```

## 这段代码的关键细节

### 1. 为什么先检查整除

每组长度固定是 `groupSize`，所有牌都必须用完。

所以总牌数必须满足：

```text
len(hand) % groupSize == 0
```

否则不用进入贪心逻辑。

### 2. 为什么堆里可能有 count 为 0 的牌

例如：

```text
hand = [1, 2, 3, 2, 3, 4]
groupSize = 3
```

处理 `1` 时会扣掉：

```text
1, 2, 3
```

这可能让 `2` 或 `3` 提前变成 `0`。但它们还留在 heap 里。

所以每次看堆顶时，要先跳过已经用完的牌：

```python
if count[first] == 0:
    heappop(min_heap)
    continue
```

### 3. 为什么不用每次真的生成一组

如果 `first` 有 `need` 张，那么必须生成 `need` 组从 `first` 开始的顺子。

批量扣减比逐组生成更直接：

```python
for card in range(first, first + groupSize):
    count[card] -= need
```

它表达的是：

```text
所有以 first 开头的顺子，一次性结算。
```

## 正确性证明

维护一个不变量：

```text
每次 while 循环开始时，所有比 heap 顶部更小的牌都已经被合法分组消耗完。
```

现在看当前最小的剩余牌 `first`。

因为没有比 `first` 更小的牌，所以 `first` 在任何合法解里都不可能出现在顺子的中间或末尾。

因此它必须作为顺子的开头。

如果 `first` 有 `need` 张，那么必须开 `need` 个顺子。每个顺子都需要：

```text
first, first + 1, ..., first + groupSize - 1
```

所以这些牌每一种都至少需要 `need` 张。

如果某一种不够，任何解都不可能存在。

如果都够，我们扣掉它们。扣完以后，所有 `first` 都被合法使用了，剩下的牌仍然是同一个问题：

```text
能不能把剩余牌分成固定长度的连续组？
```

所以继续处理新的最小牌即可。

当所有牌都被扣完时，说明每一步强制选择都成功，答案是 `True`。

## 复杂度

设：

```text
n = hand.length
u = 不同牌面的数量
g = groupSize
```

Counter 需要：

```text
O(n)
```

建堆需要：

```text
O(u)
```

每个不同牌面最多从堆里弹出一次：

```text
O(u log u)
```

批量扣减部分最多对每个“顺子起点”扫描 `groupSize` 张牌，最坏可以写成：

```text
O(u * groupSize)
```

因为 `u <= n`，面试里通常可以说：

```text
Time:  O(n log n)
Space: O(n)
```

更精确一点：

```text
Time:  O(n + u log u + u * groupSize)
Space: O(u)
```

## 另一种写法：排序去重后扫描

如果不想用 heap，也可以直接排序所有不同牌面。

思路一样：

```text
按从小到大处理每个牌面。
如果 count[x] > 0，它必须开 count[x] 个顺子。
```

代码：

```python
from collections import Counter
from typing import List

class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0:
            return False

        count = Counter(hand)

        for first in sorted(count):
            need = count[first]
            if need == 0:
                continue

            for card in range(first, first + groupSize):
                if count[card] < need:
                    return False
                count[card] -= need

        return True
```

这版更短，也很适合面试。

复杂度：

```text
Time:  O(n + u log u + u * groupSize)
Space: O(u)
```

## 手动走一遍

输入：

```text
hand = [1, 2, 3, 6, 2, 3, 4, 7, 8]
groupSize = 3
```

频率：

```text
1:1, 2:2, 3:2, 4:1, 6:1, 7:1, 8:1
```

当前最小牌是 `1`：

```text
need = 1
扣 1, 2, 3
```

剩下：

```text
2:1, 3:1, 4:1, 6:1, 7:1, 8:1
```

当前最小牌是 `2`：

```text
need = 1
扣 2, 3, 4
```

剩下：

```text
6:1, 7:1, 8:1
```

当前最小牌是 `6`：

```text
need = 1
扣 6, 7, 8
```

全部用完，返回 `True`。

## 常见坑

- 忘记先判断总长度是否能被 `groupSize` 整除。
- 只扣一张 `first`，没有批量扣掉 `count[first]` 张。
- 用 heap 时忘记跳过 `count[first] == 0` 的旧堆顶。
- 误以为最小牌可以放在顺子中间。最小牌没有前驱，所以只能开头。
- 用普通 list 每次 `pop(0)` 找最小值，导致不必要的 `O(n^2)` 开销。
- 遇到 `count[card] == 0` 才返回 false，但其实 `count[card] < need` 就已经不够。

## 面试回答模板

可以这样讲：

1. 先检查 `len(hand)` 是否能被 `groupSize` 整除。
2. 用 Counter 统计每张牌数量。
3. 贪心点是：当前最小剩余牌不可能接在任何更小牌后面，所以必须作为顺子开头。
4. 如果最小牌 `first` 有 `need` 张，就必须开 `need` 个从 `first` 开始的顺子。
5. 因此 `first, first + 1, ..., first + groupSize - 1` 每张牌都要至少有 `need` 张。
6. 不够就返回 `False`；够就批量扣减。
7. 一直处理到所有牌用完，返回 `True`。

一句话总结：

> Hand of Straights 的 Greedy 不是“随便从小开始”，而是“最小牌没有前驱，所以被迫从小开始”。

```quiz
title: 练习 1
question: Hand of Straights 里为什么当前最小牌必须作为顺子开头？
answer: B
A. 因为这样代码最短
B. 因为它没有更小的前驱牌，无法出现在顺子中间或末尾
C. 因为 heap 只能弹出最小值
explanation: 贪心正确性来自强制选择，不是来自数据结构。
```

```quiz
title: 练习 2
question: 如果当前最小牌 first 有 need 张，接下来应该怎么扣？
answer: C
A. 只扣 first 一张
B. 只扣 first + groupSize - 1 一张
C. 对 first 到 first + groupSize - 1 每张牌都扣 need 张
explanation: need 张 first 必须分别开 need 个顺子，所以后续每个连续牌面也都需要 need 张。
```
