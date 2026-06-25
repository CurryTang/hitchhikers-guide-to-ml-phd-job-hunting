# Shortest Path：Dijkstra 与带边数限制的 Bellman-Ford

## 面试目标

最短路题不要先背模板，而是先判断约束：

- 边权都非负，并且没有额外限制：优先考虑 Dijkstra。
- 有负权边，或者路径长度/边数有显式限制：考虑 Bellman-Ford。
- 图是无权图：BFS 就是最短路。
- 状态里除了节点还有别的维度，例如“用了几次中转”：要把状态扩展成 `(node, state)`。

这一节用 **Cheapest Flights Within K Stops** 作为例题，重点讲优化过的 Bellman-Ford。它不是在所有路径里找任意最短路，而是在“最多 `k` 个中转，也就是最多 `k + 1` 条边”的限制下找最短路。

## Dijkstra 回顾

Dijkstra 适用于非负权图。核心是用最小堆每次弹出当前代价最小的状态，再松弛邻居。

```text
dist[start] = 0
heap = [(0, start)]

while heap:
  d, u = heappop(heap)
  if d != dist[u]:
    continue
  for v, w in graph[u]:
    if d + w < dist[v]:
      dist[v] = d + w
      heappush(heap, (dist[v], v))
```

对于 Cheapest Flights，普通 Dijkstra 不够，因为“到同一个城市的最低价格”不一定是最终最好状态。一个更贵但用了更少航班的状态，后面可能仍然可行。因此 Dijkstra 写法需要把状态扩展为 `(cost, city, stopsUsed)`。

## 为什么 Bellman-Ford 更自然

Bellman-Ford 的语义刚好适合这题：

> 第 `i` 轮松弛之后，`prices[x]` 表示使用最多 `i` 条边到达城市 `x` 的最低价格。

题目允许最多 `k` 个 stops，也就是最多 `k + 1` 条 flight。因此我们只需要做 `k + 1` 轮松弛。

关键点是：每一轮必须从上一轮的 `prices` 读，写入 `nextPrices`。如果直接原地更新，就会在同一轮里把多条边串起来，等价于偷偷使用了超过当前轮数的航班数。

## 例题：Cheapest Flights Within K Stops

输入：

```text
n = 4
flights = [
  [0, 1, 100],
  [1, 2, 100],
  [2, 0, 100],
  [1, 3, 600],
  [2, 3, 200],
]
src = 0
dst = 3
k = 1
```

最多 `1` 个中转，所以最多可以坐 `2` 段航班。合法答案是 `0 -> 1 -> 3`，价格是 `700`。路径 `0 -> 1 -> 2 -> 3` 价格更低吗？它是 `400`，但需要 `3` 段航班，超过限制，不能用。

```bellman-demo
cheapest-flights
```

## 优化版 Bellman-Ford

基础写法是做固定的 `k + 1` 轮。优化点有两个：

1. 每轮用 `nextPrices = prices.copy()`，保证本轮只从上一轮状态转移。
2. 如果某一轮没有任何更新，说明继续松弛也不会变好，可以提前停止。

```python
from typing import List


class Solution:
    def findCheapestPrice(
        self,
        n: int,
        flights: List[List[int]],
        src: int,
        dst: int,
        k: int,
    ) -> int:
        INF = float("inf")
        prices = [INF] * n
        prices[src] = 0

        for _ in range(k + 1):
            next_prices = prices.copy()
            changed = False

            for start, end, cost in flights:
                if prices[start] == INF:
                    continue

                candidate = prices[start] + cost
                if candidate < next_prices[end]:
                    next_prices[end] = candidate
                    changed = True

            prices = next_prices
            if not changed:
                break

        return -1 if prices[dst] == INF else prices[dst]
```

## 正确性直觉

第 `0` 轮开始时，只有 `src` 的价格是 `0`，表示“不坐任何航班只能到达起点”。

第 `1` 轮只允许从第 `0` 轮的结果出发，所以只能得到所有一段航班能到达的城市。

第 `2` 轮只允许从第 `1` 轮的结果出发，所以得到最多两段航班能到达的城市。

一直做到第 `k + 1` 轮，就正好覆盖了题目允许的最大航班数。因为每轮读旧数组、写新数组，同一轮内不会发生 `0 -> 1 -> 2 -> 3` 这种连续串联。

## 复杂度

- 时间：`O((k + 1) * E)`，其中 `E` 是航班数量。
- 空间：`O(V)`，只保留 `prices` 和 `next_prices`。

这通常比把所有 `(city, stops)` 状态丢进堆更直接，也更适合解释“最多几条边”的约束。

## 常见坑

- 把 `k stops` 当成最多 `k` 条边。实际是最多 `k + 1` 条 flight。
- 原地更新 `prices`，导致同一轮串联多条边。
- 忘记跳过 `prices[start] == INF` 的航班。
- 提前返回 `dst` 的当前价格；Bellman-Ford 要等当前轮松弛完成。
- 用普通 Dijkstra 的 `dist[city]` 压掉了“更贵但 stops 更少”的状态。

```quiz
title: 练习 1
question: Cheapest Flights 中 k stops 对应最多多少条航班边？
answer: B
A. k 条
B. k + 1 条
C. 2k 条
explanation: stops 是中转城市数量，航班段数比中转数多 1。
```

```quiz
title: 练习 2
question: 为什么 Bellman-Ford 每一轮要使用 next_prices = prices.copy()？
answer: C
A. 为了让空间复杂度变成 O(E)
B. 为了支持负环检测
C. 为了避免同一轮内连续使用多条边，超过当前轮允许的航班数
explanation: 本轮只能从上一轮的结果出发，不能使用本轮刚更新出来的价格继续松弛。
```

```quiz
title: 练习 3
question: 优化版 Bellman-Ford 什么时候可以提前停止？
answer: A
A. 某一轮没有任何价格更新
B. 第一次看到 dst 可达
C. prices[src] 变成无穷大
explanation: 如果一整轮没有变化，继续增加允许边数也不会产生更优价格。
```
