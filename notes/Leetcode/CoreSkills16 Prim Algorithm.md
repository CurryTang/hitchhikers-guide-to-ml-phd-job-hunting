# Prim's Algorithm

## 面试目标

掌握最小生成树的 Prim 算法：从一个点开始，每次选择连接已访问集合和未访问集合的最小边。

## 核心流程

1. 任意选一个起点。
2. 把起点的边加入最小堆。
3. 每次弹出权重最小的边。
4. 如果边连接到未访问点，就加入 MST，并继续加入该点的边。

## 复杂度

- 邻接表 + 堆：`O(E log E)`，也常写作 `O(E log V)`。
- 空间：`O(V + E)`。

## 常见坑

- 图不连通时无法得到覆盖所有节点的 MST。
- 没有 visited，导致重复加入节点形成环。
- 把 Prim 和 Dijkstra 的 dist 含义混淆。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

Prim 维护已加入 MST 的点集，以及连接到外部点的候选边最小堆。

```text
visited = set([start])
heap = edges from start
cost = 0
while heap and len(visited) < n:
  w, u, v = heappop(heap)
  if v in visited: continue
  visited.add(v)
  cost += w
  for next, nw in adj[v]:
    if next not in visited:
      heappush(heap, (nw, v, next))
```

如果结束时 `visited` 少于 `n`，说明图不连通，没有覆盖所有点的 MST。

</details>

```quiz
title: 练习 1
question: Prim 算法每一步选择什么？
answer: A
A. 连接已访问集合到未访问节点的最小边
B. 当前最长路径
C. 入度为 0 的节点
explanation: Prim 用局部最小边逐步扩展 MST。
```

```quiz
title: 练习 2
question: Prim 主要用于解决什么问题？
answer: B
A. 单源最短路
B. 最小生成树
C. 字符串匹配
explanation: Prim 是经典 MST 算法。
```
