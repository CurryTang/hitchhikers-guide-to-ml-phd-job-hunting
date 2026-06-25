# Kruskal's Algorithm

## 面试目标

掌握最小生成树的 Kruskal 算法：按边权从小到大尝试加入，用并查集避免成环。

## 核心流程

1. 将所有边按权重排序。
2. 初始化 Union-Find。
3. 依次遍历边 `(u, v, w)`。
4. 如果 `u` 和 `v` 不连通，则加入 MST 并 union。
5. 加入 `V - 1` 条边后结束。

## 复杂度

- 排序主导：`O(E log E)`
- Union-Find 操作近似 `O(1)`。

## 常见坑

- 忘记跳过会形成环的边。
- 无向边重复加入导致排序列表翻倍。
- 图不连通时只能得到最小生成森林。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

Kruskal 先全局按边权排序，再用 Union-Find 判断一条边是否会形成环。

```text
sort edges by weight
dsu = UnionFind(n)
cost = 0
for u, v, w in edges:
  if dsu.union(u, v):
    cost += w
    used += 1
    if used == n - 1: break
```

`union` 返回 true 才说明这条边连接了两个不同连通分量，可以加入 MST。

</details>

```quiz
title: 练习 1
question: Kruskal 用什么结构判断加入一条边是否成环？
answer: B
A. 栈
B. Union-Find
C. 单调队列
explanation: 如果边两端已经连通，再加入就会成环。
```

```quiz
title: 练习 2
question: Kruskal 首先需要对什么排序？
answer: A
A. 边权
B. 节点名称
C. BFS 层数
explanation: Kruskal 从最小权重边开始贪心选择。
```
