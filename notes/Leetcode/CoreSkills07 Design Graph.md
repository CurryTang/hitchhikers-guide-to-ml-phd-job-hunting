# Design Graph

## 面试目标

实现图结构，理解邻接表、邻接矩阵、有向/无向图、加权边和遍历接口。

## 核心设计

- 邻接表适合稀疏图：`node -> neighbors`。
- 邻接矩阵适合点数较小、边查询频繁的场景。
- 无向图加边时需要同时写入 `u -> v` 和 `v -> u`。
- 加权图的邻居通常保存 `(neighbor, weight)`。

## 复杂度

- 邻接表空间：`O(V + E)`
- 邻接矩阵空间：`O(V^2)`
- 遍历全部点边：`O(V + E)`

## 常见坑

- 无向边只加了一边。
- 遍历时忘记 visited 导致环上死循环。
- 节点编号不连续时仍强行用数组下标。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

用邻接表最通用。未加权图存邻居节点；加权图存 `(neighbor, weight)`。

```text
addEdge(u, v, w=1):
  adj[u].append((v, w))
  if undirected:
    adj[v].append((u, w))

neighbors(u):
  return adj.get(u, [])
```

遍历时配合 `visited`。如果节点不是连续整数，使用 map/dictionary 保存邻接表更稳。

</details>

```quiz
title: 练习 1
question: 稀疏图通常更适合哪种表示？
answer: A
A. 邻接表
B. 邻接矩阵
C. 排序数组
explanation: 邻接表空间是 O(V + E)，稀疏图更省空间。
```

```quiz
title: 练习 2
question: 无向图 addEdge(u, v) 通常需要做什么？
answer: B
A. 只加入 u -> v
B. 同时加入 u -> v 和 v -> u
C. 删除所有旧边
explanation: 无向边可以从两个方向访问。
```
