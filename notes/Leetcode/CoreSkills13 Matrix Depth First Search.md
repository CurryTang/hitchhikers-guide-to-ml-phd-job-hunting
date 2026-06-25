# Matrix Depth-First Search

## 面试目标

掌握二维网格 DFS，常用于岛屿数量、连通块、路径搜索和 flood fill。

## 核心模板

- 遍历每个格子，遇到未访问的目标格子就启动 DFS。
- DFS 检查边界、访问状态和格子类型。
- 四方向移动：上、下、左、右。
- 可以用递归，也可以用显式栈避免递归深度问题。

## 复杂度

- 时间：`O(mn)`
- 空间：`O(mn)`，最坏递归栈或 visited。

## 常见坑

- 访问后没有立刻标记 visited，导致重复进入。
- 行列边界写反。
- 对角线是否算相邻需要看题目定义。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

DFS 函数只处理一个格子：先做越界和 visited 检查，再标记，最后递归四个方向。

```text
dfs(r, c):
  if r/c out of bounds: return
  if visited[r][c] or grid[r][c] is blocked: return
  visited[r][c] = true
  for (dr, dc) in directions:
    dfs(r + dr, c + dc)
```

如果题目要统计连通块，外层遍历每个格子，遇到新的目标格子时计数加一并启动 DFS。

</details>

```quiz
title: 练习 1
question: 网格 DFS 中 visited 应该在什么时候标记？
answer: A
A. 进入格子后尽早标记
B. 所有邻居访问完再标记
C. 永远不需要标记
explanation: 尽早标记可以避免环或相邻格子重复递归。
```

```quiz
title: 练习 2
question: m 行 n 列网格完整 DFS 的时间复杂度通常是多少？
answer: B
A. O(m+n)
B. O(mn)
C. O(log mn)
explanation: 每个格子最多被访问常数次。
```
