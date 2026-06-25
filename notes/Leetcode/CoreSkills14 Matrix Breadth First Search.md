# Matrix Breadth-First Search

## 面试目标

掌握二维网格 BFS，常用于最短步数、多源扩散、层序传播。

## 核心模板

- 用队列保存当前边界。
- 入队时立刻标记 visited。
- 每一层代表距离加一。
- 多源 BFS 可以把所有起点先同时入队。

## 复杂度

- 时间：`O(mn)`
- 空间：`O(mn)`。

## 常见坑

- 出队时才标记 visited 会导致同一格子重复入队。
- 层数统计位置错误。
- 多源 BFS 忘记初始化所有源点。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

BFS 用队列按层扩展。入队时立刻标记，避免同一格子被多个父节点重复入队。

```text
queue = all start cells
mark all starts visited
dist = 0
while queue not empty:
  repeat len(queue) times:
    r, c = pop front
    push unvisited valid neighbors
  dist += 1
```

单源最短路返回第一次到达终点的层数；多源扩散把所有源点作为第 0 层。

</details>

```quiz
title: 练习 1
question: BFS 为什么适合求无权图最短步数？
answer: A
A. 它按距离从近到远扩展
B. 它总是先访问最大权重边
C. 它会自动排序所有路径
explanation: BFS 的层序扩展保证第一次到达就是最短步数。
```

```quiz
title: 练习 2
question: 多源 BFS 的初始化方式是什么？
answer: B
A. 只放入第一个源点
B. 把所有源点同时入队
C. 先运行 DFS 再入队
explanation: 多源同时入队等价于从一个超级源点开始扩散。
```
