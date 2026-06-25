# Design Disjoint Set (Union-Find)

## 面试目标

实现并查集，用于动态维护连通分量，核心操作是 `find` 和 `union`。

## 核心设计

- `parent[x]` 指向 x 的代表节点。
- `find(x)` 沿 parent 找到根。
- 路径压缩让查找路径上的节点直接指向根。
- union by rank/size 把小树挂到大树下。

## 复杂度

- 带路径压缩和按秩合并：近似 `O(1)`，严格为反阿克曼函数级别。
- 空间：`O(n)`。

## 常见坑

- union 时没有先找根。
- find 只返回父节点，没有递归到代表元。
- 统计连通分量时重复合并也减少 count。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

并查集用 `parent` 和 `size/rank`。`find` 做路径压缩，`union` 先找两个根，再把小集合挂到大集合。

```text
find(x):
  if parent[x] != x:
    parent[x] = find(parent[x])
  return parent[x]

union(a, b):
  ra, rb = find(a), find(b)
  if ra == rb: return false
  if size[ra] < size[rb]: swap(ra, rb)
  parent[rb] = ra
  size[ra] += size[rb]
  return true
```

返回 `false` 表示两点本来就在同一集合，不能重复减少连通分量数。

</details>

```quiz
title: 练习 1
question: Union-Find 最适合解决哪类问题？
answer: A
A. 动态连通性
B. 字符串匹配
C. 数组排序稳定性
explanation: 并查集能快速判断两个元素是否属于同一集合。
```

```quiz
title: 练习 2
question: 路径压缩的作用是什么？
answer: B
A. 删除所有节点
B. 缩短 find 后续查询路径
C. 把集合变成二叉搜索树
explanation: 路径压缩让路径上的节点直接连到根。
```
