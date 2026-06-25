# Unbounded Knapsack

## 面试目标

掌握完全背包：每个物品可以选无限次，在容量限制下最大化价值或统计方案。

## 状态设计

- `dp[c]` 表示容量为 c 的最佳结果。
- 对每个物品 `(weight, value)`，容量从小到大遍历。
- 最大价值转移：`dp[c] = max(dp[c], dp[c - weight] + value)`。

## 为什么正序

正序容量允许当前物品在同一轮被重复使用，因此符合无限次选择。

## 复杂度

- 时间：`O(nW)`
- 空间：`O(W)`。

## 常见坑

- 和 0/1 背包混淆循环方向。
- 组合数和排列数问题的循环顺序不同。
- 初始化要根据最大值、可行性或计数目标调整。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

完全背包允许同一物品重复选，因此容量正序，让当前轮刚更新的 `dp[c-weight]` 可以继续参与转移。

```text
dp = [0] * (W + 1)
for weight, value in items:
  for c in range(weight, W + 1):
    dp[c] = max(dp[c], dp[c - weight] + value)
return dp[W]
```

如果题目是方案数，仍要先明确问的是组合数还是排列数；循环顺序会影响计数语义。

</details>

```quiz
title: 练习 1
question: 完全背包一维 dp 中容量通常如何遍历？
answer: A
A. 从小到大
B. 从大到小
C. 随机遍历
explanation: 正序允许当前物品被重复使用。
```

```quiz
title: 练习 2
question: Unbounded Knapsack 与 0/1 Knapsack 的关键区别是什么？
answer: B
A. 是否使用数组
B. 每个物品是否可以重复选择
C. 是否必须用递归
explanation: 完全背包允许同一物品选多次，0/1 背包最多一次。
```
