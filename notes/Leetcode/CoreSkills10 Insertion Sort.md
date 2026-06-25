# Insertion Sort

## 面试目标

理解插入排序的局部有序思想：左侧保持有序，把当前元素插入到正确位置。

## 核心流程

1. 从下标 1 开始遍历。
2. 保存当前值 `key`。
3. 从左侧有序区末尾向前移动更大的元素。
4. 把 `key` 放到空出的位置。

## 复杂度

- 最好情况：`O(n)`，数组已排序。
- 平均/最坏：`O(n^2)`。
- 空间：`O(1)`。
- 稳定性：稳定。

## 常见坑

- 覆盖 key 前没有先保存。
- while 条件导致越界。
- 把比较条件写成 `>=` 会破坏稳定性。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

左边 `[0, i)` 始终保持有序。取出 `arr[i]`，把更大的元素整体右移，最后把 key 放进空位。

```text
for i in range(1, n):
  key = arr[i]
  j = i - 1
  while j >= 0 and arr[j] > key:
    arr[j + 1] = arr[j]
    j -= 1
  arr[j + 1] = key
```

比较条件用 `>`，不要用 `>=`，这样相等元素不会互换，排序保持稳定。

</details>

```quiz
title: 练习 1
question: 插入排序在数组已经有序时复杂度是多少？
answer: A
A. O(n)
B. O(n log n)
C. O(n^2 log n)
explanation: 每个元素只需比较一次左右即可。
```

```quiz
title: 练习 2
question: 插入排序是否稳定？
answer: A
A. 稳定
B. 不稳定
C. 只能在链表上稳定
explanation: 只移动严格大于 key 的元素时，相等元素相对顺序不变。
```
