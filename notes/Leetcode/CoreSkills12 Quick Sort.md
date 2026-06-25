# Quick Sort

## 面试目标

掌握快速排序的 partition：选 pivot，把小于 pivot 的元素放一边，大于 pivot 的元素放另一边。

## 核心流程

1. 选择 pivot。
2. 通过双指针或 Lomuto/Hoare partition 重排区间。
3. 递归排序 pivot 左右两侧。
4. 使用随机 pivot 可降低退化概率。

## 复杂度

- 平均时间：`O(n log n)`
- 最坏时间：`O(n^2)`
- 递归栈空间：平均 `O(log n)`，最坏 `O(n)`。
- 稳定性：通常不稳定。

## 常见坑

- partition 后递归区间包含 pivot，导致死递归。
- 有大量重复元素时退化，可使用三路快排。
- 固定选首元素在已排序数组上容易退化。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

先 partition，再递归两侧。Lomuto 写法最容易讲清楚：`store` 指向下一个小于 pivot 的位置。

```text
quickSort(l, r):
  if l >= r: return
  p = partition(l, r)
  quickSort(l, p - 1)
  quickSort(p + 1, r)

partition(l, r):
  pivot = arr[r]
  store = l
  for i in [l, r):
    if arr[i] < pivot:
      swap(arr[i], arr[store])
      store += 1
  swap(arr[store], arr[r])
  return store
```

实际面试可以先随机选择 pivot 并换到 `r`，降低退化概率。

</details>

```quiz
title: 练习 1
question: 快速排序最坏时间复杂度是多少？
answer: C
A. O(n)
B. O(n log n)
C. O(n^2)
explanation: pivot 每次极度不平衡时会退化。
```

```quiz
title: 练习 2
question: 随机选择 pivot 的主要目的是什么？
answer: A
A. 降低遇到最坏划分的概率
B. 保证稳定排序
C. 消除递归栈
explanation: 随机化让输入顺序不容易稳定触发最坏情况。
```
