# Merge Sort

## 面试目标

掌握分治排序：递归拆分数组，排序左右两半，再线性合并。

## 核心流程

1. 如果区间长度小于等于 1，直接返回。
2. 递归排序左半和右半。
3. 使用双指针合并两个有序数组。
4. 将合并结果写回原数组或返回新数组。

## 复杂度

- 时间：`O(n log n)`
- 空间：`O(n)`
- 稳定性：稳定，取决于合并时相等元素优先取左边。

## 常见坑

- mid 计算和区间边界不统一。
- 合并后忘记拷贝剩余元素。
- 原地合并实现复杂，面试一般先写额外数组版本。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

递归排序左右两半，再用双指针合并。合并时相等元素优先取左边，可以保持稳定。

```text
mergeSort(arr):
  if len(arr) <= 1: return arr
  mid = len(arr) // 2
  left = mergeSort(arr[:mid])
  right = mergeSort(arr[mid:])
  return merge(left, right)

merge(left, right):
  use two pointers, append smaller one
  append remaining suffix
```

如果题目要求原数组排序，最后把临时数组写回原区间。

</details>

```quiz
title: 练习 1
question: 归并排序的核心思想是什么？
answer: B
A. 每轮选择最小值
B. 分治后合并有序子数组
C. 使用哈希表计数
explanation: 归并排序先递归排序子问题，再合并结果。
```

```quiz
title: 练习 2
question: 标准归并排序的额外空间复杂度通常是多少？
answer: B
A. O(1)
B. O(n)
C. O(n^2)
explanation: 合并阶段通常需要辅助数组。
```
