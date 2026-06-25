# Design Heap

## 面试目标

实现堆，掌握数组表示完全二叉树、上浮、下沉和优先队列操作。

## 核心设计

- 最小堆满足父节点值不大于子节点。
- 数组下标关系：`parent=(i-1)//2`，`left=2*i+1`，`right=2*i+2`。
- 插入：放到末尾后 bubble up。
- 删除堆顶：末尾元素换到根，再 bubble down。

## 复杂度

- peek：`O(1)`
- push/pop：`O(log n)`
- heapify：`O(n)`

## 常见坑

- 下沉时没有选择更小的子节点。
- pop 后忘记返回原堆顶。
- 空堆操作没有处理。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

数组表示完全二叉树。插入后向上和父节点比较；删除根后把最后一个元素放到根，再向下和更小的子节点交换。

```text
push(x):
  data.append(x)
  i = last index
  while i > 0 and data[i] < data[parent(i)]:
    swap(i, parent(i))
    i = parent(i)

pop():
  ans = data[0]
  data[0] = data.pop()
  heapify_down(0)
  return ans
```

`heapify_down` 每次选择左右孩子中更小的那个，只有它比当前节点更小时才交换。

</details>

```quiz
title: 练习 1
question: 最小堆的堆顶是什么？
answer: A
A. 当前最小元素
B. 当前最大元素
C. 最后插入的元素
explanation: 最小堆要求父节点不大于子节点，因此根节点最小。
```

```quiz
title: 练习 2
question: 堆中 push 一个元素的时间复杂度通常是多少？
answer: B
A. O(1)
B. O(log n)
C. O(n^2)
explanation: 新元素最多沿树高上浮，完全二叉树高度是 O(log n)。
```
