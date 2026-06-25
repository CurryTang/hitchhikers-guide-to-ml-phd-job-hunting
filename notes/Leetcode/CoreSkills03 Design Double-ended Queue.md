# Design Double-ended Queue

## 面试目标

实现双端队列，支持队头和队尾的插入、删除。常见实现是循环数组或双向链表。

## 核心设计

- 循环数组维护 `front`、`size`、`capacity`。
- 队尾位置可以由 `(front + size) % capacity` 计算。
- `pushFront` 让 `front = (front - 1 + capacity) % capacity`。
- `popBack` 只减少 `size`，无需移动元素。

## 复杂度

- 两端插入和删除：`O(1)`
- 随机访问如果实现 `get(i)`：`O(1)`
- 扩容时复制：`O(n)`，摊还后插入仍为 `O(1)`。

## 常见坑

- 负数取模导致 front 变成负数。
- 队尾下标计算 off-by-one。
- 扩容后没有把元素重新排成从 0 开始的逻辑顺序。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

循环数组版本只保存 `front` 和 `size`。逻辑下标 `i` 映射到物理下标 `(front + i) % capacity`。

```text
pushFront(x):
  grow if full
  front = (front - 1 + capacity) % capacity
  data[front] = x
  size += 1

pushBack(x):
  grow if full
  data[(front + size) % capacity] = x
  size += 1
```

扩容时按逻辑顺序复制 `data[(front+i)%old_capacity]` 到新数组的 `i`，然后把 `front` 重置为 0。

</details>

```quiz
title: 练习 1
question: 循环数组中计算下标通常为什么要取模？
answer: B
A. 为了排序
B. 为了让下标到达末尾后回到开头
C. 为了删除重复元素
explanation: 取模可以复用数组空间，形成逻辑上的环。
```

```quiz
title: 练习 2
question: pushFront 时 front 应该如何移动？
answer: A
A. 向前一格，并处理越界回绕
B. 向后一格
C. 保持不变，只修改尾指针
explanation: 队头插入需要先空出新的头位置。
```
