# Design Singly Linked List

## 面试目标

实现单链表，重点是指针更新、头节点处理、按下标遍历和插入删除边界。

## 核心设计

- 节点保存 `val` 和 `next`。
- 可以使用 dummy head 简化头部插入和删除。
- `get(index)` 从头开始走 `index` 步。
- `insertHead(val)` 新节点指向旧头，再更新 head。
- `remove(index)` 找到前驱节点后跳过目标节点。

## 复杂度

- 头部插入：`O(1)`
- 按下标访问/插入/删除：`O(n)`
- 额外空间：`O(1)`，不计新节点。

## 常见坑

- 删除第一个节点时忘记更新 head。
- 遍历条件多走或少走一步。
- 删除尾节点时没有正确断开前驱的 `next`。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

用 dummy head 可以让插入和删除头节点不需要单独分支。`getPrev(index)` 返回目标位置前一个节点。

```text
insert(index, val):
  prev = dummy
  repeat index times:
    prev = prev.next
  node = Node(val)
  node.next = prev.next
  prev.next = node

remove(index):
  prev = getPrev(index)
  prev.next = prev.next.next
```

头插就是 `insert(0, val)`；尾插可以维护 `tail` 优化，也可以遍历到末尾。

</details>

```quiz
title: 练习 1
question: 单链表按下标 get(index) 的时间复杂度通常是多少？
answer: B
A. O(1)
B. O(n)
C. O(log n)
explanation: 单链表不能随机访问，需要从头节点逐个向后走。
```

```quiz
title: 练习 2
question: dummy head 主要解决什么问题？
answer: A
A. 统一处理头节点插入和删除
B. 让链表支持二分查找
C. 自动把链表变成数组
explanation: dummy head 让真实头节点也有前驱，减少分支判断。
```
