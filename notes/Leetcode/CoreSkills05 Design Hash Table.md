# Design Hash Table

## 面试目标

实现哈希表，重点是哈希函数、桶数组、冲突处理、扩容和负载因子。

## 核心设计

- 用 `hash(key) % capacity` 定位桶。
- 冲突可用链地址法：每个桶保存一组 key-value。
- `put` 需要区分更新已有 key 和插入新 key。
- 当负载因子过高时扩容并 rehash。

## 复杂度

- 平均查找/插入/删除：`O(1)`
- 冲突严重时：`O(n)`
- 扩容 rehash：`O(n)`。

## 常见坑

- 扩容后只复制桶，没有重新计算下标。
- 删除 key 时忘记维护 size。
- 用对象引用做 key 时没有稳定哈希策略。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

链地址法最直接：桶数组中每个位置保存一个小列表，列表里是 `(key, value)`。

```text
put(key, value):
  bucket = buckets[hash(key) % capacity]
  for pair in bucket:
    if pair.key == key:
      pair.value = value
      return
  bucket.append((key, value))
  size += 1
  if size / capacity > 0.75: resize()
```

扩容时不能原样复制桶，要重新对每个 key 计算新桶下标。

</details>

```quiz
title: 练习 1
question: 哈希表扩容后为什么需要 rehash？
answer: B
A. 为了让值变大
B. capacity 改变后 key 对应的桶下标可能改变
C. 为了把所有 key 排序
explanation: 桶下标依赖 `hash(key) % capacity`，容量变化会改变映射。
```

```quiz
title: 练习 2
question: 链地址法如何处理哈希冲突？
answer: A
A. 同一个桶里保存多个 key-value
B. 直接丢弃新 key
C. 把数组改成二叉堆
explanation: 链地址法让冲突元素共存于同一个桶。
```
