# System Design 05 · 设计题基本流程

> Placeholder：这一页先记录 system design interview / design doc 的基本推进顺序，后面再逐步补每一步的展开方式、例题和模板。

## 五步流程

```text
1. Problem navigation
2. BOE
3. High-level architecture
4. Data entity
5. Deep dive
```

## 1. Problem navigation

先把题目边界讲清楚。

需要补充：

- 目标用户是谁
- 核心场景是什么
- 读写比例、数据规模、延迟目标、可用性目标
- 哪些功能先不做
- 哪些约束会改变架构选择

## 2. BOE

BOE 是 back-of-the-envelope estimation，也就是粗估。

需要补充：

- QPS / peak QPS
- read QPS / write QPS
- storage growth
- bandwidth
- cache size
- queue depth / worker count

## 3. High-level architecture

先画主链路，再逐步补组件。

需要补充：

- client / API gateway / service layer
- database / cache / queue / object store
- async worker / stream processing
- read path 和 write path
- 哪些组件是 stateless，哪些是 stateful

## 4. Data entity

把系统里真正需要保存的对象列出来。

需要补充：

- 核心表或 collection
- 主键和索引
- entity 之间的关系
- 哪些字段是 source of truth
- 哪些数据是 cache、view、derived data

## 5. Deep dive

选一两个风险最大的点深入。

需要补充：

- consistency
- sharding / hot key
- cache invalidation
- queue semantics
- failure recovery
- observability
- security / privacy

## 记忆版

```text
先问清楚题。
再估数量级。
先画大图。
再定数据模型。
最后挑瓶颈深挖。
```
