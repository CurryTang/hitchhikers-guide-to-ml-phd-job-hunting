# Topological Sort：把偏序约束排成线性答案

## 面试目标

拓扑排序不是一种“排序数组”的算法，而是把一组依赖关系排成合法执行顺序的图算法。只要题目里出现了“先后关系、依赖、课程、构建顺序、字母顺序、任务调度”，就应该先问自己：这些关系能不能建成一个有向图？

如果每条边 `u -> v` 表示 `u` 必须排在 `v` 前面，那么拓扑排序要输出一个序列，使得图里每条边都从序列左边指向右边。只有 **DAG（Directed Acyclic Graph，有向无环图）** 才一定存在拓扑序；如果存在环，比如 `a -> b -> c -> a`，就代表互相依赖，无法给出合法顺序。

## 什么是拓扑排序

给定一个有向图 `G = (V, E)`：

- `V` 是节点，例如课程、任务、字符。
- `E` 是约束，例如 `pre -> course`、`dependency -> target`、`smaller letter -> larger letter`。
- 拓扑序是一个包含所有节点的线性排列，并满足：对每条边 `u -> v`，`u` 都出现在 `v` 前面。

它解决的是 **偏序到线性序** 的问题。偏序只告诉你一部分先后关系，例如 `h < e`、`e < r`，但没有说所有字符之间都能直接比较。拓扑排序会在不违反已知约束的前提下，给出任意一个合法线性答案。

## Kahn 算法：从“没有前置依赖”的节点开始

Kahn 算法是面试里最直观的 BFS 写法。

1. 建图，并统计每个节点的入度 `indegree`。
2. 把所有入度为 `0` 的节点放入队列。
3. 每次弹出一个节点 `u`，把它加入答案。
4. 遍历 `u` 的所有后继 `v`，把 `indegree[v] -= 1`。
5. 如果某个后继入度降到 `0`，说明它的前置依赖已经全部完成，可以入队。
6. 最后如果答案长度小于节点数，说明图里有环。

```text
queue = all nodes with indegree 0
order = []

while queue:
  u = queue.pop_front()
  order.append(u)

  for v in graph[u]:
    indegree[v] -= 1
    if indegree[v] == 0:
      queue.push_back(v)

if len(order) != len(nodes):
  there is a cycle
else:
  order is a valid topological ordering
```

复杂度是 `O(V + E)`，因为每个节点入队出队一次，每条边只被处理一次。

## DFS 写法：用三色标记找环

DFS 也能做拓扑排序，核心是“后序加入答案”：

- `0 = unvisited`：还没访问过。
- `1 = visiting`：正在当前 DFS 路径上。
- `2 = visited`：这个节点和它的后继都已经处理完。

如果 DFS 时遇到 `visiting` 节点，说明当前路径形成了环；如果一个节点的所有后继都处理完，再把它加入答案。最后反转答案即可得到拓扑序。

Kahn 更适合解释“入度、依赖释放”；DFS 更适合写递归和检测环。面试中两种都可以，但要确保你能清楚解释边方向。

## 可视化：Foreign Dictionary 如何变成拓扑排序

Foreign Dictionary / Alien Dictionary 的关键不是字符串处理本身，而是从相邻单词里抽出字符之间的偏序关系。以 `words = ["hrn", "hrf", "er", "enn", "rfnn"]` 为例：

这道题的本质流程是：

1. 初始化所有出现过的字符，避免漏掉没有边的孤立节点。
2. 只比较相邻单词，因为字典整体有序时，相邻对提供的是最小必要约束。
3. 找到相邻单词的第一个不同字符 `a` 和 `b`，加入边 `a -> b`，然后停止比较这一对。
4. 如果没有找到不同字符，但前一个单词更长，例如 `["abc", "ab"]`，这是非法输入，直接返回空字符串。
5. 对字符图做拓扑排序；如果检测到环，也返回空字符串。

## Foreign Dictionary 作为例题

题目给出一个外星语言的有序词典 `words`。这些单词仍由英文字母组成，但字母大小顺序未知。你需要返回一个合法的字母顺序；如果不存在合法顺序，返回空字符串。题目源：<https://neetcode.io/problems/foreign-dictionary/question?list=neetcode150>

```topo-demo
foreign-dictionary
```

### 建图规则

对任意相邻单词 `word1` 和 `word2`：

```text
word1 = h r n
word2 = h r f
             ^
第一个不同字符是 n 和 f，所以 n 必须排在 f 前面，建边 n -> f
```

注意，只看第一个不同字符。后面的字符不能继续拿来建边，因为词典序在第一个差异处就已经决定了两个单词的大小。

### Python 解法：Kahn BFS

```python
from collections import defaultdict, deque
from typing import List


class Solution:
    def foreignDictionary(self, words: List[str]) -> str:
        graph = {char: set() for word in words for char in word}
        indegree = {char: 0 for char in graph}

        for first, second in zip(words, words[1:]):
            min_len = min(len(first), len(second))

            if len(first) > len(second) and first[:min_len] == second[:min_len]:
                return ""

            for i in range(min_len):
                if first[i] != second[i]:
                    src, dst = first[i], second[i]
                    if dst not in graph[src]:
                        graph[src].add(dst)
                        indegree[dst] += 1
                    break

        queue = deque([char for char, degree in indegree.items() if degree == 0])
        order = []

        while queue:
            char = queue.popleft()
            order.append(char)

            for nxt in graph[char]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(indegree):
            return ""

        return "".join(order)
```

时间复杂度是 `O(N + V + E)`，其中 `N` 是所有单词总长度，`V` 是不同字符数，`E` 是字符偏序边数。空间复杂度是 `O(V + E)`。

### 为什么这段代码能过

- `graph = {char: set() ...}` 先把所有字符注册成节点，保证答案包含孤立字符。
- `if dst not in graph[src]` 防止重复边把入度加多次。
- prefix invalid case 必须在比较字符前处理，否则 `["abc", "ab"]` 会被错误当成没有新约束。
- `len(order) != len(indegree)` 是 Kahn 算法的环检测；如果有环，环内节点永远不会降到入度 `0`。

## 常见坑

- 边方向写反：如果 `word1` 在 `word2` 前面，且第一个不同字符是 `a/b`，应该建 `a -> b`。
- 比较了非首个不同字符：词典序只由第一个不同字符决定。
- 没有处理 prefix invalid：`["abc", "ab"]` 必须返回 `""`。
- 用 list 存邻居但没有去重，导致入度被重复增加。
- 忘记把所有字符放进图，导致答案缺字符。
- 认为答案必须唯一；题目通常允许返回任意一个合法拓扑序。

```quiz
title: 练习 1
question: 拓扑排序要求图满足什么条件？
answer: A
A. 有向无环图
B. 任意无向图
C. 必须是完全图
explanation: 有环依赖无法给出合法线性顺序。
```

```quiz
title: 练习 2
question: Foreign Dictionary 中，比较相邻两个单词时应该用哪一对字符建边？
answer: B
A. 所有位置不同的字符
B. 第一个不同字符
C. 最后一个不同字符
explanation: 词典序在第一个差异处就已经决定大小，后面的字符不提供可靠约束。
```

```quiz
title: 练习 3
question: 输入 words = ["abc", "ab"] 为什么非法？
answer: C
A. c 的入度不是 0
B. 两个单词长度不同
C. 较短单词是较长单词的前缀，但较长单词排在前面
explanation: 合法词典序中，前缀更短的单词必须排在较长单词前。
```
