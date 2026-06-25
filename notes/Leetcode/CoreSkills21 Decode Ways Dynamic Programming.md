# Dynamic Programming：从递推到空间优化

## 面试目标

Dynamic Programming 不是“背模板”，而是把一个问题拆成稳定的子问题，然后按依赖顺序把答案算出来。

一套可复用的 DP 写法通常分三步：

1. 写出递推表达式。
2. 把递推表达式改成循环模式。
3. 根据依赖关系做空间优化。

这三步比直接写代码重要。只要递推表达式清楚，代码通常只是机械翻译。

## 什么时候想到 DP

题目有下面几个信号时，可以优先考虑 DP：

- 问“方案数 / 最小代价 / 最大收益 / 是否可行”。
- 决策过程可以拆成阶段，例如处理到第 `i` 个字符、第 `i` 个物品、第 `i` 天。
- 同一个子问题会被重复遇到。
- 当前答案可以由更小规模的答案推出。

DP 的本质是：

```text
state + transition + base case + iteration order
```

也就是：

- 状态：`dp[...]` 表示什么？
- 转移：当前状态从哪些旧状态来？
- 初始值：最小问题的答案是什么？
- 顺序：先算谁，后算谁？

## 第一步：写出递推表达式

先不要写循环。先用一句话定义状态。

常见状态形式：

```text
dp[i]       = 前 i 个元素 / 从 i 开始的后缀 的答案
dp[i][j]    = 两个维度共同决定的答案
dp[i][state]= 第 i 步处于某种状态时的答案
```

定义状态时要问两个问题：

1. 它能不能直接表达最终答案？
2. 它能不能从更小的状态转移过来？

然后写 choice。每道 DP 题的递推基本都是：

```text
dp[current] = combine(dp[previous states])
```

如果是求方案数，`combine` 通常是加法：

```text
dp[i] = dp[a] + dp[b]
```

如果是求最优值，`combine` 通常是 `min` 或 `max`：

```text
dp[i] = min(dp[a] + costA, dp[b] + costB)
```

如果是求可行性，`combine` 通常是 `or`：

```text
dp[i] = dp[a] or dp[b]
```

最后补 base case。base case 是递推的起点，不是细节。很多 DP bug 都来自 base case 没有想清楚。

## 第二步：写成循环模式

递推表达式写出来后，循环方向由依赖关系决定。

如果：

```text
dp[i] depends on dp[i - 1]
```

通常从左往右。

如果：

```text
dp[i] depends on dp[i + 1]
```

通常从右往左。

如果：

```text
dp[i][j] depends on dp[i - 1][j] and dp[i][j - 1]
```

通常双层循环，从小到大填表。

把递推写成循环时，可以按这个模板：

```text
initialize dp
set base cases
for state in valid order:
  compute dp[state] from previous states
return answer
```

注意：循环顺序不是风格问题，而是正确性问题。你必须保证当前状态需要的旧状态已经算好。

## 第三步：空间优化

空间优化不要一开始就做。先写清楚完整 DP，再观察依赖范围。

如果：

```text
dp[i] only depends on dp[i - 1]
```

可以只保留一个变量。

如果：

```text
dp[i] only depends on dp[i + 1] and dp[i + 2]
```

可以只保留两个变量。

如果二维 DP 只依赖上一行：

```text
dp[i][j] depends on dp[i - 1][...]
```

可以滚动数组，把 `O(nm)` 空间降到 `O(m)`。

空间优化的原则是：

> 只保留未来还会被用到的状态。

不要为了省空间破坏状态语义。面试里更推荐先讲完整 DP，再讲如何优化。

## Example：Decode Ways

现在用 Decode Ways 走一遍完整流程。

题目：给定只包含数字的字符串 `s`，按照：

```text
1 -> A
2 -> B
...
26 -> Z
```

问有多少种合法解码方式。

例如：

```text
s = "226"

2 | 2 | 6   -> B B F
22 | 6      -> V F
2 | 26      -> B Z

answer = 3
```

## Decode Ways：递推表达式

定义状态：

```text
dp[i] = s[i:] 这个后缀字符串的解码方案数
```

最终答案：

```text
dp[0]
```

Base case：

```text
dp[n] = 1
```

含义是：如果已经走到字符串末尾，说明前面形成了一种完整解码。

如果当前位置是 `'0'`：

```text
dp[i] = 0
```

因为 `0` 不能单独映射成任何字母。

否则当前位置有两类选择：

```text
take one digit  -> dp[i + 1]
take two digits -> dp[i + 2], if 10 <= s[i:i+2] <= 26
```

递推式：

```text
if s[i] == '0':
  dp[i] = 0
else:
  dp[i] = dp[i + 1]
  if i + 1 < n and 10 <= int(s[i:i+2]) <= 26:
    dp[i] += dp[i + 2]
```

这一步最重要。后面的循环和空间优化都只是对这个递推式的实现。

## Decode Ways：循环模式

因为 `dp[i]` 依赖 `dp[i + 1]` 和 `dp[i + 2]`，所以要从右往左算。

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        dp[n] = 1

        for i in range(n - 1, -1, -1):
            if s[i] == "0":
                dp[i] = 0
                continue

            dp[i] = dp[i + 1]

            if i + 1 < n and (
                s[i] == "1" or
                (s[i] == "2" and s[i + 1] in "0123456")
            ):
                dp[i] += dp[i + 2]

        return dp[0]
```

用 `"226"` 走一遍：

```text
dp[3] = 1

i = 2, "6":
  dp[2] = dp[3] = 1

i = 1, "26":
  dp[1] = dp[2] + dp[3] = 2

i = 0, "22":
  dp[0] = dp[1] + dp[2] = 3
```

## Decode Ways：空间优化

观察依赖：

```text
dp[i] only depends on dp[i + 1] and dp[i + 2]
```

所以只需要两个变量：

```text
one = dp[i + 1]
two = dp[i + 2]
```

每次算出：

```text
cur = dp[i]
```

然后窗口左移：

```text
two = one
one = cur
```

代码：

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        one = 1
        two = 0

        for i in range(len(s) - 1, -1, -1):
            if s[i] == "0":
                cur = 0
            else:
                cur = one

                if i + 1 < len(s) and (
                    s[i] == "1" or
                    (s[i] == "2" and s[i + 1] in "0123456")
                ):
                    cur += two

            two = one
            one = cur

        return one
```

## Example：Best Time to Buy and Sell Stock with Cooldown

这题比 Decode Ways 更适合练“状态机 DP”。

题目：给定每天股价 `prices`，可以买卖多次，但同一时间最多持有一股。卖出后，下一天不能买入，要 cooldown 一天。求最大利润。

例如：

```text
prices = [1, 2, 3, 0, 2]
answer = 3

一种最优操作：
day 0 buy at 1
day 1 sell at 2
day 2 cooldown
day 3 buy at 0
day 4 sell at 2
profit = 1 + 2 = 3
```

这题主要练两个记录技巧：

1. 先把状态记录清楚，最好画成状态机，再写状态方程。
2. 再看每一天只依赖前一天，于是把 DP 表压缩成常数空间。

## Stock Cooldown：技巧一，记录三个状态

很多人会用 `buying=True/False` 写递归，这当然可以。但 bottom-up 时，更直观的做法是记录每天结束后处于哪种状态。

定义三个状态：

```text
hold[i] = 第 i 天结束后，手里持有一股时的最大利润
sold[i] = 第 i 天结束后，今天刚卖出股票时的最大利润
rest[i] = 第 i 天结束后，手里没有股票，且不处于刚卖出的状态时的最大利润
```

这三个状态分别回答：

- `hold`：我现在手里有股票，未来可以卖。
- `sold`：我今天刚卖，明天必须 cooldown，不能买。
- `rest`：我现在手里没股票，也没有刚卖出的限制，未来可以买。

状态转移：

```text
hold[i] = max(
  hold[i - 1],              # 昨天就持有，今天继续不动
  rest[i - 1] - prices[i]   # 昨天是自由空仓，今天买入
)

sold[i] = hold[i - 1] + prices[i]
  # 昨天必须持有，今天才能卖出

rest[i] = max(
  rest[i - 1],              # 昨天自由空仓，今天继续休息
  sold[i - 1]               # 昨天刚卖出，今天 cooldown 后变成自由空仓
)
```

注意 cooldown 体现在这里：

```text
hold[i] 只能从 rest[i - 1] 买入，不能从 sold[i - 1] 买入
```

因为 `sold[i - 1]` 表示昨天刚卖，今天还在 cooldown，不能买。

Base case 可以这样理解：

```text
hold = -infinity  # 还没处理任何一天，不可能已经持股
sold = -infinity  # 还没处理任何一天，不可能已经卖出
rest = 0          # 什么都不做，利润为 0
```

最后答案不能是 `hold`，因为手里还拿着股票不算落袋利润：

```text
answer = max(sold[n - 1], rest[n - 1])
```

## Stock Cooldown：完整 DP 表

先写完整 DP，便于确认状态含义：

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        n = len(prices)
        neg_inf = float("-inf")
        hold = [neg_inf] * n
        sold = [neg_inf] * n
        rest = [0] * n

        hold[0] = -prices[0]
        sold[0] = neg_inf
        rest[0] = 0

        for i in range(1, n):
            hold[i] = max(hold[i - 1], rest[i - 1] - prices[i])
            sold[i] = hold[i - 1] + prices[i]
            rest[i] = max(rest[i - 1], sold[i - 1])

        return max(sold[n - 1], rest[n - 1])
```

用 `[1, 2, 3, 0, 2]` 走一遍：

```text
day  price  hold  sold  rest
0    1      -1    -inf  0
1    2      -1     1    0
2    3      -1     2    1
3    0       1    -1    2
4    2       1     3    2

answer = max(sold, rest) = max(3, 2) = 3
```

看第 3 天 `price = 0` 的 `hold = 1`：

```text
hold[3] = max(hold[2], rest[2] - 0)
        = max(-1, 1 - 0)
        = 1
```

这表示：第 1 天卖出赚了 `1`，第 2 天 cooldown 后进入 `rest = 1`，第 3 天可以买入价格 `0` 的股票，所以持股状态的利润仍是 `1`。

## Stock Cooldown：技巧二，空间优化

观察转移式：

```text
hold[i] only depends on hold[i - 1], rest[i - 1]
sold[i] only depends on hold[i - 1]
rest[i] only depends on rest[i - 1], sold[i - 1]
```

也就是说，第 `i` 天只用到第 `i - 1` 天的三个状态。因此不需要三条数组，只需要三个变量：

```text
hold = previous hold
sold = previous sold
rest = previous rest
```

每一天先算新状态，再整体替换：

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        hold = float("-inf")
        sold = float("-inf")
        rest = 0

        for price in prices:
            next_hold = max(hold, rest - price)
            next_sold = hold + price
            next_rest = max(rest, sold)

            hold, sold, rest = next_hold, next_sold, next_rest

        return max(sold, rest)
```

这里最重要的细节是：不要边算边覆盖旧变量。

错误写法：

```python
hold = max(hold, rest - price)
sold = hold + price  # wrong: 这里用到的是今天更新后的 hold
```

`sold` 必须来自“昨天持股，今天卖出”，所以要用旧的 `hold`。因此要先用 `next_*` 保存新状态，最后统一赋值。

## Stock Cooldown：和递归写法的关系

递归写法常用两个状态：

```text
dfs(i, buying)
```

- `buying = True`：手里没有股票，可以买。
- `buying = False`：手里有股票，可以卖。

卖出时跳到 `i + 2`：

```text
sell = prices[i] + dfs(i + 2, True)
```

三状态 bottom-up 写法把这个 `i + 2` 的 cooldown 显式拆成了 `sold -> rest`：

```text
hold --sell--> sold --cooldown one day--> rest --buy--> hold
```

所以两个写法本质相同，只是记录方式不同。面试里如果题目有“持有 / 刚卖 / 空仓可买”这种过程限制，三状态状态机会更不容易写错。

## 背包问题：0/1 Knapsack vs Complete Knapsack

背包 DP 的核心问题是：

```text
有一些物品，每个物品有 cost / weight / value。
在容量限制内，求是否可行、最大价值、最小数量、方案数。
```

面试里最重要的区别是：

```text
0/1 背包：每个物品最多用一次
完全背包：每个物品可以用无限次
```

这两个问题的代码经常只差一行循环方向，但语义完全不同。

## 0/1 背包例题：Target Sum

题目：给定整数数组 `nums` 和整数 `target`。对每个数都必须选择加号或减号，问有多少种表达式能得到 `target`。

例如：

```text
nums = [2, 2, 2], target = 2

+2 +2 -2 = 2
+2 -2 +2 = 2
-2 +2 +2 = 2

answer = 3
```

这题表面是加减号，实际可以转成 0/1 背包。

把所有被加号选中的数放进集合 `P`，被减号选中的数放进集合 `N`：

```text
sum(P) - sum(N) = target
sum(P) + sum(N) = total
```

两式相加：

```text
2 * sum(P) = target + total
sum(P) = (target + total) / 2
```

所以问题变成：

```text
从 nums 里选一些数，每个数最多选一次，使它们的和等于 bag = (target + total) / 2。
问有多少种选法。
```

这就是 0/1 背包的“方案数”版本。

什么时候无解？

```text
abs(target) > total
target + total 是奇数
```

因为 `sum(P)` 必须是一个非负整数。

## Target Sum：二维 0/1 背包

定义状态：

```text
dp[i][s] = 只看前 i 个数，凑出和 s 的方案数
```

Base case：

```text
dp[0][0] = 1
```

含义是：不用任何数，凑出 0 有一种方法，就是什么都不选。

转移：

```text
不选 nums[i - 1]：dp[i - 1][s]
选 nums[i - 1]：dp[i - 1][s - nums[i - 1]]

dp[i][s] = dp[i - 1][s] + dp[i - 1][s - nums[i - 1]]
```

代码：

```python
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if abs(target) > total or (target + total) % 2 == 1:
            return 0

        bag = (target + total) // 2
        n = len(nums)
        dp = [[0] * (bag + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            num = nums[i - 1]
            for s in range(bag + 1):
                dp[i][s] = dp[i - 1][s]
                if s >= num:
                    dp[i][s] += dp[i - 1][s - num]

        return dp[n][bag]
```

用例：

```text
nums = [2, 2, 2], target = 2
total = 6
bag = (2 + 6) / 2 = 4

问题变成：从 [2, 2, 2] 中选一些数，和为 4，有几种选法？

选第 1、2 个 2
选第 1、3 个 2
选第 2、3 个 2

answer = 3
```

注意：这里三个 `2` 是三个不同位置的数，所以选第 1、2 个和选第 1、3 个是不同方案。

## Target Sum：一维空间优化

二维转移只依赖上一行：

```text
dp[i][s] depends on dp[i - 1][s] and dp[i - 1][s - num]
```

所以可以压成一维：

```text
dp[s] = 当前已经处理过的数里，凑出和 s 的方案数
```

0/1 背包的一维循环必须倒序：

```text
for s from bag down to num:
```

原因：每个数只能用一次。倒序可以保证 `dp[s - num]` 还是“上一轮物品”的结果，而不是当前这个 `num` 刚更新出来的结果。

代码：

```python
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if abs(target) > total or (target + total) % 2 == 1:
            return 0

        bag = (target + total) // 2
        dp = [0] * (bag + 1)
        dp[0] = 1

        for num in nums:
            for s in range(bag, num - 1, -1):
                dp[s] += dp[s - num]

        return dp[bag]
```

为什么倒序这么重要？

假设只有一个数 `2`，`bag = 4`。

如果正序：

```text
dp[2] += dp[0]  # 用了一次 2
dp[4] += dp[2]  # 又立刻用刚更新的 dp[2]，等于同一个 2 用了两次
```

这就变成完全背包了，语义错了。

## 完全背包是什么

完全背包的区别是：

```text
每个物品可以用无限次
```

例如 coin change：

```text
coins = [1, 2, 5], amount = 11
```

每种硬币可以用很多枚，这就是完全背包。

完全背包的一维循环通常正序：

```text
for coin in coins:
  for s from coin to amount:
    dp[s] = combine(dp[s], dp[s - coin])
```

正序的含义是：允许当前 `coin` 被重复使用。因为当你计算 `dp[s]` 时，`dp[s - coin]` 可能已经包含当前 coin 的结果。

## 完全背包：最少硬币数

如果要求最少硬币数：

```text
dp[s] = 凑出金额 s 的最少硬币数
```

转移：

```text
dp[s] = min(dp[s], dp[s - coin] + 1)
```

代码：

```python
from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        inf = amount + 1
        dp = [inf] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for s in range(coin, amount + 1):
                dp[s] = min(dp[s], dp[s - coin] + 1)

        return -1 if dp[amount] == inf else dp[amount]
```

这里 `s` 正序，因为一枚 `coin` 可以被反复使用。

## 0/1 背包和完全背包怎么选

先问物品能不能重复用：

```text
每个元素只能选一次          -> 0/1 背包
每种物品可以选无限次        -> 完全背包
每个物品最多选 k 次         -> 多重背包
```

再问目标是什么：

```text
是否能凑出容量              -> boolean dp
有多少种凑法                -> count dp
最大价值 / 最小数量         -> max / min dp
```

最后决定循环方向：

```text
0/1 背包一维优化：
for item in items:
  for capacity from target down to item:
    use previous item layer

完全背包一维优化：
for item in items:
  for capacity from item up to target:
    allow using current item again
```

一句话记忆：

> 0/1 背包倒序，防止同一个物品重复使用；完全背包正序，主动允许同一个物品重复使用。

## 常见坑

- 没有先定义 `dp[i]` 的含义就开始写代码。
- 循环方向和依赖方向反了。
- 忘记 base case，例如 `dp[n] = 1`。
- 把非法状态当成 `1`，尤其是 Decode Ways 里的 `'0'`。
- 过早做空间优化，导致变量含义说不清楚。
- 股票 cooldown 题里，卖出后直接从 `sold` 买入，漏掉 cooldown。
- 空间优化时覆盖旧变量，导致 `sold` 使用了当天刚更新的 `hold`。
- 0/1 背包一维优化时用了正序，导致同一个物品被重复使用。
- Target Sum 忘记先判断 `(target + total)` 是否为偶数。
- Target Sum 忘记 `nums[i] = 0` 也会产生不同符号方案；背包计数写法会自然处理这个情况。

## 复杂度

Decode Ways 数组版：

- 时间：`O(n)`
- 空间：`O(n)`

空间优化版：

- 时间：`O(n)`
- 空间：`O(1)`

Stock Cooldown 三状态数组版：

- 时间：`O(n)`
- 空间：`O(n)`

Stock Cooldown 空间优化版：

- 时间：`O(n)`
- 空间：`O(1)`

Target Sum 0/1 背包：

- 时间：`O(n * bag)`，其中 `bag = (target + sum(nums)) / 2`
- 空间：二维版 `O(n * bag)`，一维版 `O(bag)`

完全背包：

- 时间：通常 `O(number_of_items * capacity)`
- 空间：一维优化后 `O(capacity)`

## 面试回答模板

DP 题可以这样讲：

1. 我先定义状态 `dp[...]` 表示什么。
2. 然后列出每一步有哪些选择。
3. 把每个选择转成对旧状态的引用，得到递推式。
4. 根据递推依赖决定循环顺序。
5. 先写完整 DP。
6. 最后观察只依赖哪些状态，再做空间优化。

```quiz
title: 练习 1
question: 写 DP 时第一步最应该明确什么？
answer: B
A. 使用哪种语言
B. dp 状态的含义
C. 是否能一行写完
explanation: 状态定义决定递推、base case 和最终答案。
```

```quiz
title: 练习 2
question: 如果 dp[i] 依赖 dp[i+1] 和 dp[i+2]，循环通常应该怎么走？
answer: A
A. 从右往左
B. 从左往右
C. 随机遍历
explanation: 必须先算出右侧依赖，才能计算当前状态。
```

```quiz
title: 练习 3
question: 什么时候适合做空间优化？
answer: C
A. 一开始就优化
B. 递推还没写出来时
C. 写出完整 DP 并确认每个状态只依赖少量旧状态后
explanation: 先保证状态语义和递推正确，再压缩空间更稳。
```
