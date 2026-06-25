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

## Kadane's Algorithm：最大子数组和

Kadane 算法解决的是 Maximum Subarray：

```text
给定数组 nums，找到一个连续子数组，使它的和最大。
```

例如：

```text
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
answer = 6

最优子数组是 [4, -1, 2, 1]
sum = 6
```

这题经常被当成贪心讲，但本质也可以看成非常简洁的一维 DP。

## Kadane：状态定义

定义状态：

```text
dp[i] = 必须以 nums[i] 结尾的最大连续子数组和
```

注意是“必须以 `i` 结尾”，不是“前 `i` 个元素里的最大答案”。这个定义让转移非常稳定。

对于 `nums[i]`，只有两种选择：

```text
1. 接在前面的子数组后面：dp[i - 1] + nums[i]
2. 从当前位置重新开始：nums[i]
```

所以状态方程：

```text
dp[i] = max(dp[i - 1] + nums[i], nums[i])
```

最终答案：

```text
answer = max(dp[i] for all i)
```

完整 DP 写法：

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        ans = nums[0]

        for i in range(1, n):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
            ans = max(ans, dp[i])

        return ans
```

## Kadane：为什么负数前缀可以丢掉

如果当前累积和是负数：

```text
curSum < 0
```

那么它接到任何未来子数组前面，只会让未来的和变小。

例如未来从 `x` 开始：

```text
curSum + x < x
```

所以负数前缀没有保留价值，可以直接从下一个位置重新开始。

这就是 Kadane 里这句的含义：

```python
if curSum < 0:
    curSum = 0
```

## Kadane：空间优化写法

因为 `dp[i]` 只依赖 `dp[i - 1]`，所以不需要完整数组，只保留一个变量：

```text
curSum = 当前正在延续的子数组和
maxSub = 到目前为止见过的最大子数组和
```

代码：

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSub = nums[0]
        curSum = 0

        for num in nums:
            if curSum < 0:
                curSum = 0
            curSum += num
            maxSub = max(maxSub, curSum)

        return maxSub
```

为什么 `maxSub = nums[0]`，而不是 `0`？

因为数组可能全是负数：

```text
nums = [-5, -2, -7]
answer = -2
```

如果把答案初始化成 `0`，就会错误地返回空子数组。但题目要求子数组非空，所以必须用第一个元素初始化答案。

## Kadane：手动走一遍

用经典例子：

```text
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
```

过程：

```text
num   curSum after update   maxSub
-2    -2                    -2
 1     1                     1      # 前面 curSum < 0，丢掉 -2
-3    -2                     1
 4     4                     4      # 前面 curSum < 0，从 4 重新开始
-1     3                     4
 2     5                     5
 1     6                     6
-5     1                     6
 4     5                     6
```

最终答案是 `6`。

## Kadane：和普通 DP 的关系

Kadane 可以这样理解：

```text
完整 DP：
dp[i] = max(dp[i - 1] + nums[i], nums[i])
ans = max(ans, dp[i])

空间优化：
curSum = dp[i - 1]
更新后 curSum = dp[i]
```

所以它不是魔法，本质就是：

```text
定义“以当前位置结尾”的状态，然后把 dp 数组压成一个变量。
```

这类题常见变形：

- 最大乘积子数组：需要同时记录最大值和最小值，因为负数会翻转符号。
- 环形最大子数组：需要比较普通最大子数组和 `total_sum - 最小子数组和`。
- 买卖股票一次最大利润：可以看成维护历史最低买入价，也可以转成最大差分子数组。

## Jump Game：DP 什么时候可以变成 Greedy

Jump Game 问的是：

```text
给定 nums，从 index 0 出发。
nums[i] 表示从 i 最多可以向右跳 nums[i] 步。
问能不能到达最后一个 index。
```

例子：

```text
nums = [2, 3, 1, 1, 4]
answer = true

0 -> 1 -> 4
```

反例：

```text
nums = [3, 2, 1, 0, 4]
answer = false

无论怎么跳，都会被 index 3 的 0 卡住。
```

这题很适合用来理解一个更重要的问题：

> 什么样的 DP 可以进一步压成 Greedy？

不是所有 DP 都能贪心。能贪心的 DP，通常有一个共同特征：

```text
一大批状态的答案，可以被一个单调边界、最优代表、或极值变量完整概括。
```

Jump Game 里的这个代表变量就是 `goal`。

## Jump Game：递归视角

先不要急着写 greedy。先从最直接的搜索开始。

定义递归函数：

```text
dfs(i) = 从 index i 出发，能不能到达最后一个 index
```

从 `i` 可以跳到：

```text
i + 1, i + 2, ..., i + nums[i]
```

所以递归关系是：

```text
dfs(i) = dfs(i + 1) or dfs(i + 2) or ... or dfs(i + nums[i])
```

代码：

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        def dfs(i):
            if i >= len(nums) - 1:
                return True

            end = min(len(nums) - 1, i + nums[i])
            for j in range(i + 1, end + 1):
                if dfs(j):
                    return True

            return False

        return dfs(0)
```

这个写法逻辑清楚，但会重复计算大量状态。比如很多路径都会问同一个问题：

```text
从 index 3 出发能不能到终点？
```

所以它会超时。

## Jump Game：Top-Down DP

加 memo 后，状态不变：

```text
dfs(i) = 从 i 出发能不能到最后
```

只是把算过的 `dfs(i)` 缓存起来。

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        memo = {}

        def dfs(i):
            if i >= len(nums) - 1:
                return True

            if i in memo:
                return memo[i]

            end = min(len(nums) - 1, i + nums[i])
            for j in range(i + 1, end + 1):
                if dfs(j):
                    memo[i] = True
                    return True

            memo[i] = False
            return False

        return dfs(0)
```

复杂度：

```text
Time:  O(n^2)
Space: O(n)
```

为什么是 `O(n^2)`？

因为每个 `i` 最多算一次，但每个 `i` 内部可能扫描很多个 `j`。

## Jump Game：Bottom-Up DP

把递归反过来，从右往左填表。

定义：

```text
dp[i] = 从 index i 出发，能不能到达最后一个 index
```

base case：

```text
dp[n - 1] = true
```

转移：

```text
dp[i] = any(dp[j] == true for j in [i + 1, i + nums[i]])
```

代码：

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [False] * n
        dp[n - 1] = True

        for i in range(n - 2, -1, -1):
            end = min(n - 1, i + nums[i])
            for j in range(i + 1, end + 1):
                if dp[j]:
                    dp[i] = True
                    break

        return dp[0]
```

到这里还是标准 DP。

关键问题来了：

```text
dp[i] 真的需要知道右边每一个 dp[j] 吗？
```

如果需要，那就不能贪心。

如果不需要，而只需要知道一个“最关键的位置”，那就有机会变成 greedy。

## Jump Game：观察 DP 表

看例子：

```text
nums = [2, 3, 1, 1, 4]
index  0  1  2  3  4
```

从右往左看。

最后一个位置一定是好位置：

```text
index  0  1  2  3  4
good              T
goal = 4
```

`i = 3`：

```text
3 + nums[3] = 3 + 1 = 4
```

它能跳到 `goal = 4`，所以 index 3 也是好位置：

```text
index  0  1  2  3  4
good           T  T
goal = 3
```

`i = 2`：

```text
2 + nums[2] = 3
```

它能跳到 `goal = 3`，所以 index 2 也是好位置：

```text
index  0  1  2  3  4
good        T  T  T
goal = 2
```

`i = 1`：

```text
1 + nums[1] = 4
```

它能跳过 `goal = 2`，当然也能到达好位置，所以：

```text
goal = 1
```

`i = 0`：

```text
0 + nums[0] = 2
```

它能到达 `goal = 1`，所以：

```text
goal = 0
```

最终 `goal == 0`，答案是 `true`。

这里发生了一件很重要的事情：

```text
DP 原本保存所有 good positions。
Greedy 只保存最左边的 good position。
```

为什么只保存最左边的好位置就够了？

因为如果某个 index `i` 能跳到最左边的好位置 `goal`，那么它一定能到达终点。

如果 `i` 连最左边的好位置都跳不到，那么它跳到更右边的好位置只会更难。

所以右侧一整段 DP 信息，可以被一个边界变量压缩：

```text
goal = 当前已经知道的、最靠左的可达终点的位置
```

## Jump Game：Greedy 写法

从右往左扫描：

```text
如果 i 能跳到 goal 或超过 goal：
    i 变成新的 goal
```

代码：

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) - 1

        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= goal:
                goal = i

        return goal == 0
```

复杂度：

```text
Time:  O(n)
Space: O(1)
```

这不是“凭感觉贪心”，而是从 DP 压缩出来的 greedy：

```text
dp array of good positions
        ↓
leftmost good position
        ↓
goal
```

## Jump Game：为什么这个 Greedy 是正确的

我们维护一个不变量：

```text
在扫描到 i 的时候，goal 是 [i + 1, n - 1] 中最靠左的好位置。
```

好位置的意思是：

```text
从这个位置出发，可以到达最后一个 index。
```

当我们检查 index `i`：

```text
i + nums[i] >= goal
```

说明 `i` 可以一步跳到 `goal`，而 `goal` 已经能到终点。

所以 `i` 也能到终点。并且 `i` 比旧的 `goal` 更靠左，于是更新：

```text
goal = i
```

如果：

```text
i + nums[i] < goal
```

说明 `i` 连最靠左的好位置都够不到。

更右边的好位置位置更大，更不可能够到，所以 `i` 不是好位置。

扫描结束后：

```text
goal == 0
```

就表示 index 0 是好位置。

## DP 什么时候可以转换为 Greedy

可以用下面这组问题判断。

### 1. DP 状态是否只在维护一个“可行集合”

Jump Game 的 DP 是：

```text
dp[i] = i 是否能到终点
```

它维护的是一组好位置：

```text
good positions = { i | dp[i] == true }
```

如果 DP 维护的是复杂数值，例如每个状态都有不同代价、不同路径、不同选择历史，通常不容易直接贪心。

### 2. 这个集合能不能被一个边界表示

Jump Game 里，我们不需要知道所有好位置，只需要知道：

```text
最左边的好位置 goal
```

原因是目标只有一个：

```text
能不能到达某个好位置
```

而 `goal` 是所有好位置里最容易够到的那个。

这就是单调边界。

### 3. 局部更新会不会破坏未来选择

当 `i` 能到 `goal` 时，我们把 `goal` 更新成 `i`。

这不会让未来更差，反而让未来更容易：

```text
旧 goal 更靠右
新 goal 更靠左
```

之后更左边的 index 想要够到一个好位置，只会更容易，不会更难。

这种性质叫贪心选择安全：

```text
local choice does not remove any globally optimal possibility
```

### 4. 是否存在“支配关系”

在 Jump Game 里：

```text
更靠左的 good position 支配更靠右的 good position
```

因为对于左侧的某个 `i` 来说：

```text
能跳到更左的 good position
=> 一定比跳到更右的 good position 更容易
```

所以我们可以丢掉所有更右的 good positions，只保留最左边那个。

这是 DP 转 greedy 最重要的信号：

> 多个状态之间存在支配关系，被支配的状态永远不可能比代表状态更有用。

## DP 转 Greedy 的通用模式

很多 greedy 不是凭空想出来的，而是这样推出来的：

```text
1. 写出完整 DP。
2. 观察 DP 保存了哪些信息。
3. 判断这些信息是否有单调性。
4. 找到能代表一整批状态的变量。
5. 证明只保留这个变量不会丢答案。
```

Jump Game 对应：

```text
完整 DP：dp[i] 表示 i 是否 good
保存信息：所有 good positions
单调性：越靠左的 good position 越容易被左侧 index reach
代表变量：leftmost good position，也就是 goal
安全性：能 reach goal 就一定能 reach end
```

所以：

```text
O(n^2) DP
  -> 保存所有 good positions
  -> 只保存 leftmost good position
  -> O(n) Greedy
```

## 另一个 Greedy 角度：从左往右维护最远可达

Jump Game 还有一个常见写法，从左往右扫：

```text
reach = 当前为止最远能到达的位置
```

如果扫描到 `i` 时：

```text
i > reach
```

说明当前位置都到不了，后面也没法继续，返回 `False`。

否则更新：

```text
reach = max(reach, i + nums[i])
```

代码：

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        reach = 0

        for i, jump in enumerate(nums):
            if i > reach:
                return False
            reach = max(reach, i + jump)

        return True
```

这个写法维护的是另一个单调边界：

```text
[0, reach] 这一段都已经可以被某种方式到达
```

只要 `reach` 不断向右扩张，最后超过 `n - 1` 就可行。

两个 greedy 写法本质一样：

```text
从右往左：维护最左 good position
从左往右：维护最远 reachable position
```

它们都把一整组 DP 状态压成了一个边界变量。

## Jump Game：面试里怎么讲

比较好的回答顺序是：

1. 我先定义 `dp[i]`：从 `i` 出发能不能到终点。
2. 递推是检查 `i` 能跳到的所有 `j`，只要某个 `dp[j]` 为 true，`dp[i]` 就为 true。
3. 这个 bottom-up DP 是 `O(n^2)`。
4. 观察到我们不需要保存所有 true 的位置，只需要保存最左边的 true 位置 `goal`。
5. 如果 `i + nums[i] >= goal`，那 `i` 可以到达一个已知好位置，所以 `i` 也是好位置。
6. 因为更左的好位置对未来更有利，所以更新 `goal = i` 是安全的。
7. 最后判断 `goal == 0`。

一句话总结：

> 这题能从 DP 变成 greedy，是因为“所有能到终点的位置”可以被“最左边的能到终点的位置”完整代表。

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
- Kadane 把 `maxSub` 初始化成 `0`，导致全负数数组返回错误。
- Kadane 忘记子数组必须连续，不能像子序列一样随意跳过中间元素。
- Jump Game 把 `nums[i]` 当成唯一目的地；它其实是最大跳跃长度，可以跳到区间 `[i + 1, i + nums[i]]` 里的任意位置。
- Jump Game 直接说“每次跳最远”但不给证明；面试里要解释为什么状态集合可以被 `goal` 或 `reach` 这个边界代表。
- Jump Game 从右往左 greedy 时忘记条件是 `i + nums[i] >= goal`，不是 `nums[i] >= goal`。

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

Kadane 最大子数组和：

- 时间：`O(n)`
- 空间：完整 DP 为 `O(n)`，空间优化后为 `O(1)`

Jump Game：

- 递归：时间指数级，空间 `O(n)`
- Top-down / bottom-up DP：时间 `O(n^2)`，空间 `O(n)`
- Greedy：时间 `O(n)`，空间 `O(1)`

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
