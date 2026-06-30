# Quant 2 · 骰子点数出现次数乘积的期望

## 题目

一枚公平的六面骰子被掷 10 次。对每个点数 $i \in \{1,2,\ldots,6\}$，令 $N_i$ 表示点数 $i$ 出现的次数。

求：

$$
\mathbb{E}[N_1N_2N_3N_4N_5N_6]
$$

---

## 核心理解

这题的随机变量不是单个 $N_i$，而是六个计数的乘积。因为 $N_1,\ldots,N_6$ 来自同一组 10 次投掷，它们不是独立的：

```text
某个点数出现得多，
其他点数能出现的总次数就会变少。
```

所以不能写成：

$$
\mathbb{E}[N_1]\mathbb{E}[N_2]\cdots\mathbb{E}[N_6]
$$

正确工具是 multinomial distribution 的 factorial moment，或者等价地用 indicator expansion 数组合。

---

## 方法一：Multinomial Falling Factorial Moment

10 次公平骰子投掷后：

$$
(N_1,\ldots,N_6) \sim \text{Multinomial}\left(10;\frac16,\ldots,\frac16\right)
$$

Multinomial 有一个很重要的公式：

$$
\mathbb{E}\left[\prod_{j=1}^k (N_j)_{a_j}\right]
=
(n)_{a_1+\cdots+a_k}\prod_{j=1}^k p_j^{a_j}
$$

其中：

$$
(x)_a = x(x-1)\cdots(x-a+1)
$$

叫 falling factorial。

这题里每个点数只取一次，所以 $a_1=\cdots=a_6=1$。因为 $(N_i)_1=N_i$，所以：

$$
\mathbb{E}[N_1N_2N_3N_4N_5N_6]
=
(10)_6\left(\frac16\right)^6
$$

展开：

$$
(10)_6 = 10\cdot 9\cdot 8\cdot 7\cdot 6\cdot 5
$$

因此答案是：

$$
\boxed{\frac{10\cdot9\cdot8\cdot7\cdot6\cdot5}{6^6}}
$$

也可以写成：

$$
\boxed{\frac{151200}{46656}=\frac{175}{54}}
$$

---

## 方法二：Indicator Expansion

这个方法更适合面试时解释为什么公式成立。

把 $N_i$ 写成 indicator 之和。设第 $t$ 次投掷是否为点数 $i$：

$$
X_{t,i} =
\mathbf{1}\{\text{第 }t\text{ 次掷出 }i\}
$$

那么：

$$
N_i = \sum_{t=1}^{10} X_{t,i}
$$

所以：

$$
N_1N_2N_3N_4N_5N_6
=
\sum_{t_1,\ldots,t_6}
X_{t_1,1}X_{t_2,2}X_{t_3,3}X_{t_4,4}X_{t_5,5}X_{t_6,6}
$$

关键点：如果两个 $t$ 相同，比如 $t_1=t_2$，那么同一次投掷不可能同时是点数 1 和点数 2，所以这一项一定是 0。

只有当：

```text
t_1, t_2, t_3, t_4, t_5, t_6
全部互不相同
```

这一项才可能非零。

有多少种互不相同的有序选择？

$$
10\cdot9\cdot8\cdot7\cdot6\cdot5 = (10)_6
$$

对每一种选择，对应 6 次指定投掷分别掷出 $1,2,3,4,5,6$，概率是：

$$
\left(\frac16\right)^6
$$

所以：

$$
\mathbb{E}[N_1N_2N_3N_4N_5N_6]
=
(10)_6\left(\frac16\right)^6
=
\frac{175}{54}
$$

---

## 为什么不是把期望相乘

每个 $N_i$ 的期望都是：

$$
\mathbb{E}[N_i] = \frac{10}{6}
$$

但不能因此得到：

$$
\left(\frac{10}{6}\right)^6
$$

原因是 $N_i$ 之间有负相关。总次数固定为 10：

$$
N_1+\cdots+N_6=10
$$

如果 $N_1$ 很大，剩下五个计数的总空间就变小。乘积期望需要处理这些依赖关系。

---

## 一句话记忆

多项分布里，普通计数乘积经常转成 falling factorial moment：

$$
\mathbb{E}[N_1N_2\cdots N_k]
=
(n)_k p_1p_2\cdots p_k
$$

前提是每个计数只出现一次。直觉上就是：

```text
先选出 k 个互不相同的 trial，
再要求它们分别落到指定的 k 个类别。
```
