# Quant 5 · 正态分布：二维正态、Cholesky 与符号相关

这一讲围绕一道经典题：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
$$

其中 $(X,Y)$ 是二维标准正态，相关系数是 $\rho$。答案是：

$$
\boxed{
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
\frac{2}{\pi}\arcsin\rho
}
$$

这题最干净的解法不是直接积分，而是把相关二维正态换成两个独立标准正态，再用圆对称性把概率变成扇形角度。

---

## 1. 一维标准正态

标准正态 $N(0,1)$ 的密度是：

$$
\phi(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}
$$

它有三个基本性质：

| 性质 | 含义 |
| --- | --- |
| 关于 0 对称 | $P(X>0)=P(X<0)=1/2$ |
| 均值为 0 | 正负偏差抵消 |
| 方差为 1 | 作为标准尺度使用 |

连续正态变量落在某个精确点的概率是 0，所以：

$$
P(X=0)=0
$$

因此 $\operatorname{sgn}(X)$ 只需要考虑正负号。

---

## 2. 二维标准正态和相关系数

如果 $(X,Y)$ 是二维标准正态，并且：

$$
\mathbb{E}X=\mathbb{E}Y=0,
\qquad
\operatorname{Var}(X)=\operatorname{Var}(Y)=1,
\qquad
\operatorname{corr}(X,Y)=\rho
$$

那么它的协方差矩阵是：

$$
\Sigma=
\begin{pmatrix}
1 & \rho\\
\rho & 1
\end{pmatrix}
$$

$\rho$ 控制椭圆的倾斜方向：

| $\rho$ | 图像直觉 |
| --- | --- |
| $\rho>0$ | 椭圆沿 $y=x$ 方向拉长，两个变量更容易同号 |
| $\rho=0$ | 圆对称，两个变量独立 |
| $\rho<0$ | 椭圆沿 $y=-x$ 方向拉长，两个变量更容易异号 |

相关二维正态的一个标准构造是：

$$
U,V\overset{i.i.d.}{\sim}N(0,1),
\qquad U\perp V
$$

定义：

$$
X=U,\qquad
Y=\rho U+\sqrt{1-\rho^2}V
$$

于是 $(X,Y)$ 就是相关系数为 $\rho$ 的二维标准正态。

<figure class="quant-svg-figure">
<svg viewBox="0 0 920 360" role="img" aria-label="Cholesky transform from independent normal variables to correlated normal variables">
  <defs>
    <marker id="arrow-normal-1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#24485a" />
    </marker>
    <linearGradient id="normal-ellipse" x1="0" x2="1">
      <stop offset="0%" stop-color="#dff1ed" />
      <stop offset="100%" stop-color="#f8ead2" />
    </linearGradient>
  </defs>
  <rect x="18" y="18" width="884" height="324" rx="16" fill="#fbfcfa" stroke="#d0dce0" />
  <text x="78" y="55" class="quant-svg-title">independent plane</text>
  <text x="610" y="55" class="quant-svg-title">correlated plane</text>
  <g transform="translate(205 190)">
    <line x1="-120" y1="0" x2="125" y2="0" stroke="#5f7b88" stroke-width="2" marker-end="url(#arrow-normal-1)" />
    <line x1="0" y1="120" x2="0" y2="-125" stroke="#5f7b88" stroke-width="2" marker-end="url(#arrow-normal-1)" />
    <circle cx="0" cy="0" r="92" fill="#e9f4f1" stroke="#2c6b7f" stroke-width="3" />
    <path d="M-70 70 C-28 25 30 -35 72 -72" fill="none" stroke="#e08d3c" stroke-width="4" />
    <text x="132" y="6" class="quant-svg-label">U</text>
    <text x="8" y="-132" class="quant-svg-label">V</text>
    <text x="-78" y="116" class="quant-svg-note">density depends only on radius</text>
  </g>
  <g transform="translate(460 180)">
    <line x1="-58" y1="0" x2="58" y2="0" stroke="#24485a" stroke-width="3" marker-end="url(#arrow-normal-1)" />
    <text x="-64" y="-16" class="quant-svg-label">Cholesky</text>
    <text x="-98" y="38" class="quant-svg-formula">[X,Y]^T = L[U,V]^T</text>
  </g>
  <g transform="translate(710 190) rotate(-30)">
    <ellipse cx="0" cy="0" rx="126" ry="64" fill="url(#normal-ellipse)" stroke="#2c6b7f" stroke-width="3" />
    <path d="M-95 0 C-45 -16 42 18 96 0" fill="none" stroke="#e08d3c" stroke-width="4" />
  </g>
  <g transform="translate(710 190)">
    <line x1="-130" y1="0" x2="136" y2="0" stroke="#5f7b88" stroke-width="2" marker-end="url(#arrow-normal-1)" />
    <line x1="0" y1="118" x2="0" y2="-125" stroke="#5f7b88" stroke-width="2" marker-end="url(#arrow-normal-1)" />
    <text x="142" y="6" class="quant-svg-label">X</text>
    <text x="8" y="-132" class="quant-svg-label">Y</text>
    <text x="-116" y="116" class="quant-svg-note">rho tilts the ellipse</text>
  </g>
</svg>
<figcaption>独立标准正态在 $(U,V)$ 平面里是圆对称的。Cholesky 线性变换把圆形等密度线拉成相关二维正态的椭圆。</figcaption>
</figure>

---

## 3. 为什么这个构造是对的

先看均值：

$$
\mathbb{E}X=0,\qquad
\mathbb{E}Y=\rho\mathbb{E}U+\sqrt{1-\rho^2}\mathbb{E}V=0
$$

再看方差：

$$
\operatorname{Var}(Y)
=
\rho^2\operatorname{Var}(U)
+
(1-\rho^2)\operatorname{Var}(V)
=
1
$$

协方差是：

$$
\operatorname{Cov}(X,Y)
=
\operatorname{Cov}(U,\rho U+\sqrt{1-\rho^2}V)
=
\rho
$$

因为 $U,V$ 独立，所以 $\operatorname{Cov}(U,V)=0$。又因为 $X,Y$ 的方差都是 1：

$$
\operatorname{corr}(X,Y)=\rho
$$

矩阵写法是：

$$
\begin{pmatrix}
X\\
Y
\end{pmatrix}
=
\begin{pmatrix}
1&0\\
\rho&\sqrt{1-\rho^2}
\end{pmatrix}
\begin{pmatrix}
U\\
V
\end{pmatrix}
$$

这个矩阵就是相关矩阵：

$$
\begin{pmatrix}
1&\rho\\
\rho&1
\end{pmatrix}
$$

的 Cholesky factor。

这一步需要 jointly normal。只知道 $X,Y$ 各自是标准正态、相关系数是 $\rho$，还不够推出这个线性表示。

---

## 4. 把符号乘积换成同号概率

因为 $P(X=0)=P(Y=0)=0$：

```text
same sign:
  sgn(X)sgn(Y) = 1

opposite sign:
  sgn(X)sgn(Y) = -1
```

因此：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
P(\text{same sign})-P(\text{opposite sign})
$$

总概率是 1，所以：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
2P(\text{same sign})-1
$$

二维正态关于原点对称：

$$
P(X>0,Y>0)=P(X<0,Y<0)
$$

令：

$$
p=P(X>0,Y>0)
$$

则：

$$
P(\text{same sign})=2p
$$

所以：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
4p-1
$$

---

## 5. 用独立正态平面算 $p$

由 Cholesky 表示：

$$
X=U,\qquad
Y=\rho U+\sqrt{1-\rho^2}V
$$

所以：

$$
p
=
P(U>0,\ \rho U+\sqrt{1-\rho^2}V>0)
$$

现在看 $(U,V)$ 平面。因为 $U,V$ 独立标准正态，它们的联合密度是：

$$
f(u,v)=\frac1{2\pi}e^{-(u^2+v^2)/2}
$$

这个密度只依赖半径：

$$
r=\sqrt{u^2+v^2}
$$

不依赖角度。因此，一个过原点的扇形区域，概率只取决于扇形角度：

$$
P((U,V)\text{ 落在角度为 }\theta\text{ 的扇形})
=
\frac{\theta}{2\pi}
$$

---

## 6. 扇形角度从哪里来

两个条件分别给出两个半平面：

$$
U>0
$$

和：

$$
\rho U+\sqrt{1-\rho^2}V>0
$$

第二个条件的边界线是：

$$
V=-\frac{\rho}{\sqrt{1-\rho^2}}U
$$

令：

$$
\alpha=\arcsin\rho
$$

那么这条边界线与正 $U$ 轴的夹角是 $-\alpha$。第一条边界 $U=0$ 对应角度 $\pi/2$。两个半平面交出来的扇形角度是：

$$
\frac{\pi}{2}+\alpha
=
\frac{\pi}{2}+\arcsin\rho
$$

<figure class="quant-svg-figure quant-svg-wide">
<svg viewBox="0 0 1060 520" role="img" aria-label="Sector geometry for the probability P of X greater than zero and Y greater than zero">
  <defs>
    <marker id="arrow-normal-2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#24485a" />
    </marker>
    <filter id="normal-card-shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="8" stdDeviation="7" flood-color="#18313f" flood-opacity="0.10" />
    </filter>
    <linearGradient id="sector-fill" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f7c66f" stop-opacity="0.86" />
      <stop offset="100%" stop-color="#df8f38" stop-opacity="0.46" />
    </linearGradient>
  </defs>
  <rect x="18" y="18" width="1024" height="484" rx="18" fill="#fbfcfa" stroke="#d0dce0" />
  <text x="52" y="58" class="quant-svg-title">sector in the independent (U,V) plane</text>
  <text x="52" y="88" class="quant-svg-note">example: rho = 0.5, alpha = arcsin(rho) = 30 degrees</text>
  <g transform="translate(330 286)">
    <circle cx="0" cy="0" r="182" fill="#f8fbfa" stroke="#d7e3e6" stroke-width="2" />
    <circle cx="0" cy="0" r="122" fill="none" stroke="#e8eff1" stroke-width="2" />
    <circle cx="0" cy="0" r="64" fill="none" stroke="#eef3f4" stroke-width="2" />
    <path d="M0 0 L157.6 91 A182 182 0 0 0 0 -182 Z" fill="url(#sector-fill)" stroke="#d1812c" stroke-width="3" />
    <line x1="-214" y1="0" x2="222" y2="0" stroke="#52707d" stroke-width="2" marker-end="url(#arrow-normal-2)" />
    <line x1="0" y1="206" x2="0" y2="-218" stroke="#52707d" stroke-width="2" marker-end="url(#arrow-normal-2)" />
    <line x1="-188" y1="-108.5" x2="188" y2="108.5" stroke="#24485a" stroke-width="3" stroke-dasharray="10 8" />
    <line x1="0" y1="190" x2="0" y2="-190" stroke="#24485a" stroke-width="3" />
    <path d="M77.9 45 A90 90 0 0 0 0 -90" fill="none" stroke="#9b4c18" stroke-width="5" stroke-linecap="round" />
    <path d="M64 0 A64 64 0 0 1 55.4 32" fill="none" stroke="#9b4c18" stroke-width="3" />
    <rect x="45" y="-141" width="112" height="32" rx="8" fill="#fff8e8" stroke="#edc77f" />
    <text x="60" y="-120" class="quant-svg-note">U&gt;0, Y&gt;0</text>
    <rect x="15" y="-214" width="74" height="28" rx="7" fill="#ffffff" stroke="#d0dce0" />
    <text x="27" y="-195" class="quant-svg-label">U = 0</text>
    <rect x="-190" y="-143" width="74" height="28" rx="7" fill="#ffffff" stroke="#d0dce0" />
    <text x="-178" y="-124" class="quant-svg-label">Y = 0</text>
    <text x="230" y="6" class="quant-svg-label">U</text>
    <text x="8" y="-225" class="quant-svg-label">V</text>
    <text x="45" y="-70" class="quant-svg-formula">pi/2 + alpha</text>
    <text x="76" y="52" class="quant-svg-formula">-alpha</text>
  </g>
  <g transform="translate(628 130)" filter="url(#normal-card-shadow)">
    <rect x="0" y="0" width="355" height="290" rx="16" fill="#ffffff" stroke="#d4e0e3" />
    <text x="24" y="42" class="quant-svg-label">What the picture is doing</text>
    <circle cx="33" cy="82" r="5" fill="#24485a" />
    <text x="52" y="87" class="quant-svg-note">U = 0 keeps the right half-plane.</text>
    <circle cx="33" cy="120" r="5" fill="#24485a" />
    <text x="52" y="125" class="quant-svg-note">Y = 0 is the slanted dashed line.</text>
    <circle cx="33" cy="158" r="5" fill="#d1812c" />
    <text x="52" y="163" class="quant-svg-note">The overlap is the shaded sector.</text>
    <line x1="24" y1="188" x2="331" y2="188" stroke="#e5edef" />
    <text x="24" y="222" class="quant-svg-formula">sector angle = pi/2 + alpha</text>
    <text x="24" y="252" class="quant-svg-formula">p = (pi/2 + alpha) / (2pi)</text>
    <text x="24" y="282" class="quant-svg-formula">alpha = arcsin(rho)</text>
  </g>
</svg>
<figcaption>在 $(U,V)$ 平面里，联合密度是圆对称的。$U>0$ 和 $Y>0$ 的交集是一个扇形，概率等于扇形角度除以 $2\pi$。</figcaption>
</figure>

所以：

$$
p
=
\frac{\frac{\pi}{2}+\arcsin\rho}{2\pi}
=
\frac14+\frac{\arcsin\rho}{2\pi}
$$

---

## 7. 代回符号乘积期望

前面得到：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
4p-1
$$

代入：

$$
p=\frac14+\frac{\arcsin\rho}{2\pi}
$$

得到：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
4\left(\frac14+\frac{\arcsin\rho}{2\pi}\right)-1
=
\frac{2}{\pi}\arcsin\rho
$$

---

## 8. 特殊值检查

| $\rho$ | 情况 | 公式结果 |
| --- | --- | --- |
| $0$ | 独立，同号和异号一样多 | $0$ |
| $1$ | $Y=X$，永远同号 | $1$ |
| $-1$ | $Y=-X$，永远异号 | $-1$ |

代入公式：

$$
\frac{2}{\pi}\arcsin 0=0
$$

$$
\frac{2}{\pi}\arcsin 1=1
$$

$$
\frac{2}{\pi}\arcsin(-1)=-1
$$

都和直觉一致。

---

## 9. 结构总结

整个推导依赖两点。

第一，相关二维正态可以写成：

$$
X=U,\qquad
Y=\rho U+\sqrt{1-\rho^2}V
$$

第二，独立标准正态平面是圆对称的。任何过原点的扇形概率只由角度决定。
