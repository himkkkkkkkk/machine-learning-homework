#set document(
  title: [机器学习第二次作业],
  author: "冯裕凯",
  date: datetime.today(),
)

#set page(
  number-align: bottom + right,
  header: align(right + horizon, text(size: 0.8em, context document.title)),
  numbering: "1",
)

#set math.equation(block: true)

#show title: set align(center)
#show title: set text(24pt)

#let pd(f, x) = $(partial #f) / (partial #x)$
#let iv(x) = $#x^tack.b$
#let ss(i, a, b) = $sum_(#i=#a)^#b$

#title()
#align(right)[#context document.author.at(0) \ #context document.date.display()]

*1.* B 因为线性函数不能作为激活函数

*2.* D Adam只是梯度下降算法的一种,不能完成提高泛化,降低过拟合的作用

*3.* C 参数数量只与通道数,核大小有关

*4.* idk

*5.* D

*6.*  C

_(i)_ 测试时bn层依然会运行,虽然无法对单个样本进行归一化

_(ii)_ Adam也可以

#v(0.5em)
*7.*

#table(
  columns: (auto, 1.2fr, 1fr),
  align: horizon,
  table.header([*Layer*], [*Activation map dimensions*], [*Number of parameters*]),
  [INPUT], [$128 times 128 times 3$], [0],
  [CONV-9-32], [$[(128-9)+1]^2 times 32 = 120 times 120 times 32$], [$32 times (9 times 9 times 3 +1)$],
  [POOL-2], [$[(120-2)/2 +1]^2 times 32 = 60 times 60 times 32$], [0],
  [CONV-5-64], [$(60-5+1)^2 times 64 = 56 times 56 times 64$], [$64 times (5 times 5 times 32 +1 )$],
  [POOL-2], [$[(56-2)/2 + 1]^2 times 64 = 28 times 28 times 64$], [0],
  [FC-3], [3], [$3 times (28 times 28 times 64 +1)$],
)

#v(0.5em)
*8.*
_forward_:
#let grid1 = table(
  columns: 4,
  align: center + horizon,
  inset: 10pt,
  stroke: 0.7pt + black,

  [4], [6], [3], [4],
  [#underline[1]], [7], [3], [4],
  [5], [8], [4], [4],
  [2], [7], [4], [4],
)

#let grid2 = table(
  columns: 2,
  align: center + horizon,
  inset: 10pt,
  stroke: 0.7pt + black,

  [$5/2$], [$-3/2$],
  [$9/4$], [$-3/4$],
)

#grid(
  columns: 5,
  align: center + horizon,
  gutter: 10pt,
  grid1, [$ limits(arrow.r)^* $], grid2, [$arrow.r$], [$5/2$],
)

_backward_:
#let grid1 = table(
  columns: 4,
  align: center + horizon,
  inset: 10pt,
  stroke: 0.7pt + black,

  [0], [$-1/4$], [0], [0],
  [#underline[$-1/4$]], [1], [$-1/4$], [0],
  [0], [$-1/4$], [0], [0],
  [0], [0], [0], [0],
)

#let grid2 = table(
  columns: 2,
  align: center + horizon,
  inset: 10pt,
  stroke: 0.7pt + black,

  [$1$], [$0$],
  [$0$], [$0$],
)

#grid(
  columns: 5,
  align: center + horizon,
  gutter: 10pt,
  grid1, [$ limits(arrow.l) $], grid2, [$arrow.l$], [$1$],
)

因此,梯度为$-1/4$

*9.*

_a_
$
  W_1 : D_(a_1) times D_(x) , b_1 : D_(a_1) times 1 \
  W_2 : 1 times D_(a_1) times 1 , b_2 : 1 times 1
$
_b_
$
  pd(L^i, z_3) & = pd(L^i, hat(y)^i) pd(hat(y)^i, z_3) \
               & = [y^i 1/hat(y)^i - (1-y^i) 1/(1-hat(y)^i)] sigma(z_3)(1-sigma (z_3)) \
               & = [y^i / sigma(z_3) - (1-y^i)/(1-sigma(z_3))]sigma(z_3)(1-sigma (z_3)) \
               & = y^i (1-sigma(z_3)) - (1-y^i)sigma(z_3) \
               & = y^i - sigma(z_3)
$
_c_
$
  pd(z_3, a) = iv(W_2)
$
_d_
$ pd(a, z_2) = pd(a, a_2) pd(a_2, z_2) = -pd(a_2, z_2) = -bb(1)_(z_2>=0) $
_e_

与_d_同理,$ pd(a, z_1) = bb(1)_(z_1>=0) $

_f_
$
  pd(L^i, W_1) & = pd(L^i, z_3)pd(z_3, a) ( pd(a, z_1) pd(z_1, W_1) + pd(a, z_2) pd(z_2, W_1) ) \
               & = (y^i - sigma(z_3)) iv(W_2) ( bb(1)_(z_1>=0) iv(x^i) - bb(1)_(z_1>=0) iv(x^(prime i)))
$
_g_
$
  pd(J, W_1) = sum_(i=1)^m pd(J, L^i)pd(L^i, W_1) & = - 1/m sum_(i=1)^m pd(L^i, W_1) \
  & = - 1/m sum_(i=1)^m (y^i - sigma(z_3)) iv(W_2) ( bb(1)_(z_1>=0) iv(x^i) - bb(1)_(z_1>=0) iv(x^(prime i)))
$
_h_
$
  W_1 <- W_1 - alpha pd(J, W_1)\
  W_2 <- W_2 - alpha pd(J, W_2)\
  b_1 <- b_1 - alpha pd(J, b_1)\
  b_2 <- b_2 - alpha pd(J, b_2)
$
