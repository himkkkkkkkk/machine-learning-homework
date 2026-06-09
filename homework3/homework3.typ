#set document(
  title: [机器学习第三次作业],
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

*1.* C

*2.*

(a)
+ 动画：
  $ Delta H = -(1/2 ln 1/2 + 1/2 ln 1/2)+3/8 (1/3 ln 1/3 + 2/3 ln 2/3) + 5/8 (2/5 ln 2/5 +3/5 ln 3/5)=0.0338 $
+ 弹出：
  $ Delta H = ln 2 + 5/8(1/5 ln 1/5 + 4/5 ln 4/5) = 0.380 $
+ 多彩：
  $ Delta H = ln 2 + 2 * 1/2 (1/4 ln 1/4 + 3/4 ln 3/4) = 0.131 $
故最大的信息收益为0.380

(b)
#import "@preview/cetz:0.5.2": canvas, draw

#canvas({
  import draw: *

  // node: thin black frame, no fill, centered content
  let node(pos, body, name) = content(
    pos, align(center, body),
    frame: "rect", stroke: 0.5pt, padding: 0.3em, name: name,
  )

  // italic Yes/No edge label placed at an absolute point
  let lab(pos, body) = content(pos, text(style: "italic", size: 9pt, body))

  // --- decision nodes ---
  node((2, 6),   [弹出?],   "pop")
  node((4, 4),   [多彩?], "col")
  node((5.5, 2), [动画?], "ani")

  // --- leaves (verdict + stored training points) ---
  node((0, 4),   [No click \ 1, 2, 3], "l1")
  node((3, 2),   [Click \ 4, 5, 7],    "l2")
  node((4.5, 0), [Click \ 6],          "l3")
  node((6.5, 0), [No click \ 8],       "l4")

  // --- edges: parent.south -> child.north ---
  line("pop.south", "l1.north", stroke: 0.5pt)
  line("pop.south", "col.north", stroke: 0.5pt)
  line("col.south", "l2.north", stroke: 0.5pt)
  line("col.south", "ani.north", stroke: 0.5pt)
  line("ani.south", "l3.north", stroke: 0.5pt)
  line("ani.south", "l4.north", stroke: 0.5pt)

  // --- edge labels ---
  lab((1.35, 5.15),  [Yes])
  lab((2.5, 5.15),  [No])
  lab((3.8, 3.15), [Yes])
  lab((4.4, 3.15), [No])
  lab((5.3, 1.15), [Yes])
  lab((5.75, 1.15),  [No])
})

(c)

错分率 $epsilon = 1/8$

则 $alpha_m = 1/2 ln (1-1/8)/(1/8) = 1/2 ln 7$

则样本8的权重 $omega_8^(\(2\)) = omega_8^(\(1\))*e^(alpha_m)=sqrt(7)/8$

