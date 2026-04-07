#set document(
  title: [机器学习第一次作业],
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
#v(0.2em)
== 求下列问题的对偶问题
#v(0.2em)
*a* $ min {1/2 || omega ||^2 + C_1 sum_(i=1)^N delta_i + C_2 sum_(i=1)^N delta_i^2} $
$
  "subject to:" & delta_i >= 0,i=1,2,dots,N \
                & y_i [iv(omega) phi(x_i) + b] >= 1-delta_i, i=1,2,dots,N
$
*answer:* $ L(b,omega,delta_i,lambda_i,zeta_i)= 1/2 || omega ||^2 + C_1 sum_(i=1)^N delta_i + C_2 sum_(i=1)^N delta_i^2\
-sum_(i=1)^N lambda_i delta_i - sum_(i=1)^N zeta_i {y_i [iv(omega) phi(x_i) + b]-1+delta_i} $
$
  & pd(L, omega) =   & omega - ss(i, 1, N) zeta_i y_i phi(x_i) = & 0 \
  & pd(L, delta_i) = &  C_1 + 2C_2 delta_i - lambda_i - zeta_i = & 0 \
  & pd(L, b) =       &                 -sum_(i=1)^N zeta_i y_i = & 0
$
带入得，
$
  theta(lambda, zeta) &= min_(b,omega,delta) L\
  &= C_1 ss(i, 1, N) delta_i - C_2 ss(i, 1, N) delta_i^2 + 1/2 ||omega||^2\
  &= C_1 ss(i, 1, N) (lambda_i+zeta_i-C_1)/(2C_2) -ss(i, 1, N) (lambda_i+zeta_i-C_1)^2/(4C_2) + 1/2 sum_(i,j) zeta_i zeta_j y_i y_j iv(phi(x_i))phi(x_j)
$

*b* $ min{1/2 ||omega||^2 + C_1 ss(i, 1, N) (delta_i + zeta_i) +C_2 ss(i, 1, N) (delta_i^2 + zeta_i^2)} $
$
  "subject to:" & iv(omega)phi(x_i) + b -y_i <= epsilon +delta_i,i=1,2,...,N \
                & y_i-iv(omega)phi(x_i) - b <=epsilon+zeta_i,i=1,2,...,N \
                & delta_i,zeta_i>=0,i=1,2,...,N
$
*answer:* $ L(omega,b,delta_i,zeta_i,lambda_i,eta_i,alpha_i,beta_i)\
= 1/2 ||omega||^2 + C_1 ss(i, 1, N) (delta_i + zeta_i) +C_2 ss(i, 1, N) (delta_i^2 + zeta_i^2)\
+ss(i, 1, N) lambda_i (iv(omega)phi(x_i) + b -y_i - epsilon - delta_i)\
+ss(i, 1, N) eta_i (y_i-iv(omega)phi(x_i) - b - epsilon - zeta_i)\
-ss(i, 1, N) alpha_i delta_i -ss(i, 1, N) beta_i zeta_i $
$
    pd(L, omega) & = ||omega|| + ss(i, 1, N) lambda_i phi(x_i) - ss(i, 1, N) eta_i phi(x_i) = 0 \
        pd(L, b) & = ss(i, 1, N) (lambda_i-eta_i) = 0 \
  pd(L, delta_i) & = C_1 + 2C_2 delta_i -lambda_i - alpha_i = 0 \
   pd(L, zeta_i) & = C_1 + 2C_2 zeta_i - eta_i - beta_i
$
带入得，
$
  theta(lambda, eta, alpha, beta) = & 1/2 (sum_(i,j)(lambda_i-eta_i)(lambda_j-eta_j)iv(phi(x_i))phi(x_j)) \
                                    & + C_1 ss(i, 1, N) frac(lambda_i+eta_i+alpha_i+beta_i-2 C_1, 2C_2) \
                                    & + C_2 ss(i, 1, N) frac((lambda_i+alpha_i-C_1)^2 + (eta_i+beta_i-c_1)^2, 4C_2^2)
$

*c* $ min{1/2 ||omega||^2+C ss(i, 1, N) delta_i^2 } $
$ "subject to:" y_i-iv(omega) phi(x_i) = delta_i $
*answer:* $ L(omega,delta_i,lambda_i)= 1/2 ||omega||^2+C ss(i, 1, N) delta_i^2+ss(i, 1, N) lambda_i (y_i-iv(omega) phi(x_i) - delta_i) $
$
  & pd(L, omega) = ||omega|| - ss(i, 1, N) lambda_i phi(x_i) = 0 \
  & pd(L, delta_i) = 2C delta_i - lambda_i = 0
$
带入得，
$ theta(lambda) = 1/2 sum_(i,j) (lambda_i lambda_j iv(phi(x_i))phi(x_j))+1/(4C) ss(i, 1, N) lambda_i^2 $
*d*
$ min{ss(i, 1, N) x_i ln(x_i)} $
$
  "subject to:" & iv(omega)x <= b \
                & ss(i, 1, N) x_i = 1
$
*answer:* $ L(x,lambda,eta) = ss(i, 1, N) x_i ln(x_i) + lambda (iv(omega)x-b) + eta(ss(i, 1, N)x_i -1) $
$ pd(L, x_i) = ln(x_i)+1+lambda omega_i+eta = 0 $
带入得，
$ theta(lambda, eta) = -ss(i, 1, N) e^(-(eta+lambda omega_i +1)) (eta+lambda omega_i +1) $

_2._
A and B

_3._

*(1)* 更容易过拟合，因为要求严格的分类容易因为一部分离群点导致分类曲线（直线）过于复杂。

*(2)* 更容易欠拟合，因为间隔无穷大也没有惩罚，机器倾向于用过于简单的曲线（直线）来（用间隔进行补偿）进行分类，导致欠拟合。

*(3.1)* 过拟合导致模型十分不稳定，方差大。

*(3.2)* 欠拟合会导致模型不准确，偏度大。

_4._

根据我们的代码,我们得到结果:\
```bash
linear AUC = 0.515
linear ERR = 0.490
poly AUC = 0.991
poly ERR = 0.031
rbf AUC = 0.999
rbf ERR = 0.004
sigmoid AUC = 0.524
sigmoid ERR = 0.473
```

因此,多项式核以及rbf核的拟合效果最好,而线性拟合以及sigmoid核拟合效果一般较差.

所有的代码和结果存储在output文件夹中.
