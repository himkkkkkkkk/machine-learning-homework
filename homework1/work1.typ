#set document(
  title:[机器学习第一次作业],
  author:"冯裕凯",
  date:datetime.today(),
)

#set page(
  number-align: bottom + right ,
  header: align(right + horizon,text(size:0.8em, context document.title)),
  numbering: "1",
)

#set math.equation(block: true)

#show title: set align(center)
#show title: set text(24pt)

#let pd(f,x) = $(partial #f) / (partial #x)$
#let iv(x) = $#x^tack.b$
#let ss(i,a,b) = $sum_(#i=#a)^#b$

#title()
#align(right)[#context document.author.at(0) \ #context document.date.display()]
#v(0.2em)
== 求下列问题的对偶问题
#v(0.2em)
*a* $ min {1/2 || omega ||^2 + C_1 sum_(i=1)^N delta_i + C_2 sum_(i=1)^N delta_i^2} $
    $ "subject to:" & delta_i >= 0,i=1,2,dots,N\ & y_i [iv(omega) phi(x_i) + b] >= 1-delta_i, i=1,2,dots,N $
*answer:* $ L(b,omega,delta_i,lambda_i,zeta_i)= 1/2 || omega ||^2 + C_1 sum_(i=1)^N delta_i + C_2 sum_(i=1)^N delta_i^2\
-sum_(i=1)^N lambda_i delta_i - sum_(i=1)^N zeta_i {y_i [iv(omega) phi(x_i) + b]-1+delta_i} $
$ &pd(L,omega) =& omega - ss(i,1,N) zeta_i y_i phi(x_i) = &0 \
&pd(L,delta_i) =& C_1 + 2C_2 delta_i - lambda_i - zeta_i = &0\
&pd(L,b) =& -sum_(i=1)^N zeta_i y_i =&0 $
带入得，
$ theta(lambda,zeta) &= min_(b,omega,delta) L\
&=ss(i,1,N) zeta_i - C_2 ss(i,1,N) delta_i^2 + 1/2 ||omega||^2\
&= ss(i,1,N) zeta_i -ss(i,1,N) (lambda_i+zeta_i-C_1)^2/(4C_2) + 1/2 sum_(i,j) zeta_i zeta_j y_i y_j iv(phi(x_i))phi(x_j) $

*b* $ min{1/2 ||omega||^2 + C_1 ss(i,1,N) (delta_i + zeta_i) +C_2 ss(i,1,N) (delta_i^2 + zeta_i^2)} $
$ "subject to:" & iv(omega)phi(x_i) + b -y_i <= epsilon +delta_i,i=1,2,...,N\
& y_i-iv(omega)phi(x_i) - b <=epsilon+zeta_i,i=1,2,...,N\
&delta_i,zeta_i>=0,i=1,2,...,N $
*answer:* $ L(omega,b,delta_i,zeta_i,lambda_i,eta_i,alpha_i,beta_i)\
= 1/2 ||omega||^2 + C_1 ss(i,1,N) (delta_i + zeta_i) +C_2 ss(i,1,N) (delta_i^2 + zeta_i^2)\
+ss(i,1,N) lambda_i (iv(omega)phi(x_i) + b -y_i - epsilon - delta_i)\
+ss(i,1,N) eta_i (y_i-iv(omega)phi(x_i) - b - epsilon - zeta_i)\
-ss(i,1,N) alpha_i delta_i -ss(i,1,N) beta_i zeta_i  $
$ pd(L,omega)&= ||omega|| + ss(i,1,N) lambda_i phi(x_i) - ss(i,1,N) eta_i phi(x_i) = 0\
  pd(L,b) &= ss(i,1,N) (lambda_i-eta_i) = 0\
  pd(L,delta_i) &= C_1 + 2C_2 delta_i -lambda_i - alpha_i = 0\
  pd(L,zeta_i) &= C_1 + 2C_2 zeta_i - eta_i - beta_i $
带入得，
$ theta(lambda,eta,alpha,beta) =& 
  1/2 (sum_(i,j)(lambda_i-eta_i)(lambda_j-eta_j)iv(phi(x_i))phi(x_j))\
  &+ C_1 ss(i,1,N) frac(lambda_i+eta_i+alpha_i+beta_i-2 C_1,2C_2)\
  &+ C_2 ss(i,1,N) frac((lambda_i+alpha_i-C_1)^2 + (eta_i+beta_i-c_1)^2,4C_2^2) $
