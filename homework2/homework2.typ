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
  W_2 : 1 times D_(a_1), b_2 : 1 times 1
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
               & = (y^i - sigma(z_3)) iv(W_2) ( bb(1)_(z_1>=0) iv(x^i) - bb(1)_(z_2>=0) iv(x^(prime i)))
$
_g_
$
  pd(J, W_1) = sum_(i=1)^m pd(J, L^i)pd(L^i, W_1) & = - 1/m sum_(i=1)^m pd(L^i, W_1) \
  & = - 1/m sum_(i=1)^m (y^i - sigma(z_3)) iv(W_2) ( bb(1)_(z_1>=0) iv(x^i) - bb(1)_(z_2>=0) iv(x^(prime i)))
$
_h_
$
  W_1 <- W_1 - alpha pd(J, W_1)\
  W_2 <- W_2 - alpha pd(J, W_2)\
  b_1 <- b_1 - alpha pd(J, b_1)\
  b_2 <- b_2 - alpha pd(J, b_2)
$

*10.*

```bash
machine-learning-homework/homework2/code HEAD*​​ 16s 
ml ❯ uv run main.py train
Test Accuracy: 90.73% (9073/10000)
Model saved to tmp/fashion_mnist/model.pt

machine-learning-homework/homework2/code HEAD*​​ 3m3s 
ml ❯ uv run main.py infer 42
pre0|exp3

machine-learning-homework/homework2/code HEAD*​​ 4s 
ml ❯ uv run main.py infer 43
pre7|exp7
```
上面是运行结果摘要。

下面将摘出任务的代码实现。

_(1)_
```py
FASHION_MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor()]
)
```
_(2)_
```py
class Fashion_Mnist_Model(nn.Module):
    def __init__(self,num_class: int = 10,dropout:float = 0.6):
        super().__init__()
        self.c1 = nn.Conv2d(1,8,3) # 28*28 -> 26*26 -> 16*16
        self.c2 = nn.Conv2d(8,16,4) # 16*16 -> 13*13 -> 8*8
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.a = nn.SiLU()
        self.pool2 = nn.AdaptiveAvgPool2d(8)
        self.pool1 = nn.AdaptiveAvgPool2d(16)
        self.fc = nn.Linear(8*8*16,num_class)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.c1(x)
        x = self.bn1(x)
        x = self.a(x)
        x = self.pool1(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.a(x)
        x = self.pool2(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        return x
```
_(3)_
```py
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    loss_f = nn.CrossEntropyLoss()
```
_(4)_
```py
for epoch in range(1,num_epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for image,label in  train:
            image,label = image.to(device),label.to(device)
            optim.zero_grad()
            output = model.forward(image)
            loss = loss_f(output,label)
            loss.backward()
            optim.step()

            total_loss += loss.item() * label.size(0)
            correct += (output.argmax(1)==label).sum().item()
            total += label.size(0)

        avg_loss = total_loss / total
        train_acc = correct / total * 100
        print(f"Epoch [{epoch}/{num_epoch}]  loss: {avg_loss:.4f}  train acc: {train_acc:.2f}%")
```
没有实现可视化曲线，运行结果如下：
```bash
Epoch [1/100]  loss: 0.5658  train acc: 80.58%
Epoch [2/100]  loss: 0.3649  train acc: 87.15%
Epoch [3/100]  loss: 0.3277  train acc: 88.28%
Epoch [4/100]  loss: 0.3027  train acc: 89.22%
Epoch [5/100]  loss: 0.2869  train acc: 89.78%
Epoch [6/100]  loss: 0.2751  train acc: 90.16%
Epoch [7/100]  loss: 0.2659  train acc: 90.41%
Epoch [8/100]  loss: 0.2599  train acc: 90.64%
Epoch [9/100]  loss: 0.2538  train acc: 90.91%
Epoch [10/100]  loss: 0.2470  train acc: 91.10%
```
对于训练过程的可视化。

_(5)_
```py
def evaluate(model: Fashion_Mnist_Model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc
```
具体的代码在code文件夹。 
