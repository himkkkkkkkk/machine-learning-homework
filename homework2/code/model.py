import torch
import torch.nn as nn

'''
OUTPUT:
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot

INPUT:
graph 28*28 px 1*256 bit
'''

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

