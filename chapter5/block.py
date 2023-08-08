import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    '''带有两个全连接层的多层感知机'''
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


if __name__ == '__main__':
    X = torch.rand(2, 20)
    net = nn.Sequential(MLP())
    # print(net(X))
    # print(net[0][1].state_dic())
    print('finish')









