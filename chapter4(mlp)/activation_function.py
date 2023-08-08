import torch
from d2l import torch as d2l


if __name__ == '__main__':

    # ReLU
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    # d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    # 下面反向传播过程中如果y是向量，则需要torch.ones_like(x)该参数
    # 设置retain_graph=True后可进行连续多次backward
    y.backward(torch.ones_like(x), retain_graph=True)

    # sigmoid
    y = torch.sigmoid(x)

    # tanh
    y = torch.tanh(x)

