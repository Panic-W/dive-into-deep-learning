import torch
from torch import nn
from d2l import torch as d2l

def relu(X):
    a = torch.zeros_like(X)     #torch.zeros_like()用于创建一个与输入张量形状相同的全零张量。
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1) # 这里“@”代表矩阵乘法
    return (H@W2 + b2)



if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    # nn.Parameter()首先可以把这个函数理解为类型转换函数，
    # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
    # 并将这个parameter绑定到这个module里面
    # torch.rand和torch.randn一个均匀分布，一个是标准正态分布
    # 取值范围(0,1)
    W1 = nn.Parameter(torch.rand(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)    #这里为什么*0.01?
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.rand(
        num_hiddens, num_outputs, requires_grad=True) * 0.01) 
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    
    params = [W1, b1, W2, b2]
    loss = nn.CrossEntropyLoss(reduction='none')    #reduction='none'表示不进行降维操作，即返回每个样本的损失值。
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    






