import torch
from torch.utils import data
from torch import nn
import torchvision
from torchvision import transforms
from IPython import display
import numpy as np
import math
import time
import random

from d2l import torch as d2l

# 从零开始实现线性回归
# def synthetic_data(w,b,num_examples):
#     '''生成数据集'''
#     X = torch.normal(0,1,(num_examples,len(w)))     #返回均值为0方差为1形状为()的随机动态分布的张量
#     y = torch.matmul(X,w) + b
#     y += torch.normal(0,0.01,y.shape)
#     return X,y.reshape((-1,1))

# def data_iter(batch_size, features, labels):
#     '''打乱并小批量读取数据集'''
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)                         #打乱列表顺序
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(
#             indices[i: min(i+batch_size, num_examples)]
#         )
#         yield features[batch_indices], labels[batch_indices]    #yield关键字，妙啊

# def linreg(X, w, b):
#     '''线性回归模型'''
#     return torch.matmul(X, w) + b

# def squared_loss(y_hat, y):
#     '''均方损失'''
#     return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2        #此处为什么reshape,难道本来不就是一个shape吗？

# def sgd(params, lr, batch_size):
#     '''小批量随机梯度下降'''
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad / batch_size
#             param.grad.zero_()


# if __name__ == '__main__':
#     print('let us start!')

#     # # 体验向量化加速
#     # n = 10000
#     # a = torch.ones(n)
#     # b = torch.ones(n)
#     # c = torch.zeros(n)
#     # d = torch.zeros(n)
#     # start = time.time()
#     # for i in range(n):
#     #     c[i] = a[i]+b[i]
#     # end = time.time()
#     # print(f'使用for循环用时：{end-start}')
#     # start = time.time()
#     # d = a + b
#     # end = time.time()
#     # print(f"；向量化加速用时：{end-start}")

#     true_w = torch.tensor([2,-3.4])
#     true_b = 4.2
#     features, labels = synthetic_data(true_w, true_b, 1000)
#     # print(f'feature:{features[0]}\nlabel:{labels[0]}')
#     # d2l.set_figsize()
#     # # d2l.plt.scatter(features[:,1].detach().numpy(), labels.detach().numpy, 1)
#     # d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)

#     batch_size = 10
#     # for X, y in data_iter(batch_size, features, labels):
#     #     print(X, '\n', y)
#     #     break

#     w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
#     b = torch.zeros(1, requires_grad=True)

#     lr = 0.03
#     num_epochs = 3
#     net = linreg
#     loss = squared_loss

#     for epoch in range(num_epochs):
#         for X, y in data_iter(batch_size, features, labels):
#             l = loss(net(X, w, b), y)
#             l.sum().backward()
#             sgd([w, b], lr, batch_size)
#         with torch.no_grad():
#             train_l = loss(net(features, w, b), labels)
#             print(f'epoch{epoch + 1}, loss {float(train_l.mean()):f}')

# # 线性回归的简单实现
# def load_array(data_arrays, batch_size, is_train=True):
#     '''构造一个PyTorch数据迭代器'''
#     dataset = data.TensorDataset(*data_arrays)                      #数据转换
#     return data.DataLoader(dataset, batch_size, shuffle=is_train)   #数据乱序和读取等

# if __name__ == '__main__':
#     true_w = torch.tensor([2, -3.4])
#     true_b = 4.2
#     features, labels = d2l.synthetic_data(true_w, true_b, 1000)
#     num_epochs = 3
#     # # 测试 loaf_array()
#     batch_size = 10
#     data_iter = load_array((features, labels), batch_size)
#     # print(next(iter(data_iter)))

#     # 定义模型
#     net = nn.Sequential(nn.Linear(2, 1))
#     # 初始化模型参数
#     net[0].weight.data.normal_(0, 0.01)         #正态初始化参数
#     net[0].bias.data.fill_(0)                   #用0填充参数
#     # 定义损失函数
#     loss = nn.MSELoss()
#     # 定义优化算法
#     trainer = torch.optim.SGD(net.parameters(), lr=0.03)
#     # 训练
#     for epoch in range(num_epochs):
#         for X, y in data_iter:
#             l = loss(net(X), y)
#             trainer.zero_grad()           #清空模型参数梯度
#             l.backward()
#             trainer.step()
#         l = loss(net(features), labels)
#         print(f'epoch{epoch+1},loss {l:f}')
    
#     w = net[0].weight.data
#     print('w的估计误差：', true_w - w.reshape(true_w.shape))
#     b = net[0].bias.data
#     print('b的估计误差：', true_b - b)

# # 读取数据集
# def get_dataloader_workers():  #@save
#     """使用4个进程来读取数据"""
#     return 4
# def load_data_fashion_mnist(batch_size, resize=None):
#     """下载Fashion-MNIST数据集，然后将其加载到内存中"""
#     # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
#     # 并除以255使得所有像素的数值均在0～1之间
#     trans = [transforms.ToTensor()]
#     if resize:
#         trans.insert(0, transforms.Tesize(resize))  #调整图片大小
#     trans = transforms.Compose(trans)           #多个步骤一起进行
#     mnist_train = torchvision.datasets.FashionMNIST(
#         root="../data", train=True, transform=trans, download=True)
#     mnist_test = torchvision.datasets.FashionMNIST(
#         root="../data", train=False, transform=trans, download=True)
#     return (data.DataLoader(mnist_train, batch_size, shuffle=False,num_workers=get_dataloader_workers()),
#             data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_dataloader_workers()))


# # 从零开始实现softmax回归
# # softmax操作对于任何输入，每个元素变成一个非负数且每行总和为1
# lr = 0.1

# class Accumulator:  #@save
#     """在n个变量上累加"""
#     def __init__(self, n):
#         self.data = [0.0] * n

#     def add(self, *args):
#         self.data = [a + float(b) for a, b in zip(self.data, args)]

#     def reset(self):
#         self.data = [0.0] * len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]
    
# class Animator:  #@save
#     """在动画中绘制数据"""
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
#                  ylim=None, xscale='linear', yscale='linear',
#                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
#                  figsize=(3.5, 2.5)):
#         # 增量地绘制多条线
#         if legend is None:
#             legend = []
#         d2l.use_svg_display()
#         self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows * ncols == 1:
#             self.axes = [self.axes, ]
#         # 使用lambda函数捕获参数
#         self.config_axes = lambda: d2l.set_axes(
#             self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#         self.X, self.Y, self.fmts = None, None, fmts

#     def add(self, x, y):
#         # 向图表中添加多个数据点
#         if not hasattr(y, "__len__"):
#             y = [y]
#         n = len(y)
#         if not hasattr(x, "__len__"):
#             x = [x] * n
#         if not self.X:
#             self.X = [[] for _ in range(n)]
#         if not self.Y:
#             self.Y = [[] for _ in range(n)]
#         for i, (a, b) in enumerate(zip(x, y)):
#             if a is not None and b is not None:
#                 self.X[i].append(a)
#                 self.Y[i].append(b)
#         self.axes[0].cla()
#         for x, y, fmt in zip(self.X, self.Y, self.fmts):
#             self.axes[0].plot(x, y, fmt)
#         self.config_axes()
#         display.display(self.fig)
#         display.clear_output(wait=True)

# # 定义softmax操作
# def softmax(X):
#     X_exp = torch.exp(X)
#     partition = X_exp.sum(1,keepdim=True)
#     return X_exp / partition

# # 定义模型
# def net(X):
#     return softmax(torch.matmul(X.reshape((-1,W.shape[0])), W) + b)

# # 定义损失函数(交叉熵)
# def cross_entropy(y_hat, y):
#     return -torch.log(y_hat[range(len(y_hat)), y])      #torch.log()用于计算张量的自然对数


# def updater(batch_size):
#     return d2l.sgd([W, b], lr, batch_size)

# # 计算预测正确的数量
# def accuracy(y_hat, y):
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = y_hat.argmax(axis = 1)
#     cmp = y_hat.type(y.dtype) == y
#     return float(cmp.type(y.dtype).sum())

# # 计算指定数据集上模型的精度
# def evaluate_accuracy(net, data_iter):
#     if isinstance(net, torch.nn.Module):    #检查一个对象是否是一个类的子类或实例
#         net.eval()  # 将模型设置为评估模式
#     metric = Accumulator(2)
#     with torch.no_grad():   #禁用梯度计算，可以提高效率。
#         for X, y in data_iter:
#             metric.add(accuracy(net(X), y), y.numel())  #numel()用来计算数组中元素个数
#         return metric[0] / metric[1]

# # 训练一个迭代周期
# def train_epoch_ch3(net, train_iter, loss, updater):
#     # 将模型设置成训练模式
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     # 训练损失总和、训练准确度总和、样本数
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         if isinstance(updater, torch.optim.Optimizer):
#             # 使用PyTorch内置的优化器和损失函数
#             updater.zero_grad()
#             l.mean().backward()
#             updater.step()
#         else:
#             # 使用定制的优化器和损失函数
#             l.sum().backward()
#             updater(X.shape[0])
#         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#     # 返回训练损失和训练精度
#     return metric[0] / metric[2], metric[1] / metric[2]

# def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
#     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
#                         legend=['train loss', 'train acc', 'test acc'])
#     for epoch in range(num_epochs):
#         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
#         test_acc = evaluate_accuracy(net, test_iter)
#         animator.add(epoch + 1, train_metrics + (test_acc,))
#     train_loss, train_acc = train_metrics
#     assert train_loss < 0.5, train_loss
#     assert train_acc <= 1 and train_acc > 0.7, train_acc
#     assert test_acc <= 1 and test_acc > 0.7, test_acc

# def predict_ch3(net, test_iter, n=6):  #@save
#     """预测标签（定义见第3章）"""
#     for X, y in test_iter:
#         break
#     trues = d2l.get_fashion_mnist_labels(y)
#     preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
#     titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
#     d2l.show_images(
#         X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# if __name__ == '__main__':
#     batch_size = 256
#     num_epochs = 10
#     train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#     #初始化模型参数
#     num_inputs = 784
#     num_outputs = 10
#     W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)       #requires_grad=True会记录当前张量计算操作，用于之后求导
#     b = torch.zeros(num_outputs, requires_grad=True)

#     train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
#     predict_ch3(net, test_iter)

# softmax回归的简单实现
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

def init_weights(m):
    '''初始化参数模型'''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) #正态分布初始化全重


if __name__ == '__main__':

    # 定义模型
    # PyTorch不会隐式地调整输入的形状。因此，
    # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状   
    net = nn.Sequential(
        nn.Flatten(),   #将连续的维度范围展平为张量
        nn.Linear(784, 10))
    net.apply(init_weights) #该用法通常用于初始化参数
    loss = nn.CrossEntropyLoss(reduction='none')    #reduction='none'表示每个位置的损失都单独保留
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    



