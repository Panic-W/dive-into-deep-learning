import torch
from torch import nn


class SSN2d(nn.Module):
    def __init__(self, num_channels, S):
        super().__init__()
        self.S = S
        shape = (1, num_channels*S, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    
    def subspectral_normalization(self, X, gamma, beta, S, moving_mean, moving_var, eps, momentum):
        N, C, T, F = X.size()
        X = X.view(N, C*S, T, F//S)
        # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
        if not torch.is_grad_enabled():
            # 如果是预测模式，直接使用传入的均值和方差
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            # 训练模式下用当前的均值和方差做标准化
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            X_hat = (X - mean) / torch.sqrt(var + eps)
            # 更新移动平均的均值和方差
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        Y = (gamma * X_hat + beta).view(N, C, T, F)
        return Y, moving_mean.data, moving_var.data
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = self.subspectral_normalization(
            X, self.gamma, self.beta, self.S, self.moving_mean, 
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
        
        











