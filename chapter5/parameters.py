import torch
from torch import nn

if __name__ == '__main__':
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    # print(net[2].state_dict())
    # print(type(net[2].bias))
    # print(net[2].bias)
    # print(net[2].bias.data)
    print(*[(name, param.shape) for name, param in net.named_parameters()])