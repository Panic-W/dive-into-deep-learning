import torch
from torch import nn
from torch.nn import functional as F

# nn

'''nn.Dropout(p = 0.3) # 表示每个神经元有0.3的可能性不被激活'''

'''
nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
normalized_shape：归一化的维度，int（最后一维）list（list里面的维度）
eps：加在方差上的数字，避免分母为0
elementwise_affine：bool，True的话会有一个默认的affine参数
'''

'''
nn.softmax()
https://zhuanlan.zhihu.com/p/397695655
'''

'''
torch.nn.Parameter()
首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个
parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变
量在学习的过程中不断的修改其值以达到最优化。

'''

'''
nn.Flatten() 用：将连续的维度范围展平为张量。 经常在nn.Sequential()中出现，一般写在某个神经网络模型之后，
用于对神经网络模型的输出进行处理，得到tensor类型的数据
'''

'''
nn.conv1d
torch.nn.Conv1d(in_channels,       "输入图像中的通道数"
                out_channels,      "卷积产生的通道数"
                kernel_size,       "卷积核的大小"
                stride,            "卷积的步幅。默认值：1"
                padding,           "添加到输入两侧的填充。默认值：0"
                dilation,          "内核元素之间的间距。默认值：1"
                groups,            "从输入通道到输出通道的阻塞连接数。默认值：1"
                bias,              "If True，向输出添加可学习的偏差。默认：True"
                padding_mode       "'zeros', 'reflect', 'replicate' 或 'circular'. 默认：'zeros'"
                )


'''






'''
torch.rand和torch.randn有什么区别？ 
一个均匀分布(0～1)，一个是标准正态分布。
'''

'''
arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    >>> torch.arange(5)  # 默认以 0 为起点
    tensor([ 0,  1,  2,  3,  4])
    >>> torch.arange(1, 4)  # 默认间隔为 1
    tensor([ 1,  2,  3])
    >>> torch.arange(1, 2.5, 0.5)  # 指定间隔 0.5
    tensor([ 1.0000,  1.5000,  2.0000])
'''


'''
param.requires_grad_(True)
requires_grad=True 的作用是让 backward 可以追踪这个参数并且计算它的梯度。最开始定义你的输入是 requires_grad=True ，那么后续对应的输出也自动具有 requires_grad=True ，如代码中的 y 和 z ，而与 z 对 x 求导无关联的 a ，其 requires_grad 仍等于 False
当你在使用 Pytorch 的 nn.Module 建立网络时，其内部的参数都自动的设置为了 requires_grad=True ，故可以直接取梯度。而我们使用反向传播时，其实根据全连接层的偏导数计算公式，可知链式求导和 w ， b 的梯度无关，而与其中一个连接层的输出梯度有关，这也是为什么冻结了网络的参数，还是可以输出对输入求导。
'''

'''
python 中 __call__ 简单介绍
实现__call__函数，这个类型就成为可调用的。 换句话说，我们可以把这个类型的对象当作函数来使用，相当于 重载了括号运算符。
'''

'''
lambda: 
lambda表达式是一行的函数。它们在其他语言中也被称为匿名函数。即，函数没有具体的名称，而用def创建的方法是有名称的。如果你不想在程序中对一个函数使用两次，你也许会想用lambda表达式，它们和普通的函数完全一样。而且当使用函数作为参数的时候，lambda表达式非常有用，可以让代码简单，简洁。
lambda表达式返回的是function类型，说明是一个函数类型。
'''

'''
for _ in range(num_preds)
for _ in range(n)中的'_' 是占位符， 表示不在意变量的值，只是用于循环遍历n次
'''

'''reshape(-1)#改成一串，没有行列'''

'''y.to(device) 这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。'''

'''
python中的super(Net, self).__init__()
首先找到Net的父类（比如是类NNet），然后把类Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数
∗的作用：函数接受实参时，按顺序分配给函数形参，如果遇到带 ∗ * ∗的形参，那么就把还未分配出去的实参以元组形式打包（pack）,分配给那个带 ∗ * ∗的形参
*args是把多个位置参数打包成元组，**kwargs是把多个关键字参数打包成字典，*args是把打包了的参数拆成单个的，依次赋值给函数的形参，**kwargs是把字典的键值拆成单个的，依次赋值给函数的形参。
'''
'''permute函数的作用是对tensor进行转置'''

'''
torch.repeat_interleave(tensor, repeats, dim=0) 是 PyTorch 中的一个函数，它可以将 tensor 重复指定次数并在指定维度上交错排列。
参数：
    tensor (Tensor) – 要重复的张量。
    repeats (int 或 List[int]) – 每个元素在 tensor 中重复的次数。
    dim (int) – 交错排列的维度。
返回值：
    新的张量，其中包含重复和交错排列后的 tensor。
例如：
>>> x = torch.tensor([1, 2, 3])
>>> torch.repeat_interleave(x, repeats=2)
tensor([1, 1, 2, 2, 3, 3])
'''

'''@property装饰器,它的作用是：将方法变成属性调用。'''


'''
传入参数很简单，两个三维矩阵而已，只是要注意这两个矩阵的shape有一些要求：

res = torch.bmm(ma, mb)
ma: [a, b, c]
mb: [a, c, d]
也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，
对于剩下的则不做要求，其实这里的意思已经很明白了，两个三维矩阵的乘法其实就是保持第一维度不变，
每次相当于一个切片做二维矩阵的乘法，
'''

'''
unsqueeze
unsqueeze()函数起升维的作用,参数dim表示在哪个地方加一个维度，
注意dim范围在:[-input.dim() - 1, input.dim() + 1]之间，比如输入input是一维，
则dim=0时数据为行方向扩，dim=1时为列方向扩，再大错误。
https://blog.csdn.net/flyingluohaipeng/article/details/125092937?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169061640416800226511570%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169061640416800226511570&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-125092937-null-null.142^v91^insertT0,239^v12^control&utm_term=unsqueeze&spm=1018.2226.3001.4187

'''

'''
损失函数中的reduction
reduction 该参数在新版本中是为了取代size_average和reduce参数的。它共有三种选项'elementwise_mean'，'sum'和'none'。
'elementwise_mean'为默认情况，表明对N个样本的loss进行求平均之后返回(相当于reduce=True，size_average=True);
'sum'指对n个样本的loss求和(相当于reduce=True，size_average=False);
'none'表示直接返回n分样本的loss(相当于reduce=False)

'''

'''
"bias=False"通常在机器学习和深度学习中使用，特别是在定义神经网络层时。这意味着在训练过程中，该层的偏置项(bias term)将被随机初始化，
并且不会被用于计算梯度。这有助于防止过拟合，因为它使得模型更加依赖于输入数据的特征，而不是任何固定的偏置值。
'''

'''
enumerate()是Python的内置函数之一，一般用于for循环。enumerate()在遍历中可以获得索引和元素值。
以下是enumerate()函数的语法：enumerate(sequence, [start=0]) 其中参数为： sequence – 一个序列；start – 下标起始位置，默认为0 。
'''

'''
torch.repart()
x = torch.tensor([[1, 2], [3, 4]])
y = x.repeat(3, 2)
print(y)
'''

'''
nn.init.xavier_uniform_ 是 PyTorch 中的一个参数初始化方法，用于初始化神经网络的权重。它是 Xavier 初始化方法的一种变体。

Xavier 初始化是一种常用的权重初始化方法，旨在解决深度神经网络训练过程中的梯度消失和梯度爆炸问题。该方法通过根据网络的输入和输出维度来初始化权重，
使得前向传播和反向传播过程中的信号保持相对一致的方差。
'''

'''
Pytorch 中的 model.apply(fn) 会递归地将函数 fn 应用到父模块的每个子模块以及model这个父模块自身。通常用于初始化模型的参数。
'''

'''
optimizer.zero_grad()函数会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，
再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。
'''

'''
a='python' 
b=a[::-1] 
print(b) #nohtyp 
c=a[::-2] 
print(c) #nhy 
#从后往前数的话，最后一个位置为-1
d=a[:-1]  #从位置0到位置-1之前的数 
print(d)  #pytho 
e=a[:-2]  #从位置0到位置-2之前的数 
print(e)  #pyth
'''

'''
with torch.no_grad的作用
在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
https://blog.csdn.net/sazass/article/details/116668755?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169102868316800213077119%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169102868316800213077119&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-116668755-null-null.142^v92^controlT0_2&utm_term=torch.no_grad%28%29&spm=1018.2226.3001.4187
'''

'''
#一般在训练模型的代码段加入：
model.train()
#在测试模型时候加入：
model.eval()
同时发现，如果不写这两个程序也可以运行，这是因为这两个方法是针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization 和 Dropout。
'''

'''
zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用*号操作符，可以将元组解压为列表。
'''

'''
torch.stack()
把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠
'''

'''
torch.matmul是tensor的乘法，输入可以是高维的。
'''

'''
F.pad()
对矩阵进行填充
http://t.csdn.cn/Ky84K
'''

'''
作用： expand()函数可以将张量广播到新的形状。
注意： 只能对维度值为1的维度进行扩展，无需扩展的维度，维度值不变，对应位置可写上原始维度大小或直接写作-1；
且扩展的Tensor不会分配新的内存，只是原来的基础上创建新的视图并返回，返回的张量内存是不连续的。
类似于numpy中的broadcast_to函数的作用。如果希望张量内存连续，可以调用contiguous函数。

a = torch.tensor([1, 0, 2])     # a -> torch.Size([3])
b1 = a.expand(2, -1)            # 第一个维度为升维，第二个维度保持原样

b1为 -> torch.Size([3, 2])
tensor([[1, 0, 2],
        [1, 0, 2]])
'''


'''
requires_grad
在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。tensor的requires_grad的属性默认为False,若一个节点（叶子变量：自己创建的tensor）requires_grad被设置为True，那么所有依赖它的节点requires_grad都为True（即使其他相依赖的tensor的requires_grad = False）

当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。

with torch.no_grad的作用
在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。

即使一个tensor（命名为x）的requires_grad = True，在with torch.no_grad计算，由x得到的新tensor（命名为w-标量）requires_grad也为False，且grad_fn也为None,即不会对w求导。
'''
'''
torch.cat
函数将两个张量（tensor）按指定维度拼接在一起，注意：除拼接维数dim数值可不同外其余维数数值需相同，方能对齐，如下面例子所示。torch.cat()函数不会新增维度，而torch.stack()函数会新增一个维度，相同的是两个都是对张量进行拼接
'''

'''
AdaptiveAvgPool2d
AdaptivePooling，自适应池化层。函数通过输入原始尺寸和目标尺寸，自适应地计算核的大小和每次
移动的步长。如告诉函数原来的矩阵是7x7的尺寸，我要得到3x1的尺寸，函数就会自己计算出核多大、该怎么运动。
'''

'''
torch.where(condition, x, y):
condition：判断条件
x：若满足条件，则取x中元素
y：若不满足条件，则取y中元素
'''

'''

'''
























