import torch

if __name__ == '__main__':
    # print('let us start!')

    # # 给出两个矩阵𝐀和𝐁，证明“它们转置的和”等于“它们和的转置”，即𝐀⊤+𝐁⊤=(𝐀+𝐁)⊤
    # A = torch.randn(3,4)
    # B = torch.randn(3,4)
    # C = A + B
    # C = C.T
    # D = A.T + B.T
    # print(C == D)

    # # 给定任意方阵𝐀，𝐀+𝐀⊤总是对称的吗?为什么?
    # A = torch.randn(5,5)
    # print(A+A.T)

    # # 本节中定义了形状(2,3,4)的张量X。len(X)的输出结果是什么？
    # A = torch.randn(2,3,4)
    # print(len(A))
    # # 答：输出为轴0的长度

    # # 运行A/A.sum(axis=1)，看看会发生什么。请分析一下原因？
    # A = torch.randn(4,5)
    # # print(A/A.sum(axis=1))
    # print(A/A.sum(axis=1,keepdim=True))

    # # 考虑一个具有形状(2,3,4)的张量，在轴0、1、2上的求和输出是什么形状?
    # # 答：应为[3,4],[2,4],[2,3],下面验证
    # A = torch.randn(2,3,4)
    # for dim in range(3):
    #     print(A.sum(axis=dim).shape)

    print('finished')
