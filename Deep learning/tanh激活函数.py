# -*- coding = utf-8 -*-
# @Time : 2023/7/3 21:18
# @Author : Juyi
# @File : tanh激活函数.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt

'''
双曲正切激活函数（hyperbolic tangent activation function）
与 Sigmoid 函数类似，Tanh 函数也使用真值，但 Tanh 函数将其压缩至-1 到 1 的区间内
与 Sigmoid 不同，Tanh 函数的输出以零为中心，因为区间在-1 到 1 之间
'''
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


x = np.linspace(-10, 10, 100)
plt.plot(x, [tanh(i) for i in x])

# 与sigmoid类似，Tanh 函数也会有梯度消失的问题，因此在饱和时（x很大或很小时）也会「杀死」梯度
# 注意：在一般的二元分类问题中，tanh 函数用于隐藏层，而 sigmoid 函数用于输出层，但这并不是固定的，需要根据特定问题进行调整