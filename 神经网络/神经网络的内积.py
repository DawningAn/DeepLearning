# -*- coding = utf-8 -*-
# @Time : 2022/8/2 21:21
# @Author : Juyi
# @File : 神经网络的内积.py
# @Software : PyCharm
import numpy as np

'''
激活函数
'''
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


# X = np.array([1, 2])
# W = np.array([[1, 3, 5], [2, 4, 6]])
#
# Y = np.dot(X, W)
# print(Y)  # 其实这里Y的结果就是Yi对应两个xi分别 乘以 连接到yi的那个权重求和

# 三层神经网络
# A(1) = XW(1) + B(1)
# 将输入信号、权重、偏置设置成任意值
X = np.array([1.0, 0.5])  # 输入层有两个x
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)  # (2, 3)
print(X.shape)  # (2,)
print(B1.shape)  # (3,)
A1 = np.dot(X, W1) + B1
# W1是2 × 3的数组，X是元素个数为2的一维数组。这里，W1和X的对应维度的元素个数也保持了一致

Z1 = Sigmoid(A1)  # 传入激活函数
print(A1)  # [0.3, 0.7, 1.1]
print(Z1)

# 实现第1层到第2层的信号传递
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)  # (3,)
print(W2.shape)  # (3, 2)
print(B2.shape)  # (2,)


# 除了第1层的输出（Z1）变成了第2层的输入这一点以外，这个实现和刚才的代码完全相同
'''
“恒等函数”将输入按原样输出
'''
def identity_function(x):
    return x


A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 最后是第2层到输出层的信号传递    ，最后的激活函数和之前的隐藏层有所不同
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)  # 或者Y = A3

'''
输出层所用的激活函数，要根据求解问题的性质决定。一般地，回
归问题可以使用恒等函数，二元分类问题可以使用 sigmoid函数，
多元分类问题可以使用 softmax函数
'''




