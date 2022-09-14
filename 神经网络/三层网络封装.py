# -*- coding = utf-8 -*-
# @Time : 2022/9/14 8:44
# @Author : Juyi
# @File : 三层网络封装.py
# @Software : PyCharm

import numpy as np

'''
激活函数
'''


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


'''
“恒等函数”将输入按原样输出
'''


def identity_function(x):
    return x


# 将整个过程进行封装(字典变量network中保存了每一层所需的参数(权重和偏置))
def init_network():
    network = {}
    network['W1'] = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.1]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.5, 0.1], [0.2, 0.3], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.2], [0.3, 0.2]])
    network['b3'] = np.array([0.1, 0.2])

    return network


# 前向过程(forward()函数中则封装了将输入信号转换为输出信号的处理过程)
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = Sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = Sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


# 样本测试
network = init_network()
x = np.array([1, 5])
y = forward(network, x)
print(y)  # [0.38709595 0.4850286 ]  结果即为y1，y2的值

# 一般而言，回归问题用恒等函数，分类问题用softmax函数
'''
Softmax的含义就在于不再唯一的确定某一个最大值，而是为每个输出分类的结果都赋予一个概率值，表示属于每个类别的可能性
'''
# 实现Softmax函数的过程
a = np.array([0.2, 2.3, 5.1])

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)  # [0.00697078 0.05692458 0.93610464]

'''
封装Softmax函数
'''


def Softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)

    return exp_x / sum_exp_x


'''
显然，Softmax并不是没有缺陷，由于指数函数的爆炸增长特性，数值太大将会导致溢出
一种改进是将每一个输出值减去输出值中最大的值
'''
# x = np.array([100, 200, 350])
# c = np.max(x)
# print(x - c)
# y = np.exp(x - c) / np.sum(np.exp(x - c))
# print(y)

'''
softmax函数的输出是0.0到1.0之间的实数。并且，softmax函数的输出值的总和是1

采用改进的函数如下
'''
def Softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)  # 防止溢出
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y

