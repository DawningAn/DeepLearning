# -*- coding = utf-8 -*-
# @Time : 2022/9/22 19:17
# @Author : Juyi
# @File : 神经网络梯度.py
# @Software : PyCharm
import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 用高斯分布进行初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

np.argmax(p) # 最大值的索引

t = np.array([0, 0, 1]) # 正确解标签
print(net.loss(x, t))


