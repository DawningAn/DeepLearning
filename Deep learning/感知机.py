# -*- coding = utf-8 -*-
# @Time : 2022/9/10 15:15
# @Author : Juyi
# @File : 感知机.py
# @Software : PyCharm
import numpy as np
# 定义一个接收参数x1和x2的AND函数
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    # 当输入的加权总和超过阈值时返回1，否则返回0。


# 测试如下(相当于实现了与门逻辑)
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

# 考虑将θ换为b，b称为偏置，w1和w2称为权重。
# 感知机会计算输入信号和权重的乘积，然后加上偏置，如果这个值大于0则输出1，否则输出0

x = np.array([0, 1])  # 输入的值x
w = np.array([0.5, 0.5])  # 权重
b = -0.7  # 偏置

print(w * x)
print(np.sum(w * x) + b)  # np.sum(w*x)再计算相乘后的各个元素的总和

# 使用权重和偏置，可以像下面这样实现与门
def AND1(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7  # 把−θ命名为偏置b
    tmp = np.sum(w*x) + b
    if tmp>0:
        return 1
    else:
        return 0
# 偏置的值决定了神经元被激活的容易程度

# 继续实现与非门和或门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # 仅权重和偏置与AND不同！
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) # 仅权重和偏置与AND不同！
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 感知机的局限性就在于它只能表示由一条直线分割的空间 (显然其难以实现异或门)
# 曲线分割而成的空间称为非线性空间，由直线分割而成的空间称为线性空间

# 这样通过三个门组合可以实现异或
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
# 异或门是一种多层结构的神经网络。