# -*- coding = utf-8 -*-
# @Time : 2022/9/12 19:56
# @Author : Juyi
# @File : ReLU函数.py
# @Software : PyCharm

# ReLU（Rectified Linear Unit）函数(修正线性单元)
# ReLU函数在输入大于0时，直接输出该值；在输入小于等于0时，输出0
# 考虑实现
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    if x > 0:
        return x
    else:
        return 0

# 改进后的实现
def ReLUFun(x):
    return np.maximum(0, x)

# 绘制
x = np.arange(-5,5,0.1)
plt.plot(x,ReLUFun(x))
plt.show()
