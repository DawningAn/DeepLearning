# -*- coding = utf-8 -*-
# @Time : 2022/9/12 18:42
# @Author : Juyi
# @File : Sigmiod函数.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
# 直接采用数学方式定义
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0,1.0,2.0])
print(Sigmoid(x))
# 绘制图像
plt.plot(x, Sigmoid(x))
plt.show()