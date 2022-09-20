# -*- coding = utf-8 -*-
# @Time : 2022/9/19 21:04
# @Author : Juyi
# @File : 偏导数.py
# @Software : PyCharm
import numpy as np
from Numerical_diff import *
import matplotlib.pylab as plt
from mpl_toolkits import mplot3d

def fun2(x):
    # 表示X0平方加上X1平方这样的双变量函数
    return x[0] **2 + x[1] ** 2

fig = plt.figure()
#创建3d绘图区域
ax = plt.axes(projection='3d')
#调用 ax.plot3D创建三维线图

x = np.array([[np.arange(0, 5, 1)],
              [np.arange(0, 5, 1)]])
z = fun2(x)

# x0 = 3, x1 = 4，关于x0的偏导数 。
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))  # 求在3.0处的导数
# 6.00000000000378