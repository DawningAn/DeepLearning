# -*- coding = utf-8 -*-
# @Time : 2022/9/18 21:40
# @Author : Juyi
# @File : 简单求导.py
# @Software : PyCharm

# 定义二次函数
def fun1(x):
    return 2 * x**2 + 0.1 * x

from Numerical_diff import *
import numpy as np
import matplotlib.pylab as plt

# 绘制fun1的图像来观察
x = np.arange(0.0, 20.0, 0.1)  # 以0.1为单位，从0到20的数组x
y = fun1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

# 计算fun1函数的导数
ret = numerical_diff(fun1,5)
print(ret)  # 求x=5处的导数
# 20.09999999998513

# 考虑绘制
diff_y = numerical_diff(fun1,5) * x + (fun1(5) - numerical_diff(fun1,5)*5)
plt.plot(x,diff_y)
plt.show()

