# -*- coding = utf-8 -*-
# @Time : 2022/9/11 21:54
# @Author : Juyi
# @File : 激活函数_阶跃函数.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt


# 阶跃函数的图
# 当输入超过0时，输出1，否则输出0
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


# 考虑这里函数参数x只能是实型，如果要传一个数组可以进行如下转化

x = np.array([-1, 2, 3, -2])
y = x > 0
print(y)  # 输出为满足x>0的布尔值，将其转化为int其实就是对应0和1


# 可以用astype()方法转换NumPy数组的类型
# y = y.astype(np.int)
# print(y)

# 优化后的阶跃函数，使之能对数组进行实现
def step_function(x):
    return np.array(x > 0, dtype=np.int64)


# 绘制阶跃函数图形
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1,1.1,1)
plt.show()
