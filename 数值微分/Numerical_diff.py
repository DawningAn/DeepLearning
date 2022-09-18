# -*- coding = utf-8 -*-
# @Time : 2022/9/18 21:30
# @Author : Juyi
# @File : Numerical_diff.py
# @Software : PyCharm
import numpy as np
# 函数numerical_diff(f, x)的名称来源于数值微分A 的英文 numerical differentiation
# 根据极限的定义，需要定义一个微小值h，即f（x + h）- f（x）/ h 求极限
# 但是在计算机中对于小数的精度是存在舍入误差的，即一个很小的数将会被判定为0，如10e-50
# 并且还存在真的导数与计算机实现的导数值有差异

# 真的导数（真的切线）和数值微分（近似切线）的值不同
# 在当前目录下可以查看图像差异（diff.png）

'''
计算函数f在(x + h)和(x − h)之间的差分。因为这种计算方法以x为中心，计算它左右两边的差分，所以也称为中心差分
（而(x + h)和x之间的差分称为前向差分）。下面，基于上述两个要改进的点来实现数值微分（数值梯度）
'''
# 利用微小的差分求导数的过程称为数值微分（numerical differentiation）

def numerical_diff(f, x):
    h = 1e-4   # 0.0001  这里来避免上述h过小而被计算机视为 0的情况
    return (f(x+h) - f(x-h)) / (2*h)