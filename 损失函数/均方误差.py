# -*- coding = utf-8 -*-
# @Time : 2022/9/18 18:14
# @Author : Juyi
# @File : 均方误差.py
# @Software : PyCharm
import numpy as np
# 均方误差是机器学习里比较常用的损失函数
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)  # 参数y和t是NumPy数组
# 简单使用
'''
考虑在数字识别时需要将“2”找出，即输出层向量如下
'''
t = [0,0,1,0,0,0,0,0,0,0]

#  “2”的概率最高的情况（0.6）
y =  [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(mean_squared_error(np.array(y),np.array(t)))




