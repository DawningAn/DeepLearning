# -*- coding = utf-8 -*-
# @Time : 2022/9/28 9:14
# @Author : Juyi
# @File : Demo01.py
# @Software : PyCharm

import numpy  as np
u = np.zeros((5,3))
print(u)

'''
np.log是计算对数函数
np.abs() 是计算数据的绝对值
np.maximum() 计算元素 y 中的最大值
np.maximum(v,0) 
v∗∗2 代表获得元素每个值得平方
1 / v 获取元素的倒数
'''
v = np.array([1,2,3,4])
u = np.exp((v))
print(u)
u = np.log(v)
print(u)

u = np.array(1/v)
print(u)

