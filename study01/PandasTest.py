# -*- coding = utf-8 -*-
# @Time : 2022/4/20 16:54
# @Author : Juyi
# @File : PandasTest.py
# @Software : PyCharm
import pandas as pd

# Series类似于通过NumPy库创建的一维数组，不同的是Series对象不仅包含数值，还包含一组索引
s1 = pd.Series(['周伯通', '王重阳', '欧阳锋'])
print(s1)

# DataFrame可以通过列表、字典或二维数组创建
s = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(s)

# 也可以设置行列的名称
ss = pd.DataFrame([[1, 2], [6, 5], [9, 8]], columns=['data', 'score'], index=['A', 'B', 'C'])
# columns代表列索引名称，index代表行索引名称
print(ss)
