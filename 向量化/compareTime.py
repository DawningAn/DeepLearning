# -*- coding = utf-8 -*-
# @Time : 2022/9/28 9:01
# @Author : Juyi
# @File : compareTime.py
# @Software : PyCharm
import numpy as np
import time

'''
通过向量化来实现，相较于for循环有很大优势
例如：实现线性回归模型，y = wx + b，当w和x都是矩阵（向量）时，向量化的计算将会有很大优势，相比较于双重for循环

np.dot(w, t) + b
'''

a = np.array([1,2,3,4])

print(a)
start = time.time()
a = np.random.rand(1000000)
b = np.random.rand(1000000) #通过round随机得到两个一百万维度的数组

c = np.dot(a,b)

end = time.time()
print(c)
print("耗时："+str(end - start)+"s")

#继续增加非向量化的版本
c = 0
start = time.time()
for i in range(1000000):
    c += a[i]*b[i]
end = time.time()
print(c)
print("耗时：" + str(end - start) + "s")  #打印for循环的版本的时间
