# -*- coding = utf-8 -*-
# @Time : 2022/9/18 18:23
# @Author : Juyi
# @File : 交叉熵误差.py
# @Software : PyCharm
import numpy as np

# log表示以e为底数的自然对数（log e）

def cross_entropy_error(y, t):
 delta = 1e-7
 return -np.sum(t * np.log(y + delta))
'''
参数y和t是NumPy数组。函数内部在计算np.log时，加上了一个微小值delta。
这是因为，当出现np.log(0)时，np.log(0)会变为负无限大的-inf，这样一来就会导致后续计算无法进行。
作为保护性对策，添加一个微小值可以防止负无限大的发生。
'''
# 使用cross_entropy_error(y, t)进行一些简单的计算
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
# 0.51082545709933802

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
# 2.3025840929945458
'''
第一个例子中，正确解标签对应的输出为0.6，此时的交叉熵误差大约为0.51
第二个例子中，正确解标签对应的输出为0.1的低值，此时的交叉熵误差大约为2.3
'''

# 要求所有训练数据的损失函数的总和
'''
如果遇到大数据，数据量会有几百万、几千万之多，这种情况下以全部数据为对象计算损失函数是不现实的。
因此，从全部数据中选出一部分，作为全部数据的“近似”。神经网络的学习也是从训练数据中选出一批数据（称为mini-batch,小批量）然后对每个mini-batch进行学习
比如，从60000个训练数据中随机选择100笔，再用这100笔数据进行学习。这种学习方式称为mini-batch学习。

编写从训练数据中随机选择指定个数的数据的代码，以进行mini-batch学习。
'''
# 如何从这个训练数据中随机抽取10笔数据呢？我们可以使用NumPy的np.random.choice()

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]



# 寻找最优参数（权重和偏置）时，要寻找使损失函数的值尽可能小的参数




