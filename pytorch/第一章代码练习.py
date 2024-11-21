# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:04:17 2024

@author: 11279
"""

import numpy as np
'''
1.1.2  利用已有数据生成数组
'''
lst1 = [3.14, 2.17, 0, 1, 2]
nd1 = np.array(lst1)
print(nd1)
print(type(nd1))

lst2 = [[3.14, 2.17, 0, 1, 2],[1, 2, 3, 4, 5]]
nd2 = np.array(lst2)

print(nd2)
print(type(nd2))


'''
1.1.3  利用random模块生成数组
'''
print(' 生成形状为（4， 4），值在0-1之间的随机数')
print(np.random.random((4,4)),end='\n\n')

#产生一个取值范围在【1，50）之间的数组，数组形状为（3， 3）
#参数起始值(low)默认为0，终止值(high)默认为1
print('生成形状(3, 3),值在low到high之间的随机整数：')
print(np.random.randint(low=1,high=50, size=(3,3)),end='\n\n')

#均匀分布的随机数
print('产生的数组元素是均匀分布的随机数：')
print(np.random.uniform(low=1,high=3,size=(3,3)),end='\n\n')

#生成满足正态分布的随机数矩阵
print('生成满足正态分布的形状为（3，3）的矩阵：')
print(np.random.randn(3,3))


#设置种子产生随机数数组
np.random.seed(10)
print("按照指定随机种子，第1次生成随机数：")
print(np.random.randint(1,5,(2,2)))  # 从1-5的随机数构成的形状为2*2的矩阵

# 想要生成同样的随机数，需在此设置相同的种子
np.random.seed(10)
print("按照相同随机种子，第2次生成随机数：")
print(np.random.randint(1,5,(2,2)))


'''
1.1.4  生成特定形状的多维数组
'''

#生成全是0的3*3矩阵
nd5 = np.zeros([3,3])

#生成与nd5形状一样的全0矩阵
nd5_1 = np.zeros_like(nd5)

#生成全是1的3*3矩阵
nd6 = np.ones([3,3])

#生成3阶的单位矩阵
nd7 = np.eye(3)

#生成3阶对角矩阵
nd8 = np.diag([1,2,3])

print("*"*6+"nd5"+"*"*6)
print(nd5)
print("*"*6+"nd6"+"*"*6)
print(nd6)
print("*"*6+"nd7"+"*"*6)
print(nd7)
print("*"*6+"nd8"+"*"*6)
print(nd8)

#暂时保存生成的数据
import numpy as np
nd9 = np.random.random([5,5])
np.savetxt(X=nd9, fname='D:\\许铭远\\课程\\第一学期(2024.09-2025.01)\\专业课\\python深度学习(基于pytorch)\\临时存储文件\\test1.txt')
nd10 = np.loadtxt('D:\\许铭远\\课程\\第一学期(2024.09-2025.01)\\专业课\\python深度学习(基于pytorch)\\临时存储文件\\test1.txt')
print(nd10)


'''
1.1.5  利用arange、linspace 函数生成数组
'''

#arange函数生成具有特殊规律的数组
#arange([start],[stop],[step],dtype=None)
print(np.arange(10))
print(np.arange(0,10))
print(np.arange(1,4,0.5))
print(np.arange(9,-1,-1))

#linspace函数可以根据输入的指定数据范围以及等份数量自动生成一个线性等分向量

print(np.linspace(0,1,10))



'''
1.2  读取数据
'''

np.random.seed(2024)

nd11 = np.random.random([10])
print(nd11)

#获取指定位置的数据，获取第四个元素
nd11[3]

#截取一段数据
nd11_1 = nd11[3:6]  #从nd11[3]开始到nd11[6]结束，不包含nd11[6]
print(nd11_1)  #实际输出nd11[3]、nd11[4]、nd11[5]

#截取固定间隔数据
nd11_2 = nd11[3:6:2]  #从nd11[3]开始，读取到nd11[6]结束，步长为2
print(nd11_2)  #实际输出nd11[3]、nd11[5]

#倒序取数
nd11_3 = nd11[::-2]  #从nd11[9]开始，向前输出，步长为2
print(nd11_3)  #实际输出（包含顺序）nd11[9]、nd11[7]、nd11[5]、nd11[3]、nd11[1]

#截取一个多维数组的某个区域内的数据
nd12 = np.arange(25).reshape([5,5]) #利用np.arange(25)生成25个随机数，并对这25个随机数使用.reshape([5,5])，使其划分为5行5列的数组矩阵
print(nd12)
nd12_1 = nd12[1:3,1:3]  #将nd12中的第2，3行第2，3列所对应的元素生成一个新的2*2数组矩阵
print(nd12_1)

#截取一个多维数组中数值在某个值域之内的数据
nd12_2 = nd12[(nd12>3)&(nd12<10)]  #将nd12中所有大于3小于10的数输出为一个数组
print(nd12_2)

#截取多维数组中指定的行，如读取第2，3行的值
nd12_3 = nd12[[1,2]]  #或nd12[1:3,:]
print(nd12_3)

#截取多维数组中指定的列，如读取第2，3列的值
nd12_4 = nd12[:,1:3] 
print(nd12_4)

#
nd12_5 = nd12[2::2,::2]  #从第三行开始步长为2，从第一列开始步长为2，选择元素
print(nd12_5)


import numpy as np
from numpy import random as nr

a = np.arange(1,25,dtype = float)
c1 = nr.choice(a,size=(3,4))  #从数组a中抽取12个元素形成3行4列的数组矩阵c1，其中元素可重复
c2 = nr.choice(a,size=(3,4),replace=False)  #在c1的基础上抽取元素不重复，即每个a中的元素仅能抽取一次
c3 = nr.choice(a,size=(3,4),p=a/np.sum(a))  #在c1的基础上赋予每个a中的元素固定的抽取概率
print("随即可重复抽取")
print(c1)
print("随机但不重复抽取")
print(c2)
print("随机按指定概率抽取")
print(c3)


'''
1.3 Numpy 的算术运算
'''
#主要分为逐元素乘积和点积


'''
1.3.1 逐元素操作
'''
#即对应元素相乘，通过np.multiply函数计算数组或矩阵对应元素乘积，输出大小和相乘数组或矩阵的大小一样
'''
标准格式：
    numpy.multiply(x1,x2,/,out=None,*,where=True,casting='same_kind',order='K',dtype=None,subok=True[,signature,extobj])
'''

A = np.array([[1, 2], [-1, 4]])
B = np.array([[2, 0], [3, 4]])
A*B
np.multiply(A,B)
print(A*2.0)
print(A/2.0)

#数组通过一些激活函数处理后，输出和输入形状一致
X = np.random.rand(2,3)
X
def sigmoid(X):
    return 1/(1+np.exp(-X))

def relu(X):
    return np.maximum(0,X)

def softmax(X):
    return np.exp(X)/np.sum(np.exp(X))

print("输入参数的形状：", X.shape)
print("激活函数 sigmoid 输出形状", sigmoid(X).shape)
print("激活函数 relu 输出形状", relu(X).shape)
print("激活函数 softmax 输出形状", softmax(X).shape)

'''
1.3.2 点积运算
'''
#基本格式: numpy.dot(a,b,out=None)  
'''
计算过程同矩阵相乘的计算方法一致
'''
X1 = np.array([[1, 2], [3, 4]])
X2 = np.array([[5, 6, 7], [8, 9, 10]])
X3 = np.dot(X1, X2)
print(X3)


'''
1.4 数组变形
'''

'''
1.4.1 修改数组的形状
'''
#arr.reshape()函数，改编维度
arr = np.arange(10)
print(arr)
#将向量arr转换为2行5列
print(arr.reshape(2, 5))
#指定维度时可以只指定行数或者列数，其他用-1代替
print(arr.reshape(5, -1))
print(arr.reshape(-1, 5))

#resize函数  修改向量本身
arr = np.arange(10)
print(arr)
#将向量arr转换为2行5列
arr.resize(2,5)
print(arr)

#T函数  转置，行变列，列变行
arr = np.arange(12).reshape(3, 4)
print(arr)
print(arr.T)

#ravel 函数   展平向量
arr = np.arange(6).reshape(2, -1)  #两行三列
print(arr)
#按照列优先，展平
print("按照列优先，展平：")
print(arr.ravel('F'))
#按照行优先，展平
print("按照行优先，展平：")
print(arr.ravel())

#flatten(order='C') 函数  将矩阵转换为向量，按照默认行优先进行展平，常用于将2，3维数组转换为一维数组
a = np.floor(10*np.random.random((3,4)))  #生成0-1之间的随机数，按照3行4列进行排布，并每个随机数的大小×10
print(a)
print(a.flatten(order='C'))

#squeeze 函数  降维，将矩阵中含1的维度去掉
arr = np.arange(3).reshape(3,1)
print(arr)
print(arr.shape)
print(arr.squeeze().shape)
print(arr.squeeze())

arr1 = np.arange(6).reshape(3, 1, 2, 1)
print(arr1)
print(arr1.shape)
print(arr1.squeeze)
print(arr1.squeeze().shape)

#transpose 函数  高纬度矩阵进行轴对换
arr2 = np.arange(24).reshape(2,3,4)
print(arr2)
print(arr2.shape)
print(arr2.transpose(1,2,0))
print(arr2.transpose(1,2,0).shape)

'''
1.4.2  合并数组
'''

#append
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.append(a, b)
print(c)

a = np.arange(4).reshape(2,2)
b = np.arange(4).reshape(2,2)
#按行合并
c = np.append(a,b,axis=0)
print("按行合并后的结果")
print(c)
print('合并后的数据维度',c.shape)
#按列合并
d = np.append(a,b,axis=1)
print("按列合并后的结果")
print(d)
print('合并后的数据维度',d.shape)

#concatenate  沿指定轴堆叠数组或矩阵
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.stack((a, b),axis=0))

#zip多用于张量运算
a = np.array([[1, 2], [3, 4]])
a
b = np.array([[5, 6], [7, 8]])
b
c=c=zip(a,b)
c
for i,j in c:
    print(i, end=",")
    print(j)

a1 = [1,2,3]
b1 = [4,5,6]
c1 = zip(a1,b1)
for i,j in c1:
    print(i, end=",")
    print(j)


'''
批处理（mini-batch）
'''
import numpy as np

data_train = np.random.randn(10000,2,3)  #生成10000个形状为2*3的矩阵
print(data_train.shape)  #第一个数为样本数量，后两个为数据形状

np.random.shuffle(data_train)  #打乱10000条数据,直接修改data_train，不会返回新的数组

batch_size = 100 #定义批量大小

for i in range(0,len(data_train),batch_size):  #进行批处理
    x_batch_sum = np.sum(data_train[i:i+batch_size])
    print("第{}批次，该批次的数据之和：{}".format(i, x_batch_sum))
    




































