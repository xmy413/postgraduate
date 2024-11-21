# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:01:11 2024

@author: 11279
"""


'''
2.4.1
'''
import torch

x = torch.tensor([1,2])
y = torch.tensor([3,4])

z = x.add(y)
print(z)
print(x)
print(y)

x.add_(y)
print(x)

'''
2.4.2  创建Tensor
'''
torch.Tensor([1,2,3,4,5,6])
torch.Tensor(2,3)

t1 = torch.Tensor(1)
t2 = torch.tensor(1)
print("t1的值{}，t1的数据类型{}".format(t1,t1.type()))
print("t2的值{}，t2的数据类型{}".format(t2,t2.type()))

torch.eye(2,2)
torch.zeros(2,3)
torch.linspace(1,10,4)
torch.rand(2,3)  #生成满足均匀分布的随机数
torch.randn(2,3)  #生成满足标准分布的随机数
torch.zeros_like(torch.rand(2,3))

'''
2.4.3 修改Tensor形状
'''
x = torch.randn(2,3)
x.size()  #查看矩阵形状
x.dim()  #查看矩阵维度
x
x.view(3,2)  #将矩阵变为3*2的形式
y = x.view(-1)
y.shape
y

z = torch.unsqueeze(y,0)  #添加一个维度
z
z.size()
z.dim()
z.numel()

'''
2.4.4  索引操作
'''
torch.manual_seed(100)
x = torch.randn(2,3)  #创建张量X，其中的元素是从标准正态分布中随机采样得到的。
x
x[0,:]  #获取x中第一行的元素
x[:,-1]  #获取x中最后一列的元素
mask = x>0
torch.masked_select(x, mask)  #获取大于0的值
torch.nonzero(mask)  #获取非0元素的下标

index = torch.LongTensor([[0,1,1]])  #创建index张量，形状为(1,3)
torch.gather(x, 0, index)

index = torch.LongTensor([[0,1,1],[1,1,1]])  
a = torch.gather(x, 1, index)
a

z = torch.zeros(2,3)
z
z.scatter_(1, index, a)
z

'''
2.4.5 广播机制
'''
import torch
import numpy as np

#Tensor 广播机制
A = np.arange(0, 40, 10).reshape(4, 1)
A
B = np.arange(0, 3)
B

A1 = torch.from_numpy(A)
A1
B1 = torch.from_numpy(B)
B1

C = A1 + B1
C

#手动实现
#先将 B1 变为 1*3（1，3） 的格式
B2 = B1.unsqueeze(0)
B2
#再将A1和B2转换为4×3的矩阵，使用expand函数
A2 = A1.expand(4,3)
A2
B3 = B2.expand(4,3)
B3
C1 = A2 + B3
C1

'''
2.4.6 逐元素操作
'''
import torch
import numpy as np

t = torch.randn(1,3)
t
t1 = torch.randn(3,1)
t1
t2 = torch.randn(1,3)
t2
#t+0.1*(t1/t2)
torch.addcdiv(t,t1,t2,value=0.1)
a=torch.tensor([[1],[2],[3]])

b=torch.tensor([1,2,3])
c=a*b
print(c)

torch.sigmoid(t)

torch.clamp(t,0,1)

t.add_(2)


'''
2.4.7 归并操作
'''
import torch

a = torch.linspace(0,10,6)  #将0-10均匀分成6份
a
a = a.view((2,3))
a
b = a.sum(dim=0)  #按照y轴方向进行相加，最后输出标量
b
b.shape
# b = a.sum(dim=1)#按照X轴方向进行相加，最后输出标量
# b
# b.shape
b = a.sum(dim=0,keepdim=True)
b
b.shape


'''
2.4.8 比较操作
'''
import torch

x = torch.linspace(0,10,6).view(2,3)
x

torch.max(x)
torch.max(x,dim=0)
torch.topk(x,1,dim=0)

'''
2.4.9 矩阵操作
'''
import torch

a = torch.tensor([2,3])
a
b = torch.tensor([3,4])
b

torch.dot(a,b)

x = torch.randint(10,(2,3))
x
y = torch.randint(6,(3,4))
y
torch.mm(x,y)

x = torch.randint(10,(2,2,3))
x
y = torch.randint(6,(2,3,4))
y
torch.bmm(x,y)

'''
2.5.3  标量的反向传播
'''

import torch

#定义输入张量X
x = torch.Tensor([2])
x
#初始化权重参数 w，偏置 b，并设置require_grad属性为True，为自动求导
w = torch.randn(1,requires_grad = True)
b = torch.randn(1,requires_grad = True)
w
b
#实现正向传播
y = torch.mul(w,x)  #等价于 w*x
y
z = torch.add(y,b)  #等价于 y+b
z
#查看x,w,b叶子节点的require_grad属性
print("x,w,b 的require_grad 属性分别为：{},{},{}".format(x.requires_grad, w.requires_grad, b.requires_grad))


#查看叶子节点、非叶子节点的其他属性

#查看非叶子节点requires_grad 属性
print("y,z 的requires_grad 属性为：{},{}".format(y.requires_grad,z.requires_grad))
#查看个节点是否为叶子节点
print("x,w,b,y,z是否为叶子节点：{},{},{},{},{}".format(x.is_leaf, w.is_leaf, b.is_leaf,y.is_leaf,z.is_leaf,))
#查看叶子节点grad_fn 属性
print("x,w,b grad_fn 属性：{},{},{}".format(x.grad_fn, w.grad_fn, b.grad_fn))
#查看非叶子节点的 grad_fn 属性
print("y,z 是否为叶子节点：{},{}".format(y.grad_fn,z.grad_fn))


#自动求导，实现梯度方向传播，即梯度的反向传播

#基于Z张量进行梯度反向传播，执行backward函数后计算图会自动清空
z.backward()
#若需要多次使用backward函数，需要修改参数retain_graph 为True,此时梯度是累加的
z.backward(retain_graph = True)

#查看叶子节点的梯度，x是叶子节点但他无需求导，故其梯度为None
print("参数w,b的梯度分别为：{},{},另一个叶子节点x的梯度为{}".format(w.grad, b.grad, x.grad))

#非叶子节点的梯度，执行backward函数之后，会自动清空
print("非叶子节点y，z的梯度分别是：{},{}".format(y.grad, z.grad))

'''
2.5.4  非标量反向传播
'''
import torch

X = torch.ones(2,requires_grad=True)
X.size()
Y = X**2 + 3
Y
Y.size()
Y.backward()

X = torch.ones(2,requires_grad=True)
Y = X**2 + 3
Y.sum().backward()
print(X.grad)

import torch

#定义叶子节点张量x，形状为 1*2 
x = torch.tensor([[2,3]],dtype=torch.float, requires_grad=True)
#初始化雅可比矩阵
J = torch.zeros(2,2)
#初始化目标张量，形状为 1*2
y = torch.zeros(1,2)
#定义y与x之间的映射关系：
y[0, 0] = x[0, 0]**2 + 3*x[0, 1]
y[0, 1] = x[0, 1]**2 + 2*x[0, 0]

#生成y1对于x的梯度
y.backward(torch.Tensor([[1,0]]),retain_graph=True)
J[0] = x.grad

#梯度是累加的，故需要对x的梯度清零
x.grad = torch.zeros_like(x.grad)

#生成y2对于x的梯度
y.backward(torch.Tensor([[0,1]]))
J[1] = x.grad

print(J)


'''
2.5.5  切断一些分支的反向传播
'''
import torch

x = torch.ones(2,requires_grad=True)
x
y = x**2 + 3
y
c = y.detach()
c
z = c*x
z
z.sum().backward()
x.grad == c
x.grad
c.grad_fn == None
c.requires_grad

x.grad.zero_()
y.sum().backward()
x.grad == 2*x 



'''
2.6  使用Numpy实现机器学习任务
'''

import numpy as np

from matplotlib import pyplot as plt

np.random.seed(100)
x = np.linspace(-1,1,100).reshape(100, 1)
x
y = 3*np.power(x,2) +2+ 0.2*np.random.rand(x.size).reshape(100,1)
y

#画图
plt.scatter(x, y)
plt.show()

#随机初始化参数
w1 = np.random.rand(1,1)
w1
b1 = np.random.rand(1,1)
b1

lr = 0.001

for i in range(800):
    # 正向传播
    y_pred = np.power(x,2)*w1 + b1  #回归方程，初始化后的系数w1和b1
    loss = 0.5 * (y_pred - y) ** 2  #损失方程：使损失方程最小（误差最小）
    loss = loss.sum()  #
    #计算梯度
    grad_w = np.sum((y_pred - y)*np.power(x,2))
    grad_b = np.sum((y_pred - y))
    #使用梯度下降法，损失值最小
    w1 -= lr * grad_w
    b1 -= lr * grad_b
    
#查看可视化结果

plt.plot(x,y_pred,'r-',label='predict',linewidth=4)
plt.scatter(x, y, color='blue',marker='o',label='true')
plt.xlim(-1,1)
plt.ylim(2,6)
plt.legend()
plt.show()
print(w1,b1)


'''
2.7 使用Tensor 及 autograd 实现机器学习任务
'''

import torch
from matplotlib import pyplot as plt 

torch.manual_seed(100)
dtype = torch.float

x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
x
y = 3*x.pow(2) +2+ 0.2*torch.rand(x.size())
y

#画图，把tensor 数据转化为numpy 数据
plt.scatter(x.numpy(), y.numpy())
plt.show()

#初始化权重参数
w = torch.randn(1, 1, dtype=dtype,requires_grad=True)
b = torch.zeros(1, 1, dtype=dtype,requires_grad=True)

#训练模型

lr = 0.001

for ii in range(800):
    y_pred = x.pow(2).mm(w) + b
    loss = 0.5*(y_pred - y)**2
    loss = loss.sum()
    
    #backward
    loss.backward()
    
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
        
        w.grad.zero_()
        b.grad.zero_()
        
plt.plot(x.numpy(), y_pred.detach().numpy(),'r-',label='predict',linewidth=4)
plt.scatter(x.numpy(), y.numpy(), color='blue',marker='o',label='true')
plt.xlim(-1, 1)
plt.ylim(2,6)

plt.legend()
plt.show()

print(w,b)





'''
2.9 把数据集转换为带批量处理功能的迭代器
'''
import numpy as np

def data_iter(features, labels, batch_size=4): #特征，标签，批次
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #随机打乱
    for i in range(0,num_examples,batch_size):
        indexs = torch.LongTensor(indices[i:min(i + batch_size,num_examples)])
        yield  features.index_select(0, indexs), labels.index_select(0, indexs)
        
# def loss_func(y_pred,labels):
#     return 0.5*(y_pred - labels)**2

loss_func = nn.MSELoss

for ii in range(1000):
    for features, labels in data_iter(x,y,10):
       
        y_pred = features.pow(2).mm(w) + b
        loss=loss_func(y_pred,labels)
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        






















































































































