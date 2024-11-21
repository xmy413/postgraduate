# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:10:29 2024

@author: 11279
"""

import numpy as np

# 生成正态分布的随机数
def generate_random_numbers(mu, sigma, num_samples, decimals=None):
    random_numbers = np.random.normal(mu, sigma, num_samples)
    if decimals is not None:
        random_numbers = np.round(random_numbers, decimals=decimals)
    return random_numbers


def sigmoid(x): #激活函数
    return 1 / (1 + np.exp(-x))

'''
隐藏层神经单元定义：
'''

def cell_yc1(x):
    '''
    隐藏层第1个神经单元，通过对输入数据进行加权得到Zysr_list[0]，后再针对神经单元进行激活，生成Aysc_list[0]
    '''
    Zysr_list[0] = sum(wy_list[0][j] * x[j] for j in range(12)) + by_list[0]
    Aysc_list[0] = 1+(np.exp(-1*(Zysr_list[0])))
    return Zysr_list[0], Aysc_list[0]

def cell_yc2(x):
    '''
    隐藏层第2个神经单元，通过对输入数据进行加权得到Zysr_list[1]，后再针对神经单元进行激活，生成Aysc_list[1]
    '''
    Zysr_list[1] = sum(wy_list[1][j] * x[j] for j in range(12)) + by_list[1]
    Aysc_list[1] = 1+(np.exp(-1*(Zysr_list[1])))
    return Zysr_list[1], Aysc_list[1]

def cell_yc3(x):
    '''
    隐藏层第3个神经单元，通过对输入数据进行加权得到Zysr_list[2]，后再针对神经单元进行激活，生成Aysc_list[2]
    '''
    Zysr_list[2] = sum(wy_list[2][j] * x[j] for j in range(12)) + by_list[2]
    Aysc_list[2] = 1+(np.exp(-1*(Zysr_list[2])))
    return Zysr_list[2], Aysc_list[2]



'''
输出层神经单元定义：
'''

def cell_sc1():
    '''
    输出层第1个神经单元，通过对输入数据进行加权得到Zcsr_list[0]，后再针对神经单元进行激活，生成Acsc_list[0]
    '''
    Zcsr_list[0] = ws_list[0][0]*Aysc_list[0]+ws_list[0][1]*Aysc_list[1]+ws_list[0][1]*Aysc_list[2]+bs_list[0]
    Acsc_list[0] = 1/(1+np.exp(-1*(Zcsr_list[0])))
    return Zcsr_list[0], Acsc_list[0]

def cell_sc2():
    '''
    输出层第2个神经单元，通过对输入数据进行加权得到Zcsr_list[1]，后再针对神经单元进行激活，生成Acsc_list[1]
    '''
    Zcsr_list[1] = ws_list[1][0]*Aysc_list[0]+ws_list[1][1]*Aysc_list[1]+ws_list[1][1]*Aysc_list[2]+bs_list[1]
    Acsc_list[1] = 1/(1+np.exp(-1*(Zcsr_list[1])))
    return Zcsr_list[1], Acsc_list[1]


'''
神经网络误差计算
'''

#平方误差
def Wucha(flag):
    '''
    根据输入的图像不同正解变量不同
    '''
    if flag == 0:
        C = [1, 0]
        return 1/2 * (((C[0]-Acsc_list[0])**2) + ((C[1]-Acsc_list[1])**2))
    elif flag ==1:
        C = [0, 1]
        return 1/2 * (((C[0]-Acsc_list[0])**2) + ((C[1]-Acsc_list[1])**2))


#神经单元误差

#输出层的神经单元误差dc

def BP_sc(flag):  #输出层神经单元误差计算函数
    if flag == 0:
        C = [1, 0]
        dsc[0] = (Acsc_list[0]-C[0])*(Acsc_list[0])*(1-Acsc_list[0])
        dsc[1] = (Acsc_list[1]-C[1])*(Acsc_list[1])*(1-Acsc_list[1])
        return dsc
    elif flag == 1:
        C = [0, 1]
        dsc[0] = (Acsc_list[0]-C[0])*(Acsc_list[0])*(1-Acsc_list[0])
        dsc[1] = (Acsc_list[1]-C[1])*(Acsc_list[1])*(1-Acsc_list[1])
        return dsc

#隐藏层的神经单元误差dy
def BP_yc():  #隐藏层神经单元误差计算函数
    dyc[0] = (dsc[0]*ws_list[0][0]+dsc[1]*ws_list[1][0])*(Aysc_list[0])*(1-(Aysc_list[0]))
    dyc[1] = (dsc[0]*ws_list[0][1]+dsc[1]*ws_list[1][1])*(Aysc_list[1])*(1-(Aysc_list[1]))
    dyc[2] = (dsc[0]*ws_list[0][2]+dsc[1]*ws_list[1][2])*(Aysc_list[2])*(1-(Aysc_list[2]))
    return dyc



'''
对代价函数使用梯度下降法
'''

#对所有权值进行偏导求解

def Quanzhi_pd():  #权值求导
    '''
    对代价函数C中所有权值求解偏导
    '''
    #输出层权值偏导
    for i in range(2):
        for j in range(3):
            Qz_pd[3][0][j] = dsc[0] * Aysc_list[j]
    #隐藏层权值偏导
    for i in range(3):
        for j in range(12):
            Qz_pd[i][j] = dyc[i]*x[j]
    return Qz_pd
       
#对所有偏置进行偏导求解

def Pianzhi_pd():  #偏置求导
    '''
    对代价函数C中所有偏置求解偏导
    '''
    #隐藏层偏置求导
    for i in range(3):
        Pz_pd[0][i] = dyc[i]
    #输出层偏执求导
    for i in range(2):
        Pz_pd[1][i] = dsc[i]
    return Pz_pd


def Tidu():  #梯度下降法
    for h in range(50):
        #隐藏层权值梯度下降
        for i in range(3):
            for j in range(12):
                wy_list[i][j] = wy_list[i][j] - n*Qz_pd[i][j]
        #输出层权值梯度下降
        for i in range(2):
            for j in range(3):
                ws_list[i][j] = ws_list[i][j] - n*Qz_pd[2][i][j]
        #隐藏层偏置梯度下降
        for i in range(3):
            by_list[i] = by_list[i] - n*Pz_pd[0][i]
        #输出层偏置梯度下降
        for i in range(2):
            bs_list[i] = bs_list[i] - n*Pz_pd[1][i]
    return wy_list, ws_list, by_list, bs_list
         


'''
主部分
'''
x = [1,1,1,1,0,1,1,0,1,1,1,1]
flag = 0
# x1 = [1,1,0,0,1,0,0,1,0,1,1,1]
# flag = 1

# 生成符合正态分布的随机数,用作隐藏层的权值
mu = 0  # 均值
sigma = 2  # 标准差
decimals=3  #需要保留的小数位

# 隐藏层初始化参数
num_samples = 36  #需要的权重个数
random_numbers = generate_random_numbers(mu, sigma, num_samples)  #生成符合正态分布的权重
random_numbers_rounded = np.round(random_numbers, decimals)  #对生成的权重进行四舍五入放入新的列表中
wy_list = [tuple(random_numbers_rounded[i:i + 12]) for i in range(0, num_samples, 12)]  #将生成的权重按照12个一组分成3组
by_list = generate_random_numbers(mu, sigma, 3, decimals)  #生成3个偏置
Zysr_list = [None, None, None]  #存放隐藏层加权输入
Aysc_list = [None, None, None]  #存放隐藏层激活输出

# 输出层初始化参数
num_samples = 6  #需要的权重和偏置的个数
random_numbers = generate_random_numbers(mu, sigma, num_samples)  #生成符合正态分布的权重
random_numbers_rounded = np.round(random_numbers, decimals=6)  #对生成的权重进行四舍五入放入新的列表中
ws_list =[tuple(random_numbers_rounded[i:i + 3]) for i in range(0, num_samples, 3)]  #将生成的权重按照3个一组分成2组
bs_list = generate_random_numbers(mu, sigma, 2, decimals=2)  #生成2个偏置
Zcsr_list = [None, None]  #存放输出层加权输入
Acsc_list = [None, None]  #存放输出层激活输出

dsc = [None, None]  #输出层神经单元误差
dyc = [None, None, None]  #隐藏层神经单元误差
Qz_pd = [(None,None,None,None,None,None,None,None,None,None,None,None),(None,None,None,None,None,None,None,None,None,None,None,None),(None,None,None,None,None,None,None,None,None,None,None,None),((None,None,None),(None,None,None))]  ##所有权重的偏导数
Pz_pd = [(None,None,None),(None,None)]  #所有偏置的偏导数
n = 0.2  #学习率
 
# 神经单元计算
Acsc_list = [cell_sc1()[1], cell_sc2()[1]]

# 判断
if Acsc_list[0] - 1 < Wucha(flag):
    print("该图像是数字0")
elif Acsc_list[1] - 1 < Wucha(flag):
    print("该图像是数字1")






























































