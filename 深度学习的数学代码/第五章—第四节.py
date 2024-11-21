# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:43:37 2024

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
    return 1 / (1 + np.exp(-1 * x))

'''
卷积层过滤器权重设置
'''

mu = 0  # 均值
sigma = 2  # 标准差
decimals=3  #需要保留的小数位

# 生成符合正态分布的随机数,用作过滤器的权值
num_samples = 27
random_numbers = generate_random_numbers(mu, sigma, num_samples)  #生成符合正态分布的权重
random_numbers_rounded = np.round(random_numbers, decimals)  #对生成的权重进行四舍五入放入新的列表中
Glqw_list = []

for i in range(3):  #生成三个过滤器
    Glq = []   
    # 每个过滤器中有3行
    for j in range(3):
        Glq_h = []
        # 每行有三列
        for k in range(3):
            index = i * 9 + j * 3 + k
            Glq_h.append(random_numbers_rounded[index])
        Glq.append(Glq_h)
    Glqw_list.append(Glq)
print(Glqw_list)

# 用作卷积层的偏置
Jjcb_list = generate_random_numbers(mu, sigma, 3, decimals) 
# print(Jjcb_list)

# 用作输出层的权值
num_samples = 36
random_numbers = generate_random_numbers(mu, sigma, num_samples)  #生成符合正态分布的权重
random_numbers_rounded = np.round(random_numbers, decimals)  #对生成的权重进行四舍五入放入新的列表中
Sccw_list = []
#将输出层的权值按照输出神经元进行分组（3个），然后按池化层进行分类（3个），每个池化组中四个权值
for i in range(3):
    sc = []
    # 每组中有3个小组
    for j in range(3):
        ch = []
        # 每个小组中有四个数
        for k in range(4):
            index = i * 12 + j * 4 + k
            ch.append(random_numbers_rounded[index])
        sc.append(ch)
    Sccw_list.append(sc)
# print(Sccw_list)

# 用作输出层的偏置
Sccb_list = generate_random_numbers(mu, sigma, 3, decimals) 
# print(Sccb_list)

x = [[0,0,0,1,0,0],
     [0,0,0,1,0,0],
     [0,0,0,1,0,0],
     [0,0,0,1,0,0],
     [0,0,0,1,0,0],
     [0,0,0,1,0,0]]
#设置标志辅助判断真实值
flag = 1

'''
卷积层计算过程
'''

JQsr_list = [
             [[None,None,None,None],
              [None,None,None,None],
              [None,None,None,None],
              [None,None,None,None]],
             [[None,None,None,None],
              [None,None,None,None],
              [None,None,None,None],
              [None,None,None,None]],
             [[None,None,None,None],
              [None,None,None,None],
              [None,None,None,None],
              [None,None,None,None]]
             ]   

JJsc_list = [
             [[None,None,None,None],
              [None,None,None,None],
              [None,None,None,None],
              [None,None,None,None]],
             [[None,None,None,None],
              [None,None,None,None],
              [None,None,None,None],
              [None,None,None,None]],
             [[None,None,None,None],
              [None,None,None,None],
              [None,None,None,None],
              [None,None,None,None]]
             ]



def Juanji(x,Glqw_list,Jjcb_list,JQsr_list):
    #卷积过程 
    
    #计算卷积层加权输入 
    # tag = [None,None,None,None,None,None,None,None,None]
    for i in range(3):  #选择过滤器
        for j in range(4):  #卷积层的行
            for k in range(4):  #卷积层的列
                JQsr_list[i][j][k] = (Glqw_list[i][0][0]*x[j][k] + Glqw_list[i][0][1]*x[j][k+1] + Glqw_list[i][0][2]*x[j][k+2] + 
                                      Glqw_list[i][1][0]*x[j+1][k] + Glqw_list[i][1][1]*x[j+1][k+1] + Glqw_list[i][1][2]*x[j+1][k+2] + 
                                      Glqw_list[i][2][0]*x[j+2][k] + Glqw_list[i][2][1]*x[j+2][k+1] + Glqw_list[i][2][2]*x[j+2][k+2] + Jjcb_list[i])
                # JQsr_list[i][j][k] = sum(value for row in tag for value in row)   
            # JQsr_list[i][j][k] = sum[tag]
                    
    #计算卷积层输出 
    for i in range(3):
        for j in range(4):
            for k in range(4):
                JJsc_list[i][j][k] = sigmoid(JQsr_list[i][j][k])
                
    return JJsc_list

'''
池化层计算
'''
#池化层结果存放
# Chihua_list = [
#                 [0, 0, 0, 0],
#                 [0, 0, 0, 0],
#                 [0, 0, 0, 0]
# ]

Chihua_list = [[[0] * 4 for _ in range(4)] for _ in range(3)]

#池化操作
def Chihua(JJsc_list):
    
    for i in range(3):  #选择卷积子层
        for j in range(2):  #选择池化层中的一行
            for k in range(2):  #选择池化层中的一列
                Yucun_list = []  
            #扫描所选卷积子层中的四个元素：
                for h in range(j,j+1):  
                    for g in range(k,k+1):
                        Yucun_list.append(JJsc_list[i][j + h][k + g])  #将搜索到的四个卷积子层中的元素存入Yuncun_list中
                # if Yucun_list:
                #     Chihua_list[i][j][k] = max(Yucun_list)  #最大池化
                # else:
                #     Chihua_list[i][j][k] = 0  #如果临时列表为空，可以设置为一个特定的值，这里设置为0
                Chihua_list[i][j][k] = max(Yucun_list)  #最大池化
                # Yucun_list.clear()
    return Chihua_list
# def Chihua(JJsc_list):
#     Yucun_list = []
#     for i in range(3):  #选择卷积子层
#         for j in range(2):  #选择池化层中的一行
#             for k in range(2):  #选择池化层中的一列
#                 #扫描所选卷积子层中的四个元素:
#                 for h in range(j, j+1):
#                     for g in range(k, k+1):
#                         try:
#                             print(f"Accessing JJsc_list[{i}][{h}][{g}]")
#                             Yucun_list.append(JJsc_list[i][h][g])
#                         except IndexError as e:
#                             print(f"IndexError: {e}")
#                 if Yucun_list:
#                     try:
#                         print(f"Calculating max for Yucun_list: {Yucun_list}")
#                         Chihua_list[i][j][k] = max(Yucun_list)  #最大池化
#                     except Exception as e:
#                         print(f"Error during max calculation: {e}")
#                 else:
#                     Chihua_list[i][j][k] = 0  #如果临时列表为空，可以设置为一个特定的值，这里设置为0
#                 Yucun_list.clear()
#     return Chihua_list

#输出层

SC_JQsr_list = [0, 0, 0]  #存放加权输入
Sc_list = [0, 0, 0]  #存放输出层结果

def Shuchu(Sccw_list, JJsc_list):
    #加权输入
    Chihua1 = [0, 0, 0]
    for i in range(3):  #确定输出层的神经单元
        for j in range(3):  #依次扫描池化层
            Chihua1[j] = Chihua_list[j][0][0]*Sccw_list[i][j][0] + Chihua_list[j][0][1]*Sccw_list[i][j][1] + Chihua_list[j][0][2]*Sccw_list[i][j][2] + Chihua_list[j][0][3]*Sccw_list[i][j][3]  
        SC_JQsr_list[i] = Chihua1[0]+Chihua1[1]+Chihua1[2]+Sccb_list[i]
        Chihua1 = [None,None,None]
    for i in range(3):  #调用激活函数
        Sc_list[i] = sigmoid(SC_JQsr_list[i])
    return Sc_list

# def Shuchu(Sccw_list, JJsc_list):
#     for i in range(3):
#         Chihua1 = [0] * 3
#         for j in range(3):
#             Chihua1[j] = (Chihua_list[j][0] * Sccw_list[i][j][0] + Chihua_list[j][1] * Sccw_list[i][j][1] +
#                           Chihua_list[j][2] * Sccw_list[i][j][2] + Chihua_list[j][3] * Sccw_list[i][j][3])
#         SC_JQsr_list[i] = sum(Chihua1) + Sccb_list[i]
#     for i in range(3):
#         Sc_list[i] = sigmoid(SC_JQsr_list[i])
#     return Sc_list

'''
平方误差计算
'''
def Wucha(Sc_list,flag):
    if flag == 1:
        C = [1,0,0]
        return 1/2 * ((C[0]-Sc_list[0])**2+(C[1]-Sc_list[1])**2+(C[2]-Sc_list[2])**2)
    elif flag == 2:
        C = [0,1,0]
        return 1/2 * ((C[0]-Sc_list[0])**2+(C[1]-Sc_list[1])**2+(C[2]-Sc_list[2])**2)
    elif flag == 3:
        C = [0,0,1]
        return 1/2 * ((C[0]-Sc_list[0])**2+(C[1]-Sc_list[1])**2+(C[2]-Sc_list[2])**2)
    

Juanji(x,Glqw_list,Jjcb_list,JQsr_list)
Chihua(JJsc_list)
Shuchu(Sccw_list, JJsc_list)
Wucha(Sc_list,flag)

# 判断
if Sc_list[0] - 1 < Wucha(Sc_list,flag):
    print("该图像是数字1")
elif Sc_list[1] - 1 < Wucha(Sc_list,flag):
    print("该图像是数字2")
elif Sc_list[2] - 1 < Wucha(Sc_list,flag):
    print("该图像是数字3")

