# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:48:24 2024

@author: 25051
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:16:28 2024

@author: 25051
"""

# 定义函数f(x, y)
def f(x, y):
    return y - (2 * x) / y

# Euler显式方法
def euler_explicit(x0, y0, h, x_end):
    x = []
    y = []
    x.append(x0)
    y.append(y0)
    
    while x[-1] < x_end:
        x_next = x[-1] + h
        y_next = y[-1] + h * f(x[-1], y[-1])
        x.append(x_next)
        y.append(y_next)
        
    return x, y

# 初始条件和参数
x0 = 0
y0 = 1
h = 0.1  # 步长
x_end = 1

# 执行Euler显式方法
x,y = euler_explicit(x0, y0, h, x_end)

# 打印结果
print("欧拉显式结果为：")
for i in range(len(x)-1):
    print(f"x = {x[i]:.2f}, y = {y[i]:.4f}")
    
    
    
    
    
    
    
    
#定义函数f(x, y)
def f(x, y):
    return y - (2 * x) / y

# Euler显式方法
def euler_explicit(x0, y0, h, x_end):
    x = []
    y = []
    x.append(x0)
    y.append(y0)
    
    while x[-1] < x_end:
        x_next = x[-1] + h
        y_next = y[-1] + h * f(x[-1], y[-1])
        x.append(x_next)
        y.append(y_next)
        
    return x, y

# 初始条件和参数
x0 = 0
y0 = 1
h = 0.1  # 步长
x_end = 1

# 执行Euler显式方法
x,y = euler_explicit(x0, y0, h, x_end)

# 打印结果
print("欧拉隐式结果为：")
for i in range(len(x)-1):
    print(f"x = {x[i]:.2f}, y = {y[i]:.4f}")