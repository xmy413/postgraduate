



#法一
from scipy.integrate import quad
import numpy as np

# 定义被积函数
def integrand0(x):
    return 1
def integrand1(x):
    return x
def integrand2(x):
    return x**2
def integrand3(x):
    return x**3
def integrand4(x):
    return x**4
# 计算定积分
result0, error = quad(integrand0, 0, 3)
result1, error = quad(integrand1, 0, 3)
result2, error = quad(integrand2, 0, 3)
result3, error = quad(integrand3, 0, 3)
result4, error = quad(integrand4, 0, 3)


import numpy as np
from fractions import Fraction

xishu = ([[1,1,1,1],[0,1,2,3],[0,1,4,9],[0,1,8,27]])

zengguang = ([result0,result1,result2,result3])

x = np.linalg.solve(xishu, zengguang)

x_fraction_2 = [Fraction(item).limit_denominator() for item in x]
print(f'方法1本题中的积分约等于{x_fraction_2[0]}*f(0)+{x_fraction_2[1]}*f(1)+{x_fraction_2[2]}*f(2)+{x_fraction_2[3]}*f(3)')

if(result4 != x_fraction_2[0]*0+x_fraction_2[1]*1+x_fraction_2[2]*2**4+x_fraction_2[3]*3**4+3/8*8):
    print("精度为4时，等式不成立，所以精度为3")














#法二
import numpy as np
from fractions import Fraction

xishu = ([[1,1,1,1],[0,1,2,3],[0,1,4,9],[0,1,8,27]])

zengguang = ([3,9/2,9,81/4])

x = np.linalg.solve(xishu, zengguang)

x_fraction_2 = [Fraction(item).limit_denominator() for item in x]

print(f'方法2本题中的积分约等于{x_fraction_2[0]}*f(0)+{x_fraction_2[1]}*f(1)+{x_fraction_2[2]}*f(2)+{x_fraction_2[3]}*f(3)')





#解法三：机械求积-插值型求积公式

from sympy import *

x = 3

A = [None, None, None, None]

def Chazhi ():
    A[0] = Rational((-1/6)*((1/4 * (x)**4) - (2*(x)**3) + (11/2 * (x)**2) - (6*x)))
    A[1] = Rational((1/2)*(((1/4)*(x**4))-((5/3)*(x**3))+(3*(x**2))))
    A[2] = Rational((-1/2)*(((1/4)*(x**4))-((4/3)*(x**3))+((3/2)*(x**2))))
    A[3] = Rational((1/6)*((1/4)*(x**4)-x**3+x**2))

Chazhi()

print(f'方法3本题中的积分约等于{A[0]}*f(0)+{A[1]}*f(1)+{A[2]}*f(2)+{A[3]}*f(3)')

