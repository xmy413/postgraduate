from sympy import*
import math

x = symbols('x')

y = exp(x)

#计算基函数与原函数的内积
neiji = [None, None, None, None]
neiji[0] = integrate(y,(x,-1,1))
neiji[0] = round(neiji[0],4)
neiji[1] = integrate(x*y,(x,-1,1))
neiji[1] = round(neiji[1],4)
expr2 = (((3*((x)**2))-1)/2)*y
neiji[2] = integrate(expr2,(x,-1,1))
neiji[2] = round(neiji[2],5)
expr3 = (((5*((x)**3))-3)/2)*y
neiji[3] = integrate(expr2,(x,-1,1))
neiji[3] = round(neiji[3],5)

for i in range(4):
    print(neiji[i])

#计算基函数的系数
a = [None, None, None, None]
for i in range(4):
    a[i] = ((2*i+1)/2) * neiji[i]
    a[i] = round(a[i],4)
for i in range(4):
    print(a[i])
        
#三次最佳平方逼近多项式
s = a[0]*1 + a[1]*x + a[2]*(((3*((x)**2))-1)/2) + a[3]*(((5*((x)**3))-3)/2)

#误差计算
wc0 = integrate(y**2,(x,-1,1))
wc0 = round(wc0, 6)

wclist = [None, None, None, None]
wc1 = 0.000000
for i in range(4):
    wclist[i] = (2/(2*i+1))*(a[i])**2
    wc1 = wc1 + wclist[i]
    wc1 = round(wc1, 6)

wc = round(wc0-wc1, 6)
wc




# from sympy import*
# import math

# x = symbols('x')

# y = exp(x)

# def Neiji():
#     neiji0 = integrate(y,(x,-1,1))
#     neiji1 = integrate(x*y,(x,-1,1))
#     expr2 = (((3*((x)**2))-1)/2)*y
#     neiji2 = integrate(expr2,(x,-1,1))
#     #expr3 = (((5/2)*(x)**3)-(3/2))*y
#     expr3 = (((5*((x)**3))-3)/2)*y
#     neiji3 = integrate(expr3,(x,-1,1))
#     # neiji0 = integrate(y,(x,-1,1))
#     # neiji1 = integrate(x*y,(x,-1,1))
#     # expr2 = (((3/2)*(x)**2)-(1/2))*y
#     # neiji2 = integrate(expr2,(x,-1,1))
#     # expr3 = (((5/2)*(x)**3)-(3/2))*y
#     # neiji3 = integrate(expr3,(x,-1,1))
#     # print(f'neiji0: {neiji0}')
#     # print(f'neiji1: {neiji1}')
#     # print(f'neiji2: {neiji2}')
#     # print(f'neiji3: {neiji3}')
#     #neiji[None, None, None, None]
#     #neiji[0] = integrate(y,(x,-1,1))
#     #neiji[1] = integrate(x*y,(x,-1,1))
#     #neiji[2] = integrate(((((3/2)*(x)**2))-(1/2))*y,(x,-1,1))
#     #neiji[3] = integrate(((((5/2)*(x)**3))-(3/2))*y,(x,-1,1))
#    #for i in range(4):
#         #print(f'neiji[i]')

# Neiji()
