
import math
    
xn = [(math.pi/6, 1/2), (math.pi/4, 1/(2**(1/2))), (math.pi/3, (3**(1/2)/2))]


x = None

def lagrange_one(flag):
    """
    拉格朗日插值法函数（n=1）:
    """
    if flag == 1:
        lagrange = (x-xn[1][0])/(xn[0][0]-xn[1][0])*(xn[0][1]) + (x-xn[0][0])/(xn[1][0]-xn[0][0])*(xn[1][1])
    elif flag == 2:
        lagrange = (x-xn[2][0])/(xn[1][0]-xn[2][0])*(xn[1][1]) + (x-xn[1][0])/(xn[2][0]-xn[1][0])*(xn[2][1])
    return lagrange
        
def lagrange_two():
    """
    拉格朗日插值法函数（n=2）
    Returns
    -------
    lagrange : TYPE
        DESCRIPTION.
    """
    lagrange = ((x-xn[1][0])*(x-xn[2][0]))/((xn[0][0]-xn[1][0])*(xn[0][0]-xn[2][0]))*xn[0][1] + ((x-xn[0][0])*(x-xn[2][0]))/((xn[1][0]-xn[0][0])*(xn[1][0]-xn[2][0]))*xn[1][1] + ((x-xn[0][0])*(x-xn[1][0]))/((xn[2][0]-xn[0][0])*(xn[2][0]-xn[1][0]))*xn[2][1]
    return lagrange

            
x = float(input("请输入需要估算的sin函数的自变量x:"))*(math.pi/180)

lagrange_one(1)
print(f'利用x0,x1可推导出sin{x}°的值为{lagrange_one(1)}')
lagrange_one(2)
print(f'利用x1,x2可推导出sin{x}°的值为{lagrange_one(2)}')
lagrange_two()
print(f'利用sinx的2次lagrange插值计算可推导出sin{x}°的值为{lagrange_two()}')
def erro(flag):  #误差接受范围存在问题
    if flag == 1:
        """
          当 flag=1 时，计算lagrange_one(1) 所产生的误差。
        """
        erroYou = ((x)-(xn[0][0]))*((x-(xn[1][0]))/2)*(-1/2)
        erroZuo= ((x)-(xn[0][0]))*((x-(xn[1][0]))/2)*(-((3**(1/2))/2))
        erroFinal = math.sin(x) - lagrange_one(1)
        if erroFinal > erroZuo and erroFinal < erroYou:
            print(f"利用x0,x1推导出sin{x}°的值的误差在可接受范围内，且误差为:{erroFinal}")
        else:
            print(f"利用x0,x1推导出sin{x}°的值的误差过大，请重新设计。（误差为：{erroFinal}）")
    elif flag == 2:
        """
          当 flag=2 时，计算lagrange_one(2) 所产生的误差。
        """
        erroZuo = ((x)-(xn[1][0]))*((x-(xn[2][0]))/2)*(-1/(2**(1/2)))
        erroYou = ((x)-(xn[1][0]))*((x-(xn[2][0]))/2)*(-((3**(1/2))/2))       
        erroFinal = math.sin(x) - lagrange_one(2)
        if erroFinal > erroZuo and erroFinal < erroYou:
            print(f"利用x1,x2推导出sin{x}°的值的误差在可接受范围内，且误差为:{erroFinal}")
        else:
            print(f"利用x1,x2推导出sin{x}°的值的误差过大，请重新设计。（误差为：{erroFinal}）")
    elif flag == 3:
        """
          当 flag=3 时，计算lagrange_two() 所产生的误差。
        """
        erroYou = ((x)-(xn[0][0]))*(((x)-(xn[1][0])))*((x)-(xn[2][0]))*(-(3**(1/2))/2)/(3*2*1)
        erroZuo = ((x)-(xn[0][0]))*(((x)-(xn[1][0])))*((x)-(xn[2][0]))*(-1/2)/(3*2*1)
        erroFinal = math.sin(x) - lagrange_two()
        if erroFinal > erroZuo and erroFinal < erroYou:
            print(f"利用sinx的2次lagrange插值计算可推导出sin{x}°的值的误差在可接受范围内，且误差为:{erroFinal}")
        else:
            print(f"利用sinx的2次lagrange插值计算可推导出sin{x}°的值的误差过大，请重新设计。（误差为：{erroFinal}）")
    
erro(1)
erro(2)
erro(3)







