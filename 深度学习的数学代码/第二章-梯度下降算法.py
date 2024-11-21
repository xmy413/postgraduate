
def Ofunc(x,y):
    return x**2 + y**2

def Tfunc(x,y):
    dx = 2 * x
    dy = 2 * y
    return dx, dy

def dfunc():
    x = 3.00
    y = 2.00
    z = Ofunc(x, y)
    n = 0.1
    i = 1
    e = 0.0001
    while True:
       print("第{}次迭代:  x={:.2f},  y={:.2f},  z={:.2f}".format((i), x, y, z))
       
       gx,gy = Tfunc(x, y)
       print("第{}次迭代:  gx={:.2f},  gy={:.2f}".format((i), gx, gy))
       x = x - n*2*gx
       y = y - n*2*gy
       z = Ofunc(x, y)
       i = i + 1
       if gx < e and gy < e :
           break
            
dfunc()
        