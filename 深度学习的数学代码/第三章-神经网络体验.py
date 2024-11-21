import numpy as np

# 生成符合正态分布的随机数,用作隐藏层的权值
mu = 0  # 均值
sigma = 2  # 标准差
num_samples = 36
random_numbers = np.random.normal(mu, sigma, num_samples)

random_numbers_rounded = np.round(random_numbers, decimals=3)

# 将随机数每12个一组放入元组，再将元组放入列表
wy_list = [tuple(random_numbers_rounded[i:i + 12]) for i in range(0, num_samples, 12)]  #列表推导式

print(wy_list)



# 生成符合正态分布的随机数,用作隐藏层的偏置
mu = 0  # 均值
sigma = 2  # 标准差
num_samples = 3
random_numbers = np.random.normal(mu, sigma, num_samples)

# random_numbers_rounded = np.round(random_numbers, decimals=3)
by_list = np.round(random_numbers, decimals=3)
# 将随机数每12个一组放入元组，再将元组放入列表
# by_list = random_numbers_rounded

print(by_list)


# 生成符合正态分布的随机数,用作输出层的权值

mu = 0  # 均值
sigma = 2  # 标准差
num_samples = 3
random_numbers = np.random.normal(mu, sigma, num_samples)

random_numbers_rounded = np.round(random_numbers, decimals=3)

# 将随机数每12个一组放入元组，再将元组放入列表
ws_list = random_numbers_rounded

print(ws_list)


# 生成符合正态分布的随机数,用作输出层的偏置

mu = 0  # 均值
sigma = 2  # 标准差
num_samples = 3
random_numbers = np.random.normal(mu, sigma, num_samples)

random_numbers_rounded = np.round(random_numbers, decimals=3)

# 将随机数每12个一组放入元组，再将元组放入列表
bs_list = random_numbers_rounded

print(bs_list)


aysc_list = [None, None, None]
x = [1,1,1,1,0,1,1,0,1,1,1,1]
# for i in range(12):
#     x = x.append(input())

def Yinsc(x):
    for i in range(12):
        aysc_list[0] = wy_list[0][i] * x[i] + by_list[0]
        aysc_list[1] = wy_list[1][i] * x[i] + by_list[1]
        aysc_list[2] = wy_list[2][i] * x[i] + by_list[2]
    return aysc_list


acsc_list = [None, None]

def Chusc(aysc_list):
    for i in range(3):
        # acsc_list[i] = aysc_list[i]*ws_list[i] + bs_list[i]
        acsc_list[0] = 1/(1+np.exp(-aysc_list[i]))
        acsc_list[1] = 1/(1+np.exp(-aysc_list[i]))
    return acsc_list

def Wucha(acsc_list):
    C = [0, 1]
    wucha = 1/2 * (((C[0]-acsc_list[0])**2) + ((C[1]-acsc_list[1])**2))
    return wucha

Yinsc(x)

Chusc(aysc_list)

Wucha(acsc_list)

if acsc_list[0] - 1 < Wucha(acsc_list) :
    print("该图像是数字0")
elif acsc_list[1] - 1 < Wucha(acsc_list):
    print("该图像是数字1")
    




