# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:51:41 2024

@author: 11279
"""

#6.3.1 局部池化
import torch
import torch.nn as nn

m1 = nn.MaxPool2d(3, stride=2)

m2 = nn.MaxPool2d((3,2), stride=(2,1))

input = torch.randn(20, 16, 50, 32)
output = m2(input)

print(output.shape)

#6.3.2 全局池化

import torch
import torch.nn as nn

m = nn.AdaptiveMaxPool2d((5,7))  #输出大小为5*7
input = torch.randn(1, 64, 8, 9)
output = m(input)
print(output.size())

m = nn.AdaptiveMaxPool2d(7)  #输出大小为7*7的正方形
input = torch.randn(1, 64, 10, 9)
output = m(input)
print(output.size())

m = nn.AdaptiveMaxPool2d((None,7))  #输出大小为10*7
input = torch.randn(1, 64, 10, 9)
output = m(input)
print(output.size())

m = nn.AdaptiveMaxPool2d((1))  #输出大小为1*1
input = torch.randn(1, 64, 10, 9)
output = m(input)
print(output.size())


















