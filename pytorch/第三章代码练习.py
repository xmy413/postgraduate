# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:36:27 2024

@author: 11279
"""

'''
3.3.1  继承nn.module 基类构建模型
'''
import torch
from torch import nn
import torch.nn.functional as F

#构建模型
class Model_Seq(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Model_Seq, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_dim, n_hidden_1)
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.linear2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)
        self.out = nn.Linear(n_hidden_2, out_dim)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x,dim=1)
        return x
    
 #查看模型
in_dim, n_hidden_1, n_hidden_2, out_dim=28 * 28, 300, 100, 10
model_seq = Model_Seq(in_dim, n_hidden_1, n_hidden_2, out_dim)
print(model_seq)


'''
3.3.2  使用nn.Sequential 按层顺序构建模型
'''

#利用可变参数
#python函数的参数个数是可变的，pytorch中也有函数类似：nn.Sequential(*args)

import torch
from torch import nn

in_dim, n_hidden_1, n_hidden_2, out_dim=28 * 28, 300, 100, 10

Seq_arg = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_dim, n_hidden_1),
    nn.BatchNorm1d(n_hidden_1),
    nn.ReLU(),
    nn.Linear(n_hidden_1, n_hidden_2),
    nn.BatchNorm1d(n_hidden_2),
    nn.ReLU(),
    nn.Linear(n_hidden_2, out_dim),  
    nn.Softmax(dim=1)
    )  #缺点：不能给每个层指定名称

print(Seq_arg)


#使用 add_module 方法

import torch
from torch import nn

in_dim, n_hidden_1, n_hidden_2, out_dim=28 * 28, 300, 100, 10

Seq_module = nn.Sequential()
Seq_module.add_module("flatten", nn.Flatten())
Seq_module.add_module("linear1", nn.Linear(in_dim, n_hidden_1))
Seq_module.add_module("bn1", nn.BatchNorm1d(n_hidden_1))
Seq_module.add_module("relu1", nn.ReLU())
Seq_module.add_module("linear2", nn.Linear(n_hidden_1, n_hidden_2))
Seq_module.add_module("bn2", nn.BatchNorm1d(n_hidden_2))
Seq_module.add_module("relu2", nn.ReLU())
Seq_module.add_module("out", nn.Linear(n_hidden_2, out_dim))
Seq_module.add_module("softmax", nn.Softmax(dim=1))

print(Seq_module)


#使用 OrderedDict 方法

import torch
from torch import nn
from collections import OrderedDict


Seq_dict = nn.Sequential(OrderedDict([
    ("flatten",nn.Flatten()),
    ("linear1",nn.Linear(in_dim, n_hidden_1)),
    ("bn1",nn.BatchNorm1d(n_hidden_1)),
    ("relu1",nn.ReLU()),
    ("linear2",nn.Linear(n_hidden_1, n_hidden_2)),
    ("bn2",nn.BatchNorm1d(n_hidden_2)),
    ("relu2",nn.ReLU()),
    ("out",nn.Linear(n_hidden_2, out_dim)),
    ("softmax",nn.Softmax(dim=1))]))

in_dim, n_hidden_1, n_hidden_2, out_dim=28 * 28, 300, 100, 10

print(Seq_dict)


'''
3.3.3  继承nn.Module 基类并应用模型容器来构建模型
'''

import torch
from torch import nn  #从库里（包里）调用一个类
import torch.nn.functional as F

class Model_lay(nn.Module):
    '''
    使用 nn.Sequential 构建网络，Sequential() 函数的功能是将网络的层组合到一起
    '''
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Model_lay, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),nn.BatchNorm1d(n_hidden_2))
        self.out = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.out(x),dim=1)
        return x
    
in_dim, n_hidden_1, n_hidden_2, out_dim=28 * 28, 300, 100, 10
model_lay = Model_lay(in_dim, n_hidden_1, n_hidden_2, out_dim)
print(model_lay)
    

#nn.ModuleList 模型容器
import torch
from torch import nn
import torch.nn.functional as F

class Model_lst(nn.Module):
    
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Model_lst, self).__init__()
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(),
            nn.Linear(n_hidden_2, out_dim),
            nn.Softmax(dim=1)
            ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

in_dim, n_hidden_1, n_hidden_2, out_dim=28 * 28, 300, 100, 10
model_lst = Model_lst(in_dim, n_hidden_1, n_hidden_2, out_dim)
print(model_lst)


#nn.ModuleDict() 模型容器

import torch 
from torch import nn

class Model_dict(nn.Module):
    
    def __init__(self,in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Model_dict, self).__init__()
        self.layers_dict = nn.ModuleDict({"flatten":nn.Flatten(),
                                          "linear1":nn.Linear(in_dim, n_hidden_1),
                                          "bn1":nn.BatchNorm1d(n_hidden_1),
                                          "relu":nn.ReLU(),
                                          "linear2":nn.Linear(n_hidden_1, n_hidden_2),
                                          "bn2":nn.BatchNorm1d(n_hidden_2),
                                          "out":nn.Linear(n_hidden_2, out_dim),
                                          "softmax":nn.Softmax(dim=1)
                                          })
    
    def forward(self, x):
        layers = ["flatten","linear1","bn1","relu","linear2","bn2","relu","out","softmax"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x

in_dim, n_hidden_1, n_hidden_2, out_dim=28 * 28, 300, 100, 10
model_dict = Model_dict(in_dim, n_hidden_1, n_hidden_2, out_dim)
print(model_dict)    



'''
3.3.4 自定义网络模块
'''
import torch
import torch.nn as nn
from torch.nn import functional as F


#3-4a残差块网络结构（不含1*1卷积层）
class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

 

#3-4b残差块网络结构（含1*1卷积层）
class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)



#组合这两个模块得到现代经典的 RestNet18 网络结构：
class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(
            RestNetBasicBlock(64, 64, 1),RestNetBasicBlock(64, 64, 1))
        
        self.layer2 = nn.Sequential(
            RestNetDownBlock(64, 128, [2, 1]), RestNetBasicBlock(128, 128, 1))
        
        self.layer3 = nn.Sequential(
            RestNetDownBlock(128, 256, [2, 1]), RestNetBasicBlock(256, 256, 1))
        
        self.layer4 = nn.Sequential(
            RestNetDownBlock(256, 512, [2, 1]), RestNetBasicBlock(512, 512, 1))
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.fc = nn.Linear(512, 10)
        
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.reshape(x.shape[0],-1)
        out = self.fc(out)
        return out
    


























       