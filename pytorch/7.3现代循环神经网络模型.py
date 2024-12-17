# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 08:46:18 2024

@author: 11279
"""

import torch
import torch.nn as nn

# 一个典型RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,input,hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output,hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
n_hidden = 128
#rnn = RNN(n_letters, n_hidden, n_categories)


#LSTM

#标准LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
# 第一层相关权重参数形状
print("wih 形状 {}，whh 形状 {}，bin 形状 {}".format(lstm.weight_ih_l0.shape,lstm.weight_hh_l0.shape,lstm.bias_hh_l0.shape))

input = torch.randn(100, 32, 10)
h_0 = torch.randn(2, 32, 20)
h0 = (h_0,h_0)
output,h_n = lstm(input,h0)

print(output.size(),h_n[0].size(),h_n[1].size())


#p168 图7-9的实现

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, output_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.gate = nn.Linear(input_size + hidden_size, cell_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden, cell):
        combined = torch.cat((input,hidden), 1)
        f_gate = self.sigmoid(self.gate(combined))
        i_gate = self.sigmoid(self.gate(combined))
        o_gate = self.sigmoid(self.gate(combined))
        z_state = self.tanh(self.gate(combined))
        cell = torch.add(torch.mul(cell, f_gate), torch.mul(z_state, i_gate))
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        output = self.softmax(output)
        return output, hidden, cell
    
    def initHidden(self):
        return torch.zeros(1,self.hidden_size)
    
    def initCell(self):
        return torch.zeros(1, self.cell_size)
    
lstmcell = LSTMCell(input_size=10, hidden_size=20, cell_size=20, output_size=10)
input = torch.randn(32,10)
h_0 = torch.randn(32,20)

output,hn,cn = lstmcell(input,h_0,h_0)
print(output.size(),hn.size(),cn.size())

#GRU 的实现

class GRUCell(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size+hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),1)
        z_gate = self.sigmoid(self.gate(combined))
        r_gate = self.sigmoid(self.gate(combined))
        combined01 = torch.cat((input,torch.mul(hidden, r_gate)),1)
        h1_state = self.tanh(self.gate(combined01))
        
        h_state = torch.add(torch.mul((1-z_gate),hidden), torch.mul(h1_state, z_gate))
        output = self.output(h_state)
        output = self.softmax(output)
        return output, h_state
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
grucell = GRUCell(input_size=10, hidden_size=20, output_size=10)
input = torch.randn(32,10)
h_0 = torch.randn(32,20)

output, hn = grucell(input,h_0)
print(output.size(), hn.size())


'''
7.5 文本数据处理
'''
import jieba
raw_text = """我爱上海 她喜欢北京"""
stoplist = [' ', '\n']

words = list(jieba.cut(raw_text))

words = [i for i in words if i not in stoplist]
words

word_to_ix = { i: word for i , word in enumerate(set(words))}
word_to_ix

from torch import nn
import torch

embeds = nn.Embedding(6, 8)
lists = []

for k, v in word_to_ix.items():
    tensor_value = torch.tensor(k)
    lists.append((embeds(tensor_value).data))

lists


 











































































