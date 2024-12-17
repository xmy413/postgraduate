# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 08:59:05 2024

@author: 11279
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


training_data = [
    ("The cat ate the fish".split(),["DET", "NN", "V", "DET", "NN"]),
    ("They read that book".split(),["NN", "V", "DET", "NN"])
    ]

testing_data = [("They ate the fish".split())]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            
print(word_to_ix)

#手工设置词性的索引字典
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

'''
   构建网络
'''
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger,self).__init__()
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        self.hidden2tag = nn.Linear(hidden_dim,tagset_size)
        self.hidden = self.init_hidden()
        
    #初始化隐含状态 State 及 C
    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_dim),
                torch.zeros(1,1,self.hidden_dim))
    
    def forward(self, sentence):
        #获得词嵌入矩阵
        embeds = self.word_embeddings(sentence)
        #按照 LSTM 格式，修改embeds 的形状
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence),1,-1),self.hidden)
        #修改隐含状态的形状，作为全连接层的输入
        tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))
        #计算每个单词属于各个词性的概率
        tag_scores = F.log_softmax(tag_space,dim=1)
        return tag_scores
    
    
# 将输入数据转换为torch.LongTensor 张量

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return tensor

'''
   模型网络训练
'''
#定义超参数
EMBEDDING_DIM = 10
HIDDEN_DIM = 3  # 词性个数为3个
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 简单运行
# model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
# loss_function = nn.NLLLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# inputs = prepare_sequence(training_data[0][0], word_to_ix)
# inputs
# tag_scores = model(inputs)
# print(training_data[0][0])
# print(inputs)
# print(inputs.shape)
# print(tag_scores)
# print(torch.max(tag_scores, 1))


# 正式训练
for epoch in range(400):
    for sentence, tags, in training_data:
        # 先清除网络先前的梯度值
        model.zero_grad()
        #重新初始化隐含层数据
        model.hidden = model.init_hidden()
        # 按照网络的要求格式输入数据和真实的标签数据
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        # 实例化模型
        tag_scores = model(sentence_in)
        # 计算损失，反向传递梯度及更新模型参数
        loss = loss_function(tag_scores,targets)
        loss.backward()
        optimizer.step()
        
# 查看模型训练效果
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(training_data[0][0])
print(tag_scores)
print(torch.max(tag_scores, 1))

# 测试模型
test_inputs = prepare_sequence(testing_data[0], word_to_ix)
tag_scores01 = model(test_inputs)

print(testing_data[0])
print(test_inputs)
print(tag_scores01)
print(torch.max(tag_scores01, 1))




































