# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:31:52 2024

@author: 11279
"""

import torch
import numpy as np
#  导入PyTorch 中的MNIST 数据
from torchvision.datasets import mnist
#  导入预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#  导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

'''
  定义超参数
'''
train_batch_size = 64
test_batch_size = 128
# learning_rate = 0.01
num_epoches = 50
lr = 0.01
momentum = 0.5

'''
  下载数据并对数据进行预处理
'''
#定义预处理函数
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
#下载数据，并对数据进行预处理
train_dataset = mnist.MNIST('D:\许铭远\课程\第一学期(2024.09-2025.01)\专业课\python深度学习(基于pytorch)\临时存储文件',train=True, transform=transform, download=True)
test_dataset = mnist.MNIST('D:\许铭远\课程\第一学期(2024.09-2025.01)\专业课\python深度学习(基于pytorch)\临时存储文件', train=False, transform=transform)
#得到一个生成器
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

'''
  可视化源数据
'''  
import matplotlib.pyplot as plt

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
    
'''
  构建模型
'''
class Net(nn.Module):
    '''
    使用nn.Sequential 构建网络,模型容器构造
    '''
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.out = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        
    def forward(self,x):
        x = self.flatten(x)
        # print(x.size)
        x = F.relu(self.layer1(x))
        # print(x.size)
        x = F.relu(self.layer2(x))
        # print(x.size)
        x = F.softmax(self.out(x), dim=1)
        # print(x.size)
        return x
    

'''
  实例化网络
'''
#检测是否有可用的GPU, 有则使用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#实例化网络
model = Net(28*28, 300, 100, 10)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


'''
  训练模型
'''
#开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
writer = SummaryWriter(log_dir='logs',comment='train-loss')

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    #动态修改参数学习率
    if epoch%5 == 0:
      optimizer.param_groups[0]['lr'] *= 0.9
    print("学习率：{:.6f}".format(optimizer.param_groups[0]['lr']))
    for img,label in train_loader:
        img = img.to(device)
        label = label.to(device)
      #正向传播
        out = model(img)
        loss = criterion(out, label)
      #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      #记录误差
        train_loss += loss.item()
      #保存loss 的数据与epoch数值
        writer.add_scalar('Train', train_loss/len(train_loader), epoch)
      #计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
    
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))

    #在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    #net.eval() 将模型改为预测模式
    model.eval()
    for img,label in test_loader:
        
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
      #记录误差
        eval_loss += loss.item()
      #记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
    
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {},Train Loss: {:4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc:{:.4f}'.format(epoch, train_loss/len(train_loader), train_acc/len(train_loader), eval_loss/len(test_loader), eval_acc/len(test_loader)))

# plt.title('train loss')
# plt.plot(np.arange(len(losses)), losses)
# plt.legend(['Train Loss'], loc='upper right')



















































































































































































