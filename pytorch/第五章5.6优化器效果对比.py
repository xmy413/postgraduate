import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

'''
定义超参数
'''
train_batch_size = 64
test_batch_size = 128
num_epoches = 20
lr = 0.01
momentum = 0.5
alpha = 0.99  # 用于RMSProp和Adam优化器的超参数

'''
下载数据并对数据进行预处理
'''
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = mnist.MNIST('D:/许铭远/课程/第一学期(2024.09-2025.01)/专业课/python深度学习(基于pytorch)/临时存储文件', train=True, transform=transform, download=True)
test_dataset = mnist.MNIST('D:/许铭远/课程/第一学期(2024.09-2025.01)/专业课/python深度学习(基于pytorch)/临时存储文件', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

'''
构建模型
'''
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.out = nn.Linear(n_hidden_2, out_dim)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.log_softmax(self.out(x), dim=1)
        return x

'''
实例化网络
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(28*28, 300, 100, 10).to(device)
criterion = nn.CrossEntropyLoss()

'''
训练模型
'''
def train_model(model, optimizer, num_epoches, train_loader, test_loader, device, optimizer_name):
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    for epoch in range(num_epoches):
        model.train()
        train_loss = 0
        train_acc = 0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = out.max(1)
            train_acc += pred.eq(label).sum().item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        losses.append(train_loss)
        acces.append(train_acc)
        
        model.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                img = img.view(img.size(0), -1)
                out = model(img)
                loss = criterion(out, label)
                eval_loss += loss.item()
                _, pred = out.max(1)
                eval_acc += pred.eq(label).sum().item()
        eval_loss /= len(test_loader)
        eval_acc /= len(test_loader.dataset)
        eval_losses.append(eval_loss)
        eval_acces.append(eval_acc)
        
        print(f'Epoch {epoch+1}, {optimizer_name} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {eval_loss:.4f}, Test Acc: {eval_acc:.4f}')
    return losses, acces, eval_losses, eval_acces

# 定义优化器
sgd_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)  # 普通的SGD
momentum_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # 带动量的SGD
rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)  # RMSProp优化器
adam_optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)  # Adam优化器

# 训练模型
sgd_losses, sgd_acces, sgd_eval_losses, sgd_eval_acces = train_model(model, sgd_optimizer, num_epoches, train_loader, test_loader, device, 'SGD')
momentum_losses, momentum_acces, momentum_eval_losses, momentum_eval_acces = train_model(model, momentum_optimizer, num_epoches, train_loader, test_loader, device, 'Momentum')
rmsprop_losses, rmsprop_acces, rmsprop_eval_losses, rmsprop_eval_acces = train_model(model, rmsprop_optimizer, num_epoches, train_loader, test_loader, device, 'RMSProp')
adam_losses, adam_acces, adam_eval_losses, adam_eval_acces = train_model(model, adam_optimizer, num_epoches, train_loader, test_loader, device, 'Adam')

# 可视化结果
def plot_results(sgd_losses, sgd_acces, momentum_losses, momentum_acces, rmsprop_losses, rmsprop_acces, adam_losses, adam_acces):
    epochs = range(1, num_epoches + 1)
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, sgd_losses, 'r', label='SGD Loss')
    plt.plot(epochs, momentum_losses, 'b', label='Momentum Loss')
    plt.plot(epochs, rmsprop_losses, 'g', label='RMSProp Loss')
    plt.plot(epochs, adam_losses, 'y', label='Adam Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, sgd_acces, 'r', label='SGD Accuracy')
    plt.plot(epochs, momentum_acces, 'b', label='Momentum Accuracy')
    plt.plot(epochs, rmsprop_acces, 'g', label='RMSProp Accuracy')
    plt.plot(epochs, adam_acces, 'y', label='Adam Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制评估损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, sgd_eval_losses, 'r', label='SGD Loss')
    plt.plot(epochs, momentum_eval_losses, 'b', label='Momentum Loss')
    plt.plot(epochs, rmsprop_eval_losses, 'g', label='RMSProp Loss')
    plt.plot(epochs, adam_eval_losses, 'y', label='Adam Loss')
    plt.title('Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制评估准确率曲线
    plt.subplot(2, 2, 4)
    plt.plot(epochs, sgd_eval_acces, 'r', label='SGD Accuracy')
    plt.plot(epochs, momentum_eval_acces, 'b', label='Momentum Accuracy')
    plt.plot(epochs, rmsprop_eval_acces, 'g', label='RMSProp Accuracy')
    plt.plot(epochs, adam_eval_acces, 'y', label='Adam Accuracy')
    plt.title('Evaluation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_results(sgd_losses, sgd_acces, momentum_losses, momentum_acces, rmsprop_losses, rmsprop_acces, adam_losses, adam_acces)
