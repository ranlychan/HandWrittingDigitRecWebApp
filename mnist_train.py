import datetime
import time

import numpy as np
from dateutil.parser import parse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# 如果gpu可用
from scipy.interpolate import make_interp_spline

from MLP_DNN import NeuralNet
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 设置超参数
input_size = 784
hidden_size = 500
output_size = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001

# MNIST 下载，分为训练集和测试集共4个：训练图片，训练标签。测试图片，测试标签。即数据：图片。标签：图片对应的数字
train_dataset = torchvision.datasets.MNIST(root='./datam',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./datam',
                                          train=False,
                                          transform=transforms.ToTensor())
# 数据加载
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = NeuralNet(input_size, hidden_size, output_size).to(device)  # 类的实例化
print(model)
params = list(model.parameters())
print(len(params))
print(params[0].size())
# 损失函数和优化算法
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 训练模型
total_step = len(train_loader)  # 训练数据的大小，也就是含有多少个barch


#start_time = datetime.datetime.now()
acc = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  ##返回值和索引，这里索引即为标签
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)  ## -1 是指模糊控制的意思，即固定784列，不知道多少行
        labels = labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # 测试模型
    # 在测试阶段，不用计算梯度
    with torch.no_grad():
        correct = 0
        total = 0
        test_imgset_one = []
        for images, labels in test_loader:
            test_imgset_one = images
            images = images.reshape(-1, 28 * 28)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            '''
                    for i in range(20):
                        plt.figure("Image")  # 图像窗口名称
                        plt.imshow(test_imgset_one[i].reshape(28, 28),cmap='gray')
                        plt.axis('on')  # 关掉坐标轴为 off
                        plt.title('hand write digit rec')  # 图像题目
                        plt.show()
            '''
            total += labels.size(0)  ##更新测试图片的数量   size(0),返回行数
            correct += (predicted == labels).sum().item()  ##
        acc_this = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: {} %，total test case number:{}'.format(acc_this, total))
        acc.append(acc_this)

epoch_num = np.arange(1, num_epochs+1, 1)#np.linspace(1, epoch, 1)

plt.figure()
plt.plot(epoch_num,acc, 'b', label='acc')
plt.title("Test acc in different EPOCH (EPOCH-Acc plot)")
plt.ylabel('acc %')
plt.xlabel('epoch_num')
plt.show()


x_smooth = np.linspace(epoch_num.min(), epoch_num.max(), 300)#list没有min()功能调用
y_smooth = make_interp_spline(epoch_num, acc)(x_smooth)
plt.plot(x_smooth, y_smooth)
plt.scatter(epoch_num, acc, marker='o')#绘制散点图
plt.show()
#end_time = datetime.datetime.now()
#print('Train time: {} second'.format((end_time - start_time).seconds))


torch.save(model.state_dict(), 'model.ckpt')