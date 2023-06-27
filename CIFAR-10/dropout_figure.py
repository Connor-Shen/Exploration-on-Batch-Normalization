import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import netron
import time 
import os



BATCH_SIZE = 128
EPOCH = 10

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train = True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size = BATCH_SIZE,
                                          shuffle = True)

testset = torchvision.datasets.CIFAR10(root='./data',train = False,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size = BATCH_SIZE,
                                          shuffle = False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# plt.imshow(trainset.data[85])
# plt.show()

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                     out_channels = 64,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 3,
                                       stride = 2)
        self.conv2 = torch.nn.Conv2d(64, 64, 5)
        # using FC layer
        self.fc1 = torch.nn.Linear(64*4*4, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # torch.Size([batch_size, 3, 32, 32])
        x = self.pool(F.relu(self.conv1(x)))
        # torch.Size([batch_size, 64, 13, 13])
        x = self.pool(F.relu(self.conv2(x)))
        # torch.Size([batch_size, 64, 4, 4])
        x = x.view(-1, 64*4*4)
        # torch.Size([batch_size, 1024])
        x = F.relu(self.fc1(x))
        # torch.Size([batch_size, 384])
        x = F.relu(self.fc2(x))
        # torch.Size([batch_size, 192])
        x = self.fc3(x)
        # torch.Size([batch_size, 10])
        return x


class CNN_dropout(torch.nn.Module):
    def __init__(self):
        super(CNN_dropout, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                     out_channels = 64,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 3,
                                       stride = 2)
        self.conv2 = torch.nn.Conv2d(64, 64, 5)
        # using FC layer
        self.fc1 = torch.nn.Linear(64*4*4, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # torch.Size([batch_size, 3, 32, 32])
        x = self.pool(F.relu(self.conv1(x)))
        # torch.Size([batch_size, 64, 13, 13])
        x = self.pool(F.relu(self.conv2(x)))
        # torch.Size([batch_size, 64, 4, 4])
        x = x.view(-1, 64*4*4)
        # torch.Size([batch_size, 1024])
        x = F.relu(self.dropout(self.fc1(x)))
        # torch.Size([batch_size, 384])
        x = F.relu(self.dropout(self.fc2(x)))
        # torch.Size([batch_size, 192])
        x = self.fc3(x)
        # torch.Size([batch_size, 10])
        return x


net = CNN()
net_dropout = CNN_dropout()

net_list = [net, net_dropout]
losses = []
import torch.optim as optim

optimizer = optim.SGD(net.parameters(),lr=0.1,momentum=0.9)
loss_func =torch.nn.CrossEntropyLoss()

for n in net_list:
    loss_values = []
    for epoch in range(EPOCH):
        running_loss = 0.0
        for step, data in enumerate(trainloader):
            b_x, b_y = data
            outputs = n.forward(b_x)
            loss = loss_func(outputs, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印状态信息
            running_loss += loss.item()
        running_loss = running_loss/50000
        loss_values.append(running_loss)
    losses.append(loss_values)

print('Finished Training')

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# images_comb = torchvision.utils.make_grid(images)
# images_comb_unnor = (images_comb*0.5+0.5).numpy()
# plt.imshow(np.transpose(images_comb_unnor, (1, 2, 0)))
# plt.show()

# predicts=net.forward(images)


########测试集精度#######
correct = 0
total = 0
with torch.no_grad():
    #不计算梯度，节省时间
    for (images,labels) in testloader:
        outputs = net(images)
        numbers,predicted = torch.max(outputs.data,1)
        total +=labels.size(0)
        correct+=(predicted==labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# 绘制损失率曲线
plt.figure()
colors = ['r', 'b']
labels = ['without dropout', 'with dropout']

for i, loss_values in enumerate(losses):
    plt.plot(loss_values, color=colors[i], label=labels[i])

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss decrease with and without dropout')

if not os.path.isdir('dropout'):
    os.mkdir('dropout')
plt.savefig(f'./dropout/figure.png')