import torch
import torchvision
import torchvision.transforms as T
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time 
import os
import netron


transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
transform_test = T.Compose(
    [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
train_batch_size = 256
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=train_batch_size, shuffle=True, num_workers=0
)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
test_batch_size = 256
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=test_batch_size, shuffle=False, num_workers=0
)

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from RESNET import ResNet18
net = ResNet18().to(device)

"""输出网络结构"""
# features, targets = next(iter(train_loader)) #从dataloader中取出一个batch
# modelData = "./demo.pth"  # 定义模型数据保存的路径
# torch.onnx.export(net, features, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
# netron.start(modelData)  # 输出网络结构

learning_rate = 0.1
epochs = 4

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# 在发现loss不再降低或者acc不再提高之后，降低学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
)


best_acc = 0

# Training
def train(epoch):
    net.train()
    train_loss = 0
    accu = 0
    total = 0
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        accu += predicted.eq(targets).sum().item()

    t1 = time.time()
    training_time = t1 - t0
    epoch_loss = train_loss / ((batch_idx + 1) * train_batch_size)
    epoch_acc = 100.0 * accu / total

    return epoch_loss, epoch_acc, training_time


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = test_loss / ((batch_idx + 1) * test_batch_size)
    epoch_acc = 100.0 * correct / total

    if epoch_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': epoch_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = epoch_acc

    return epoch_loss, epoch_acc

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
lr_schedule = []

for epoch in range(epochs):
    train_loss, train_acc, training_time = train(epoch)
    test_loss, test_acc = test(epoch)
    # 更新优化器的学习率
    scheduler.step(test_loss)

    # Print log info
    print("============================================================")
    print(f"Epoch: {epoch + 1}")
    # optimizer.param_groups: [‘params’, ‘lr’, ‘betas’, ‘eps’, ‘weight_decay’, ‘amsgrad’, ‘maximize’]
    print(f"Time: {training_time:.3f}s")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%"
    )
    print("============================================================")

    # Logging
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    lr_schedule.append(optimizer.param_groups[0]['lr'])


# Plot
plt.figure(figsize=(16, 8))

plt.subplot2grid((2, 4), (0, 0), colspan=3)
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot2grid((2, 4), (1, 0), colspan=3)
plt.plot(train_accuracies, label='train')
plt.plot(test_accuracies, label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot2grid((2, 4), (0, 3), rowspan=2)
plt.plot(lr_schedule, label='lr')
plt.legend()
plt.title('Learning Rate')
plt.xlabel('Epoch')

if not os.path.isdir('log'):
    os.mkdir('log')
plt.savefig(f'./log/figure.png')