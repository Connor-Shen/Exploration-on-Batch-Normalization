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
    train_set, batch_size=train_batch_size, shuffle=True, num_workers=2
)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
test_batch_size = 256
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=test_batch_size, shuffle=False, num_workers=2
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

learning_rate = 0.1
epochs = 4


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
        targets_onehot = F.one_hot(targets, num_classes=10).to(torch.float)
        optimizer.zero_grad()
        outputs = net(inputs)
        if criterion != nn.CrossEntropyLoss():
            loss = criterion(outputs, targets_onehot)
        else:
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
            targets_onehot = F.one_hot(targets, num_classes=10).to(torch.float)
            outputs = net(inputs)
            if criterion != nn.CrossEntropyLoss():
                loss = criterion(outputs, targets_onehot)
            else:
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

train_ls = []
train_accu = []
test_ls = []
test_accu = []

criterion_func = [nn.MSELoss(), nn.CrossEntropyLoss(), nn.SmoothL1Loss(), nn.BCEWithLogitsLoss()]

for criter in criterion_func:
    net = ResNet18().to(device)
    criterion = criter
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # 在发现loss不再降低或者acc不再提高之后，降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
    )

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
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
    train_ls.append(train_losses)
    train_accu.append(train_accuracies)
    test_ls.append(test_losses)
    test_accu.append(test_accuracies)


plt.figure(figsize=(16, 8))
colors = ['r', 'g', 'b', 'y', 'k', 'm', 'c']
labels = ["MSELoss", "CrossEntropyLoss", "SmoothL1Loss", "BCEWithLogitsLoss"]

plt.subplot2grid((2, 2), (0, 0))
for i, loss_values in enumerate(train_ls):
    plt.plot(loss_values, color=colors[i], label=labels[i])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Train Loss')

plt.subplot2grid((2, 2), (0, 1))
for i, accu in enumerate(train_accu):
    plt.plot(accu, color=colors[i], label=labels[i])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')

plt.subplot2grid((2, 2), (1, 0))
for i, loss_values in enumerate(test_ls):
    plt.plot(loss_values, color=colors[i], label=labels[i])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Test Loss')

plt.subplot2grid((2, 2), (1, 1))
for i, accu in enumerate(test_accu):
    plt.plot(accu, color=colors[i], label=labels[i])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')

if not os.path.isdir('loss_figure'):
    os.mkdir('loss_figure')
plt.savefig(f'./loss_figure/figure.png')
