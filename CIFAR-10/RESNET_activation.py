import torch
import torch.nn as nn
import torch.nn.functional as F

activation_func = [nn.ReLU(), nn.Sigmoid(), nn.LeakyReLU(), nn.Tanh(), nn.Softplus()]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, activation, stride=1):
        super(BasicBlock, self).__init__()
        # 2 3*3 conv2d layer
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = activation

        self.shortcut = nn.Sequential()
        # use 1*1 conv2d to change to the right planes
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


# use bottleneck to reduce Parameters

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, activation, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.activation = activation

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, activation, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # block defines which resblock to use and num_blocks is a [a, b, c, d] list contains num
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, activation=activation)
        # 通道数为512 * block.expansion
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.activation = activation

    def _make_layer(self, block, planes, num_blocks, stride, activation):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, activation=activation, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # after pool -> (512 * block.expansion, 1, 1)
        out = out.view(out.size(0), -1)
        # reshape to fc layer
        out = self.linear(out)
        return out


def ResNet18(activation):
    return ResNet(BasicBlock, [2, 2, 2, 2], activation)


def ResNet34(activation):
    return ResNet(BasicBlock, [3, 4, 6, 3], activation)


def ResNet50(activation):
    return ResNet(Bottleneck, [3, 4, 6, 3], activation)


def ResNet101(activation):
    return ResNet(Bottleneck, [3, 4, 23, 3], activation)


def ResNet152(activation):
    return ResNet(Bottleneck, [3, 8, 36, 3], activation)