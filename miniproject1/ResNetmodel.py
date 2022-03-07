import torch
import torch.nn as nn
import netconfig
import torch.nn.functional as F

import numpy as np

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, conv_kernel_size=3, skip_kernel_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=conv_kernel_size,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=conv_kernel_size,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          planes,
                          kernel_size=skip_kernel_size,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = netconfig.C[0]

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(netconfig.C[0])

        res_layers = []
        for i in range(0, netconfig.N):
            if i == 0:
                stride = 1
            else:
                stride = 2
            layer = self._make_layer(block,
                                     netconfig.C[i],
                                     netconfig.B[i],
                                     stride=stride,
                                     conv_kernel_size=netconfig.F[i],
                                     skip_kernel_size=netconfig.K[i])
            res_layers.append(layer)
        self.reslayer = nn.Sequential(*res_layers)
        self.linear = nn.Linear(netconfig.C[-1], num_classes)


    def _make_layer(self, block, planes, num_blocks, stride, conv_kernel_size=3, skip_kernel_size=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            newblock = block(self.in_planes, planes, stride, conv_kernel_size, skip_kernel_size)
            layers.append(newblock)
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.reslayer(out)
        out = F.avg_pool2d(out, netconfig.P)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


model = ResNet(BasicBlock).to(netconfig.device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=netconfig.lr)

