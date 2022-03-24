import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import numpy as np

class config():
    def __init__(self):
        self.N = 3# Residual Layers
        self.B = [4, 4, 1]  # Residual blocks in Residual Layer i
        self.C1 = 64
        self.C = self.C1 * (2 ** np.arange(0, self.N, 1))  # channels in Residual Layer i
        self.F = [5, 5, 3]  # Conv. kernel size in Residual Layer i
        self.K = [5, 3, 1] # K Skip connection kernel size in Residual Layer i
        self.P = 8 # P Average pool kernel size
        self.device = "cpu"

netconfig = config()

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, conv_kernel_size=3, skip_kernel_size=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=conv_kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=conv_kernel_size,
                               stride=1,
                               padding='same',
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
        #print(out.shape, self.shortcut(x).shape)
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
        for i in range(0, len(strides)):
            stride = strides[i]
            if i == 0 and strides[0] != 1:
                padding = 1
            else:
                padding = 'same'
            newblock = block(self.in_planes, planes, stride, conv_kernel_size, skip_kernel_size, padding)
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

def project1_model():
    model = ResNet(BasicBlock).to(netconfig.device)
    #torchsummary.summary(model, (3,32,32))
    return model


