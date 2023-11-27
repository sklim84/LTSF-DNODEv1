import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)#nn.GroupNorm(planes//16, planes) #nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)#nn.GroupNorm(planes//16, planes)#nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)#nn.GroupNorm(self.expansion*planes//16, self.expansion*planes)#
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, dim):
        super(BasicBlock2, self).__init__()
        in_planes = dim
        planes = dim
        stride = 1
        self.nfe = 0
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) #nn.GroupNorm(planes//16, planes)#
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) #nn.GroupNorm(planes//16, planes)#

        self.shortcut = nn.Sequential()

    def forward(self,t, x):
        self.nfe += 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        #out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10, ODEBlock_ = None):
        super(ResNet, self).__init__()
        self._planes = 64
        self.in_planes = self._planes
        self.ODEBlock = ODEBlock_

        self.conv1 = nn.Conv2d(3, self._planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self._planes)
        self.layer1_1 = self._make_layer(self._planes, 1, stride=1)
        self.layer1_2 = self._make_layer2(self._planes, num_blocks[0]-1, stride=1)

        self.layer2_1 = self._make_layer(self._planes*2, 1, stride=2)
        self.layer2_2 = self._make_layer2(self._planes*2, num_blocks[1]-1, stride=1)

        self.layer3_1 = self._make_layer(self._planes*4, 1, stride=2)
        self.layer3_2 = self._make_layer2(self._planes*4, num_blocks[2]-1, stride=1)

        self.layer4_1 = self._make_layer(self._planes*8, 1, stride=2)
        self.layer4_2 = self._make_layer2(self._planes*8, num_blocks[3]-1, stride=1)
        self.linear = nn.Linear(self._planes*8 * block.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _make_layer2(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        #layers.append(nn.BatchNorm2d(self.in_planes))
        for stride in strides:
            layers.append(self.ODEBlock(BasicBlock2(self.in_planes)))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1_1(out)
        out = self.layer1_2(out)
        out = self.layer2_1(out)
        out = self.layer2_2(out)
        out = self.layer3_1(out)
        out = self.layer3_2(out)
        out = self.layer4_1(out)
        out = self.layer4_2(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(ODEBlock):
      return ResNet(BasicBlock, [2,2,2,2], ODEBlock_ = ODEBlock)


