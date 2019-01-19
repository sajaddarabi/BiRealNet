import torch.nn as nn
import torchvision.transforms as transforms
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, do_bntan=False):
        super(BasicBlock, self).__init__()
        self.act0 = nn.Hardtanh(inplace=False)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.Hardtanh(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.act0(x)
        out = self.conv1(out)
        out = self.bn1(out)

        identity2 = identity + out

        out = self.act1(identity2)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity2

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1, do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                conv1x1(self.inplanes, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=BasicBlock, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inflate = 4
        self.inplanes = 16*self.inflate
        self.conv1 = nn.Conv2d(3, 16*self.inflate, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16*self.inflate,  layers[0])
        self.layer2 = self._make_layer(block, 32*self.inflate,  layers[1], stride=2, do_bntan=True)
        self.layer3 = self._make_layer(block, 64*self.inflate,  layers[2], stride=2, do_bntan=True)
        self.layer4 = self._make_layer(block, 128*self.inflate, layers[3], stride=2, do_bntan=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init_model(self)

def resnet_bireal(**kwargs):
    dataset = kwargs.get('dataset', 'imagenet')
    if dataset == 'imagenet' or 'cifar10':
        if dataset == 'cifar10':
            num_classes = num_classes or 10
        elif dataset == 'imagenet':
            num_classes = num_classes or 1000
        depth = depth or 50

        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2], reg_type=reg_type)
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
