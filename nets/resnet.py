import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, n_class=1000,
                 batch_norm=None):
        """
        :param block: block类型，BasicBlock或者BottleneckBlock
        :param num_blocks: layer1-4中包含block的个数，取列表的前4个元素
        :param in_channels: 输入通道，默认是3
        :param n_class: 分类数
        :param batch_norm: 指定bn
        """
        super(ResNet, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d
        self._batch_norm = batch_norm  # bn不希望被外部使用

        self.in_channels = 64  # 各layer的输入channel，在_make_layer中更新

        self.conv1 = nn.Conv2d(in_channels, self.in_channels, 7, stride=2,
                               padding=3, bias=False)  # 后面接bn，不要bias
        self.bn1 = batch_norm(self.in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_blocks[0], 64)  # 不进行下采样

        self.layer2 = self._make_layer(block, num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(block, num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(block, num_blocks[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化输出1x1
        self.fc = nn.Linear(block.expansion * 512, n_class)
        pass

    def _make_layer(self, block, num_blocks, channels, stride=1):
        """
        :param block: block类型
        :param num_blocks: 本层layer有几个block
        :param in_channels: 本层layer输入的channel数
        :param channels: 本层第一个卷积输出channel数
        :return:
        """
        batch_norm = self._batch_norm

        # 第一个block要单独考虑下采样，是否project
        project = None
        if stride != 1 or self.in_channels != block.expansion * channels:
            project = nn.Sequential(
                conv1x1(self.in_channels, block.expansion * channels, stride),
                nn.BatchNorm2d(block.expansion * channels)
            )
            pass
        layer = [
            block(self.in_channels, channels, stride=stride, project=project,
                  batch_norm=batch_norm)
        ]

        self.in_channels = block.expansion * channels  # 第一个block之后更新输入channel

        # 其他block，都不进行下采样
        for _ in range(1, num_blocks):  # 其他层
            layer.append(
                block(self.in_channels, channels, batch_norm=batch_norm))
            pass
        return nn.Sequential(*layer)

    def forward(self, x):
        print('in', x.shape)

        x = self.conv1(x)  # 1/2
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.maxpool(x)  # 1/4
        x = self.layer1(x)

        x = self.layer2(x)  # 1/8
        x = self.layer3(x)  # 1/16
        x = self.layer4(x)  # 1/32

        x = self.avgpool(x)  # 张量
        x = x.view(1, -1)  # 拉成向量
        x = self.fc(x)
        return x


def conv3x3(in_channels, out_channels, stride=1):
    """
    不将conv和bn合在一起，是因为可以指定bn

    :param in_channels: 输入channel
    :param out_channels: 输出channel
    :param stride: 默认1不进行下采样，2进行下采样
    :return: 3x3 same 卷积
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)  # 接bn，不要bias


def conv1x1(in_channels, out_channels, stride=1):
    """
    不将conv和bn合在一起，是因为可以指定bn

    :param in_channels: 输入channel
    :param out_channels: 输出channel
    :param stride: 默认1不进行下采样，2进行下采样
    :return: 1x1 卷积
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     bias=False)  # 接bn，不要bias


class BasicBlock(nn.Module):
    """
    Basic Block 每个block有2个卷积
    """
    expansion = 1  # 两个卷积的输出channel一样大，所以乘以1

    def __init__(self, in_channels, channels, stride=1, project=None,
                 batch_norm=None):
        """
        Basic Block 每个block有2个卷积
        :param in_channels: block的输入channel数
        :param channels: 第一个（也是每一个）卷积的输出channel数
        :param stride: 默认1，不进行下采样；2，第一个卷积进行下采样
        :param project: 加x的投影，默认为空直接加
        :param batch_norm: 没指定bn就直接加bn
        """
        super(BasicBlock, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d

        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = batch_norm(channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(channels, channels)  # 第二个都不进行下采样
        self.bn2 = batch_norm(channels)

        self.project = project
        pass

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.project is not None:
            identity = self.project(x)

        out += identity
        out = self.relu(out)

        return out

    pass


def resnet18():
    return ResNet(BasicBlock, num_blocks=[2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, num_blocks=[3, 4, 6, 3])


class BottleneckBlock(nn.Module):
    """
    Bottleneck Block 每个block有3个卷积
    """
    expansion = 4  # 第三个（最后一个）卷积的输出channel是第一个卷积输出channel的4倍，所以乘以4

    def __init__(self, in_channels, channels, stride=1, project=None,
                 batch_norm=None):
        """
        Bottleneck Block 每个block有3个卷积
        :param in_channels: block输出channel数
        :param channels: 每个block第一个卷积输出channel数
        :param stride: 默认1，不进行下采样；2，第二个卷积进行下采样
        :param project: 加x的投影，默认为空直接加
        :param batch_norm: 没指定bn就直接加bn
        """
        super(BottleneckBlock, self).__init__()

        if batch_norm is None:
            batch_norm = nn.BatchNorm2d

        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = batch_norm(channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(channels, channels, stride)  # 3x3 负责下采样
        self.bn2 = batch_norm(channels)

        self.conv3 = conv3x3(channels,
                             self.expansion * channels)  # bottoleneck第三个卷积输出channel是4倍
        self.bn3 = batch_norm(self.expansion * channels)

        self.project = project

        pass

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

        if self.project is not None:
            identity = self.project(x)

        out += identity
        out = self.relu(out)
        return out

    pass


def resnet50():
    return ResNet(BottleneckBlock, num_blocks=[3, 4, 6, 3])


def resnet101():
    return ResNet(BottleneckBlock, num_blocks=[3, 4, 23, 3])


def resnet152():
    return ResNet(BottleneckBlock, num_blocks=[3, 8, 36, 3])


if __name__ == '__main__':
    # net = resnet18()
    # net = resnet34()
    net = resnet50()
    in_data = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.float32)
    out_data = net(in_data)
    # print(net)
