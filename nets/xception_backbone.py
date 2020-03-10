import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        """
        深度可分离卷积
        第一个卷积在spatial层面上，每个channel单独进行卷积，用group=out_channels实现
        第二个卷积在cross-channel层面上，相当于用1x1卷积调整维度
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        :param kernel_size: depthwise conv的kernel_size
        :param stride: depthwise conv的stride
        :param padding: depthwise conv的padding
        :param bias: 两个卷积的bias
        """
        super(SeparableConv2d, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, groups=in_channels, bias=bias)
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               bias=bias)
        pass

    def forward(self, x):
        return self.pconv(self.dconv(x))  # 先depthwise conv，后pointwise conv

    pass


class ResidualConnection(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        论文中的Residual Connection，来自ResNet的启发，也叫project、skip等
        调整维数
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        :param stride: 下采样，默认不下采样，只调整channel
        """
        super(ResidualConnection, self).__init__(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        pass

    pass


"""
论文《Deeplab v3+：Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation》

摘自DeepLabV3+论文 Fig 4 的一段话
Fig. 4. We modify the Xception as follows: 
(1) more layers (same as MSRA's modifcation except the changes in Entry flow).
(2) all the max pooling operations are replaced by depthwise separable convolutions with striding.
(3) extra batch normalization and ReLU are added after each 3x3 depthwise convolution, similar to MobileNet.

按照DeepLabV3+论文中对Xception的改进实现XceptionBackbone，为DeepLabV+做准备：

对EntryFlow：
第1个module，也就是普通卷积单独实现。
第2-第4个module结构相似，都是stride=2的SeprableConv下采样，用_ComvEntryBlock实现。

对MiddleFlow：
第5-第12个module结构相似，都不进行下采样，用_ConvMiddleBlock实现。

对ExitFlow：
第14个module有stride=2的SeprableConv下采样，通道数不同，用_ConvExitBlock实现。
第15个module拆开，3个卷积单独实现

全局平均池化单独实现。后面接全连接层。
"""


class _ConvEntryBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Entry Flow的3个下采样module
        按论文所说，每个Conv和Separable Conv都需要跟BN，ReLU
        每个module的Separable Conv的out_channels一样，stride=2的SeprableConv做下采样
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        """
        super(_ConvEntryBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)
        convs = [SeparableConv2d(in_channels, out_channels, 3, padding=1,  # 第1个SeparableConv2d,不下采样
                                 bias=False),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(out_channels, out_channels, 3, padding=1,  # 第2个SeparableConv2d,不下采样
                                 bias=False),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(out_channels, out_channels, 3, stride=2,  # 第2个SeparableConv2d,stride=2,下采样2倍
                                 padding=1, bias=False),

                 nn.BatchNorm2d(out_channels), ]
        self.convs = nn.Sequential(*convs)
        pass

    def forward(self, x):
        identity = self.project(x)  # residual connection 准备
        x = self.convs(x)  # 下采样2倍
        x = x + identity  # residual connection 相加
        return F.relu(x, inplace=True)

    pass


class _ConvMiddleBlock(nn.Module):
    def __init__(self, inplanes=728):
        """
        Middle Flow中重复的block，channels和spatial都不发生变化
        :param inplanes: 输入channels
        """
        super(_ConvMiddleBlock, self).__init__()
        convs = [SeparableConv2d(inplanes, inplanes, 3, padding=1, bias=False),
                 nn.BatchNorm2d(inplanes),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(inplanes, inplanes, 3, padding=1, bias=False),
                 nn.BatchNorm2d(inplanes),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(inplanes, inplanes, 3, padding=1, bias=False),
                 nn.BatchNorm2d(inplanes), ]
        self.convs = nn.Sequential(*convs)
        pass

    def forward(self, x):
        x = x + self.convs(x)  # channels和spatial都没有发生变化，Residual Connection直接相加
        return F.relu(x, inplace=True)

    pass


class _ConvExitBlock(nn.Module):
    def __init__(self, in_channels=728, out_channels=1024):
        """
        Exit Flow的第1个module
        前两个Separable Conv都不做下采样
        最后一个Separable Conv下采样2倍
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        """
        super(_ConvExitBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)
        convs = [SeparableConv2d(in_channels, in_channels, 3, padding=1,
                                 bias=False),  # 728->728，不下采样
                 nn.BatchNorm2d(in_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(in_channels, out_channels, 3, padding=1,
                                 bias=False),  # 728->1024，不下采样
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(out_channels, out_channels, 3, stride=2,
                                 padding=1, bias=False),  # 1024->1024，下采样2倍
                 nn.BatchNorm2d(out_channels), ]
        self.convs = nn.Sequential(*convs)
        pass

    def forward(self, x):
        identity = self.project(x)  # residual connection 准备
        x = self.convs(x)  # 下采样2倍
        x = x + identity  # residual connection 相加
        return F.relu(x, inplace=True)

    pass


class XceptionBackbone(nn.Module):
    def __init__(self, in_channels=3, n_class=1000):
        super(XceptionBackbone, self).__init__()
        # 以下Entry Flow
        conv1 = [nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True), ]
        self.entry_conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(32, 64, 3, padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True), ]
        self.entry_conv2 = nn.Sequential(*conv2)

        self.entry_block1 = _ConvEntryBlock(64, 128)
        self.entry_block2 = _ConvEntryBlock(128, 256)
        self.entry_block3 = _ConvEntryBlock(256, 728)

        # 以下Middle Flow
        self.middle_flow = nn.ModuleList([_ConvMiddleBlock(728)] * 16)  # 改进之一，middle block有16个

        # 以下Exit Flow
        self.exit_block = _ConvExitBlock(728, 1024)

        conv1 = [SeparableConv2d(1024, 1536, 3, padding=1, bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv1 = nn.Sequential(*conv1)

        conv2 = [SeparableConv2d(1536, 1536, 3, padding=1, bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv2 = nn.Sequential(*conv2)

        conv3 = [SeparableConv2d(1536, 2048, 3, padding=1, bias=False),
                 nn.BatchNorm2d(2048),
                 nn.ReLU(inplace=True), ]
        self.exit_conv3 = nn.Sequential(*conv3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(2048, n_class),
                                nn.ReLU(inplace=True))

        pass

    def forward(self, x):
        # Entry Flow

        x = self.entry_conv1(x)  # 2x
        print(x.shape)
        x = self.entry_conv2(x)
        print(x.shape)
        x = self.entry_block1(x)  # 4x
        print(x.shape)
        low_level_features = x  # low-level features
        x = self.entry_block2(x)  # 8x
        print(x.shape)
        x = self.entry_block3(x)  # 16x
        print(x.shape)

        # Middle Flow
        for block in self.middle_flow:
            x = block(x)  # 16x
            print(x.shape)

        # Exit Flow
        x = self.exit_block(x)  # 32x
        print(x.shape)
        x = self.exit_conv1(x)  # 32x
        print(x.shape)
        x = self.exit_conv2(x)  # 32x
        print(x.shape)
        x = self.exit_conv3(x)  # 32x
        print(x.shape)

        # 输出主干特征x和low-level特征
        return x, low_level_features

    pass


if __name__ == '__main__':
    # device = torch.device('cuda:6')
    device = torch.device('cpu')

    net = XceptionBackbone(3, 8).to(device)
    print('in:', net)

    in_data = torch.randint(0, 256, (24, 3, 299, 299), dtype=torch.float)
    print(in_data.shape)
    in_data = in_data.to(device)

    high_level, low_level = net(in_data)
    high_level = high_level.cpu()
    low_level = low_level.cpu()
    print('out:', high_level.shape, low_level.shape)
    pass
