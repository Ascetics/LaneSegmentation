import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.resnet_backbone import resnet18_backbone, resnet34_backbone
from nets.resnet_backbone import resnet50_backbone, resnet101_backbone, resnet152_backbone


class ASPPConv1x1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """
        ASPP用的5个处理之1，1个1x1卷积
        :param in_channels: 输入channels，是backbone产生的主要特征的输出channels
        :param out_channels: 输出channels，论文建议取值256
        """
        modules = [nn.Conv2d(in_channels, out_channels, 1, bias=False),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True), ]
        super(ASPPConv1x1, self).__init__(*modules)
        pass

    pass


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        """
        ASPP用的5个处理之3，3个dilation conv，都是3x3的same卷积
        :param in_channels: dilation conv的输入channels，是backbone产生的主要特征的输出channels
        :param out_channels: dilation conv的输出channels，论文建议取值256
        :param dilation: 膨胀率，论文建议取值6,12,18
        """
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                             padding=dilation, dilation=dilation, bias=False),  # same卷积padding=dilation*(k-1)/2
                   nn.BatchNorm2d(out_channels),  # 有BN，卷积bias=False
                   nn.ReLU(inplace=True), ]  # 激活函数
        super(ASPPConv, self).__init__(*modules)
        pass

    pass


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """
        ASPP用的5个处理之1，Image Pooling
        :param in_channels: 输入channels，是backbone产生的主要特征的输出channels
        :param out_channels: 输出channels，论文建议取值256
        """
        modules = [nn.AdaptiveAvgPool2d(1),  # 全局平均池化，输出spatial大小1
                   nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1卷积调整channels
                   nn.BatchNorm2d(out_channels),  # 有BN，卷积bias=False
                   nn.ReLU(inplace=True), ]  # 激活函数
        super(ASPPPooling, self).__init__(*modules)
        pass

    def forward(self, x):
        size = x.shape[-2:]  # 记录下输入的大小
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  # 双线性差值上采样到原spatial大小

    pass


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        ASPP，对backbone产生的主干特征进行空间金字塔池化。
        金字塔有5层：1个1x1卷积，3个3x3 dilation conv，1个全局平均池化
        将5层cat后再调整channels输出。
        这里不进行upsample，因为不知道low-level的spatial大小。
        :param in_channels: 输入channels，是backbone产生的主要特征的输出channels
        :param out_channels: 输出channels，论文建议取值256
        """
        super(ASPP, self).__init__()
        modules = [ASPPConv1x1(in_channels, out_channels),  # 1个1x1卷积
                   ASPPConv(in_channels, out_channels, dilation=6),  # 3x3 dilation conv，dilation=6
                   ASPPConv(in_channels, out_channels, dilation=12),  # 3x3 dilation conv，dilation=12
                   ASPPConv(in_channels, out_channels, dilation=18),  # 3x3 dilation conv，dilation=18
                   ASPPPooling(in_channels, out_channels), ]  # 全局平均池化Image Pooling
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0.5))  # 将5层cat后再调整channels输出，但不知道为什么Dropout
        pass

    def forward(self, x):
        output = []
        for mod in self.convs:
            output.append(mod(x))
            pass
        x = torch.cat(output, dim=1)
        x = self.project(x)
        return x

    pass


def resnet_backbone(in_channels, resnet_type='resnet101'):
    """
    使用ResNet作为backbone
    :param in_channels: 输出channels也就是图像的channels
    :param resnet_type: 默认和论文一致ResNet101
    :return: 返回backbone，主干特征channels，low-level特征channels
    """
    if resnet_type == 'resnet18':
        backbone = resnet18_backbone(in_channels=in_channels)
        atrous_channels = 512
        low_level_channels = 64
    elif resnet_type == 'resnet34':
        backbone = resnet34_backbone(in_channels=in_channels)
        atrous_channels = 512
        low_level_channels = 64
    elif resnet_type == 'resnet50':
        backbone = resnet50_backbone(in_channels=in_channels)
        atrous_channels = 2048
        low_level_channels = 256
    elif resnet_type == 'resnet101':
        backbone = resnet101_backbone(in_channels=in_channels)
        atrous_channels = 2048
        low_level_channels = 256
    elif resnet_type == 'resnet152':
        backbone = resnet152_backbone(in_channels=in_channels)
        atrous_channels = 2048
        low_level_channels = 256
    else:
        raise ValueError('resnet type error!')
    return backbone, atrous_channels, low_level_channels


class DeepLabV3P(nn.Module):
    aspp_out_channels = 256  # ASPP最终输出channels=256
    reduce_to_channels = 48  # 论文中说low-level特征减少channels到48

    def __init__(self, backbone_type, in_channels, n_class):
        super(DeepLabV3P, self).__init__()
        if backbone_type == 'resnet':  # 目前只支持ResNet101作为backbone
            backbone, aspp_in_channels, low_level_in_channels = resnet_backbone(in_channels)
        else:
            raise ValueError('backbone type error!')
        self.backbone = backbone
        self.aspp = ASPP(aspp_in_channels, self.aspp_out_channels)  # 论文建议channels=256

        reduce_modules = [nn.Conv2d(low_level_in_channels, self.reduce_to_channels, 1, bias=False),
                          nn.BatchNorm2d(self.reduce_to_channels),
                          nn.ReLU(inplace=True), ]
        self.reduce_channels = nn.Sequential(*reduce_modules)

        decode_modules = [nn.Conv2d(self.aspp_out_channels + self.reduce_to_channels, 256, 3, padding=1, bias=False),
                          nn.BatchNorm2d(256),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(256, n_class, 3, padding=1, bias=False),
                          nn.BatchNorm2d(n_class),
                          nn.ReLU(inplace=True), ]
        self.decode = nn.Sequential(*decode_modules)
        pass

    def forward(self, x):
        size1 = x.shape[-2:]  # 图像原始大小

        high_level, low_level = self.backbone(x)  # 提取特征，主干特征high-level和低级特征low-level

        low_level = self.reduce_channels(low_level)  # low-level feature减少channels到48
        size2 = low_level.shape[-2:]  # low-level feature大小，aspp上采样目标大小

        high_level = self.aspp(high_level)  # 空间金字塔池化
        high_level = F.interpolate(high_level, size=size2, mode='bilinear',
                                   align_corners=False)  # 上采样和low-level的spatial大小一致

        x = torch.cat([high_level, low_level], dim=1)  # cat融合一下
        x = self.decode(x)  # 后面跟一系列3x3卷积，本实现选择2个3x3卷积

        return F.interpolate(x, size=size1, mode='bilinear', align_corners=False)  # 上采样和原图像大小一致

    pass


if __name__ == '__main__':
    net = DeepLabV3P('resnet', 3, 8)
    print(net)

    in_data = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.float)
    print('in data:', in_data.shape)

    out_data = net(in_data)
    print('out_data:', out_data.shape)
    pass
