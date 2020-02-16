import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1)
        )
        self.adjust_channel = nn.Conv2d(5 * out_channels, out_channels, 1)
        pass

    def forward(self, x):
        c = []

        size = x.shape[2:]
        h = self.pool(x)
        h = F.interpolate(h, size=size, mode='bilinear', align_corners=False)
        c.append(h)

        c.append(self.aspp18(x))
        c.append(self.aspp12(x))
        c.append(self.aspp6(x))
        c.append(self.aspp1(x))

        x = torch.cat(c, dim=1)
        x = self.adjust_channel(x)
        return x

    pass


class DeepLabV3P(nn.Module):
    def __init__(self, n_class):
        super(DeepLabV3P, self).__init__()
        # # backbone 使用了resnet50
        resnet50 = torchvision.models.resnet50(pretrained=True)

        # backbone 使用了resnet50， in_channels=3， out_channels=64
        self.start = nn.Sequential(
            resnet50.conv1,  # out_channel2=64
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool
        )

        # 从另一段代码看来的，不知道为什么是这样产生low-level feature
        # 也不知道为什么out_channels是48
        self.low_feature = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 256, 1)
        )

        # backbone的后半部分
        self.layers = nn.Sequential(
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4  # out_channels=2048
        )

        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.adjust_channel = nn.Conv2d(256, n_class, 1)
        pass

    def forward(self, x):
        origin_size = x.shape[2:]  # 记录输入图像spatial大小

        x = self.start(x)  # backbone前半部分
        low_feature = self.low_feature(x)  # low-level feature，已经conv1x1，out_channel=256
        x = self.layers(x)  # backbone后半部分

        low_feature_size = low_feature.shape[2:]
        x = self.aspp(x)
        x = F.interpolate(x, low_feature_size, mode='bilinear', align_corners=False)  # spatial一致
        x = torch.cat((low_feature, x), dim=1)

        x = self.conv(x)
        x = F.interpolate(x, origin_size, mode='bilinear', align_corners=False)  # 恢复spatial大小

        x = self.adjust_channel(x)

        return x

    pass


if __name__ == '__main__':
    in_data = torch.randint(0, 255, (3, 224, 224), dtype=torch.float32) \
        .view((1, 3, 224, 224))
    net = DeepLabV3P(n_class=2)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net.to(device)
    # in_data = in_data.to(device)

    out_data = net(in_data)
    print(out_data.shape)
    pass
