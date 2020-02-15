import torch
import torch.nn as nn
import torchvision


class UNetFactory(nn.Module):
    """
    UNet网络工厂
    """

    def __init__(self, encode_blocks, encode_bottom,
                 decode_ups, decode_blocks, n_class):
        """
        生成Unet网络
        :param encode_blocks: encoder的一部分，blocks可以替换成ResNet等，每个block产生一个shortcut
        :param encode_bottom: encoder的一部分，位于UNet底部，可以替换成ResNet等，不产生shortcut
        :param decode_ups: 上采样块，可以用转置卷积、双线性差值
        :param decode_blocks: decoder部分，可以替换成ResNet等
        :param n_class: 分类数
        """
        super(UNetFactory, self).__init__()
        self.encoder = UNetEncoder(encode_blocks)
        self.bottom = encode_bottom
        self.decoder = UNetDecoder(decode_ups, decode_blocks)
        self.classifier = nn.Conv2d(64, n_class, 1)
        pass

    def forward(self, x):
        x, shortcuts = self.encoder(x)
        if self.bottom is not None:
            x = self.bottom(x)
            # print('bottom', x.shape)
        x = self.decoder(x, shortcuts)
        x = self.classifier(x)
        return x

    pass


def _double_conv3x3(in_channels, out_channels):
    """
    连续两个3x3卷积，后面接BatchNormalization和ReLU
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :return: 连续两个3x3卷积子网络
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _encode_block(in_channels, out_channels):
    """
    论文中的encoder block
    :param in_channels:输入通道数
    :param out_channels:输出通道数
    :return: 池化后再2个3x3卷积
    """
    return nn.Sequential(
        nn.MaxPool2d(2, stride=2, ceil_mode=True),
        _double_conv3x3(in_channels, out_channels)
    )


def _make_encoder(in_channels):
    """
    论文中encoder
    :param in_channels:
    :return: 论文中encoder
    """
    return [
        _double_conv3x3(in_channels, 64),  # 2个3x3卷积
        _encode_block(64, 128),  # 池化后再2个3x3卷积
        _encode_block(128, 256),
        _encode_block(256, 512),
    ]


class UNetEncoder(nn.Module):
    def __init__(self, encode_blocks):
        super(UNetEncoder, self).__init__()
        self.encode_blocks = nn.ModuleList(encode_blocks)
        pass

    def forward(self, x):
        shortcuts = []
        for i, block in enumerate(self.encode_blocks):
            x = block(x)
            shortcuts.append(x)
            # print('en', i, x.shape)
        return x, shortcuts

    pass


def _decode_ups_deconv():
    """
    论文中的上采样，用转置卷积实现
    :return:
    """
    return [
        nn.ConvTranspose2d(1024, 512, 2, stride=2),
        nn.ConvTranspose2d(512, 256, 2, stride=2),
        nn.ConvTranspose2d(256, 128, 2, stride=2),
        nn.ConvTranspose2d(128, 64, 2, stride=2),
    ]


def _decode_ups_bilinear():
    """
    论文中的上采样，用双线性差值实现
    :return:
    """
    return [
        nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, 1)
        ),
        nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 1)
        ),
        nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 1)
        ),
        nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 1)
        ),
    ]


def _decode_blocks():
    return [
        _double_conv3x3(1024, 512),
        _double_conv3x3(512, 256),
        _double_conv3x3(256, 128),
        _double_conv3x3(128, 64),
    ]


class UNetDecoder(nn.Module):
    def __init__(self, decode_ups, decode_blocks):
        super(UNetDecoder, self).__init__()
        self.decode_ups = decode_ups
        self.decode_blocks = decode_blocks
        pass

    @staticmethod
    def _crop(x, shortcut):
        """
        按照x和shortcut最小值剪裁
        :param x: 上采样结果
        :param shortcut: 就是shortcut
        :return: 剪裁后的x和shortcut
        """
        _, _, h_x, w_x = x.shape
        _, _, h_s, w_s = shortcut.shape
        h, w = min(h_x, h_s), min(w_x, w_s)  # 取最小是
        hc_x, wc_x = (h_x - h) // 2, (w_x - w) // 2  # x要剪裁掉的值
        hc_s, wc_s = (h_s - h) // 2, (w_s - w) // 2  # shortcut要剪裁掉的值
        return x[..., hc_x:hc_x + h, wc_x: wc_x + w], \
               shortcut[..., hc_s:hc_s + h, wc_s:wc_s + w]

    def forward(self, x, shortcuts):
        for i, (up, block) in enumerate(zip(self.decode_ups, self.decode_blocks)):
            x = up(x)  # 上采样
            x, s = self._crop(x, shortcuts[-(i + 1)])  # 剪裁
            x = torch.cat((x, s), dim=1)  # concatenate特征融合
            x = block(x)
            # print('de', i, x.shape)
        return x

    pass


def unet(in_channels, n_class, upmode='deconv'):
    encode_blocks = _make_encoder(in_channels=in_channels)
    encode_bottom = _encode_block(512, 1024)
    if upmode == 'deconv':
        decode_ups = _decode_ups_deconv()
    elif upmode == 'bilinear':
        decode_ups = _decode_ups_bilinear()
    decode_blocks = _decode_blocks()
    return UNetFactory(encode_blocks, encode_bottom, decode_ups, decode_blocks, n_class)


def unet_resnet(in_channels, n_class, upmode='deconv'):
    # TODO ResNet实现
    return


if __name__ == '__main__':
    channel = 1
    size = 572
    n_class = 2
    net = unet(channel, n_class)
    x = torch.randint(0, 255, (size, size), dtype=torch.float32) \
        .view((1, channel, size, size))
    out = net(x)
