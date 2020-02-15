import torch
import torch.nn as nn


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
            print('bottom', x.shape)
        x = self.decoder(x, shortcuts)
        x = self.classifier(x)
        return x

    pass


class UNetEncoder(nn.Module):
    """
    Encoder处理
    """

    def __init__(self, encode_blocks):
        super(UNetEncoder, self).__init__()
        self.encode_blocks = nn.ModuleList(encode_blocks)
        pass

    def forward(self, x):
        """
        Encoder逐个block调用
        :param x:
        :return: 最终输出x和shortcuts
        """
        shortcuts = []
        for i, block in enumerate(self.encode_blocks):
            x = block(x)
            shortcuts.append(x)
            print('en', i, x.shape)
        return x, shortcuts

    pass


class UNetDecoder(nn.Module):
    """
    Decoder处理
    """

    def __init__(self, decode_ups, decode_blocks):
        super(UNetDecoder, self).__init__()
        self.decode_ups = nn.ModuleList(decode_ups)
        self.decode_blocks = nn.ModuleList(decode_blocks)
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
        """
        Decoder逐个block调用
        :param x: Encoder生成的x
        :param shortcuts: Encoder生成的shortcuts
        :return: Decoder结果
        """
        for i, (up, block) in enumerate(zip(self.decode_ups, self.decode_blocks)):
            x = up(x)  # 上采样
            x, s = self._crop(x, shortcuts[-(i + 1)])  # 剪裁
            x = torch.cat((x, s), dim=1)  # concatenate特征融合
            x = block(x)
            print('de', i, x.shape)
        return x

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


################################################################################

class _UNetEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, maxpool=False):
        super(_UNetEncodeBlock, self).__init__()
        self.maxpool = None
        if maxpool:
            self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        pass

    def forward(self, x):
        if self.maxpool:
            x = self.maxpool(x)
        x = self.conv(x)
        return x

    pass


class _UNetDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_UNetDecodeBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        pass

    def forward(self, x):
        x = self.conv(x)
        return x

    pass


def unet(in_channels, n_class, upmode='deconv'):
    """
    生成论文的UNet
    :param in_channels: 输入通道数
    :param n_class: 分类数
    :param upmode: 上采样模式，默认转置卷积，'bilinear'双线性差值
    :return: 论文UNet网络
    """
    # Encoder
    encode_blocks = [
        _UNetEncodeBlock(in_channels, 64),
        _UNetEncodeBlock(64, 128, maxpool=True),
        _UNetEncodeBlock(128, 256, maxpool=True),
        _UNetEncodeBlock(256, 512, maxpool=True),
    ]
    encode_bottom = _UNetEncodeBlock(512, 1024, maxpool=True)

    # Upsample
    decode_ups = None
    if upmode == 'deconv':
        decode_ups = _decode_ups_deconv()
    elif upmode == 'bilinear':
        decode_ups = _decode_ups_bilinear()

    # Decoder
    decode_blocks = [
        _UNetDecodeBlock(1024, 512),
        _UNetDecodeBlock(512, 256),
        _UNetDecodeBlock(256, 128),
        _UNetDecodeBlock(128, 64),
    ]
    return UNetFactory(encode_blocks, encode_bottom, decode_ups, decode_blocks, n_class)


################################################################################


class _UNetEncodeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, maxpool=False):
        super(_UNetEncodeResBlock, self).__init__()
        self.maxpool = None
        if maxpool:
            self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.adjust = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        if self.maxpool:
            x = self.maxpool(x)
        c = self.conv(x)  # 卷积
        x = self.adjust(x)  # 调整channel数
        x = c + x  # residual
        x = self.relu(x)  # relu
        return x

    pass


class _UNetDecodeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_UNetDecodeResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.adjust = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        c = self.conv(x)
        x = self.adjust(x)
        x = c + x
        x = self.relu(x)
        return x


def unet_res_block(in_channels, n_class, upmode='deconv'):
    """
    用Residual Block替换论文中的两层卷积形成的UNet
    :param in_channels: 输入通道数
    :param n_class: 分类数
    :param upmode: 上采样模式，默认转置卷积，'bilinear'双线性差值
    :return: 用Residual Block替换论文中的两层卷积形成的UNet
    """
    # Encoder
    encode_blocks = [
        _UNetEncodeResBlock(in_channels, 64),
        _UNetEncodeResBlock(64, 128, maxpool=True),
        _UNetEncodeResBlock(128, 256, maxpool=True),
        _UNetEncodeResBlock(256, 512, maxpool=True),
    ]
    encode_bottom = _UNetEncodeResBlock(512, 1024, maxpool=True)

    # Upsample
    decode_ups = None
    if upmode == 'deconv':
        decode_ups = _decode_ups_deconv()
    elif upmode == 'bilinear':
        decode_ups = _decode_ups_bilinear()

    # Decoder
    decode_blocks = [
        _UNetDecodeResBlock(1024, 512),
        _UNetDecodeResBlock(512, 256),
        _UNetDecodeResBlock(256, 128),
        _UNetDecodeResBlock(128, 64),
    ]
    return UNetFactory(encode_blocks, encode_bottom, decode_ups, decode_blocks, n_class)


################################################################################

if __name__ == '__main__':
    """
    单元测试
    """
    channel = 1
    n_class = 2
    # net = unet(channel, n_class)
    net = unet_res_block(channel, n_class)

    size = 572
    x = torch.randint(0, 255, (size, size), dtype=torch.float32) \
        .view((1, channel, size, size))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    x = x.to(device)
    out = net(x)
