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
        assert len(encode_blocks) == len(decode_blocks)  # 有shortcut的encode和decode数量一致
        assert len(decode_blocks) == len(decode_ups)  # 上采样和decode数量一致

        super(UNetFactory, self).__init__()
        self.encoder = UNetEncoder(encode_blocks)  # encoder有shortcut
        self.bottom = encode_bottom  # 最后一个encode没有shortcut，单独列出
        self.decoder = UNetDecoder(decode_ups, decode_blocks)  # decoder包含上采样和decode
        self.classifier = nn.Conv2d(64, n_class, 1)  # 最终输出
        pass

    def forward(self, x):
        x, shortcuts = self.encoder(x)  # encoder生成特征x和shortcut
        if self.bottom is not None:
            x = self.bottom(x)  # 最后一个encoder不产生shortcut的下采样单独列出
        x = self.decoder(x, shortcuts)  # decoder
        x = self.classifier(x)  # 最终分类
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
        for block in self.encode_blocks:
            x = block(x)  # 逐个调用encode
            shortcuts.append(x)  # 记录每个block输出为shortcut
        return x, shortcuts

    pass


class UNetDecoder(nn.Module):
    """
    Decoder处理
    """

    def __init__(self, decode_ups, decode_blocks):
        super(UNetDecoder, self).__init__()
        self.decode_ups = nn.ModuleList(decode_ups)  # 上采样
        self.decode_blocks = nn.ModuleList(decode_blocks)  # decode
        pass

    @staticmethod
    def _crop(x, shortcut):
        """
        按照x和shortcut最小值剪裁
        :param x: 上采样结果
        :param shortcut: 就是shortcut
        :return: 剪裁后的x和shortcut
        """
        _, _, h_x, w_x = x.shape  # 取特征的spatial大小
        _, _, h_s, w_s = shortcut.shape  # 取shortcut的spatial大小
        h, w = min(h_x, h_s), min(w_x, w_s)  # 取最小spatial
        hc_x, wc_x = (h_x - h) // 2, (w_x - w) // 2  # x要剪裁掉的值
        hc_s, wc_s = (h_s - h) // 2, (w_s - w) // 2  # shortcut要剪裁掉的值
        x = x[..., hc_x:hc_x + h, wc_x: wc_x + w]  # center crop
        shortcut = shortcut[..., hc_s:hc_s + h, wc_s:wc_s + w]  # center crop
        return x, shortcut

    def forward(self, x, shortcuts):
        """
        Decoder逐个block调用
        :param x: Encoder生成的x
        :param shortcuts: Encoder生成的shortcuts
        :return: Decoder结果
        """
        z = zip(self.decode_ups, self.decode_blocks)  # 上采样和decode一一对应
        for i, (up, block) in enumerate(z):
            x = up(x)  # 上采样
            x, s = self._crop(x, shortcuts[-(i + 1)])  # 剪裁，shortcut顺序-1，-2，-3,-4，后出现的先融合
            x = torch.cat((x, s), dim=1)  # concatenate特征融合
            x = block(x)  # decode
        return x

    pass


def upconv(in_channels, out_channels):
    """
    论文中的上采样，用转置卷积实现，上次采样2倍
    :return:
    """
    return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)


def upsample(in_channels, out_channels):
    """
    论文中的上采样，用双线性差值实现，上采样2倍，再调整channel数
    :return:
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_channels, out_channels, 1)
    )


################################################################################

class _UNetEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, maxpool=False, padding=0):
        """
        论文中的encode block，每个block有2个3x3卷积
        两个3x3卷积默认不加padding，后面有bn所以bias=False
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        :param maxpool: 第一个block没有maxpool，其余的block有maxpool
        :param padding: 论文没有padding
        """
        super(_UNetEncodeBlock, self).__init__()
        self.maxpool = None
        if maxpool:
            self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 是否下采样
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding, bias=False),
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
    def __init__(self, in_channels, out_channels, padding=0):
        """
        论文中的decode block，不包括上采样和特征融合，每个block有2个3x3卷积
        两个3x3卷积默认不加padding，后面有bn所以bias=False
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        :param padding: 论文不加padding
        """
        super(_UNetDecodeBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        pass

    def forward(self, x):
        x = self.conv(x)
        return x

    pass


def unet(in_channels, n_class, upmode='upconv', padding=0):
    """
    生成论文的UNet
    :param in_channels: 输入通道数
    :param n_class: 分类数
    :param upmode: 上采样模式，默认'upconv'转置卷积，'upsample'双线性差值
    :return: 论文UNet网络
    """
    assert upmode == 'upconv' or upmode == 'upsample'

    # Encoder
    encode_blocks = [
        _UNetEncodeBlock(in_channels, 64, padding=padding),  # 第一个encode没有下采样，有shortcut
        _UNetEncodeBlock(64, 128, padding=padding, maxpool=True),  # 第二个encode有下采样，有shortcut
        _UNetEncodeBlock(128, 256, padding=padding, maxpool=True),  # 第三个encode有下采样，有shortcut
        _UNetEncodeBlock(256, 512, padding=padding, maxpool=True),  # 第四个encode有下采样，有shortcut
    ]
    encode_bottom = _UNetEncodeBlock(512, 1024, maxpool=True, padding=padding)  # 第五个encode有下采样，没有shortcut

    # Upsample
    decode_ups = None  # 上采样可以二选一
    if upmode == 'upconv':  # 转置卷积
        decode_ups = [upconv(1024, 512), upconv(512, 256),
                      upconv(256, 128), upconv(128, 64)]
    elif upmode == 'upsample':  # 双线性差值
        decode_ups = [upsample(1024, 512), upsample(512, 256),
                      upsample(256, 128), upsample(128, 64)]

    # Decoder
    decode_blocks = [
        _UNetDecodeBlock(1024, 512, padding=padding),  # 第一个decode
        _UNetDecodeBlock(512, 256, padding=padding),  # 第二个decode
        _UNetDecodeBlock(256, 128, padding=padding),  # 第三个decode
        _UNetDecodeBlock(128, 64, padding=padding),  # 第四个decode
    ]
    return UNetFactory(encode_blocks, encode_bottom, decode_ups, decode_blocks,
                       n_class)


################################################################################

# TODO 待ResNet完成后修改
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


def unet_res_block(in_channels, n_class, upmode='upconv'):
    """
    用Residual Block替换论文中的两层卷积形成的UNet
    :param in_channels: 输入通道数
    :param n_class: 分类数
    :param upmode: 上采样模式，默认转置卷积，'bilinear'双线性差值
    :return: 用Residual Block替换论文中的两层卷积形成的UNet
    """
    assert upmode == 'upconv' or upmode == 'upsample'

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
    if upmode == 'upconv':
        decode_ups = [upconv(1024, 512), upconv(512, 256),
                      upconv(256, 128), upconv(128, 64)]
    elif upmode == 'upsample':
        decode_ups = [upsample(1024, 512), upsample(512, 256),
                      upsample(256, 128), upsample(128, 64)]

    # Decoder
    decode_blocks = [
        _UNetDecodeResBlock(1024, 512),
        _UNetDecodeResBlock(512, 256),
        _UNetDecodeResBlock(256, 128),
        _UNetDecodeResBlock(128, 64),
    ]
    return UNetFactory(encode_blocks, encode_bottom, decode_ups, decode_blocks,
                       n_class)


################################################################################

if __name__ == '__main__':
    """
    单元测试
    """
    channel = 1
    num_class = 2
    net = unet(channel, num_class, padding=1)
    print(net)

    # size = 572
    # in_data = torch.randint(0, 255, (size, size), dtype=torch.float32)
    # in_data = in_data.view((1, channel, size, size))


