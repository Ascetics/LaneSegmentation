import torch
import torch.nn as nn


class FCN16s(nn.Module):
    """
    FCN16s模型，backbone为VGG16
    """
    def __init__(self, n_class):
        """
        FCN16s的__init__
        :param n_class: 分类个数
        """
        super(FCN16s, self).__init__()
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        )

        # conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        )

        # conv4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        )

        # fc6
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        # fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        # fc8
        self.fc8 = nn.Conv2d(4096, n_class, 1)

        # upsample 2 times
        self.upsample2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)

        # adjust channel conv4
        self.adjust4 = nn.Conv2d(512, n_class, 1)

        # upsample 16 times
        self.upsample16 = nn.ConvTranspose2d(n_class, n_class, 32, stride=16, bias=False)
        pass

    def forward(self, x):
        # 保留输入，上采样16倍后剪裁用
        h = x

        # 下采样
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        conv4 = h  # 保留conv4用于特征融合
        h = self.conv5(h)
        h = self.fc6(h)
        h = self.fc7(h)
        h = self.fc8(h)

        # 上采样
        h = self.upsample2(h)  # 上采样2倍
        adjust4 = self.adjust4(conv4)  # 调整conv4的channel，与2倍上采样channel一致
        adjust4 = adjust4[..., 5:5 + h.shape[2], 5:5 + h.shape[3]]  # 剪裁
        h = h + adjust4  # 特征融合

        h = self.upsample16(h)  # 上采样16倍
        h = h[..., 27:27 + x.shape[2], 27:27 + x.shape[3]].contiguous()  # 剪裁
        return h

    pass


if __name__ == '__main__':
    fcn16s = FCN16s(8)
    in_data = torch.randint(0, 10, (1, 3, 224, 224)).type(torch.float32)
    out_data = fcn16s(in_data)
