import torch
import torch.nn as nn


class FCN32s(nn.Module):
    """
    FCN32s模型，backbone为VGG16
    """
    def __init__(self, n_class):
        """
        FCN32s的__init__
        :param n_class: 分类个数
        """
        super(FCN32s, self).__init__()
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

        # upsample
        self.upsample = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
        pass

    def forward(self, x):
        # 保留输入，用于最后剪裁
        h = x

        # 下采样
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.fc6(h)
        h = self.fc7(h)
        h = self.fc8(h)

        # 上采样
        h = self.upsample(h)

        # 因与输入大小不一致，需要crop
        h = h[..., 19:19 + x.shape[2], 19:19 + x.shape[3]]
        return h

    pass


if __name__ == '__main__':
    fcn32s = FCN32s(8)
    in_data = torch.randint(0, 10, (1, 3, 224, 224)).type(torch.float32)
    out_data = fcn32s(in_data)
