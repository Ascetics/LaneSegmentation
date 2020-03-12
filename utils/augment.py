import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image


class PairCrop(object):
    def __init__(self, offsets=None, size=None):
        """
        剪裁图像
        :param offsets: 剪裁的偏移量，(H,W)类型，None表示不偏移
        :param size: 剪裁的大小，(H,W)类型，None表示不剪裁
        """
        super(PairCrop, self).__init__()
        # 偏移量可以为空或大于等于0，None或等于0表示不偏移
        assert offsets is None or (offsets[0] is None or offsets[0] >= 0) and (offsets[1] is None or offsets[1] >= 0)
        # 剪裁的大小，必须是正数或者None不剪裁
        assert size is None or (size[0] is None or size[0] > 0) and (size[1] is None or size[1] > 0)

        if offsets is None:  # HW都不偏移
            offsets = (0, 0,)
        self.start = (0 if offsets[0] is None else offsets[0],  # H或W不偏移
                      0 if offsets[1] is None else offsets[1],)

        if size is None:  # HW都不剪裁
            size = (None, None,)
        self.stop = (self.start[0] + size[0] if size[0] is not None else size[0],  # H或W不剪裁
                     self.start[1] + size[1] if size[1] is not None else size[1],)
        pass

    def __call__(self, image, label):
        """
        剪裁图像
        :param image: [H,W,C] PIL Image RGB
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB,  [H,W] PIL Image trainId
        """

        image = np.asarray(image)
        label = np.asarray(label)
        assert image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1]

        h, w = image.shape[0], image.shape[1]
        assert 0 <= self.start[0] < h and (self.stop[0] is None or 0 <= self.stop[0] < h)  # 剪裁大小不超过原图像大小
        assert 0 <= self.start[1] < w and (self.stop[1] is None or 0 <= self.stop[1] < w)

        hslice = slice(self.start[0], self.stop[0])  # H方向剪裁量
        wslice = slice(self.start[1], self.stop[1])  # W方向剪裁量

        image = Image.fromarray(image[hslice, wslice])
        label = Image.fromarray(label[hslice, wslice])
        return image, label

    pass


class PairRandomLeftRightFlip(object):
    def __init__(self):
        """
        随机图像左右翻转
        """
        super(PairRandomLeftRightFlip, self).__init__()
        pass

    def __call__(self, image, label):
        """
        随机图像左右翻转
        :param image: [H,W,C] PIL Image RGB
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB,  [H,W] PIL Image trainId
        """
        if random.random() < 0.5:  # 50%的概率会翻转
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # PIL的接口，左右翻转，上下用FLIP_TOP_BOTTOM
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return image, label

    pass


class PairAdjust(object):
    def __init__(self, factors=(0.3, 2.)):
        super(PairAdjust, self).__init__()
        self.factors = factors
        pass

    def __call__(self, image, label):
        """
        调整亮度、对比度、饱和度
        只调整image，不调整label
        :param image: [H,W,C] PIL Image RGB 0~255
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB 0~255,  [H,W] PIL Image trainId
        """
        brightness_factor = random.uniform(*self.factors)
        contrast_factor = random.uniform(*self.factors)
        saturation_factor = random.uniform(*self.factors)

        image = F.adjust_brightness(image, brightness_factor)
        image = F.adjust_contrast(image, contrast_factor)
        image = F.adjust_saturation(image, saturation_factor)
        return image, label

    pass


class PairResize(object):
    def __init__(self, size):
        """
        图像等比缩放
        :param size: 图像等比缩放后，短边的大小
        """
        super(PairResize, self).__init__()
        self.size = size
        pass

    def __call__(self, image, label):
        """
        图像等比缩放
        :param image: [H,W,C] PIL Image RGB
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB,  [H,W] PIL Image trainId
        """
        image = F.resize(image, self.size, interpolation=Image.BILINEAR)
        label = F.resize(label, self.size, interpolation=Image.NEAREST)  # label要用邻近差值
        return image, label

    pass


class PairNormalizeToTensor(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        IMAGE_NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet统计的RGB mean
        IMAGE_NORM_STD = [0.229, 0.224, 0.225]  # ImageNet统计的RGB std
        LABEL_NORM_MEAN = [0.5]  # ImageNet统计的GRAY mean
        LABEL_NORM_STD = [0.5]  # ImageNet统计的GRAY std
        :param mean: 正则化的平均值mean
        :param std: 正则化的标准差std
        """
        super(PairNormalizeToTensor, self).__init__()
        self.mean = mean
        self.std = std
        pass

    def __call__(self, image, label):
        """
        归一化，只对image除以255，label不动
        :param image: [H,W,C] PIL Image RGB 0~255
        :param label: [H,W] PIL Image trainId
        :return: [C,H,W] tensor RGB -1.0~0.0,  [H,W] tensor trainId
        """
        # torchvision.transform的API，对PIL Image类型image归一化，也就是除以255
        # 并转为tensor，维度变为[C,H,W]
        # image [C,H,W]tensor RGB 0.0~1.0
        image = F.to_tensor(image)

        # 正则化，x=(x-mean)/std
        # 只对image正则化, image [C,H,W]tensor RGB -1.0~1.0
        image = F.normalize(image, self.mean, self.std)

        # 先转为ndarray，再转为tensor，不归一化，维度保持不变
        # label [H,W]tensor trainId
        label = torch.from_numpy(np.asarray(label))

        return image, label

    pass


if __name__ == '__main__':
    x = np.array([[[53, 170, 134],
                   [92, 111, 202]],
                  [[235, 126, 244],
                   [107, 46, 15]]], dtype=np.uint8)  # 模拟一个RGB图像
    y = np.array([[2, 4],
                  [3, 5]], dtype=np.uint8)  # 模拟一个trainId的label
    print('np', x)
    print('np', y)

    x = Image.fromarray(x)
    y = Image.fromarray(y)
    x, y = PairNormalizeToTensor()(x, y)  # 测试PairNormalizeToTensor
    print('tensor', x)  # x应该是-1.0~1.0
    print('tensor', y)  # y应该是trainId
    """
    tensor tensor([[[-1.2103, -0.5424],
         [ 1.9064, -0.2856]],

        [[ 0.9405, -0.0924],
         [ 0.1702, -1.2304]],

        [[ 0.5311,  1.7163],
         [ 2.4483, -1.5430]]])
    tensor tensor([[2, 4],
    """

    im = Image.open('Z:/Python资料/AI/cv_lane_seg_初赛/'
                    'Road04/ColorImage_road04/ColorImage/Record002/Camera 6/'
                    '171206_054227243_Camera_6.jpg')
    lb = Image.open('Z:/Python资料/AI/cv_lane_seg_初赛/'
                    'Gray_Label/Label_road04/Label/Record002/Camera 6/'
                    '171206_054227243_Camera_6_bin.png')

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    ax = ax.flatten()
    ax[0].imshow(im)
    ax[1].imshow(lb, cmap='gray')

    crop = PairCrop(offsets=(690, None), size=(None, None))
    random_lr_flip = PairRandomLeftRightFlip()
    adjust = PairAdjust()
    resize = PairResize(256)
    ts = [
        crop,
        random_lr_flip,
        adjust,
        resize,
    ]
    for t in ts:
        im, lb = t(im, lb)
        pass
    ax[2].imshow(im)
    ax[3].imshow(lb, cmap='gray')
    plt.tight_layout()
    plt.show()

    im, lb = PairNormalizeToTensor()(im, lb)
    print(im.shape, lb.shape)

    pass
