import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image


class Crop(object):
    def __init__(self, offsets=None, size=None):
        """
        剪裁图像
        :param offsets: 剪裁的偏移量，(H,W)类型，None表示不偏移
        :param size: 剪裁的大小，(H,W)类型，None表示不剪裁
        """
        super(Crop, self).__init__()
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
        逐个剪裁图像
        :param image: [H,W,C],PIL Image图像
        :param label: [H,W,C],PIL Image图像
        :return: 剪裁后的图像HWC
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


class RandomLeftRightFlip(object):
    def __init__(self):
        """
        随机图像左右翻转
        """
        super(RandomLeftRightFlip, self).__init__()
        pass

    def __call__(self, image, label):
        """
        随机图像左右翻转
        :param image: [H,W,C],PIL Image图像
        :param label: [H,W,C],PIL Image图像
        :return:
        """
        if random.random() < 0.5:  # 50%的概率会翻转
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # PIL的接口，左右翻转，上下用FLIP_TOP_BOTTOM
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return image, label

    pass


class Adjust(object):
    def __init__(self, factors=(0.3, 2.)):
        super(Adjust, self).__init__()
        self.factors = factors
        pass

    def __call__(self, image, label):
        """
        调整亮度、对比度、饱和度
        :param image: [H,W,C],PIL Image图像，只调整image
        :param label: [H,W,C],PIL Image图像，不调整label
        :return:
        """
        brightness_factor = random.uniform(*self.factors)
        contrast_factor = random.uniform(*self.factors)
        saturation_factor = random.uniform(*self.factors)

        image = F.adjust_brightness(image, brightness_factor)
        image = F.adjust_contrast(image, contrast_factor)
        image = F.adjust_saturation(image, saturation_factor)
        return image, label

    pass


if __name__ == '__main__':
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

    crop = Crop(offsets=(690, None), size=(None, None))
    random_lr_flip = RandomLeftRightFlip()
    adjust = Adjust()
    ts = [
        crop,
        random_lr_flip,
        adjust,
    ]
    for t in ts:
        im, lb = t(im, lb)
        pass
    ax[2].imshow(im)
    ax[3].imshow(lb, cmap='gray')
    plt.tight_layout()
    plt.show()
    pass
