import matplotlib.pyplot as plt
import numpy as np
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

    def __call__(self, images):
        """
        逐个剪裁图像
        :param images: iterable，图像HWC
        :return: 剪裁后的图像HWC
        """
        res = []
        for im in images:
            im = np.array(im)
            if im.ndim == 3:
                h, w, _ = im.shape
            if im.ndim == 2:
                h, w = im.shape
            assert 0 <= self.start[0] < h and (self.stop[0] is None or 0 <= self.stop[0] < h)  # 剪裁大小不超过原图像大小
            assert 0 <= self.start[1] < w and (self.stop[1] is None or 0 <= self.stop[1] < w)

            hslice = slice(self.start[0], self.stop[0])  # H方向剪裁量
            wslice = slice(self.start[1], self.stop[1])  # W方向剪裁量
            res.append(im[hslice, wslice])
        return res

    pass


if __name__ == '__main__':
    image = Image.open('Z:/Python资料/AI/cv_lane_seg_初赛/'
                       'Road04/ColorImage_road04/ColorImage/Record002/Camera 6/'
                       '171206_054227243_Camera_6.jpg')
    label = Image.open('Z:/Python资料/AI/cv_lane_seg_初赛/'
                       'Gray_Label/Label_road04/Label/Record002/Camera 6/'
                       '171206_054227243_Camera_6_bin.png')

    # t = Crop(offsets=None, size=None)
    # t = Crop(offsets=None, size=(None, None))
    # t = Crop(offsets=None, size=(None, 100))
    # t = Crop(offsets=None, size=(100, None))
    # t = Crop(offsets=None, size=(500, 500))
    # t = Crop(offsets=(None, None), size=None)
    # t = Crop(offsets=(None, 100), size=None)
    # t = Crop(offsets=(100, None), size=None)
    # t = Crop(offsets=(100, 100), size=None)

    t = Crop(offsets=(690, None), size=(None, None))
    im_crop, lb_crop = t((image, label))

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    ax = ax.flatten()
    ax[0].imshow(image)
    ax[1].imshow(im_crop)
    ax[2].imshow(label, cmap='gray')
    ax[3].imshow(lb_crop, cmap='gray')
    plt.tight_layout()
    plt.show()
    pass
