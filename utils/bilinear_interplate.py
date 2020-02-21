import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
全部自己写的代码
https://github.com/Ascetics/LaneSegmentation/blob/master/utils/bilinear_interplate.py
"""

def bilinear_interpolate(src, dst_size, align_corners=False):
    """
    双线性差值
    :param src: 原图像张量 NCHW
    :param dst_size: 目标图像spatial大小(H,W)
    :param align_corners: 换算坐标的不同方式
    :return: 目标图像张量NCHW
    """
    src_n, src_c, src_h, src_w = src.shape
    dst_n, dst_c, (dst_h, dst_w) = src_n, src_c, dst_size

    if src_h == dst_h and src_w == dst_w:
        return src.copy()
    """将dst的H和W坐标映射到src的H和W坐标"""
    hd = torch.arange(0, dst_h)
    wd = torch.arange(0, dst_w)
    if align_corners:  # 按照4个顶角对齐的映射算法
        h = float(src_h - 1) / (dst_h - 1) * hd
        w = float(src_w - 1) / (dst_w - 1) * wd
    else:  # 按照中心对齐的映射算法
        h = float(src_h) / dst_h * (hd + 0.5) - 0.5
        w = float(src_w) / dst_w * (wd + 0.5) - 0.5

    h = torch.clamp(h, 0, src_h - 1)  # 防止越界，0相当于上边界padding
    w = torch.clamp(w, 0, src_w - 1)  # 防止越界，0相当于左边界padding

    h = h.view(dst_h, 1)  # 1维dst_h个，变2维dst_h*1个
    w = w.view(1, dst_w)  # 1维dst_w个，变2维1*dst_w个
    h = h.repeat(1, dst_w)  # H方向重复1次，W方向重复dst_w次
    w = w.repeat(dst_h, 1)  # H方向重复dsth次，W方向重复1次

    """求出四点坐标"""
    h0 = torch.clamp(torch.floor(h), 0, src_h - 2)  # -2相当于下边界padding
    w0 = torch.clamp(torch.floor(w), 0, src_w - 2)  # -2相当于右边界padding
    h0 = h0.long()  # torch坐标必须是long
    w0 = w0.long()  # torch坐标必须是long

    h1 = h0 + 1
    w1 = w0 + 1

    """求出四点值"""
    q00 = src[..., h0, w0]  # 双线性插值的q00
    q01 = src[..., h0, w1]  # 双线性插值的q01
    q10 = src[..., h1, w0]  # 双线性插值的q10
    q11 = src[..., h1, w1]  # 双线性插值的q11

    """公式计算"""
    r0 = (w1 - w) * q00 + (w - w0) * q01  # 双线性插值的r0
    r1 = (w1 - w) * q10 + (w - w0) * q11  # 双线性差值的r1
    dst = (h1 - h) * r0 + (h - h0) * r1  # 双线性差值的q

    return dst


if __name__ == '__main__':
    def unit_test4():
        src = torch.arange(1, 1 + 27).view((1, 3, 3, 3)).type(torch.float32)  # 原有张量
        dst_size = (4, 4)  # 目标张量大小
        align_corner = False  # 两种模式
        dst = bilinear_interpolate(src, dst_size=dst_size,
                                   align_corners=align_corner)  # 自己实现的双线性差值
        pt_dst = F.interpolate(src.float(), size=dst_size, mode='bilinear',
                               align_corners=align_corner)  # pytorch API实现的双线性差值
        if torch.equal(dst, pt_dst):
            print('ok')  # 如果自己实现的和官方实现一致说明正确

        image_file = os.path.join(os.getcwd(), 'test.jpg')
        image = mpimg.imread(image_file)  # 读取一个图片

        image_in = torch.from_numpy(image.transpose(2, 0, 1))  # HWC转换为CHW
        image_in = torch.unsqueeze(image_in, 0)  # 转换为NCHW
        image_out = bilinear_interpolate(image_in, (256, 256))  # 差值到256x256
        image_out = torch.squeeze(image_out, 0).numpy().astype(int)  # 转换为CHW
        image_out = image_out.transpose(1, 2, 0)  # 转换为HWC

        fig, axes = plt.subplots(1, 2, figsize=(8, 10))
        axes = axes.flatten()
        axes[0].imshow(image)  # 原图
        axes[1].imshow(image_out)  # 双线性差值
        axes[0].axis([0, image.shape[1], image.shape[0], 0])
        axes[1].axis([0, image_out.shape[1], image_out.shape[0], 0])
        fig.tight_layout()
        plt.show()


    unit_test4()
