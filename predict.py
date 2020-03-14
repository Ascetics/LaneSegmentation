import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import time

from PIL import Image
from train import get_model
from utils.laneseg_dataset import LaneSegDataset
from utils.process_label import gray_to_rgb
from utils.augment import PairCrop, PairResize, PairNormalizeToTensor
from config import Config

early_weight = ('/root/private/LaneSegmentation/weight/'
                'deeplabv3p_xception-2020-03-13 17:12:17.867786-epoch-14.pth')  # 读取训练好的参数
print('loading model...')
model = get_model('deeplabv3p_xception', in_channels=3, n_class=8,
                  early_weight=early_weight)  # 加载读取的参数
model.eval()  # 准备测试
device = torch.device('cuda:3')  # 选择一个可用的GPU
model.to(device)  # 模型装入GPU

test_set = LaneSegDataset(Config.DATA_LIST['test'])  # 不剪裁，不缩放的测试集，读取PIL Image


def _get_miou(pred, label, n_class=8):
    """
    计算IoU
    :param pred: [H,W]ndarray
    :param label: [H,W]ndarray
    :return: float, Mean Intersection Over Union of pred and label
    """
    pred = np.asarray(pred)
    label = np.asarray(label)
    assert pred.shape == label.shape

    mask = (label >= 0) & (label < n_class)
    cm = np.bincount(label[mask] * n_class + pred[mask], minlength=n_class ** 2)  # 计算混淆矩阵
    cm = cm.reshape((n_class, n_class))  # 计算混淆矩阵
    iou = np.diag(cm) / (np.sum(cm, axis=0) + np.sum(axis=1) - np.diag(cm))
    return np.nanmean(iou)


def predict(image, label, resize_to=256, name=None):
    _, h0 = image.size[:2]  # 记录下原来大小,PIL Image大小W,H
    pair_resize = PairResize(size=resize_to)
    image, label = pair_resize(image, label)  # 缩放到指定大小
    w, h = image.size[:2]  # 记录下缩放后大小，PIL Image大小W,H
    offset = int(690 * h / h0)  # 缩放前剪裁690对应缩放后剪裁的大小
    pair_crop = PairCrop(offsets=(offset, None))  # 剪裁
    pair_norm_to_tensor = PairNormalizeToTensor(norm=True)  # 归一化并正则化

    fig, ax = plt.subplots(2, 2, figsize=(20, 15))  # 画布
    ax = ax.flatten()

    ax[0].imshow(image)  # 左上角显示原图
    ax[0].set_title('Input Image', fontsize=16)  # 标题

    ax[1].imshow(gray_to_rgb(np.asarray(label)))  # 右上角显示 Grand Truth
    ax[1].set_title('Grand Truth', fontsize=16)  # 标题

    im_tensor, _ = pair_crop(image, label)  # 剪裁最上边没有特征的部分,PIL Image
    im_tensor, _ = pair_norm_to_tensor(im_tensor, label)  # PIL Image转换为[C,H,W]tensor
    im_tensor = im_tensor.to(device)  # 装入GPU
    im_tensor = im_tensor.unsqueeze(0)  # 转换为[N,C,H,W]tensor
    output = model(im_tensor)  # 经过模型输出[N,C,H,W]tensor
    output = output.squeeze(0)  # [C,H,W]tensor
    pred = output.cpu().numpy()  # [C,H,W]ndarray
    pred = np.argmax(pred, axis=0)  # [H,W]ndarray

    supplement = np.zeros((offset, w), dtype=np.long)  # [H,W]ndarray,补充成背景
    pred = np.append(supplement, pred, axis=0)  # 最终的估值，[H,W]ndarray,在H方向cat，给pred补充被剪裁的背景

    mIoU = _get_miou(pred, label)  # 计算mIoU
    fig.suptitle('mIoU:{:.4f}'.format(mIoU), fontsize=16)  # 用mIoU作为大标题

    mask = (pred != 0).astype(np.long) * 255  # H,W]ndarray,alpha融合的mask

    pred = gray_to_rgb(pred)  # [H,W,C=3]ndarray RGB
    ax[3].imshow(pred)  # 右下角显示Pred
    ax[3].imshow('Pred', fontsize=16)  # 标题

    mask = mask[..., np.newaxis]  # [H,W,C=1]ndarray
    pred = np.append(pred, mask, axis=2)  # [H,W,C=4]ndarray，RGB+alpha变为RGBA
    im_comp = Image.alpha_composite(image.convert('RGBA'),
                                    Image.fromarray(pred).convert('RGBA'))  # alpha融合
    # im_comp = Image.blend(image.convert('RGBA'),
    #                       Image.fromarray(pred).convert('RGBA'),
    #                       alpha=0.5)  # 将Input Image和Pred RGB融合
    ax[2].imshow(im_comp)  # 左下角显示融合图像
    ax[2].set_title('Pred over Input', fontsize=16)  # 标题

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)

    if name is None:
        name = time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime())
    plt.savefig('/root/private/imfolder/pred-{:s}.jpg'.format(name))  # 保存图像
    plt.close(fig)
    pass


while True:
    s = input('>>>')
    if s == 'q':  # 按q退出
        break
    else:
        try:
            i = int(s)
            if i < 0 or i > len(test_set):
                continue
        except ValueError:
            print('input error, please input number or \'q\' for quit.')
            continue
        pass
    im, lb = test_set[i]  # PIL Image
    predict(im, lb)
    pass
