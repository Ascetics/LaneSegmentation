import torch
import torchvision.transforms as tsfs
import numpy as np
from PIL import Image


class Config(object):
    """
    配置类
    """
    # 设备   ####################################################################
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')
    DEVICE = torch.device('cuda:6')

    # 数据集 ####################################################################
    DATASETS_ROOT = '/root/private/torch_datasets'  # Pytorch数据集根目录

    IMAGE_BASE = '/root/data/LaneSeg/Image_Data'  # image文件的根目录
    LABEL_BASE = '/root/data/LaneSeg/Gray_Label'  # label文件的根目录

    TRAIN_RATE = 0.035  # 数据集划分，训练集占整个数据集的比例
    VALID_RATE = 0.010  # 数据集划分，验证集占整个数据集的比例

    DATALIST_TRAIN = '/root/private/LaneSegmentation/data_list/train.csv'  # 车道线分割训练集csv文件路径
    DATALIST_VALID = '/root/private/LaneSegmentation/data_list/valid.csv'  # 车道线分割验证集csv文件路径
    DATALIST_TEST = '/root/private/LaneSegmentation/data_list/test.csv'  # 车道线分割测试集csv文件路径

    # 训练结果###################################################################
    WEIGHT_SAVE_PATH = '/root/private/LaneSegmentation/weight'  # weight保存路径
    TEST_BATCH_SIZE = 1  # 测试集batch为1，这样可以一张图一张图的看

    # 超参数 ####################################################################
    TRAIN_BATCH_SIZE = 1  # batch大小
    LEARN_RATE = 0.003  # 学习率
    WEIGHT_DECAY = 0.0001
    EPOCHS = 20  # 训练次数

    # 数据处理 ####################################################################
    IMAGE_NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet统计的mean
    IMAGE_NORM_STD = [0.229, 0.224, 0.225]  # ImageNet统计的std
    LABEL_NORM_MEAN = [0.5]  # label的mean
    LABEL_NORM_STD = [0.5]  # label的std
    RESIZE = 448

    # 训练、验证、测试集的image的transforms，用法Config.IMAGE_TRANSFORMS['train']等
    # 经过转换后的image是NCHW的张量
    IMAGE_TRANSFORMS = {
        'train': tsfs.Compose([
            tsfs.Resize(size=RESIZE),  # 缩放
            tsfs.ToTensor(),  # 转换成Tensor，同时除以255归一化
        ]),
        'valid': tsfs.Compose([
            tsfs.Resize(size=RESIZE),  # 缩放
            tsfs.ToTensor(),  # 转换成Tensor，同时除以255归一化
        ]),
        'test': tsfs.Compose([
            tsfs.Resize(size=RESIZE),  # 测试一般不进行缩放
            tsfs.ToTensor(),  # 转换成Tensor，同时除以255归一化
        ]),
    }

    # 训练、验证、测试集的label的transforms，用法Config.LABEL_TRANSFORMS['train']等
    # 经过转换后的label是NHW的张量
    LABEL_TRANSFORMS = {
        'train': tsfs.Compose([
            tsfs.Resize(size=RESIZE, interpolation=Image.NEAREST),  # 缩放，为不产生错误label用NEAREST
            np.array,  # PIL转为ndarray
            torch.from_numpy  # 再转成tensor
        ]),
        'valid': tsfs.Compose([
            tsfs.Resize(size=RESIZE, interpolation=Image.NEAREST),  # 缩放，为不产生错误label用NEAREST
            np.array,  # PIL转为ndarray
            torch.from_numpy  # 再转成tensor
        ]),
        'test': tsfs.Compose([
            tsfs.Resize(size=RESIZE, interpolation=Image.NEAREST),  # 缩放，为不产生错误label用NEAREST
            np.array,  # PIL转为ndarray
            torch.from_numpy  # 再转成tensor
        ]),
    }

    pass
