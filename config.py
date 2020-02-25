import torch
import torchvision.transforms as tsfs


class Config(object):
    """
    配置类
    """
    # 数据集 ####################################################################
    DATASETS_ROOT = '/root/private/torch_datasets'  # Pytorch数据集根目录

    IMAGE_BASE = '/root/data/LaneSeg/Image_Data'  # image文件的根目录
    LABEL_BASE = '/root/data/LaneSeg/Gray_Label'  # label文件的根目录

    TRAIN_RATE = 0.0001  # 数据集划分，训练集占整个数据集的比例
    VALID_RATE = 0.0001  # 数据集划分，验证集占整个数据集的比例

    DATALIST_TRAIN = '/root/private/LaneSegmentation/data_list/train.csv'  # 车道线分割训练集csv文件路径
    DATALIST_VALID = '/root/private/LaneSegmentation/data_list/valid.csv'  # 车道线分割验证集csv文件路径
    DATALIST_TEST = '/root/private/LaneSegmentation/data_list/test.csv'  # 车道线分割测试集csv文件路径

    # 训练结果###################################################################
    WEIGHT_SAVE_PATH = '/root/private/LaneSegmentation/weight'  # weight保存路径

    # 超参数 ####################################################################
    TRAIN_BATCH_SIZE = 1  # batch大小
    LEARN_RATE = 0.1  # 学习率
    EPOCHS = 10  # 训练次数

    # 数据处理 ####################################################################
    IMAGE_NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet统计的mean
    IMAGE_NORM_STD = [0.229, 0.224, 0.225]  # ImageNet统计的std
    LABEL_NORM_MEAN = [0.5]  # label的mean
    LABEL_NORM_STD = [0.5]  # label的std

    # 训练、验证、测试集的image的transforms，用法Config.IMAGE_TRANSFORMS['train']等
    IMAGE_TRANSFORMS = {
        'train': tsfs.Compose([
            tsfs.Resize(size=224),  # 缩放
            tsfs.ToTensor(),  # 转换成Tensor
            tsfs.Normalize(IMAGE_NORM_MEAN, IMAGE_NORM_STD),  # 归一化
        ]),
        'valid': tsfs.Compose([
            tsfs.Resize(size=224),  # 缩放
            tsfs.ToTensor(),  # 转换成Tensor
            tsfs.Normalize(IMAGE_NORM_MEAN, IMAGE_NORM_STD),  # 归一化
        ]),
        'test': tsfs.Compose([
            tsfs.Resize(size=224),  # 测试一般不进行缩放
            tsfs.ToTensor(),  # 转换成Tensor
            tsfs.Normalize(IMAGE_NORM_MEAN, IMAGE_NORM_STD),  # 归一化
        ]),
    }

    # 训练、验证、测试集的label的transforms，用法Config.LABEL_TRANSFORMS['train']等
    LABEL_TRANSFORMS = {
        'train': tsfs.Compose([
            tsfs.Resize(size=224),  # 缩放
            tsfs.ToTensor(),  # 转换成Tensor
            # trainId不进行归一化
        ]),
        'valid': tsfs.Compose([
            tsfs.Resize(size=224),  # 缩放
            tsfs.ToTensor(),  # 转换成Tensor
            # trainId不进行归一化
        ]),
        'test': tsfs.Compose([
            tsfs.Resize(size=224),  # 测试一般不进行缩放
            tsfs.ToTensor(),  # 转换成Tensor
            # trainId不进行归一化
        ]),
    }

    # 设备   ####################################################################
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')
    DEVICE = torch.device('cuda:5')
    pass
