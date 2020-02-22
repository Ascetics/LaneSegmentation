import torch


class Config(object):
    """
    配置类
    """
    # 数据集
    DATASETS_ROOT = '/root/private/torch_datasets'  # 数据集根目录

    # 训练结果
    WEIGHT_SAVE_PATH = '/root/private/LaneSegmentation/weight'  # weight保存路径

    # 超参数
    TRAIN_BATCH_SIZE = 10  # batch大小
    LEARN_RATE = 0.1  # 学习率
    EPOCHS = 10  # 训练次数

    # 设备
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')
    DEVICE = torch.device('cuda:5')
    pass
