import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from config import Config
from utils.process_label import id_to_trainid
from PIL import Image


class LaneSegDataset(data.Dataset):
    """
    根据输入的csv文件，读取image和label
    对label做id到trainid的转换
    并输出image和label，都是PIL Image
    """

    def __init__(self, data_list, image_transform=None, label_transform=None):
        """
        :param data_list: csv文件的绝对路径，csv文件有两列，第一列是image，第二列是label
        :param image_transform: optional，对image做转换
        :param label_transform: optional，对label做转换
        """
        super(LaneSegDataset, self).__init__()
        self._data_frame = pd.read_csv(data_list)  # 读取传入的csv文件形成data_frame
        self._image_transform = image_transform  # 对image的转换
        self._label_transform = label_transform  # 对label的转换
        pass

    def __len__(self):
        return len(self._data_frame)  # Dataset子类必须实现，返回数据集长度

    def __getitem__(self, index):
        return self._get_data(index)  # Dataset子类必须实现，通过key返回value，这里的key是索引

    def _get_data(self, index):
        image_path = self._data_frame['image'][index]  # 记录下要返回的image的路径
        label_path = self._data_frame['label'][index]  # 记录下要返回的label的路径

        image = Image.open(image_path)  # 读取image为PIL Image
        label = Image.open(label_path)  # 读取label为PIL Image
        label = np.asarray(label)  # label从PIL Image转换为ndarray
        label = id_to_trainid(label)  # label的Id转换为TrainId
        label = Image.fromarray(label)  # label从ndarray转换为PIL Image

        if self._image_transform is not None:
            image = self._image_transform(image)  # image更多的转换
        if self._label_transform is not None:
            label = self._label_transform(label)  # label更多的转换
        return image, label

    pass


if __name__ == '__main__':
    # 训练、验证、测试dataset
    image_datasets = {
        'train': LaneSegDataset(data_list=Config.DATA_LIST['train'],
                                image_transform=Config.IMAGE_TRANSFORMS['train'],
                                label_transform=Config.LABEL_TRANSFORMS['train']),
        'valid': LaneSegDataset(data_list=Config.DATA_LIST['valid'],
                                image_transform=Config.IMAGE_TRANSFORMS['valid'],
                                label_transform=Config.LABEL_TRANSFORMS['valid']),
        'test': LaneSegDataset(data_list=Config.DATA_LIST['test'],
                               image_transform=Config.IMAGE_TRANSFORMS['test'],
                               label_transform=Config.LABEL_TRANSFORMS['test']),
    }

    # 训练、验证、测试dataloader
    data_loaders = {
        'train': data.DataLoader(dataset=image_datasets['train'],
                                 batch_size=Config.TRAIN_BATCH_SIZE),
        'valid': data.DataLoader(dataset=image_datasets['valid'],
                                 batch_size=Config.TRAIN_BATCH_SIZE),
        'test': data.DataLoader(dataset=image_datasets['test'],
                                batch_size=Config.TEST_BATCH_SIZE),
    }

    # 逐个读取，查看读取的内容，验证dataloader可用
    for i, (im, lb) in enumerate(data_loaders['train']):
        s = input('>>>')
        if s == 'q':
            break
        print(i)
        print(type(im), im.shape)
        print(type(lb), lb.shape, np.bincount(lb.flatten()))