import torch
import torch.utils.data as data
import torchvision.transforms as tsfs
import pandas as pd
import matplotlib.image as mpimg
from config import Config
from utils.process_label import id_to_trainid


class DatasetReader(data.IterableDataset):
    """
    根据输入的csv文件，读取image和label
    对label做id到trainid的转换
    并迭代输出
    """

    def __init__(self, data_list, image_transform=None, label_transform=None):
        """
        :param data_list: csv文件的绝对路径，csv文件有两列，第一列是image，第二列是label
        :param image_transform: optional，对image做转换
        :param label_transform: optional，对label做转换
        """
        super(DatasetReader, self).__init__()
        self._data_frame = pd.read_csv(data_list)  # 读取传入的csv文件形成data_frame
        self._index = 0  # 迭代器索引
        self.transform = image_transform  # 对image的转换
        self.target_transform = label_transform  # 对label的转换
        pass

    def __iter__(self):
        return self  # 实现__next__方法，直接返回self

    def __next__(self):
        """
        迭代器协议
        :return:
        """
        if self._index == len(self._data_frame):
            raise StopIteration  # 一次迭代结束

        image_path = self._data_frame['image'][self._index]  # 记录下要返回的image
        label_path = self._data_frame['label'][self._index]  # 记录下要返回的label
        self._index += 1  # 迭代器索引指向下一个

        image = mpimg.imread(image_path)  # 用matplotlib读取image
        label = mpimg.imread(label_path)  # 用matplotlib读取label，因为id_to_trainid输入是ndarray
        label = id_to_trainid(label)  # 将Id转换为TrainId

        t = tsfs.ToPILImage()  # 转换器
        image = t(image)  # 将ndarray转换为PIL.Image因为torchvision.transforms.Compose要求输入是PIL.Image
        label = t(label)  # 将ndarray转换为PIL.Image因为torchvision.transforms.Compose要求输入是PIL.Image

        if self.transform is not None:
            image = self.transform(image)  # image更多的转换
        if self.target_transform is not None:
            label = self.target_transform(label)  # label更多的转换
        return image, label

    pass


if __name__ == '__main__':
    # 训练、验证、测试dataset
    image_datasets = {
        'train': DatasetReader(data_list=Config.DATALIST_TRAIN,
                               image_transform=Config.IMAGE_TRANSFORMS['train'],
                               label_transform=Config.LABEL_TRANSFORMS['train']),
        'valid': DatasetReader(data_list=Config.DATALIST_VALID,
                               image_transform=Config.IMAGE_TRANSFORMS['valid'],
                               label_transform=Config.LABEL_TRANSFORMS['valid']),
        'test': DatasetReader(data_list=Config.DATALIST_TEST,
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
                                batch_size=Config.TRAIN_BATCH_SIZE),
    }

    # 逐个读取，查看读取的内容，验证dataloader可用
    for im, lb in data_loaders['test']:
        s = input('>>>')
        if s == 'q':
            break
        print(type(im), im.shape)
        print(type(lb), lb.shape)
        a = lb.type(torch.int).sum()
        if a > 0:
            print('error')
        else:
            print('ok')
        pass
