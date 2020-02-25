import torch.utils.data as data
import pandas as pd
from PIL import Image
from config import Config


class DatasetReader(data.IterableDataset):
    def __init__(self, data_list, image_transform=None, label_transform=None):
        super(DatasetReader, self).__init__()
        self._data_frame = pd.read_csv(data_list)
        self._index = 0
        self.transform = image_transform
        self.target_transform = label_transform
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self._data_frame):
            raise StopIteration

        image_path = self._data_frame['image'][self._index]
        label_path = self._data_frame['label'][self._index]
        self._index += 1
        image = Image.open(image_path)
        label = Image.open(label_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    pass


if __name__ == '__main__':
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

    data_loaders = {
        'train': data.DataLoader(dataset=image_datasets['train'],
                                 batch_size=Config.TRAIN_BATCH_SIZE),
        'valid': data.DataLoader(dataset=image_datasets['valid'],
                                 batch_size=Config.TRAIN_BATCH_SIZE),
        'test': data.DataLoader(dataset=image_datasets['test'],
                                batch_size=Config.TRAIN_BATCH_SIZE),
    }
    for im, lb in data_loaders['test']:
        s = input('>>>')
        if s == 'q':
            break
        print(type(im), im.shape)
        print(type(lb), lb.shape)
        pass
