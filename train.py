import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datetime import datetime

from nets.fcn8s import FCN8s
from nets.unet import unet_resnet
from nets.deeplabv3p import DeepLabV3P
from utils.laneseg_dataset import LaneSegDataset
from utils.loss_func import MySoftmaxCrossEntropyLoss, FocalLoss, DiceLoss, SoftIoULoss, MulticlassDiceLoss
from utils.make_list import make_data_list
from utils.tools import log, epoch_timer, save_weight
from config import Config


class SemanticSegmentationTrainer(object):
    def __init__(self, net, loss_func, optimizer, train_data, valid_data,
                 n_class, device, name):
        """
        语义分割训练器
        :param net: 被训练的网络
        :param loss_func: 训练使用的loss函数
        :param optimizer: 训练使用的优化器
        :param train_data: 训练集
        :param valid_data: 验证集
        :param n_class: n种分裂
        :param device: 训练用GPU或CPU
        :param name: 保存模型用的名字
        """
        super(SemanticSegmentationTrainer, self).__init__()
        self.net = net
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_data = train_data
        self.valid_data = valid_data
        self.n_class = n_class
        self.device = device
        self.name = name
        pass

    def _get_confusion_matrix(self, pred, label):
        """
        计算图像的混淆矩阵
        cm = torch.bincount(n_class * label[mask] + pred[mask], minlength=n_class ** 2)
        n_class * label[mask] 将0,1,2,...,n映射到0,1,2,..,n*n
        n_class * label[mask] + pred[mask] 将分类映射到不同的grand true
        比如
        n=8
        label=0,pred=0,1,2,...,7映射后是0,1,2,...,7
        label=1,pred=0,1,2,...,7映射后是8,9,10,...,15
        ...
        label=7,pred=0,1,2,...,7映射后是56,57,58,...,63
        bincount 统计以后就会对应到0,1,2,...,63，再reshape就得到了混淆矩阵
        :param pred: 估值 维度NHW
        :param label: grand true 维度NHW
        :return: 混淆矩阵
        """
        pred, label = pred.type(torch.int), label.type(torch.int)  # 必须转化为int类型否则bincount报错
        mask = (label >= 0) & (label < self.n_class)  # 生成掩码，经过掩码以后tensor变为1维tensor
        cm = torch.bincount(self.n_class * label[mask] + pred[mask], minlength=self.n_class ** 2)  # 统计0-n*n的个数
        return cm.reshape((self.n_class, self.n_class))  # 返回reshape成矩阵的confusion matrix

    @staticmethod
    def _get_miou(cm):
        """
        计算mIoU
        diag是对角线元素，也就是各个分类的TP
        torch.sum(cm, axis=0)也就是各个分类的TP+FP
        torch.sum(cm, axis=1)也就是各个分类的TP+FN
        返回是mIoU将各分类IoU相加再除以分类数
        :param cm: 混淆矩阵。dim=0是grand truth，dim=1是pred。
        :return: 平均交并比
        """
        cm = cm.cpu().numpy()
        iou = np.diag(cm) / (np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm))
        return np.nanmean(iou)

    @epoch_timer
    def _epoch_train(self):
        """
        训练一个epoch
        :return:
        """
        self.net.train()  # 训练

        total_loss = 0.  # 一个epoch训练的loss
        confusion_matrix = torch.zeros((self.n_class, self.n_class)).to(self.device)  # 一个epoch的混淆矩阵

        for i, (im, lb) in enumerate(self.train_data):
            im = im.to(self.device)  # 一个训练batch image NCHW
            lb = lb.to(self.device)  # 一个训练batch label NHW

            self.optimizer.zero_grad()  # 清空梯度

            output = self.net(im)  # 前向传播，计算一个训练batch的output NCHW
            loss = self.loss_func(output, lb.type(torch.int64))  # 计算一个训练batch的loss
            total_loss += loss.detach().item()  # 累加训练batch的loss
            loss.backward()  # 反向传播
            self.optimizer.step()  # 优化器迭代

            pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # 将输出转化为dense prediction，减少一个C维度 NHW
            # lb = lb.squeeze(1)  # label也减少一个C维度 NHW
            confusion_matrix += self._get_confusion_matrix(pred, lb)  # 计算混淆矩阵并累加
            del im, lb, pred  # 节省内存
            pass
        total_loss /= len(self.train_data)  # 求取一个epoch训练的loss
        mean_iou = self._get_miou(confusion_matrix)
        return total_loss, mean_iou

    @epoch_timer
    def _epoch_valid(self):
        """
        验证一个epoch
        :return:
        """
        self.net.eval()  # 验证

        total_loss = 0.  # 一个epoch验证的loss和正确率acc
        confusion_matrix = torch.zeros((self.n_class, self.n_class)).to(self.device)

        for i, (im, lb) in enumerate(self.valid_data):
            im = im.to(self.device)  # 一个验证batch image
            lb = lb.to(self.device)  # 一个验证batch label

            output = self.net(im)  # 前项传播，计算一个验证batch的output
            loss = self.loss_func(output, lb.type(torch.int64))  # 计算一个验证batch的loss
            total_loss += loss.detach().item()  # 累加验证batch的loss

            # 验证的时候不进行反向传播
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # 将输出转化为dense prediction
            confusion_matrix += self._get_confusion_matrix(pred, lb)  # 计算混淆矩阵并累加
            del im, lb, pred  # 节省内存
            pass
        total_loss /= len(self.valid_data)  # 求取一个epoch验证的loss
        mean_iou = self._get_miou(confusion_matrix)
        return total_loss, mean_iou

    def train(self, epochs=Config.EPOCHS):
        """
        训练
        :param epochs: 训练多少个epoch
        :return:
        """
        for e in range(1, epochs + 1):
            epoch_str = '{:s}|Epoch: {:02d}|'.format(str(datetime.now()), e)
            log(epoch_str)
            log('\n')

            # 一个epoch训练
            t_loss, t_miou = self._epoch_train()
            train_str = 'Train Loss: {:.4f}|Train mIoU: {:.4f}|'.format(t_loss, t_miou)
            log(train_str)

            # 一个epoch验证
            v_loss, v_miou = self._epoch_valid()
            valid_str = 'Valid Loss: {:.4f}|Valid mIoU: {:.4f}|'.format(v_loss, v_miou)
            log(valid_str)
            log('\n')

            save_dir = save_weight(self.net, self.name, e)  # 每个epoch的参数都保存
            log(save_dir)  # 日志记录
            log('\n')
            pass
        pass

    pass


def get_data(data_type, batch_size=Config.TRAIN_BATCH_SIZE):
    """
    获取数据集
    :param data_type: 可以是'train', 'valid', 'test'数据集
    :param batch_size: 默认和配置文件里的BATCH_SIZE一致
    :return: 数据集DataLoader
    """
    assert data_type in ('train', 'valid', 'test')
    image_dataset = LaneSegDataset(Config.DATA_LIST[data_type],
                                   Config.IMAGE_TRANSFORMS[data_type],
                                   Config.LABEL_TRANSFORMS[data_type])
    data_loader = DataLoader(image_dataset, batch_size=batch_size)
    return data_loader


def get_model(model_name, in_channels, n_class, early_weight=None):
    if model_name == 'fcn8s':
        model = FCN8s(n_class)
    elif model_name == 'resnet152':
        model = unet_resnet('resnet152', in_channels, n_class, pretrained=True)
    elif model_name == 'deeplabv3p_resnet':
        model = DeepLabV3P('resnet', in_channels, n_class)
    else:
        raise ValueError('model name error!')

    if early_weight:
        if os.path.exists(early_weight):  # 有训练好的模型就加载
            print(early_weight, 'exists!')
            model.load_state_dict(torch.load(early_weight))
        else:  # 否则重新生成数据训练
            print('no early weight')
            s = input('reproduce for regenerate data list:')
            if s == 'reproduce':  # 输入reproduce重新生成data list
                make_data_list(train_path=Config.DATA_LIST['train'],
                               valid_path=Config.DATA_LIST['valid'],
                               test_path=Config.DATA_LIST['test'],
                               train_rate=Config.TRAIN_RATE,
                               valid_rate=Config.VALID_RATE)  # 生成csv文件
        pass

    return model


if __name__ == '__main__':
    name = 'deeplabv3p_resnet'
    num_class = 8
    load_file = '/root/private/LaneSegmentation/weight/deeplabv3p_resnet-2020-03-04 05:45:34.725327-epoch-40.pkl'
    custom_model = get_model(name, 3, num_class, load_file)
    custom_model.to(Config.DEVICE)

    # custom_loss_func = cross_entropy2d  # loss函数
    # custom_loss_func = MySoftmaxCrossEntropyLoss(num_class)
    # custom_loss_func = SoftIoULoss(num_class)
    # custom_loss_func = MulticlassDiceLoss()
    custom_loss_func = nn.CrossEntropyLoss()
    custom_loss_func.to(Config.DEVICE)

    custom_optimizer = torch.optim.Adam(params=custom_model.parameters(),
                                        lr=Config.LEARN_RATE)  # 将模型参数装入优化器

    trainer = SemanticSegmentationTrainer(custom_model,
                                          custom_loss_func,
                                          custom_optimizer,
                                          get_data('train'),
                                          get_data('valid'),
                                          n_class=num_class,
                                          device=Config.DEVICE,
                                          name=name)
    trainer.train(epochs=1)  # 开始训（炼）练（丹）
