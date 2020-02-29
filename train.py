import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from datetime import datetime
from nets.fcn8s import FCN8s
from nets.unet import unet
from utils.laneseg_dataset import LaneSegDataset
from utils.loss_func import MySoftmaxCrossEntropyLoss, FocalLoss, DiceLoss, SoftIoULoss, MulticlassDiceLoss
from config import Config


def cross_entropy2d(output, target, weight=None, size_average=True):
    channels = output.shape[1]
    output = output.view(output.shape[0], output.shape[1], -1)
    output = output.permute((0, 2, 1))
    output = output.contiguous().view(-1, channels)
    target = target.type(torch.int64).view(-1)
    return nn.CrossEntropyLoss()(output, target)
    # # input: (n, c, h, w), target: (n, c, h, w)但target的c=1
    # n, c, h, w = output.size()
    # target = target.squeeze(1).type(torch.int64)  # target(n, h, w)
    #
    # # log_p: (n, c, h, w)
    # log_p = F.softmax(output, dim=1)
    # # log_p: (n*h*w, c)
    # log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    # log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    # log_p = log_p.view(-1, c)
    # # target: (n*h*w,)
    # mask = target >= 0
    # target = target[mask]
    # loss = F.cross_entropy(log_p, target, weight=weight, reduction='sum')
    # if size_average:
    #     loss /= mask.data.sum()
    # return loss


def epoch_timer(func):
    """
    装饰器。epoch计时器，记录一个epoch用时并打印
    :param func: 被装饰函数，是epoch_train
    :return:
    """

    def timer(*args, **kwargs):  # func的所有入参
        begin_time = datetime.now()  # 开始时间
        res = func(*args, **kwargs)  # 执行func，记录func返回值
        end_time = datetime.now()  # 结束时间
        mm, ss = divmod((end_time - begin_time).seconds, 60)  # 秒换算成分、秒
        hh, mm = divmod(mm, 60)  # 分钟换算成时、分
        print('Time: {:02d}:{:02d}:{:02d}'.format(hh, mm, ss))  # HH:mm:ss
        return res  # 返回func返回值

    return timer


def get_confusion_matrix(pred, label, n_class):
    """
    计算一张图像的混淆矩阵
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
    :param pred: 估值
    :param label: grand true
    :param n_class: n种分类
    :return: 混淆矩阵
    """
    pred, label = pred.type(torch.int), label.type(torch.int)  # 必须转化为int类型否则bincount报错
    mask = (label >= 0) & (label < n_class)  # 生成掩码，经过掩码以后tensor变为1维tensor
    cm = torch.bincount(n_class * label[mask] + pred[mask], minlength=n_class ** 2)  # 统计0-n*n的个数
    return cm.reshape((n_class, n_class))  # 返回reshape成矩阵的confusion matrix


def get_acc(cm, n_class):
    return


def get_miou(cm, n_class):
    """
    计算mIoU
    diag是对角线元素，也就是各个分类的TP
    np.sum(cm, axis=0)也就是各个分类的TP+FP
    np.sum(cm, axis=1)也就是各个分类的TP+FN
    iou是各个分类的IoU=TP/(TP+FP+FN)
    返回是平均IoU将各分类IoU相加再除以分类数
    :param cm:
    :param n_class:
    :return:
    """
    iou = torch.diag(cm) / (torch.sum(cm, axis=0) + torch.sum(cm, axis=1) - torch.diag(cm))
    return torch.sum(iou) / n_class


@epoch_timer  # 记录一个epoch时间并打印
def epoch_train(net, loss_func, optimizer, dataset, n_class):
    """
    一个epoch训练过程，分成两个阶段：先训练，再验证
    :param net: 使用的模型
    :param loss_func: loss函数
    :param optimizer: 优化器
    :param dataset: 训练集
    :param valid_data: 验证集
    :return: 一个epoch的训练loss、训练acc、验证loss、验证acc
    """
    net.train()  # 训练

    total_loss = 0.  # 一个epoch训练的loss
    confusion_matrix = torch.zeros((n_class, n_class)).to(Config.DEVICE)  # 一个epoch的混淆矩阵

    for i, (im, lb) in enumerate(dataset):
        im = im.to(Config.DEVICE)  # 一个训练batch image
        lb = lb.to(Config.DEVICE)  # 一个训练batch label

        optimizer.zero_grad()  # 清空梯度

        output = net(im)  # 前向传播，计算一个训练batch的output
        loss = loss_func(output, lb.squeeze(1).type(torch.int64))  # 计算一个训练batch的loss
        total_loss += loss.detach().item()  # 累加训练batch的loss

        loss.backward()  # 反向传播
        optimizer.step()  # 优化器迭代

        pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # 将输出转化为dense prediction，减少一个C维度
        lb = lb.squeeze(1)  # label也减少一个C维度
        confusion_matrix += get_confusion_matrix(pred, lb, n_class)  # 计算混淆矩阵并累加
        pass
    total_loss /= len(dataset)  # 求取一个epoch训练的loss
    mean_iou = get_miou(confusion_matrix, n_class)

    return total_loss, mean_iou


@epoch_timer
def epoch_valid(net, loss_func, dataset, n_class):
    net.eval()  # 验证

    total_loss = 0.  # 一个epoch验证的loss和正确率acc
    confusion_matrix = torch.zeros((n_class, n_class)).to(Config.DEVICE)

    for i, (im, lb) in enumerate(dataset):
        im = im.to(Config.DEVICE)  # 一个验证batch image
        lb = lb.to(Config.DEVICE)  # 一个验证batch label

        output = net(im)  # 前项传播，计算一个验证batch的output
        loss = loss_func(output, lb.squeeze(1).type(torch.int64))  # 计算一个验证batch的loss
        total_loss += loss.detach().item()  # 累加验证batch的loss

        # 验证的时候不进行反向传播
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # 将输出转化为dense prediction
        lb = lb.squeeze(1)  # label转换为numpy
        temp = get_confusion_matrix(pred, lb, n_class)
        # print(i, temp)
        confusion_matrix += temp  # 计算混淆矩阵并累加
        pass
    total_loss /= len(dataset)  # 求取一个epoch验证的loss
    mean_iou = get_miou(confusion_matrix, n_class)
    return total_loss, mean_iou


def save_checkpoint(net, name, epoch):
    """
    保存模型参数
    :param net: 模型
    :param save_dir: 保存的路径
    :param name: 参数文件名
    :param epoch: 训练到第几个epoch的参数
    :return:
    """
    save_dir = os.path.join(os.getcwd(), 'weight')
    save_dir = os.path.join(save_dir, name + '-' + str(epoch) + '.pkl')
    torch.save(net.state_dict(), save_dir)
    pass


def train(net, loss_func, optimizer, train_data, valid_data, n_class, name, epochs=5):
    """
    训练
    :param net: 被训练的模型
    :param loss_func: loss函数
    :param optimizer: 优化器
    :param train_data: 训练集
    :param valid_data: 验证集
    :param name: 保存参数文件名
    :param epochs: epoch数，默认是5
    :return:
    """
    for e in range(epochs):
        print('Epoch: {:d}'.format(e + 1))

        # 一个epoch训练
        t_loss, t_miou = epoch_train(net, loss_func, optimizer, train_data, n_class)
        train_str = 'Train Loss: {:.4f} | Train mIoU: {:.4f}'
        print(train_str.format(t_loss, t_miou))

        # 一个epoch验证
        v_loss, v_miou = epoch_valid(net, loss_func, valid_data, n_class)
        valid_str = 'Valid Loss: {:.4f} | Valid mIoU: {:.4f}'
        print(valid_str.format(v_loss, v_miou))

        # save_checkpoint(net, name, e)  # 每个epoch的参数都保存
        pass
    pass


pass

if __name__ == '__main__':
    """
    单元测试
    """
    image_datasets = {
        'train': LaneSegDataset(data_list=Config.DATALIST_TRAIN,
                                image_transform=Config.IMAGE_TRANSFORMS['train'],
                                label_transform=Config.LABEL_TRANSFORMS['train']),
        'valid': LaneSegDataset(data_list=Config.DATALIST_VALID,
                                image_transform=Config.IMAGE_TRANSFORMS['valid'],
                                label_transform=Config.LABEL_TRANSFORMS['valid']),
        'test': LaneSegDataset(data_list=Config.DATALIST_TEST,
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

    num_class = 8
    # custom_model = FCN8s(n_class=num_class)
    custom_model = unet(3, num_class, upmode='upsample', padding=1)
    print(custom_model)
    custom_model.to(Config.DEVICE)

    # custom_loss_func = cross_entropy2d  # loss函数
    # custom_loss_func = MySoftmaxCrossEntropyLoss(num_class)
    # custom_loss_func = SoftIoULoss(num_class)
    custom_loss_func = MulticlassDiceLoss()
    custom_loss_func.to(Config.DEVICE)

    custom_optimizer = torch.optim.Adam(params=custom_model.parameters(),
                                        weight_decay=Config.WEIGHT_DECAY)  # 将模型参数装入优化器

    train(custom_model, custom_loss_func, custom_optimizer,
          data_loaders['train'], data_loaders['valid'],
          n_class=num_class, name='fcn8s', epochs=2)  # 开始训（炼）练（丹）
