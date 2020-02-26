import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from datetime import datetime
from nets.fcn8s import FCN8s
from nets.unet import unet
from utils.dataset_reader import DatasetReader
from config import Config


def cross_entropy2d(output, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, c, h, w)但target的c=1
    n, c, h, w = output.size()
    target = target.squeeze(1).type(torch.int64)  # target(n, h, w)

    # log_p: (n, c, h, w)
    log_p = F.log_softmax(output, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


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


@epoch_timer  # 记录一个epoch时间并打印
def epoch_train(net, loss_func, optimizer, train_data, valid_data):
    """
    一个epoch训练过程，分成两个阶段：先训练，再验证
    :param net: 使用的模型
    :param loss_func: loss函数
    :param optimizer: 优化器
    :param train_data: 训练集
    :param valid_data: 验证集
    :return: 一个epoch的训练loss、训练acc、验证loss、验证acc
    """
    device = Config.DEVICE  # 使用GPU，没有GPU就用CPU
    net.to(device)  # 模型装入GPU
    # loss_func.to(device)  # loss函数装入CPU

    """训练"""
    train_loss, train_acc = 0., 0.  # 一个epoch训练的loss和正确率acc
    net.train()  # 训练
    for i, (train_image, train_label) in enumerate(train_data):
        train_image = Variable(train_image.to(device))  # 一个训练batch image
        train_label = Variable(train_label.to(device))  # 一个训练batch label

        train_output = net(train_image)  # 前向传播，计算一个训练batch的output
        loss = loss_func(train_output, train_label)  # 计算一个训练batch的loss
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 步进

        train_loss += loss.detach().cpu().numpy()  # 累加训练batch的loss
        train_acc += get_acc(train_output, train_label)  # 累加训练batch的acc
        pass
    train_loss /= len(train_data)  # 求取一个epoch训练的loss
    train_acc /= len(train_data)  # 求取一个epoch训练的acc

    """验证"""
    valid_loss, valid_acc = 0., 0.  # 一个epoch验证的loss和正确率acc
    net.eval()  # 验证
    for i, (valid_image, valid_label) in enumerate(valid_data):
        valid_image = Variable(valid_image.to(device))  # 一个验证batch image
        valid_label = Variable(valid_label.to(device))  # 一个验证batch label

        valid_output = net(valid_image)  # 前项传播，计算一个验证batch的output
        loss = loss_func(valid_output, valid_label)  # 计算一个验证batch的loss
        # 验证的时候不进行反向传播

        valid_loss += loss.detach().cpu().numpy()  # 累加验证batch的loss
        valid_acc += get_acc(valid_output, valid_label)  # 累加验证batch的acc
        pass
    valid_loss /= len(valid_data)  # 求取一个epoch验证的loss
    valid_acc /= len(valid_data)  # 求取一个epoch验证的acc
    return train_loss, train_acc, valid_loss, valid_acc


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


def train(net, loss_func, optimizer, train_data, valid_data, name, epochs=5):
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
        t_loss, t_acc, v_loss, v_acc = epoch_train(net, loss_func, optimizer,
                                                   train_data, valid_data)  # 一个epoch训练
        epoch_str = ('Epoch: {:d} | '
                     'Train Loss: {:.4f} | Train Acc: {:.4f} | '
                     'Valid Loss: {:.4f} | Valid Acc: {:.4f}')
        print(epoch_str.format(e + 1, t_loss, t_acc, v_loss, v_acc))  # 打印一个epoch的loss和acc
        save_checkpoint(net, name, e)  # 每个epoch的参数都保存
        pass
    pass


pass


def get_acc(output, label):
    """
    计算正确率
    :param output: 模型的输出
    :param label: label
    :return: 正确率
    """
    # total = output.shape[0]  # 总数
    # pred = torch.argmax(output, dim=1)  # 模型channel方向最大值的索引也就是估值
    # num_correct = (pred == label).sum().cpu().numpy()  # 估值与label一致的总数
    # return num_correct / total  # 正确率=估值正确总数/总数
    return 1


if __name__ == '__main__':
    """
    单元测试
    """
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

    model = FCN8s(n_class=8)
    # model = unet(3, 8, upmode='upsample', padding=31)
    print(model)
    custom_loss_func = cross_entropy2d  # loss函数
    custom_optimizer = torch.optim.Adam(model.parameters())  # 将模型参数装入优化器

    train(model, custom_loss_func, custom_optimizer,
          data_loaders['train'], data_loaders['valid'], name='fcn8s')  # 开始训（炼）练（丹）
