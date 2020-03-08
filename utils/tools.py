import os
import random
import time
import torch
from datetime import datetime
from config import Config


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
        time_str = 'Time: {:02d}:{:02d}:{:02d}|'.format(hh, mm, ss)  # HH:mm:ss
        log(time_str)  # 记录到日志里面
        return res  # 返回func返回值

    return timer


def log(msg, logfile=Config.LOG_FILE):
    """
    日志记录
    :param msg: 要记录的内容
    :param logfile: 日志文件，如果没有回创建
    :return:
    """
    log_f = open(logfile, mode='a')
    log_f.write(msg)  # 写到日志
    log_f.flush()  # 很有必要的马上写
    log_f.close()
    print(msg, end='', flush=True)  # 马上打印到终端
    pass


def save_weight(net, name, epoch, save_dir=Config.WEIGHT_SAVE_PATH):
    """
    保存模型参数，模型参数文件名格式 {模型名}-{保存日期时间}-epoch-{第几个epoch}.pkl
    :param net: 要保存参数的模型
    :param name: 模型的名字
    :param epoch: 训练到第几个epoch的参数
    :param save_dir: 保存模型参数的路径，不包含文件名
    :return:
    """
    filename = '{:s}-{:s}-epoch-{:02d}.pkl'.format(name, str(datetime.now()),
                                                   epoch)  # 模型参数文件{模型名}-{保存日期时间}-epoch-{第几个epoch}.pkl
    save_dir = os.path.join(save_dir, filename)  # 保存的文件绝对路径
    torch.save(net.state_dict(), save_dir)  # 保存模型参数
    return save_dir


if __name__ == '__main__':
    @epoch_timer
    def test_train():  # 模拟一个epoch训练
        time.sleep(random.randint(0, 5) / 5)
        return random.random(), random.random()


    @epoch_timer
    def test_valid():  # 模拟一个epoch验证
        time.sleep(random.randint(0, 5) / 5)
        return random.random(), random.random()


    epochs = 5
    for e in range(1, epochs + 1):
        epoch_str = '{:s}|Epoch: {:02d}|'.format(str(datetime.now()), e)
        log(epoch_str, './test.log')

        # 模拟一个epoch训练
        t_loss, t_miou = test_train()
        train_str = 'Train Loss: {:.4f}|Train mIoU: {:.4f}|'.format(t_loss, t_miou)
        log(train_str, './test.log')

        # 模拟一个epoch验证
        v_loss, v_miou = test_valid()
        valid_str = 'Valid Loss: {:.4f}|Valid mIoU: {:.4f}|'.format(v_loss, v_miou)
        log(valid_str, './test.log')
        log('\n', './test.log')
        pass
    pass
