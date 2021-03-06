import os
import pandas as pd
from sklearn.utils import shuffle
from config import Config


def _get_image_label_dir():
    """
    全部是自己手写的代码
    https://github.com/Ascetics/LaneSegmentation/blob/master/utils/make_list.py

    遍历服务器image和label目录，将image和label一一对应
    :return: 生成器（image绝对路径，label绝对路径）
    """
    data_err = 'data error. check!'
    image_base = os.path.join(Config.IMAGE_BASE)  # 服务器上Image根目录
    label_base = os.path.join(Config.LABEL_BASE)  # 服务器上Label根目录

    for road in os.listdir(image_base):  # 遍历根目录下所有目录
        image_road = os.path.join(image_base, road)  # image的Road02-Road04
        label_road = os.path.join(label_base, 'Label_' + str.lower(road))  # label的Label_road02-Label_road04
        if not (os.path.isdir(image_road) and
                os.path.exists(label_road) and
                os.path.isdir(label_road)):
            print(image_road, label_road, data_err)  # 路径不存在打印显示，跳过
            continue
        for record in os.listdir(image_road):  # 遍历road下所有目录
            image_record = os.path.join(image_road, record)  # image的 Record001-Record007
            label_record = os.path.join(label_road, 'Label/' + record)  # label的Record001-Record007，比image多了一层Label
            if not (os.path.isdir(image_record) and
                    os.path.exists(label_record) and
                    os.path.isdir(label_record)):
                print(image_record, label_record, data_err)  # 路径不存在打印显示，跳过
                continue
            for camera in os.listdir(image_record):  # 遍历record下所有目录
                image_camera = os.path.join(image_record, camera)  # image的Camera5-Camera6
                label_camera = os.path.join(label_record, camera)  # label的Camera5-Camera6
                if not (os.path.isdir(image_camera) and
                        os.path.exists(label_camera) and
                        os.path.isdir(label_camera)):
                    print(image_camera, label_camera, data_err)  # 路径不存在打印显示，跳过
                    continue
                for image in os.listdir(image_camera):  # 遍历Camera下所有图片
                    image_abspath = os.path.join(image_camera, image)  # image
                    label_abspath = os.path.join(label_camera,
                                                 image.replace('.jpg', '_bin.png'))  # label名字比image多_bin，格式png
                    if not (os.path.isfile(image_abspath) and
                            os.path.exists(label_abspath) and
                            os.path.isfile(label_abspath)):
                        print(image_abspath, label_abspath, data_err)  # 图片不存在或不对应打印显示，跳过
                        continue
                    yield image_abspath, label_abspath  # 生成器函数返回
    pass


def make_data_list(train_path, valid_path, test_path, train_rate=0.7, valid_rate=0.2):
    """
    打乱顺序，生成data_list的csv文件。
    :param train_path: 训练集保存路径
    :param valid_path: 验证集保存路径
    :param test_path: 测试集保存路径
    :param train_rate: 训练集占比，默认0.7
    :param valid_rate: 验证集占比，默认0.2
    :return:
    """
    g = _get_image_label_dir()  # 获取生成器
    abspaths = list(g)  # 将生成器转换为列表

    df = pd.DataFrame(
        data=abspaths,  # csv文件数据，每个元素是一条数据
        columns=['image', 'label']  # 两列 image、label
    )

    df_shuffle = shuffle(df)  # 随机打乱顺序

    # 70%做训练，20%做推断，剩余10%做测试
    train_size = int(df_shuffle.shape[0] * train_rate)
    valid_size = int(df_shuffle.shape[0] * valid_rate)

    print('total: {:d} | train: {:d} | val: {:d} | test: {:d}'.format(
        df_shuffle.shape[0], train_size, valid_size,
        df_shuffle.shape[0] - train_size - valid_size))

    df_train = df_shuffle[0: train_size]  # train数据集
    df_valid = df_shuffle[train_size: train_size + valid_size]  # valid数据集
    df_test = df_shuffle[train_size + valid_size:]  # test数据集

    df_train.to_csv(os.path.join(train_path), index=False)  # 保存train.csv文件
    df_valid.to_csv(os.path.join(valid_path), index=False)  # 保存valid.csv文件
    df_test.to_csv(os.path.join(test_path), index=False)  # 保存test.csv文件


if __name__ == '__main__':
    """
    单元测试
    """
    make_data_list(train_path=Config.DATA_LIST['train'],
                   valid_path=Config.DATA_LIST['valid'],
                   test_path=Config.DATA_LIST['test'],
                   train_rate=Config.TRAIN_RATE,
                   valid_rate=Config.VALID_RATE)  # 生成csv文件


    def test__get_image_label_dir():
        """
        单元测试，每次按键都应该输出一一对应的image名和label名。按q退出
        :return: 无
        """
        g = _get_image_label_dir()
        for image, label in g:
            s = input('>>>')
            if s == 'q':
                break
            print(image)
            print(label)
            pass

    # test__get_image_label_dir()
