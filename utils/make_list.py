import os
import pandas as pd
from sklearn.utils import shuffle


def _get_image_label_dir():
    """
    遍历服务器image和label目录，将image和label一一对应
    :return: 生成器（image绝对路径，label绝对路径，image文件名，label文件名）
    """
    data_err = 'data error. check!'
    image_base = os.path.join('/root/data/LaneSeg/Image_Data')
    label_base = os.path.join('/root/data/LaneSeg/Gray_Label')
    for road in os.listdir(image_base):
        image_road = os.path.join(image_base, road)
        label_road = os.path.join(label_base, 'Label_' + str.lower(road))
        if not (os.path.isdir(image_road) and
                os.path.exists(label_road) and
                os.path.isdir(label_road)):
            print(image_road, label_road, data_err)
            continue
        for record in os.listdir(image_road):
            image_record = os.path.join(image_road, record)
            label_record = os.path.join(label_road, 'Label/' + record)  # 服务器上label多了一层Label
            if not (os.path.isdir(image_record) and
                    os.path.exists(label_record) and
                    os.path.isdir(label_record)):
                print(image_record, label_record, data_err)
                continue
            for camera in os.listdir(image_record):
                image_camera = os.path.join(image_record, camera)
                label_camera = os.path.join(label_record, camera)
                if not (os.path.isdir(image_camera) and
                        os.path.exists(label_camera) and
                        os.path.isdir(label_camera)):
                    print(image_camera, label_camera, data_err)
                    continue
                for image in os.listdir(image_camera):
                    image_abspath = os.path.join(image_camera, image)
                    label_abspath = os.path.join(label_camera, image.replace('.jpg', '_bin.png'))
                    if not (os.path.isfile(image_abspath) and
                            os.path.exists(label_abspath) and
                            os.path.isfile(label_abspath)):
                        print(image_abspath, label_abspath, data_err)
                    yield image_abspath, label_abspath
    pass


def _make_data_list(save_path):
    """
    打乱顺序，生成data_list的csv文件。
    :param save_path: 保存的路径
    :return:
    """
    g = _get_image_label_dir()
    abspaths = list(g)

    df = pd.DataFrame(
        data=abspaths,
        columns=['image', 'lable']
    )

    df_shuffle = shuffle(df)
    df_shuffle.to_csv(save_path, index=False)  # 不保存行索引（行号）

    # 70%做训练，20%做推断，10%做测试
    train_size = int(df_shuffle.shape[0] * 0.7)
    valid_size = int(df_shuffle.shape[0] * 0.2)

    print('total: {:d} | train: {:d} | val: {:d} | test: {:d}'.format(df_shuffle.shape[0], train_size, valid_size,
                                                                      df_shuffle.shape[0] - train_size - valid_size))

    df_train = df_shuffle[0: train_size]
    df_valid = df_shuffle[train_size: train_size + valid_size]
    df_test = df_shuffle[train_size + valid_size:]

    df_train.to_csv(os.path.join(os.path.dirname(save_path), 'train.csv'), index=False)
    df_valid.to_csv(os.path.join(os.path.dirname(save_path), 'valid.csv'), index=False)
    df_test.to_csv(os.path.join(os.path.dirname(save_path), 'test.csv'), index=False)


if __name__ == '__main__':
    """
    单元测试
    """

    save_path = os.path.join(os.pardir, 'data_list')
    save_path = os.path.join(save_path, 'all.csv')
    _make_data_list(save_path)


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
