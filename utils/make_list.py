import os
import pandas as pd
from sklearn.utils import shuffle

IMAGE_BASE = '/root/data/LaneSeg/Image_Data'
LABEL_BASE = '/root/data/LaneSeg/Gray_Label'


def _get_image_label_dir():
    """
    遍历服务器image和label目录，将image和label一一对应
    :return: 生成器（image绝对路径，label绝对路径，image文件名，label文件名）
    """
    image_base = os.path.abspath(os.path.join(IMAGE_BASE))  # image的base目录
    label_base = os.path.abspath(os.path.join(LABEL_BASE))  # label的base目录
    ld_image_base = os.listdir(image_base)  # 所有image的road目录
    ld_label_base = os.listdir(label_base)  # 所有label的road目录
    ld_image_base.sort()  # 排序
    ld_label_base.sort()  # 排序
    ld_base = zip(ld_image_base, ld_label_base)  # road一一对应
    for image_road, label_road in ld_base:  # 一一对应遍历所有road目录
        image_road = os.path.join(image_base, image_road)
        label_road = os.path.join(label_base, label_road)
        label_road = os.path.join(label_road, 'Label')  # 服务器上label多了一层Label目录
        ld_image_road = os.listdir(image_road)
        ld_label_road = os.listdir(label_road)
        ld_image_road.sort()
        ld_label_road.sort()
        ld_road = zip(ld_image_road, ld_label_road)
        for image_record, label_record in ld_road:  # 一一对应遍历所有record目录
            image_record = os.path.join(image_road, image_record)
            label_record = os.path.join(label_road, label_record)
            ld_image_record = os.listdir(image_record)
            ld_label_record = os.listdir(label_record)
            ld_image_record.sort()
            ld_label_record.sort()
            ld_record = zip(ld_image_record, ld_label_record)
            for image_camera, label_camera in ld_record:  # 一一对应遍历所有camera目录
                image_camera = os.path.join(image_record, image_camera)
                label_camera = os.path.join(label_record, label_camera)
                ld_image_camera = os.listdir(image_camera)
                ld_label_camera = os.listdir(label_camera)
                ld_image_camera.sort()
                ld_label_camera.sort()
                ld_camera = zip(ld_image_camera, ld_label_camera)
                for image, label in ld_camera:  # 一一对应遍历所有的image和label
                    image_name = image
                    label_name = label
                    image = os.path.join(image_camera, image)
                    label = os.path.join(label_camera, label)
                    yield image, label, image_name, label_name
    pass


def _make_data_list(save_path):
    """
    打乱顺序，生成data_list的csv文件。
    :param save_path: 保存的路径
    :return:
    """
    g = _get_image_label_dir()
    image_list = [image for _, _, image, _ in g]  # 生成器转换为image列表
    g = _get_image_label_dir()
    label_list = [label for _, _, _, label in g]  # 生成器转换为label列表
    df = pd.DataFrame({'image': image_list, 'label': label_list})
    df_shuffle = shuffle(df)
    df_shuffle.to_csv(save_path, index=False)  # 不保存行索引（行号）

    # 70%做训练，30%做测试
    train_size = int(df_shuffle.shape[0] * 0.8)
    df_train = df_shuffle[0:train_size]
    df_test = df_shuffle[train_size:]

    df_train.to_csv(os.path.join(os.path.dirname(save_path), 'train.csv'), index=False)
    df_test.to_csv(os.path.join(os.path.dirname(save_path), 'test.csv'), index=False)


if __name__ == '__main__':
    """
    单元测试
    """
    # 生成csv
    save_path = os.path.join(os.pardir, 'data_list')
    save_path = os.path.join(save_path, 'all.csv')
    _make_data_list(save_path)


    def test__make_data_list():
        """
        单元测试，将csv保存在data_list下的make_test.csv
        :return:
        """
        save_path = os.path.join(os.pardir, 'data_list')
        save_path = os.path.join(save_path, 'make_test.csv')
        _make_data_list(save_path)
        pass


    # test__make_data_list()

    def test__get_image_label_dir():
        """
        单元测试，每次按键都应该输出一一对应的image名和label名。按q退出
        :return: 无
        """
        g = _get_image_label_dir()
        for _, _, img, lbl in g:
            s = input('>>>')
            if s == 'q':
                break
            print(img)
            print(lbl)
            pass

    # test__gen_image_label_dir()
