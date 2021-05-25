# -*- coding: utf-8 -*-
# !/usr/bin/env python
from torch.utils import data
import torch
import h5py
import numpy as np
import time


class DataSet(data.Dataset):
    def __init__(self, radar_list, load_file_count, start_index=0, test=False):
        """
        radar_list: 雷达数据的路径列表
        load_file_count: 一次加载的列表数据数量
        start_index=0: 起始索引
        test=False: False为处理训练集,True为处理测试集
        """
        self.radar_list = radar_list
        self.test = test
        self.radar_data = []  # 雷达数据
        self.target_cla = []  # 分类标签
        self.target_reg = []  # 回归标签

        # 处理整图
        for i in range(start_index, start_index + load_file_count):
            start_time = time.time()
            # 读取整图
            with h5py.File(self.radar_list[i]) as fhandle:
                radar_data = fhandle[u'data'][:].astype(np.float32)
                label = fhandle[u'label'][:].astype(np.float32)

            # 测试
            if self.test:
                # 整图的大小为270*350
                for m in range(0, 270 - 54 + 1):
                    for n in range(0, 350 - 54 + 1):
                        data1 = radar_data[:, m:m + 54, n:n + 54]  # 裁剪雷达数据 54*54
                        data2 = label[:, m:m + 54, n:n + 54]  # 裁剪雷达标签 54*54

                        self.radar_data.append(data1)  # 将数据添加进列表

                        # 处理标签,以每张图的中心点的4个值的均值作为样本的真实标签
                        # l1: 30分钟的标签
                        # l2: 60分钟的标签
                        x = data2[4::5]
                        z = []
                        l_1 = (x[0, 26, 26] + x[0, 26, 27] + x[0, 27, 26] + x[
                            0, 27, 27]) / 4.0 * 85.0 - 10.0
                        l_2 = (x[1, 26, 26] + x[1, 26, 27] + x[1, 27, 26] + x[
                            1, 27, 27]) / 4.0 * 85.0 - 10.0
                        if l_1 > 35:
                            l_1 = 1
                        else:
                            l_1 = 0
                        if l_2 > 35:
                            l_2 = 1
                        else:
                            l_2 = 0

                        z.append([l_1, l_2])
                        self.target_cla.append(z)  # 将分类标签添加进列表
                        self.target_reg.append(x[:, 3:51, 3:51])  # 将回归标签添加进列表
                print('load the file use %.6f time' % (time.time() - start_time))

        # 将列表转成数组
        self.radar_data = np.array(self.radar_data)
        self.target_cla = np.array(self.target_cla)
        self.target_reg = np.array(self.target_reg)

    def __getitem__(self, index):
        return torch.from_numpy(self.radar_data[index]).unsqueeze(0).float(), \
               torch.from_numpy(self.target_cla[index:index + 1]).squeeze(), \
               torch.tensor(self.target_reg[index:index + 1], dtype=torch.float32).squeeze()

    # 裁剪后小图的数量
    def __len__(self):
        return len(self.target_cla)


if __name__ == '__main__':
    # dataset list
    with open('./dataset_path/path_train_radar.txt', 'r') as fhandle:
        radar_dataset_list = fhandle.read().split('\n')
    radar_dataset_list.pop()

    # 从路径列表加载1条数据
    data_set = DataSet(radar_dataset_list, 1, test=True)

    # 输出裁剪后的小图的数量
    print('data_set num :%d' % len(data_set))

    train_loader = data.DataLoader(data_set, batch_size=256)  # 测试集

    for ii, (input, label_cla, label_reg) in enumerate(train_loader):
        print(input.size())
        print(label_cla.size())
        print(label_reg.size())
