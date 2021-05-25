# -*- coding: utf-8 -*-
"""
@Author: ZhangRui
@Date: 2021/5/19
@Desc:

"""
import torch
import h5py
from torch.utils.data import Dataset, DataLoader


class VdrasDataset(Dataset):
    def __init__(self, path):
        self.temp = -1 # 记录加载进内存的文件
        self.num = 200 * 280 # 每张大图可以生成200*280个patch
        with open(path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()] # 路径列表，每一条数据是一张大图的绝对路径

    def __len__(self):
        return len(self.samples) * self.num # 大图的数量*每张大图可以产生的patch的数量=总的样本数量

    def __getitem__(self, item):
        file_no = item // self.num  # 确定在哪个文件
        self.index = item % self.num  # 确定文件中的第几条数据
        self.row = self.index // 280  # 确定path左上角所在行
        self.col = self.index - self.row * 280  # 确定patch左上角所在的列
        #print("item:{}, file_no:{}, row:{}, col:{}".format(item,file_no,self.row,self.col))
        if self.temp == file_no:
            # 说明文件已经在内存，直接取值返回
            data = self.train_data[:, :, self.row:self.row + 37, self.col:self.col + 37]
        else:
            # 说明文件不在内存, 需要加载
            self.temp = file_no
            with h5py.File(self.samples[file_no], 'r') as f:
                self.train_data = f['data'][()]
            # f.close()
            #print(self.train_data.shape)
            data = self.train_data[:, :, self.row:self.row + 37, self.col:self.col + 37]
        return torch.from_numpy(data).float() # 将numpy类型的数据转成tensor类型的数据，并指定为float类型


if __name__ == '__main__':
    path = "/media/data8T_1/zr/bj_data/test_200*280*120/test_path.txt"
    dataset = VdrasDataset(path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    i = 0
    for item in dataloader:
        assert item.shape == (1, 3, 120, 37, 37) #断言，判断输出是否符合预期
        i += 1
        if i >= 200 * 280:
            break
