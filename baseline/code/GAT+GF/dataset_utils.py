import os
import torch
import random
import numpy as np
from torch.utils.data import random_split #torch.utils.data 中的 random_split 用于将数据集随机划分为训练集和测试集。

#MSLE（Mean Squared Logarithmic Error）损失函数：计算输出和目标值的对数差的平方均值。这对于处理对数尺度的目标值非常有用。
def msle_loss(output, target):
    output = torch.log(output + 1)
    target = torch.log(target + 1)
    return torch.mean(torch.square(output - target))

#MAPE（Mean Absolute Percentage Error）损失函数：计算输出和目标值之间的绝对百分比误差的均值。
def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target))

#MAE（Mean Absolute Error）损失函数：计算输出和目标值之间的绝对误差的均值。
def mae_loss(output, target):
    return torch.mean(torch.abs(target - output))

'''
生成数据集函数：从给定的目录 dataset_dir 中加载指定名称列表 dataset_name_list 中的数据集文件，并将它们合并到一个列表中返回。
os.path.join 用于构建文件路径，os.path.isfile 检查文件是否存在，torch.load 加载数据集文件。
如果 print_info 为 True，会打印加载的文件路径。
'''
def generate_dataset(dataset_dir, dataset_name_list, print_info=False):
    dataset_list = list()
    for ds in dataset_name_list:
        ds_path = os.path.join(dataset_dir, ds)
        if os.path.isfile(ds_path):
            tem_data = torch.load(ds_path)
            dataset_list = dataset_list + tem_data
            if print_info:
                print(ds_path)
    return dataset_list

'''
划分数据集函数：将数据集 all_list 划分为训练集和测试集。
首先，打印划分前前 10 个样本的 y 值。
如果 shuffle 为 True 并且提供了 seed，则使用固定的随机种子 seed 来打乱数据集。如果没有提供种子，则使用默认的随机打乱方法。
再次打印划分后前 10 个样本的 y 值。
使用 torch.utils.data.random_split 按照 80% 和 20% 的比例将数据集划分为训练集和测试集，并使用固定的随机种子 42 来保证划分的可重复性。
'''
def split_dataset(all_list, shuffle=True, seed=6666):
    first_10_y = []
    for i in all_list[0:10]:
        first_10_y.append(i.y)
    print("first ten train graphs Y before shuffle:", first_10_y)

    if shuffle and seed is not None:
        np.random.RandomState(seed=seed).shuffle(all_list)
        print("seed number:", seed)
    elif shuffle and seed is None:
        random.shuffle(all_list)
        print("seed number:", seed)

    first_10_y = []
    for i in all_list[0:10]:
        first_10_y.append(i.y)
    print("first ten train graphs Y after shuffle:", first_10_y)

    train_ds, test_ds = random_split(all_list, [round(0.8 * len(all_list)), round(0.2 * len(all_list))],
                                     generator=torch.Generator().manual_seed(42))

    return train_ds, test_ds
