import torch
from torch.utils.data import Dataset
import numpy as np
import os

class DomainDataset(Dataset):
    """
    参考：https://github.com/FangXiuwen/FSMAFL/blob/main/collaborate_train.py 中的 DomainDataset
    """

    def __init__(self, publicadataset, privatedataset, localindex, step1=True):
        imgs = []
        if step1:
            for index in range(len(publicadataset)):
                imgs.append((publicadataset[index][0], 10))
            for index in range(len(privatedataset)):
                imgs.append((privatedataset[index][0], localindex))
        else:
            for index in range(len(publicadataset)):
                imgs.append((publicadataset[index][0], localindex))
            for index in range(len(privatedataset)):
                imgs.append((privatedataset[index][0], 10))
        self.imgs = imgs

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its domain labels
        """
        image, domain_label = self.imgs[index]
        return image, domain_label

    def __len__(self):
        return len(self.imgs)

def divide_dataset_epoch(dataset, epochs, num_samples_per_epoch=5000):
    """
    这个函数将会返回一个名为epoch_group的字典，用来指定模型异构联邦学习的过程中每个client在不同epoch中使用公共数据集的哪些样本
    key是epoch编号: 0，2，..., collaborative_epoch-1
    values是一个list: 保存着对应的epoch要用到的公共数据集的样本编号。
    Note:让每个client确定每一轮用哪些公共数据集的样本直觉上看起来并非是一个高效的做法，但是为了加快实现速度，我暂时没有对这一部分做改进
    """
    dict_epoch, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(epochs):
        dict_epoch[i] = set(np.random.choice(all_idxs, num_samples_per_epoch, replace=False))
        all_idxs = list(set(all_idxs) - dict_epoch[i])
    return dict_epoch

from torch.utils.data import Dataset

class DigestDataset(Dataset):
    def __init__(self, original_dataset, new_labels):
        self.original_dataset = original_dataset
        self.new_labels = new_labels

    def __getitem__(self, index):
        # 获取原始数据集中的数据
        data, _ = self.original_dataset[index]
        # 获取新的标签
        new_label = self.new_labels[index]
        return data, new_label

    def __len__(self):
        return len(self.original_dataset)

