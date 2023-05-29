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
    Divide MNIST
    """
    dict_epoch, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(epochs):
        dict_epoch[i] = set(np.random.choice(all_idxs, num_samples_per_epoch, replace=False))
        all_idxs = list(set(all_idxs) - dict_epoch[i])
    return dict_epoch
