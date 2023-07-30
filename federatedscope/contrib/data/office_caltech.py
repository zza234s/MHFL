import os
import pickle
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.data.base_data import ClientData
from federatedscope.core.auxiliaries.utils import setup_seed
from torchvision import datasets, transforms
from federatedscope.core.data import DummyDataTranslator
from PIL import Image
"""
基于FedPCL的源码构造的office caltech数据集
用以验证FedPCL复现的正确性
The office caltech dataset constructed based on the source code of FedPCL.
It is used to verify the correctness of FedPCL recurrence.
source:https://github.com/yuetan031/FedPCL/blob/main/lib/utils.py
"""

def prepare_data_caltech_noniid(config, client_cfgs=None):
    # Prepare data
    transform_office = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])

    data_root = config.data.root
    num_users = config.federate.client_num
    assert config.data['splitter_args'][0]['alpha']
    alpha = config.data['splitter_args'][0]['alpha']
    # caltech
    caltech_trainset = OfficeDataset(data_root + 'office/', 'caltech', transform=transform_office, train=True)
    caltech_testset = OfficeDataset(data_root + 'office/', 'caltech', transform=transform_test, train=False)

    # generate train idx and test idx
    K = config.model.num_classes
    idx_batch = [[] for _ in range(num_users)]
    y = np.array(caltech_trainset.labels)
    N = y.shape[0]
    df = np.zeros([num_users, K])
    for k in range(K):
        idx_k = np.where(y == k)[0]
        idx_k = idx_k[0:30 * num_users]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        j = 0
        for idx_j in idx_batch:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1

    user_groups = {}
    user_groups_test = {}
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])
        num_samples = len(idx_batch[i])
        train_len = int(num_samples)
        # train_len = int(num_samples / 2)
        user_groups[i] = idx_batch[i][:train_len]
        user_groups_test[i] = idx_batch[i][train_len:]

    # Convert to the data format of federatedscope
    # data = {}
    data = {
        0: {'train':caltech_trainset, 'val':None, 'test':caltech_testset}
    }
    idxs_users = np.arange(config.federate.client_num)
    for client_id in idxs_users:
        idx_train = user_groups[client_id]
        idx_test = user_groups_test[client_id]
        train = DatasetSplit(caltech_trainset, idx_train)
        test = DatasetSplit(caltech_testset, idx_test)
        client_id = client_id + 1  # In federatedscope, the ID of the server is 0, and the ID of the client is 1 to client_num
        if config.data.local_eval_whole_test_dataset:
            data[client_id] = {'train': train,
                           'val': None,
                           'test': caltech_testset
                           }
        else:
            data[client_id] = {'train': train,
                               'val': None,
                               'test': test
                               }

    translator = DummyDataTranslator(config, client_cfgs)
    data = translator(data)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(base_path + '{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(base_path + '{}_test.pkl'.format(site), allow_pickle=True)

        for i in range(len(self.paths)):
            tmp = self.paths[i].split('/')[1:]
            self.paths[i] = '/'.join(tmp)

        label_dict = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4, 'laptop_computer': 5,
                      'monitor': 6, 'mouse': 7, 'mug': 8, 'projector': 9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label)


def call_file_data(config, client_cfgs):
    if config.data.type == "office_caltech":
        # All the data (clients and servers) are loaded from one unified files
        data, modified_config = prepare_data_caltech_noniid(config, client_cfgs)
        return data, modified_config

register_data("office_caltech_fedpcl", call_file_data)
