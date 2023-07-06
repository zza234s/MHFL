import os
import pickle
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed
from torchvision import datasets, transforms

"""
This dataset division is consistent with the official code of FedProto to verify 
whether the code we reproduced is correct
该数据划分和FedProto官方代码保持一致，用来验证我们所复现的代码是否是正确的

###########################################################################################################
对于FedProto源码中变量的个人理解如下
user_group: 一个字典，用来索引每个客户端的训练数据集。每个key，对应的value是完整数据集的训练集里的样本编号
classes_list和classes_list_gt：形式为[[1,2,3],[0,2,4,5],....],分别保存每个client的训练集包含哪些类的样本[1,2,3]
                               对应第一个client的训练样本只来自1，2，3这些类；classes_list和classes_list_gt是完全相同的
user_groups_lt：类似user_group，用来索引每个客户端的测试集（value是原始数据集的测试集的样本编号构成的数组）;
                注意，按照代码逻辑，如果某个client训练集里的样本只数据class：1,2,3，那么其测试集的样本也只来自这几个class
"""

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])


# TODO: 和原文统一seed
# TODO: dataloader需要和原文尽可能地统一
def load_data_from_file(config, client_cfgs=None):
    from federatedscope.core.data import DummyDataTranslator
    file_path = config.data.root
    client_num = config.federate.client_num

    if not os.path.exists(file_path):
        raise ValueError(f'The file {file_path} does not exist.')

    train_dataset = datasets.CIFAR10(file_path, train=True, download=True, transform=trans_cifar10_train)
    test_dataset = datasets.CIFAR10(file_path, train=False, download=True, transform=trans_cifar10_val)

    # TODO 在fedproto源代码 以及这里固定 n_list 和k_list后进行实验
    n_list = np.random.randint(max(2, config.fedproto.ways - config.fedproto.stdev),
                               min(10, config.fedproto.ways + config.fedproto.stdev + 1),
                               config.federate.client_num)
    k_list = np.random.randint(config.fedproto.shots - config.fedproto.stdev + 1,
                               config.fedproto.shots + config.fedproto.stdev - 1, config.federate.client_num)

    # sample training data amongst users
    if config.fedproto.iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, config.federate.client_num)
    else:
        # Sample Non-IID user data from Mnist
        if config.fedproto.unequal:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups, classes_list, classes_list_gt = cifar10_noniid(config, train_dataset,
                                                                        config.federate.client_num, n_list, k_list)
            user_groups_lt = cifar10_noniid_lt(config, test_dataset, config.federate.client_num, n_list, k_list,
                                               classes_list)

    data = {}
    # data_dict = {
    #     0: ClientData(self.global_cfg, train=train_dataset, val=None, test=test_dataset)
    # }
    idxs_users = np.arange(config.federate.client_num)
    for client_id in idxs_users:
        idx_train = user_groups[client_id]
        idx_test = user_groups_lt[client_id]
        train = DatasetSplit(train_dataset, idx_train)
        test = DatasetSplit(test_dataset, idx_test)
        client_id = client_id + 1
        data[client_id] = {'train': train,
                           'val': None,
                           'test': test
                           }

    # The shape of data is expected to be:
    # (1) the data consist of all participants' data:
    # {
    #   'client_id': {
    #       'train/val/test': {
    #           'x/y': np.ndarray
    #       }
    #   }
    # }
    # (2) isolated data
    # {
    #   'train/val/test': {
    #       'x/y': np.ndarray
    #   }
    # }
    translator = DummyDataTranslator(config, client_cfgs)
    data = translator(data)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


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


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar10_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 5000
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    classes_list = []
    classes_list_gt = []
    k_len = args.fedproto.train_shots_max
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        classes = random.sample(range(0, args.model.out_channels), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt


def cifar10_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 1000
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(test_dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    for i in range(num_users):
        k = args.fedproto.test_shots
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data

    return dict_users

def call_file_data(config, client_cfgs):
    if config.data.type == "CIFAR10_fedproto":
        # All the data (clients and servers) are loaded from one unified files
        data, modified_config = load_data_from_file(config, client_cfgs)
        return data, modified_config

register_data("CIFAR10_fedproto", call_file_data)