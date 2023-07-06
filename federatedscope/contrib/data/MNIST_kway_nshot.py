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
该数据划分和FedProto原文保持一致，用来验证我们所复现的代码是否是正确的

###########################################################################################################
对于FedProto变量的个人理解
user_group: 一个字典，用来索引每个客户端的训练数据集。每个key，对应的value是完整数据集的训练集里的样本编号
classes_list和classes_list_gt：形式为[[1,2,3],[0,2,4,5],....],分别保存每个client的训练集包含哪些类的样本[1,2,3]
                               对应第一个client的训练样本只来自1，2，3这些类；classes_list和classes_list_gt是完全相同的
user_groups_lt：类似user_group，用来索引每个客户端的测试集（value是原始数据集的测试集的样本编号构成的数组）;
                注意，按照代码逻辑，如果某个client训练集里的样本只数据class：1,2,3，那么其测试集的样本也只来自这几个class

"""


def load_data_from_file(config, client_cfgs=None):
    from federatedscope.core.data import DummyDataTranslator
    file_path = config.data.root
    if not os.path.exists(file_path):
        raise ValueError(f'The file {file_path} does not exist.')

    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(file_path, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(file_path, train=False, download=True,
                                  transform=apply_transform)

    # TODO 在fedproto源代码 以及这里固定 n_list 和k_list后进行实验
    n_list = np.random.randint(max(2, config.fedproto.ways - config.fedproto.stdev),
                               min(10, config.fedproto.ways + config.fedproto.stdev + 1),
                               config.federate.client_num)
    k_list = np.random.randint(config.fedproto.shots - config.fedproto.stdev + 1,
                               config.fedproto.shots + config.fedproto.stdev - 1, config.federate.client_num)

    '''
    
    '''
    # sample training data amongst users
    if config.fedproto.iid:
        # Sample IID user data from Mnist
        user_groups = mnist_iid(train_dataset, config.federate.client_num)
    else:
        # Sample Non-IID user data from Mnist
        if config.fedproto.unequal:
            # Chose uneuqal splits for every user
            user_groups = mnist_noniid_unequal(config, train_dataset, config.federate.client_num)
        else:
            # Chose euqal splits for every user

            user_groups, classes_list = mnist_noniid(config, train_dataset, config.federate.client_num, n_list, k_list)
            user_groups_lt = mnist_noniid_lt(config, test_dataset, config.federate.client_num, n_list, k_list,
                                             classes_list)
            classes_list_gt = classes_list

    # TODO:
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
    # TODO: dataloader需要和原文尽可能地统一

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

    # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
    # data = convert_data_mode(data, config)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


def call_file_data(config, client_cfgs):
    if config.data.type == "MNIST_fedproto":
        # All the data (clients and servers) are loaded from one unified files
        data, modified_config = load_data_from_file(config, client_cfgs)
        return data, modified_config


register_data("MNIST_fedproto", call_file_data)


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


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
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


def mnist_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    dict_users: key为client_id（编号从0开始），value为dataset的样本编号组成的列表
    classes_list: 存储每个client的训练集里的样本包含哪些class，例如[[0,1,2,3],[0,1,4,9],...],每个子list对应一个client
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 6000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()
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
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.fedproto.train_shots_max
        classes = random.sample(range(0, 10), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # begin = i*10 + label_begin[each_class.item()]
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def mnist_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    dict_users: key为client_id,value为一组测试集的样本索引
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = test_dataset.train_labels.numpy()
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
        k = 40  # 每个类选多少张做测试 #TODO 原文中这里指定了40
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * 40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data

    return dict_users


def train_val_test(self, dataset, idxs):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    idxs_train = idxs[:int(1 * len(idxs))]
    trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                             batch_size=self.args.local_bs, shuffle=True, drop_last=True)

    return trainloader
