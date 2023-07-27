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
from federatedscope.core.data import DummyDataTranslator
from PIL import Image
import scipy.io as scio

"""
基于FSFL源码构造的EMNIST数据集,用以验证FSFL复现的正确性
"""


def pre_handle_femnist_mat(dataroot):
    """
    Preprocessing EMNIST_mat
    """
    mat = scio.loadmat(f'{dataroot}/emnist-letters.mat', verify_compressed_data_integrity=False)
    data = mat["dataset"]
    writer_ids_train = data['train'][0, 0]['writers'][0, 0]
    writer_ids_train = np.squeeze(writer_ids_train)
    X_train = data['train'][0, 0]['images'][0, 0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order="F")
    y_train = data['train'][0, 0]['labels'][0, 0]
    y_train = np.squeeze(y_train)
    y_train -= 1
    writer_ids_test = data['test'][0, 0]['writers'][0, 0]
    writer_ids_test = np.squeeze(writer_ids_test)
    X_test = data['test'][0, 0]['images'][0, 0]
    X_test = X_test.reshape((X_test.shape[0], 28, 28), order="F")
    y_test = data['test'][0, 0]['labels'][0, 0]
    y_test = np.squeeze(y_test)
    y_test -= 1
    return X_train, y_train, writer_ids_train, X_test, y_test, writer_ids_train, writer_ids_test


def generate_bal_private_data(X, y, N_parties=10, classes_in_use=range(11), N_samples_per_class=3, data_overlap=False,
                              data_root='./data'):
    """
    Generate private data
    """
    if False:
        priv_data = np.load('Temp/priv_data_72.npy')
        priv_data = priv_data.tolist()

        with open('Temp/total_priv_data_72.pickle', 'rb') as handle:
            total_priv_data = pickle.load(handle)
        # f = open('Src/Temp/total_priv_data.txt', 'r')
        # a = f.read()
        # total_priv_data = eval(a)
        # f.close()
    else:
        priv_data = [None] * N_parties
        combined_idx = np.array([], dtype=np.int16)
        for cls in classes_in_use:
            # Get the index of eligible data
            idx = np.where(y == cls)[0]
            # Randomly pick a certain number of indices
            idx = np.random.choice(idx, N_samples_per_class * N_parties,
                                   replace=data_overlap)
            # np.r_/np.c_: It is to connect two matrices by column/row, that is, add the two matrices up and down/left and right,
            # requiring the same number of columns/rows, similar to concat()/merge() in pandas.
            combined_idx = np.r_[combined_idx, idx]

            for i in range(N_parties):
                idx_tmp = idx[i * N_samples_per_class: (i + 1) * N_samples_per_class]
                if priv_data[i] is None:
                    tmp = {}
                    tmp["X"] = X[idx_tmp]
                    tmp["y"] = y[idx_tmp]
                    tmp["idx"] = idx_tmp
                    priv_data[i] = tmp
                else:
                    priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                    priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                    priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]

        priv_data_save = np.array(priv_data)
        np.save(f'{data_root}/priv_data_72.npy', priv_data_save)

        total_priv_data = {}
        total_priv_data["idx"] = combined_idx
        total_priv_data["X"] = X[combined_idx]
        total_priv_data["y"] = y[combined_idx]

        with open(f'{data_root}/total_priv_data_72.pickle', 'wb') as handle:
            pickle.dump(total_priv_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return priv_data, total_priv_data


def generate_partial_femnist(X, y, class_in_use=None, verbose=False):
    """
    Generate partial femnist as test set
    """
    if class_in_use is None:
        idx = np.ones_like(y, dtype=bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis=0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose == True:
        print("Selected X shape :", X_incomplete.shape)
        print("Selected y shape :", y_incomplete.shape)
    return X_incomplete, y_incomplete


def prepare_data_EMNIST(config, client_cfgs=None):
    data_root = config.data.root
    num_users = config.federate.client_num

    apply_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])

    X_train, y_train, writer_ids_train, X_test, y_test, writer_ids_train, writer_ids_test = pre_handle_femnist_mat(
        data_root)
    y_train += len(config.fsfl.public_classes)
    y_test += len(config.fsfl.public_classes)

    """
    Create test dataset
    """
    # TODO: 官方代码中设定私有数据集的变量名为femnist，但是需要注意这个femnist和LEAF的femnist有所不同（不过也有可能是相同的？）
    femnist_X_test, femnist_y_test = generate_partial_femnist(X=X_test, y=y_test,
                                                              class_in_use=config.fsfl.private_classes,
                                                              verbose=False)
    femnist_bal_data_test = FemnistValTest(femnist_X_test, femnist_y_test, apply_transform)
    """
    Create train dataset
    """
    private_bal_femnist_data, total_private_bal_femnist_data = \
        generate_bal_private_data(X=X_train, y=y_train, N_parties=num_users,
                                  classes_in_use=config.fsfl.private_classes,
                                  N_samples_per_class=config.fsfl.N_samples_per_class, data_overlap=False,
                                  data_root=data_root)  # TODO:待看


    # Convert to the data format of federatedscope
    data = {}
    # data_dict = {
    #     0: ClientData(self.global_cfg, train=train_dataset, val=None, test=test_dataset)
    # }
    idxs_users = np.arange(config.federate.client_num)
    for client_id in idxs_users:
        train = Mydata(private_bal_femnist_data[client_id], apply_transform)
        test = femnist_bal_data_test
        client_id = client_id + 1  # In federatedscope, the ID of the server is 0, and the ID of the client is 1 to client_num
        data[client_id] = {'train': train,
                           'val': None,
                           'test': test
                           }
    translator = DummyDataTranslator(config, client_cfgs)
    data = translator(data)

    # # generate train idx and test idx
    # K = config.model.num_classes
    # idx_batch = [[] for _ in range(num_users)]
    # y = np.array(caltech_trainset.labels)
    # N = y.shape[0]
    # df = np.zeros([num_users, K])
    # for k in range(K):
    #     idx_k = np.where(y == k)[0]
    #     idx_k = idx_k[0:30 * num_users]
    #     np.random.shuffle(idx_k)
    #     proportions = np.random.dirichlet(np.repeat(1.0, num_users))  # TODO: 注意，这里alph固定为1.0
    #     proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
    #     proportions = proportions / proportions.sum()
    #     proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
    #     idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    #     j = 0
    #     for idx_j in idx_batch:
    #         if k != 0:
    #             df[j, k] = int(len(idx_j))
    #         else:
    #             df[j, k] = int(len(idx_j))
    #         j += 1
    #
    # user_groups = {}
    # user_groups_test = {}
    # for i in range(num_users):
    #     np.random.shuffle(idx_batch[i])
    #     num_samples = len(idx_batch[i])
    #     train_len = int(num_samples / 2)
    #     user_groups[i] = idx_batch[i][:train_len]
    #     user_groups_test[i] = idx_batch[i][train_len:]
    #
    # # Convert to the data format of federatedscope
    # data = {}
    # # data_dict = {
    # #     0: ClientData(self.global_cfg, train=train_dataset, val=None, test=test_dataset)
    # # }
    # idxs_users = np.arange(config.federate.client_num)
    # for client_id in idxs_users:
    #     idx_train = user_groups[client_id]
    #     idx_test = user_groups_test[client_id]
    #     train = DatasetSplit(caltech_trainset, idx_train)
    #     test = DatasetSplit(caltech_testset, idx_test)
    #     client_id = client_id + 1  # In federatedscope, the ID of the server is 0, and the ID of the client is 1 to client_num
    #     data[client_id] = {'train': train,
    #                        'val': None,
    #                        'test': test
    #                        }
    # translator = DummyDataTranslator(config, client_cfgs)
    # data = translator(data)

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


class Mydata():
    """
    An abstract dataset class
    """

    def __init__(self, data_list, transform):
        data_X_list = data_list["X"]
        data_Y_list = data_list["y"]
        imgs = []
        for index in range(len(data_X_list)):
            imgs.append((data_X_list[index], data_Y_list[index]))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.imgs[index]
        if self.transform is not None:
            image = self.transform(image)
        image = list(image)
        image[0] = image[0].squeeze(0)
        image = tuple(image)
        label = label.astype(np.int32)
        #        <class 'tuple'> <class 'numpy.int32'>
        return torch.Tensor(image), torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)


class FemnistValTest():
    """
    Femnist test dataset class
    """

    def __init__(self, data_X_list, data_Y_list, transform):
        imgs = []
        for index in range(len(data_X_list)):
            imgs.append((data_X_list[index], data_Y_list[index]))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.imgs[index]
        if self.transform is not None:
            image = self.transform(image)
        image = list(image)
        image[0] = image[0].squeeze(0)
        image = tuple(image)
        label = label.astype(np.int32)
        return image, label

    def __len__(self):
        return len(self.imgs)


def call_file_data(config, client_cfgs):
    if config.data.type == "EMNIST_for_FSFL":
        # All the data (clients and servers) are loaded from one unified files
        data, modified_config = prepare_data_EMNIST(config, client_cfgs)
        return data, modified_config


register_data("EMNIST_FSFL", call_file_data)
