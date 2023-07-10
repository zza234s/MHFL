import os
import pickle
import torchvision.transforms as transforms
import numpy as np

from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed

from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Tuple
import os
class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None,subset_train_num=7,subset_capacity=10):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            #加载具有类别子文件夹结构的图像数据集
            self.imagefolder_obj = ImageFolder(self.root+'OfficeHome/'+self.data_name+'/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root+'OfficeHome/'+self.data_name+'/', self.transform, self.target_transform)

        all_data=self.imagefolder_obj.samples
        self.train_index_list=[]
        self.test_index_list=[]
        for i in range(len(all_data)):
            if i%subset_capacity<=subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        if self.train:
            return len(self.train_index_list)
        else:
            return len(self.test_index_list)

    def __getitem__(self, index):

        if self.train:
            used_index_list=self.train_index_list
        else:
            used_index_list=self.test_index_list

        path = self.samples[used_index_list[index]][0]
        target = self.samples[used_index_list[index]][1]
        target = int(target)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

#用于处理transform的一个函数
def get_normalization_transform():
    transform = transforms.Normalize((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))
    return transform
#自己定义的dataloader
def partition_office_domain_skew_loaders(train_datasets: list, test_datasets: list,
                                         config, percent_dict):

    ini_len_dict={}
    not_used_index_dict = {}
    data = dict()
    for i in range(len(train_datasets)):
        name = train_datasets[i].data_name
        if name not in not_used_index_dict:
            all_train_index = train_datasets[i].train_index_list
            not_used_index_dict[name] = np.arange(len(all_train_index))
            ini_len_dict[name] = len(all_train_index)

    for index in range(len(test_datasets)):
        test_dataset = test_datasets[index].imagefolder_obj
        test_loader = DataLoader(test_dataset, batch_size=config.dataloader.batch_size)
        data[index + 1] = dict()
        data[index+1]['test'] = test_loader

    for index in range(len(train_datasets)):
        name = train_datasets[index].data_name
        train_dataset = train_datasets[index].imagefolder_obj
        idxs = np.random.permutation(not_used_index_dict[name])
        percent = percent_dict[name]
        selected_idx = idxs[0:int(percent * ini_len_dict[name])]
        not_used_index_dict[name] = idxs[int(percent * ini_len_dict[name]):]
        train_sampler = SubsetRandomSampler(selected_idx)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.dataloader.batch_size, sampler=train_sampler)
        data[index+1]['train'] = train_loader

    return data

def load_data(config, client_cfgs=None):
    #一些准备工作
    path = config.data.root
    NAME = 'fl_officehome'
    SETTING = 'domain_skew'
    DOMAINS_LIST = ['Art', 'Clipart', 'Product', 'Real World']
    percent_dict = {'Art': 0.8, 'Clipart': 0.7, 'Product': 0.8, 'Real World': 0.7}
    N_SAMPLES_PER_Class = None
    N_CLASS = 65
    Nor_TRANSFORM = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])
    #设置domain
    selected_domain_list = []
    domains_len = len(DOMAINS_LIST)
    for i in range(config.federate.client_num):  # 根据客户端将domain放入selected_domain_list中 即一个客户端一个域
        index = i % domains_len
        selected_domain_list.append(DOMAINS_LIST[index])
    using_list = DOMAINS_LIST if selected_domain_list == [] else selected_domain_list
    #
    train_dataset_list = []
    test_dataset_list = []
    # 将输入的图像调整为大小为 (32, 32) 的尺寸 转换为张量格式 返回一个用于归一化图像的数据转换操作
    test_transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), get_normalization_transform()])
    # 根据domain加载train数据集数据
    for _, domain in enumerate(using_list):
        train_dataset = ImageFolder_Custom(data_name=domain, root=path, train=True,
                                           transform=Nor_TRANSFORM)
        train_dataset_list.append(train_dataset)
    # 根据domain加载test数据集数据
    for _, domain in enumerate(using_list):
        test_dataset = ImageFolder_Custom(data_name=domain, root=path, train=False,
                                          transform=test_transform)
        test_dataset_list.append(test_dataset)
    data = partition_office_domain_skew_loaders(train_dataset_list, test_dataset_list, config, percent_dict)
    return data, config


def call_data(config, client_cfgs):
    if config.data.type == "officehome":
        # All the data (clients and servers) are loaded from one unified files
        data, modified_config = load_data(config, client_cfgs)
        return data, modified_config


register_data("officehome", call_data)
