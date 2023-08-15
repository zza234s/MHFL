import os
import pickle
import numpy as np

from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed
import torch

def load_data_from_file(config, client_cfgs=None):
    from federatedscope.core.data import DummyDataTranslator

    file_path = config.data.file_path
    client_num = config.federate.client_num
    data = {}


    for client_id in range(1, client_num + 1):
        train_data = read_mnist(file_path, idx=client_id, is_train=True)

        test_data = read_mnist(file_path, idx=client_id, is_train=False)
        data[client_id]={
            'train': train_data,
            'val': None,
            'test': test_data
        }

    data[0]={
        'train': train_data,
        'val': None,
        'test': test_data
    }

    translator = DummyDataTranslator(config, client_cfgs)
    data = translator(data)

    # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
    data = convert_data_mode(data, config)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


def read_mnist(file_path, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(file_path, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'

        if not os.path.exists(train_file):
            raise ValueError(f'The file {train_file} does not exist.')

        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]

        return train_data

    else:
        test_data_dir = os.path.join(file_path, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'

        if not os.path.exists(test_file):
            raise ValueError(f'The file {test_file} does not exist.')

        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def call_file_data(config, client_cfgs):
    if config.data.type == "hfl_mnist":
        data, modified_config = load_data_from_file(config, client_cfgs)
        return data, modified_config


register_data("hfl_mnist", call_file_data)
