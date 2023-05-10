import os
import pickle

from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed
from torchvision import datasets, transforms
import femnist
import numpy as np



def load_data_from_file(config, client_cfgs=None):
    '''
    femnist
    参考自fedproto的k-way n-shot划分方式
    Args:
        config:
        client_cfgs:
    Returns:
    '''
    n_list = np.random.randint(max(2, client_cfgs.data.ways - client_cfgs.data.stdev),
                               min(client_cfgs.data.num_classes, client_cfgs.data.ways + client_cfgs.data.stdev + 1),
                               client_cfgs.federate.client_num)
    k_list = np.random.randint(client_cfgs.data.shots - client_cfgs.data.stdev + 1,
                               client_cfgs.data.shots + client_cfgs.data.stdev + 1, client_cfgs.federate.client_num)

    from federatedscope.core.data import DummyDataTranslator
    file_path = config.data.file_path

    if not os.path.exists(file_path):
        raise ValueError(f'The file {file_path} does not exist.')

    # with open(file_path, 'br') as file:
    #     data = pickle.load(file)
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = femnist.FEMNIST(client_cfgs, file_path, train=True, download=True,
                                    transform=apply_transform)
    test_dataset = femnist.FEMNIST(client_cfgs, file_path, train=False, download=True,
                                   transform=apply_transform)
    user_groups, classes_list, classes_list_gt = femnist_noniid(client_cfgs, client_cfgs.federate.client_num, n_list,
                                                                k_list)
    user_groups_lt = femnist_noniid_lt(args, args.num_users, classes_list)

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

    # translator = DummyDataTranslator(config, client_cfgs)
    # data = translator(data)

#     # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
#     data = convert_data_mode(data, config)
#
#     # Restore the user-specified seed after the data generation
#     setup_seed(config.seed)
#
#     return data, config
#
#
# def call_file_data(config, client_cfgs):
#     if config.data.type == "file":
#         # All the data (clients and servers) are loaded from one unified files
#         data, modified_config = load_data_from_file(config, client_cfgs)
#         return data, modified_config
#
#
# register_data("file", call_file_data)
