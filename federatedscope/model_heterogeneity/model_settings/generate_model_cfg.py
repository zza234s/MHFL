import random
import numpy as np
import torch
from federatedscope.core.configs.config import global_cfg, CfgNode, CN
from collections import Counter
import logging
logger = logging.getLogger(__name__)

def generate_models_cfg(init_cfg, models_cfgs, ratios=[0.2, 0.2, 0.2, 0.1, 0.1], shuffle=False):
    """
    这个函数被用于在客户端数量较多的情况下（例如200个客户端），为每个客户端分配不同的模型,每种模型至少会出现一次
    Args:
        init_cfg: 主cfg的配置文件; (CfgNode对象）
        models_cfgs: 存放所有模型种类的cfg文件；（CfgNode对象）
        ratios: 每种模型的比例。例如:当client总数为10，模型共两种，ration为[0.4,0.6]时，4个client会是第一种模型，6个client会是第2种模型
        shuffle: 是否打乱所分配的模型编号。例子：为False时 [1,1,1,1,0,0,0,0,0,0]，为True则打乱这个列表
    Returns:
        分配好模型的client_cfg
    """
    assert len(models_cfgs) == len(ratios)  # 比例列表的长度要和模型类型的总数一致
    client_num = init_cfg.federate.client_num
    # Ensure client_num is at least the length of proportions
    if client_num < len(ratios):
        raise ValueError(f' client_num {client_num} is less than the number of the model categories {len(ratios)}.')
    # Start by giving each category one instance
    counts_per_model_type = [1 for _ in range(len(ratios))]
    remaining = client_num - len(ratios)

    # Distribute the remaining instances according to the proportions
    for i in range(len(ratios)):
        num_to_add = int(np.floor(ratios[i] * remaining))
        counts_per_model_type[i] += num_to_add

    # Handle any rounding errors
    while sum(counts_per_model_type) < client_num:
        for i in range(len(ratios)):
            if sum(counts_per_model_type) < client_num:
                counts_per_model_type[i] += 1
            else:
                break
    # 分配模型ID
    assignment = []
    for i in range(len(counts_per_model_type)):
        assignment.extend(
            [i + 1] * counts_per_model_type[i])  # 此时assignment的长度为client_num，每个元素是模型的ID,ID的范围 是1至len(model_list)）

    if shuffle:
        np.random.shuffle(assignment)

    # 为每个client生成cfg文件
    client_cfgs = CN()
    client_cfgs.clear_aux_info()
    type_list = []
    for idx in range(1, client_num + 1):
        model_id = assignment[idx - 1]
        temp_cfg = models_cfgs[f'type_{model_id}'].clone()
        client_cfgs[f'client_{idx}'] = temp_cfg
        type_list.append(temp_cfg.model.type)

    logger.info(f'每种模型的数量 \n{Counter(type_list)}')

    return client_cfgs
