import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import logging
import pandas as pd
from datetime import datetime
import os
logger = logging.getLogger(__name__)


def get_public_dataset(dataset):
    if dataset == 'mnist':
        data_dir = './data'
    apply_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])  # TODO:学习Normalize的值应该如何设置
    data_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=apply_transform)
    data_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=apply_transform)
    return data_train, data_test


def train_CV(model, optimizer, criterion, train_loader, device, client_id, epoch):
    """
        本函数封装本地模型在public dataset或是private dataset上的训练过程(针对计算机视觉数据集)
        被用于FSFL
    """
    model.train()
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # TODO:FSFL的模型输出维度和标签对不上
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    logger.info(f'Local Model {client_id}\t Train Epoch: {epoch + 1} \t Loss: {train_loss}')
    return train_loss


@torch.no_grad()
def eval_CV(model, criterion, test_loader, device, client_id, epoch):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            gt_labels.append(labels.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    val_loss = val_loss / len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)  # TODO:查看federateddscope里的计算精度的代码
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
    logger.info(f'Epoch: {epoch} \tValidation Loss: {val_loss}, Accuracy: {acc}')
    return val_loss, acc


class EarlyStopMonitor(object):
    def __init__(self, max_round=10, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round

    def reset(self):
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None


def result_to_csv(result, init_cfg):
    # 获取当前时间
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M")
    out_dict = {
        'exp_time': [time_string],
        'seed': [init_cfg.seed],
        'method': [init_cfg.federate.method],
        'batch_size': [init_cfg.dataloader.batch_size],
        'datasets': [init_cfg.data.type],
        'splitter': [init_cfg.data.splitter],
        'client_num': [init_cfg.federate.client_num],
        'local_updates': [init_cfg.train.local_update_steps],
        'test_acc': [result['client_summarized_avg']['test_acc']]
    }
    # if out_dict['method'][0] == 'fedproto':
    #     out_dict['server_lr'] = init_cfg.model.fedgsl.server_lr
    #     out_dict['loc_gnn_outsize'] = init_cfg.model.fedgsl.loc_gnn_outsize
    #     out_dict['glob_gnn_outsize'] = init_cfg.model.fedgsl.glob_gnn_outsize
    #     out_dict['gsl_gnn_hids'] = init_cfg.model.fedgsl.gsl_gnn_hids
    #     out_dict['k_for_knn'] = init_cfg.model.fedgsl.k
    #     out_dict['pretrain_out_channels'] = init_cfg.model.fedgsl.pretrain_out_channels
    #     out_dict['pretrain_epoch'] = init_cfg.model.fedgsl.pretrain_epoch
    df = pd.DataFrame(out_dict, columns=out_dict.keys())
    folder_path = init_cfg.result_floder
    csv_path = f'{folder_path}/{init_cfg.exp_name}.csv'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 如果已存在csv，则在csv末尾添加本次的实验记录
    if not os.path.exists(csv_path) or not os.path.getsize(csv_path):
        df.to_csv(csv_path, mode='a', index=False, header=True)
        logger.info(f'本次试验记录已保存至:{csv_path}文件')
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)
        logger.info(f'添加本次实验记录至:{csv_path}文件')
    print(df)

    return df
