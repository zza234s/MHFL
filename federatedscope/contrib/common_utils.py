import copy
import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import logging
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from PIL import Image

logger = logging.getLogger(__name__)


def plot_num_of_samples_per_classes(data, modified_cfg, scaling=10.0):
    client_num = modified_cfg.federate.client_num
    class_num = modified_cfg.model.out_channels
    client_list = [i for i in range(1, client_num + 1)]
    train_label_distribution = {i: {j: 0 for j in range(class_num)} for i in range(1, client_num + 1)}
    test_label_distribution = copy.deepcopy(train_label_distribution)
    fig, axs = fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    # 输出训练集标签分布
    for idx in range(1, client_num + 1):
        train_dataset = data[idx].train_data
        train_label_distribution_new = [j[1].item() if isinstance(j[1], torch.Tensor) else j[1] for j in train_dataset]
        train_label_distribution_new = dict(Counter(train_label_distribution_new))
        train_label_distribution[idx].update(train_label_distribution_new)

        size = [count / scaling for count in list(train_label_distribution[idx].values())]

        axs[0].scatter([idx] * class_num, list(train_label_distribution[idx].keys()),
                       s=size, color='red')
    axs[0].set_title(f'Train Dataset label distribution')

    # 输出验证集标签分布
    if modified_cfg.data.local_eval_whole_test_dataset:
        dataset = data[1].test_data
        test_label_distribution_new = [j[1] for j in dataset]
        print(Counter(test_label_distribution_new))
    else:
        for idx in range(1, client_num + 1):
            test_dataset = data[idx].test_data
            test_label_distribution_new = [j[1].item() if isinstance(j[1], torch.Tensor) else j[1] for j in
                                           test_dataset]
            test_label_distribution_new = dict(Counter(test_label_distribution_new))
            test_label_distribution[idx].update(test_label_distribution_new)
            axs[1].scatter([idx] * class_num, list(test_label_distribution[idx].keys()),
                           s=list(test_label_distribution[idx].values()), color='blue')

    axs[1].set_title(f'Test Dataset label distribution')
    plt.show()


def divide_dataset_epoch(dataset, epochs, num_samples_per_epoch=5000):
    """
    这个用来指定每个client在不同的epoch中使用公共数据集的哪些样本，从而确保在每一个通信轮次中，不同client上传的logits是基于同样的样本算出来的
    适用方法：FedMD，FSFL （在他们的源码中，没有模拟client-server的通信过程）

    key是epoch编号: 0，2，..., collaborative_epoch-1
    values是一个list: 保存着对应的epoch要用到的公共数据集的样本编号。
    Note:让每个client确定每一轮用哪些公共数据集的样本直觉上看起来并非是一个高效的做法，但是为了加快实现速度，我暂时没有对这一部分做改进
    """
    dict_epoch, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(epochs):
        dict_epoch[i] = set(np.random.choice(all_idxs, num_samples_per_epoch, replace=False))
        # all_idxs = list(set(all_idxs) - dict_epoch[i])
    return dict_epoch


def get_public_dataset(dataset):
    # TODO:核对每个数据集Normalize的值是否正确
    data_dir = './data'
    if dataset == 'mnist':
        apply_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        data_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=apply_transform)
        data_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=apply_transform)
    elif dataset == 'cifar100':
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        data_train = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
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
    logger.info(f'Train Epoch: {epoch} \t Train Loss: {train_loss}')
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
    logger.info(f'Eval Epoch: {epoch} \tTest Loss: {val_loss}, Test Acc: {acc}')
    return val_loss, acc



def test(model, test_loader, device):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    # print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
    #       .format(test_loss, acc))
    return acc, test_loss


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


class Ensemble(torch.nn.Module):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/synthesizers.py
    """

    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


def pack_images(images, col=None, channel_last=False, padding=1):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class UnlabeledImageDataset(torch.utils.data.Dataset):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """

    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(self.root)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
            self.root, len(self), self.transform)


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            if img.shape[0] == 1:
                img = Image.fromarray(img[0])
            else:
                img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '-%d.png' % (idx))


class ImagePool(object):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform)


class DeepInversionHook():
    '''
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):  # hook_fn(module, input, output) -> None
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()


def average_weights(w):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class KLDiv(nn.Module):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """

    def __init__(self, T=1.0, reduction='batchmean'):
        """

        :rtype: object
        """
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/heter_fl.py
    """
    if is_best:
        torch.save(state, filename)


def result_to_csv(result, init_cfg, best_round):
    # 获取当前时间
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M")
    out_dict = {
        'exp_time': [time_string],
        'seed': [init_cfg.seed],
        'method': [init_cfg.federate.method],
        'batch_size': [init_cfg.dataloader.batch_size],
        'optimizer': [init_cfg.train.optimizer.type],
        'lr': [init_cfg.train.optimizer.lr],
        'datasets': [init_cfg.data.type],
        'splitter': [init_cfg.data.splitter],
        'client_num': [init_cfg.federate.client_num],
        'local_updates': [init_cfg.train.local_update_steps],
        'test_acc': [result['client_summarized_avg']['test_acc']],
        'best_round': [best_round]
    }

    if len(init_cfg.data.splitter_args) != 0 and 'alpha' in init_cfg.data.splitter_args[0]:
        out_dict['alpha'] = init_cfg.data.splitter_args[0]['alpha']

    if out_dict['method'][0] == 'fedproto':
        out_dict['proto_weight'] = init_cfg.fedproto.proto_weight

    out_dict['local_eval_whole_test_dataset'] = [init_cfg.data.local_eval_whole_test_dataset]

    df = pd.DataFrame(out_dict, columns=out_dict.keys())
    folder_path = init_cfg.result_floder
    csv_path = f'{folder_path}/{init_cfg.exp_name}.csv'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 如果已存在csv，则在csv末尾添加本次的实验记录
    if not os.path.exists(csv_path) or not os.path.getsize(csv_path):
        df.to_csv(csv_path, mode='a', index=False, header=True)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)
    logger.info(f'The results of the experiment have been saved to: {csv_path} file')
    print(df)

    return df
