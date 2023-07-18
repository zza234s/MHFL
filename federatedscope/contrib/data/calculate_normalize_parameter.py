import torch
from torchvision import datasets
import os
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import CIFAR100
from PIL import Image
# from public_dataset import PublicDataset,random_loaders
from typing import Tuple
# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#     ])


def compute_mean_std(dataset):
    mean = np.zeros(3)
    std = np.zeros(3)
    num_samples = len(dataset)

    for i in range(num_samples):
        data, _ = dataset[i]
        mean += np.mean(data.numpy(), axis=(1, 2))
        std += np.std(data.numpy(), axis=(1, 2))

    mean /= num_samples
    std /= num_samples

    return mean, std

class MyCifar100(CIFAR100):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])  #先将图像转换成张量并将图像串联

        super(MyCifar100, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

def load_Cifar100(pubaug, path):
    if pubaug == 'weak':
        selected_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             ])
    elif pubaug == 'strong':
        selected_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.RandomApply([
                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
             ], p=0.8),
             transforms.RandomGrayscale(p=0.2),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821))])
    train_dataset = MyCifar100(path, train=True, transform=selected_transform, download=True)
    return  train_dataset

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    print(os.getcwd())
    # train_dataset = datasets.cifar.CIFAR100(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
    train_dataset = load_Cifar100('weak','../../data',)
    print(getStat(train_dataset))
    # data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, num_workers=16)
    # mean = 0.
    # std = 0.
    # for images, _ in data_loader:
    #     batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    #     std += images.std(2).sum(0)
    #
    # mean /= len(data_loader.dataset)
    # std /= (len(data_loader.dataset)-1)
    #
    # print(f'Mean: {mean}')
    # print(f'STD: {std}')
    # print(compute_mean_std(train_dataset))
    #
    # # train_dataset = ImageFolder(root=r'/data1/sharedata/leafseg/', transform=transforms.ToTensor())
    # # print(getStat(train_dataset))
