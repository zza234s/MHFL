import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torchvision
from PIL import Image
#
# class MyMNIST(Dataset):
#     def __init__(self, root, transform=None):
#         self.transform = transform
#         self.data, self.targets = torch.load(root)
#
#     def __getitem__(self, index):
#         img, target = self.data[index], int(self.targets[index])
#         img = Image.fromarray(img.numpy(), mode='L')
#
#         if self.transform is not None:
#             img = self.transform(img)
#         img = transforms.ToTensor()(img)
#
#         sample = {'img': img, 'target': target}
#         return sample
#
#     def __len__(self):
#         return len(self.data)

def load_minist(pubaug, path):
    if pubaug == 'weak':
        selected_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    #TODO:strong直接复制了weak
    elif pubaug == 'strong':
        selected_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(path, train=True,
                                     download=True, transform=selected_transform)
    return train_dataset