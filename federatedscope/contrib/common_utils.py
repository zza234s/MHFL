import torch
import torch.nn as nn
from torchvision import datasets,transforms
def get_public_dataset(dataset):
    if dataset =='mnist':
        data_dir ='./data'
    apply_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]) #TODO:学习Normalize的值应该如何设置
    data_train = datasets.MNIST(root=data_dir,train=True,download=True,transform=apply_transform)
    data_test = datasets.MNIST(root=data_dir,train=False,download=True,transform=apply_transform)
    return data_train,data_test