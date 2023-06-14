from torch import nn
import torch.nn.functional as F
from federatedscope.register import register_model
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
'''
VGG11:"Very Deep Convolutional Networks for Large-Scale Image Recognition"
按照FedProto的思路，额外输出倒数第二层的representation
'''
class VGG11_proto(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=128,
                 class_num=10,
                 dropout=.0):
        super(VGG11_proto, self).__init__()

        self.conv1 = Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = BatchNorm2d(64)

        self.conv2 = Conv2d(64, 128, 3, padding=1)
        self.bn2 = BatchNorm2d(128)

        self.conv3 = Conv2d(128, 256, 3, padding=1)
        self.bn3 = BatchNorm2d(256)

        self.conv4 = Conv2d(256, 256, 3, padding=1)
        self.bn4 = BatchNorm2d(256)

        self.conv5 = Conv2d(256, 512, 3, padding=1)
        self.bn5 = BatchNorm2d(512)

        self.conv6 = Conv2d(512, 512, 3, padding=1)
        self.bn6 = BatchNorm2d(512)

        self.conv7 = Conv2d(512, 512, 3, padding=1)
        self.bn7 = BatchNorm2d(512)

        self.conv8 = Conv2d(512, 512, 3, padding=1)
        self.bn8 = BatchNorm2d(512)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 512,
            hidden)
        self.fc2 = Linear(hidden, hidden)
        self.fc3 = Linear(hidden, class_num)

        self.dropout = dropout

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn7(self.conv7(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn8(self.conv8(x)))
        x = self.maxpool(x)

        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.relu(self.fc2(x))
        x = F.dropout(x1, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x,x1

def call_vgg11_proto(model_config, local_data):
    if 'VGG11_proto' in model_config.type:
        input_shape=local_data
        model = VGG11_proto(in_channels=input_shape[-3],
                         h=input_shape[-2],
                         w=input_shape[-1],
                         hidden=model_config.hidden,
                         class_num=model_config.out_channels,
                         dropout=model_config.dropout)
        return model

register_model('vgg11_proto', call_vgg11_proto)

