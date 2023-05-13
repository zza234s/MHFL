import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedscope.register import register_model
from torchsummary import summary
import copy

class Net0(nn.Module):
    def __init__(self,
                 in_channels=3,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True,
                 dropout=.0):
        super(Net0, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(12544, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        inter_out = x
        x = self.dropout2(x)
        output = self.fc2(x)
        return output,inter_out


class Net1(nn.Module):
    def __init__(self,
                 in_channels=3,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True,
                 dropout=.0):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(14*14*64, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        inter_out = x
        x_new = self.dropout2(inter_out)
        x_new = self.fc2(x_new)
        return x_new, inter_out


class Net2(nn.Module):
    def __init__(self,
                 in_channels=3,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True,
                 dropout=.0):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(14*14*32, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        inter_out = x
        x_new = self.dropout2(inter_out)
        x_new = self.fc2(x_new)
        return x_new, inter_out


class Net3(nn.Module):
    def __init__(self,
                 in_channels=3,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True,
                 dropout=.0):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(14*14*32, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        inter_out = x
        x_new = self.dropout2(inter_out)
        x_new = self.fc2(x_new)
        return x_new, inter_out


def call_CNN_FedHeNN_CIFAR(model_config, input_shape):
    # TODO 根据local_data来控制模型的形状
    # input_shape for the CIFAR10 dataset:(B,3,32,32)
    if 'CNN' in model_config.type and 'fedhenn' in model_config.type and 'cifar' in model_config.type:
        if 'net0' in model_config.type:
            model = Net0(hidden=model_config.hidden)
        elif 'net1' in model_config.type:
            model = Net1(hidden=model_config.hidden)
        elif 'net2' in model_config.type:
            model = Net2(hidden=model_config.hidden)
        else:
            model = Net3(hidden=model_config.hidden)
        return model

register_model('CNN_FedHeNN_CIFAR', call_CNN_FedHeNN_CIFAR)

# if __name__ == "__main__":
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Net3().to(device)
#
#     summary(model, (1, 28, 28))
