import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedscope.register import register_model

class CNN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True,
                 dropout=.0):
        super(CNN, self).__init__()
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

def call_CNN_proto(model_config, local_data):
    if 'CNN_proto' in model_config.type:
        model = CNN(in_channels=local_data[1],hidden=model_config.hidden)
        return model


register_model('CNN_proto', call_CNN_proto)