import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedscope.register import register_model
#TODO: 用公式计算T
class LeNet5(nn.Module):
    def __init__(self,input_channels,out_channels,T=4):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * T * T, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_channels)
        self.T=T
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * self.T * self.T)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def call_LeNet5(model_config, local_data):
    if 'LeNet5' in model_config.type:
        model = LeNet5(input_channels=local_data[1],out_channels=model_config.out_channels,T=5)
        return model

register_model('LeNet5', call_LeNet5)