from torch import nn
import torch.nn.functional as F
from federatedscope.register import register_model
'''
来源于fedproto源码：
论文中控制fedproto_femnist_channel_temp变量来为每个client设置不同的模型
'''

class CNNFemnist(nn.Module):
    def __init__(self, args):
        super(CNNFemnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, args.fedproto_femnist_channel_temp, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(16820/20*args.fedproto_femnist_channel_temp), 50)
        self.fc2 = nn.Linear(50, args.out_channels)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return x, x1

def call_CNNFemnist_proto(model_config, local_data):
    if 'CNNFemnist_proto' in model_config.type:
        model = CNNFemnist(model_config)
        return model

register_model('CNNFemnist_proto', call_CNNFemnist_proto)