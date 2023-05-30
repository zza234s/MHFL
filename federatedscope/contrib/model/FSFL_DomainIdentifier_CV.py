import torch.nn as nn
import torch.nn.functional as F


class DomainIdentifier(nn.Module):
    """
    源代码参考自：https://github.com/FangXiuwen/FSMAFL/blob/main/models.py
    需要根据每个客户端的模型来修改resize_layer_zero中第一个FC层的输入维度。
    resize_layer_one代表第1个客户端对应的DI
    """
    def __init__(self):
        super(DomainIdentifier, self).__init__()
        self.resize_layer_zero = nn.Sequential(nn.Linear(43264, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_layer_one = nn.Sequential(nn.Linear(43264, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_layer_two = nn.Sequential(nn.Linear(43264, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_layer_three = nn.Sequential(nn.Linear(43264, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_layer_four = nn.Sequential(nn.Linear(86528, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_layer_five = nn.Sequential(nn.Linear(2304, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_layer_six = nn.Sequential(nn.Linear(1728, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_layer_seven = nn.Sequential(nn.Linear(2304, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_layer_eight = nn.Sequential(nn.Linear(1152, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_layer_nine = nn.Sequential(nn.Linear(1728, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.resize_dict = {0: self.resize_layer_zero, 1: self.resize_layer_one, 2: self.resize_layer_two,
                            3: self.resize_layer_three,
                            4: self.resize_layer_four, 5: self.resize_layer_five, 6: self.resize_layer_six,
                            7: self.resize_layer_seven,
                            8: self.resize_layer_eight, 9: self.resize_layer_nine}
        self.resize_list = [self.resize_layer_zero, self.resize_layer_one, self.resize_layer_two,
                            self.resize_layer_three,
                            self.resize_layer_four, self.resize_layer_five, self.resize_layer_six,
                            self.resize_layer_seven,
                            self.resize_layer_eight, self.resize_layer_nine]
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 11)

    def forward(self, x, index):
        x = x.view(x.shape[0], -1)
        x = self.resize_list[index](x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x
