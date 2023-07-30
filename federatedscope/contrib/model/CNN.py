import torch.nn.functional as F
from federatedscope.register import register_model
from torch.nn import Module
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU


class CNN_2layers(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 filters=[128, 256],
                 use_bn=True,
                 dropout=.0,
                 return_proto=False):
        super(CNN_2layers, self).__init__()

        n1 = filters[0]
        n2 = filters[1]

        self.conv1 = Conv2d(in_channels, n1, 5, padding=2)
        self.conv2 = Conv2d(n1, n2, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(n1)
            self.bn2 = BatchNorm2d(n2)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * n2, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout
        self.return_proto = return_proto

    def forward(self, x, GAN=False):
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = Flatten()(x)
        if GAN:
            return x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.relu(self.fc1(x))
        x = F.dropout(x1, p=self.dropout, training=self.training)
        x = self.fc2(x)

        if self.return_proto:
            return x, x1
        else:
            return x


class CNN_3layers(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 filters=[64, 128, 256],
                 use_bn=True,
                 dropout=.0,
                 return_proto=False):
        super(CNN_3layers, self).__init__()

        n1 = filters[0]
        n2 = filters[1]
        n3 = filters[2]

        self.conv1 = Conv2d(in_channels, n1, 5, padding=2)
        self.conv2 = Conv2d(n1, n2, 5, padding=2)
        self.conv3 = Conv2d(n2, n3, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(n1)
            self.bn2 = BatchNorm2d(n2)
            self.bn3 = BatchNorm2d(n3)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * n3, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout
        self.return_proto = return_proto

    def forward(self, x, GAN=False):
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = self.bn3(self.conv3(x)) if self.use_bn else self.conv3(x)
        x = self.relu(x)

        x = Flatten()(x)
        if GAN:
            return x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.relu(self.fc1(x))
        x = F.dropout(x1, p=self.dropout, training=self.training)
        x = self.fc2(x)

        if self.return_proto:
            return x, x1
        else:
            return x


class CNN_4layers(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 filters=[64, 64, 64, 64],
                 use_bn=True,
                 dropout=.0,
                 return_proto=False):
        super(CNN_4layers, self).__init__()

        n1 = filters[0]
        n2 = filters[1]
        n3 = filters[2]
        n4 = filters[3]

        self.conv1 = Conv2d(in_channels, n1, 5, padding=2)
        self.conv2 = Conv2d(n1, n2, 5, padding=2)
        self.conv3 = Conv2d(n2, n3, 5, padding=2)
        self.conv4 = Conv2d(n3, n4, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(n1)
            self.bn2 = BatchNorm2d(n2)
            self.bn3 = BatchNorm2d(n3)
            self.bn4 = BatchNorm2d(n4)

        self.fc1 = Linear((h // 2 // 2 // 2) * (w // 2 // 2 // 2) * n4, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout

        self.return_proto = return_proto

    def forward(self, x, GAN=False):
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = self.bn3(self.conv3(x)) if self.use_bn else self.conv3(x)
        x = self.maxpool(self.relu(x))
        x = self.bn4(self.conv4(x)) if self.use_bn else self.conv4(x)
        x = self.relu(x)

        x = Flatten()(x)
        if GAN:
            return x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.relu(self.fc1(x))
        x = F.dropout(x1, p=self.dropout, training=self.training)
        x = self.fc2(x)

        if self.return_proto:
            return x, x1
        else:
            return x

def call_our_cnn(model_config, input_shape):
    if 'CNN_2layers' in model_config.type:
        model = CNN_2layers(in_channels=input_shape[-3],
                            w=input_shape[-2],
                            h=input_shape[-1],
                            hidden=model_config.hidden,
                            class_num=model_config.out_channels,
                            filters=model_config.filter_channels,
                            dropout=model_config.dropout,
                            return_proto=model_config.return_proto)
        return model
    elif 'CNN_3layers' in model_config.type:
        model = CNN_3layers(in_channels=input_shape[-3],
                            w=input_shape[-2],
                            h=input_shape[-1],
                            hidden=model_config.hidden,
                            class_num=model_config.out_channels,
                            filters=model_config.filter_channels,
                            dropout=model_config.dropout,
                            return_proto=model_config.return_proto)
        return model

    elif 'CNN_4layers' in model_config.type:
        model = CNN_4layers(in_channels=input_shape[-3],
                            w=input_shape[-2],
                            h=input_shape[-1],
                            hidden=model_config.hidden,
                            class_num=model_config.out_channels,
                            filters=model_config.filter_channels,
                            dropout=model_config.dropout,
                            return_proto=model_config.return_proto)
        return model

register_model('call_our_cnn', call_our_cnn)
