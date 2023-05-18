from torch import nn
import torch.nn.functional as F
import copy
from federatedscope.register import register_model

"""
参考:https://github.com/FangXiuwen/FSMAFL/blob/main/models.py 
"""

class CNN_2layer_fc_model(nn.Module):
    def __init__(self, params):
        super(CNN_2layer_fc_model, self).__init__()
        n_one = params["n1"]
        n_two = params["n2"]
        # dropout_rate = params["dropout_rate"]
        self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1, kernel_size=3, out_channels=n_one, padding=1),
                                       nn.BatchNorm2d(n_one),
                                       nn.ReLU(),
                                       # nn.Dropout(dropout_rate),
                                       nn.AvgPool2d(kernel_size=2, stride=1)
                                       )
        self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=n_one, stride=2, kernel_size=3, out_channels=n_two),
                                       nn.BatchNorm2d(n_two),
                                       nn.ReLU()
                                       # nn.Dropout(keep_prob=dropout_rate)
                                       )
        self.FC1 = nn.Linear(169 * n_two, 16)

    def forward(self, x, GAN=False):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = x.view((x.shape[0], -1))
        if GAN:
            return x
        else:
            x = self.FC1(x)
            return nn.LogSoftmax()(x)


def call_CNN_FSFL_MNIST(model_config, local_data):
    if 'CNN' in model_config.type and 'fsfl' in model_config.type and 'mnist' in model_config.type and '2_layer' in model_config.type:
        #参考的github仓库代码里仅修改了两个CNN的输出通道数用以模型异构设定
        param = dict()
        param['n1'] = model_config.fsfl_cnn_layer1_out_channels
        param['n2'] = model_config.fsfl_cnn_layer2_out_channels
        model = CNN_2layer_fc_model(param)
        return model


register_model('CNN_FSFL_MNIST', call_CNN_FSFL_MNIST)
