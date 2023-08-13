from federatedscope.register import register_model
from torch import nn
import torch.nn.functional as F
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, drop_rate, num_classes,return_features):
        super(MLP, self).__init__()

        # init
        self.num_layers = num_layers
        self.num_classes = num_classes

        # define layers.
        for i in range(1, self.num_layers + 1):
            in_features = input_size if i == 1 else hidden_size
            out_features = hidden_size

            layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(p=drop_rate),
            )
            setattr(self, "layer{}".format(i), layer)

        self.FC = nn.Linear(hidden_size, self.num_classes, bias=False)

        self.return_features = return_features
    def _decide_input_feature_size(self):
        if "cifar" in self.dataset:
            return 32 * 32 * 3
        elif "mnist" in self.dataset:
            return 28 * 28
        else:
            raise NotImplementedError

    def forward(self, x):
        out = x.view(x.size(0), -1)
        for i in range(1, self.num_layers + 1):
            out = getattr(self, "layer{}".format(i))(out)
        res = self.FC(out)
        if self.return_features:
            return res, out
        else:
            return res


def call_mlp(model_config, local_data):
    if 'MLP' in model_config.type:
        input_size = local_data[-1] * local_data[-2] * local_data[-3]
        model = MLP(input_size=input_size, num_layers=model_config.layer, hidden_size=model_config.hidden,
                    drop_rate=model_config.dropout, num_classes=model_config.out_channels,return_features=model_config.return_proto)
        return model

register_model('MLP', call_mlp)
