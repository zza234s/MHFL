import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import os
from federatedscope.register import register_model

class ProjandDeci(nn.Module):
    def __init__(self, in_d, out_d, num_classes):
        super(ProjandDeci, self).__init__()
        self.fc1 = nn.Linear(in_d, out_d)
        self.fc2 = nn.Linear(out_d, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.normalize(x, dim=1)
        x = F.relu(self.fc2(x1))
        x = F.normalize(x, dim=1)
        return F.log_softmax(x, dim=1), x1

def call_fecpcl_mlp(model_config, local_data):
    if 'proj_and_deci' in model_config.type:
        in_d = model_config.fedpcl.input_size
        out_d = model_config.fedpcl.output_dim
        num_classes = model_config.out_channels

        model = ProjandDeci(in_d, out_d, num_classes)
        return model

register_model('fedpcl_mlp', call_fecpcl_mlp)
