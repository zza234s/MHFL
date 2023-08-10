import torch
import torch.nn as nn
class FedGH_FC(nn.Module):
    def __init__(self, in_dim=500, out_dim=10):
        super(FedGH_FC, self).__init__()
        self.FC = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        o = self.FC(x)
        return o
