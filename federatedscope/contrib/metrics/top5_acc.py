import torch

from federatedscope.register import register_metric

METRIC_NAME = 'top5'

# def Top5(ctx, y_true, y_pred, **kwargs ):
#     _,max5 = torch.topk(y_pred, 5, dim=-1)
#     y_true = y_true.view(-1,1)
#     top5 += (y_true == max5).sum().item()