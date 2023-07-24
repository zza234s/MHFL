from federatedscope.register import register_metric
import torch
import torch.nn as nn

METRIC_NAME = 'acc_based_on_local_prototype'


def acc_based_on_local_prototype(ctx, **kwargs):
    loss_mse = nn.MSELoss()
    labels = torch.Tensor(kwargs['y_true']).to(ctx.device).view(-1)
    features = torch.Tensor(torch.cat(ctx.ys_feature)).to(ctx.device)

    num_classes = ctx.cfg.model.num_classes
    local_protos = ctx.agg_protos
    sample_size = kwargs['y_prob'].shape[0]
    a_large_num = 100
    dist = a_large_num * torch.ones(size=(sample_size, num_classes)).to(ctx.device)  # initialize a distance matrix
    # input()
    for i in range(sample_size):
        for j in range(num_classes):
            if j in local_protos.keys():
                d = loss_mse(features[i, :], local_protos[j])  # compare with local protos
                dist[i, j] = d
    _, pred_labels = torch.min(dist, 1)

    pred_labels = pred_labels.view(-1)
    correct = torch.sum(torch.eq(pred_labels, labels)).item()
    total = len(labels)
    acc = correct / total
    return acc


def call_my_metric(types):
    if METRIC_NAME in types:
        the_larger_the_better = True
        metric_builder = acc_based_on_local_prototype
        return METRIC_NAME, metric_builder, the_larger_the_better


register_metric(METRIC_NAME, call_my_metric)
