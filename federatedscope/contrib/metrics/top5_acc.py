import torch
import  numpy as np

from federatedscope.register import register_metric

METRIC_NAME = 'top5'

def Top5(ctx, **kwargs ):
    max5 = np.argsort(ctx.ys_prob,axis=-1)[:, -5:]
    y_true = ctx.ys_true[:, np.newaxis]
    top5 = np.sum(y_true == max5)
    top5acc = round(100 * top5 / len(y_true), 2)
    return top5acc

def call_my_metric(types):
    if METRIC_NAME in types:
        the_larger_the_better = True
        metric_builder = Top5
        return METRIC_NAME, metric_builder, the_larger_the_better


register_metric(METRIC_NAME, call_my_metric)