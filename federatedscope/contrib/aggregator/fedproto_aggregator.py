import os
import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor

class PrototypeAggregator(Aggregator):

    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def aggregate(self, agg_info):
        local_protos_list=dict()
        client_feedback = agg_info["client_feedback"]

        for k,v in enumerate(client_feedback):
            local_protos_list[k] = v[0][1]

        agg_protos_label = self._proto_aggregation(local_protos_list)
        return agg_protos_label

    def _proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label
