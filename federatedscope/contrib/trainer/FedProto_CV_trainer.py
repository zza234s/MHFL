from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar,MODE
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.message import Message
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
import torch.nn as nn
import copy
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Build your trainer here.
class FedProto_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedProto_Trainer, self).__init__(model, data, device, config,
                                              only_for_eval, monitor)
        self.loss_mse = nn.MSELoss()
        self.register_hook_in_train(self._hook_on_fit_end_agg_proto,
                                    "on_fit_end")

        self.register_hook_in_train(self._hook_on_epoch_start_for_proto,
                                    "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_epoch_start_for_proto,
                                    "on_epoch_start")

    def _hook_on_batch_forward(self, ctx):
        x, labels = [_.to(ctx.device) for _ in ctx.data_batch]

        pred, protos = ctx.model(x)
        if len(labels.size()) == 0:
            labels = labels.unsqueeze(0)
        loss1 = ctx.criterion(pred, labels)

        if len(ctx.global_protos) == 0:
            loss2 = 0 * loss1
        else:
            proto_new = copy.deepcopy(protos.data)
            i = 0
            for label in labels:
                if label.item() in ctx.global_protos.keys():
                    proto_new[i, :] = ctx.global_protos[label.item()][0].data
                i += 1
            loss2 = self.loss_mse(proto_new, protos)
        loss = loss1 + loss2 * 1.0  #TODO: 将1.0变成变量超参--》parser.add_argument('--ld', type=float, default=1, help="weight of proto loss")

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        for i in range(len(labels)):
            if labels[i].item() in ctx.agg_protos_label:
                ctx.agg_protos_label[labels[i].item()].append(protos[i, :])
            else:
                ctx.agg_protos_label[labels[i].item()] = [protos[i, :]]

    def update(self, global_proto,strict=False):
        self.ctx.global_proto = global_proto

    def _hook_on_epoch_start_for_proto(self,ctx):
        """定义一些fedproto需要用到的全局变量"""
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}
        agg_protos_label = {}
        ctx.epoch_loss = CtxVar(epoch_loss, LIFECYCLE.EPOCH)
        ctx.agg_protos_label = CtxVar(agg_protos_label, LIFECYCLE.ROUTINE)


    def _hook_on_fit_end_agg_proto(self, ctx):
        protos = ctx.agg_protos_label
        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]

        setattr(ctx, "agg_protos", protos)


    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_protos


    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set):
        for batch_i in tqdm(range(
                getattr(self.ctx, f"num_{self.ctx.cur_split}_batch"))):
            self.ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH)

            for hook in hooks_set["on_batch_start"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_forward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_backward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_end"]:
                hook(self.ctx)

            # Break in the final epoch
            if self.ctx.cur_mode in [
                    MODE.TRAIN, MODE.FINETUNE
            ] and self.ctx.cur_epoch_i == self.ctx.num_train_epoch - 1:
                if batch_i >= self.ctx.num_train_batch_last_epoch - 1:
                    break

def call_my_trainer(trainer_type):
    if trainer_type == 'fedproto_trainer':
        trainer_builder = FedProto_Trainer
        return trainer_builder


register_trainer('fedproto_trainer', call_my_trainer)
