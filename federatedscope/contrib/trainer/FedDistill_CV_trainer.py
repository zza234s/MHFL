from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar, MODE
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.message import Message
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
import torch.nn as nn
import copy
import logging
import torch
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from collections import OrderedDict, defaultdict


class FedDistill_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedDistill_Trainer, self).__init__(model, data, device, config,
                                                 only_for_eval, monitor)

        self.register_hook_in_train(self._hook_on_fit_end_agg_local_logits,
                                    "on_fit_end")
        self.gamma = config.FedDistill.gamma
        self.ctx.global_logits = None
        self.mse_loss =nn.MSELoss()
    def _hook_on_batch_forward(self, ctx):
        x, labels = [_.to(ctx.device) for _ in ctx.data_batch]
        pred = ctx.model(x)
        if len(labels.size()) == 0:
            labels = labels.unsqueeze(0)

        CE_loss = ctx.criterion(pred, labels)
        KD_loss = torch.tensor(0.0, device=ctx.device)

        if ctx.global_logits is not None:
            logits_label = torch.empty_like(pred)  # TODO: 检查该操作是否正确
            for i, label in enumerate(labels):
                label_item = label.item()
                if label_item in ctx.global_logits:
                    logits_label[i] = ctx.global_logits[label_item].data
            # KD_loss = ctx.criterion(pred, logits_label)
            KD_loss=self.mse_loss(logits_label,pred)
        loss = CE_loss + self.gamma * KD_loss

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
    def _hook_on_epoch_start_variable_definition(self, ctx):
        ctx.avg_logits_dict = CtxVar(dict(), LIFECYCLE.ROUTINE)  # local-average logit vectors

    @torch.no_grad()
    def _hook_on_fit_end_agg_local_logits(self, ctx):
        st_time = time.time()
        avg_logits_dict = dict()

        all_preds = []  # Collect all predictions
        all_labels = []  # Collect all labels

        ctx.model.eval()
        for batch_idx, (images, labels) in enumerate(ctx.data['train']):
            images = images.to(ctx.device)
            labels = labels.to(ctx.device)

            pred = ctx.model(images)
            all_preds.append(pred)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)  # Concatenate all predictions
        all_labels = torch.cat(all_labels)  # Concatenate all labels

        owned_classes = all_labels.unique()
        for cls in owned_classes:
            filted_preds = all_preds[all_labels == cls]
            avg_logits_dict[cls.item()]=filted_preds.mean(dim=0)

        ctx.avg_logits_dict = avg_logits_dict
        time_cost = time.time() - st_time
        logger.info(f"client:{ctx.client_ID},本地聚合时间开销:{time_cost}s")
    def update(self, global_logits, strict=False):
        self.ctx.global_logits = global_logits

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.avg_logits_dict


def call_my_trainer(trainer_type):
    if trainer_type == 'feddistill_trainer':
        trainer_builder = FedDistill_Trainer
        return trainer_builder


register_trainer('feddistill_trainer', call_my_trainer)
