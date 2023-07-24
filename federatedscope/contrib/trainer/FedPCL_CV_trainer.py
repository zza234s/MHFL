from federatedscope.register import register_trainer
import torch
import torch.nn as nn
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar
from federatedscope.contrib.loss.MHFL_losses import ConLoss
import logging
import numpy as np
import copy
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def agg_local_proto(protos):
    """
    Average the protos for each local user
    """
    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = torch.stack(proto_list)
        agg_protos[label] = torch.mean(proto, dim=0).data

    return agg_protos

class FedPCL_CV_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedPCL_CV_Trainer, self).__init__(model, data, device, config,
                                                only_for_eval, monitor)
        self.loss_mse = nn.MSELoss()
        self.loss_CL = ConLoss(temperature=0.07)
        self.nll_loss = nn.NLLLoss().to(device)
        self.num_users = config.federate.client_num
        self.device = device
        self.register_hook_in_train(self._hook_on_fit_end_agg_local_proto,
                                    "on_fit_end",insert_pos=0)
        self.register_hook_in_train(self._hook_on_fit_start_init_additionaly,
                                    "on_fit_start")
        self.register_hook_in_eval(self._hook_on_fit_start_init_additionaly,
                                   "on_epoch_start")
        self.debug = config.fedpcl.debug
    def _hook_on_batch_forward(self, ctx):
        image, labels = [_.to(ctx.device) for _ in ctx.data_batch]
        images = torch.cat([image, image.clone()], dim=0)  # 源代码每次读取样本时，会返回两个相同的image（使用了TwoCropTransform）

        if self.debug:
            # 根据fedpcl源代码进行前向传播，用来验证复现的正确性
            with torch.no_grad():  # TODO: 请注意，改论文假设每个client端可以拥有多个骨干网络
                for i in range(len(ctx.backbone_list)):
                    backbone = ctx.backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)
            log_probs, features = ctx.model(reps)
            bsz = labels.shape[0]
            lp1, lp2 = torch.split(log_probs, [bsz, bsz],
                                   dim=0)  # lp1==lp2 完全相等 ，猜测lp代表log_probs: [batchsize, num_class]
            loss1=self.nll_loss(lp1, labels)
        else:
            probs, features = ctx.model(images)
            features = F.normalize(features, dim=1)

            bsz = labels.shape[0]
            lp1, lp2 = torch.split(probs, [bsz, bsz], dim=0)  # lp1==lp2 完全相等 ，猜测lp代表log_probs: [batchsize, num_class]
            loss1 = ctx.criterion(lp1, labels)  # cross-entropy loss

        # compute regularized loss term
        L_g = 0 * loss1
        L_p = 0 * loss1
        if len(ctx.global_protos) == self.num_users:
            # compute global proto based CL loss
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # L_g = self.loss_CL(features, labels, ctx.global_avg_protos)#TODO: 取消注释
            for i in range(1, self.num_users + 1):
                for label in ctx.global_avg_protos.keys():
                    if label not in ctx.global_protos[i].keys():
                        ctx.global_protos[i][label] = ctx.global_avg_protos[label]
                L_p += self.loss_CL(features, labels, ctx.global_protos[i])
        else:
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        # TODO: 源代码中 loss=loss2 (L_p) ，没用上loss1 (CE_LOSS);
        # TODO: 此外，正如用户TsingZ0在FedPCL的仓库的issue里提到的：--原文中的基于全局原型的损失Eq.(8)在源代码中缺失了
        loss = L_p
        logger.info(
            f'client#{ctx.client_ID} {ctx.cur_split} round:{ctx.cur_state} \t CE_loss:{loss1}, \t L_p:{L_p}\t total_loss:{loss}')

        # loss = loss1 + L_p + L_g
        # logger.info(
        #     f'client#{ctx.client_ID} {ctx.cur_split} round:{ctx.cur_state} \t CE_loss:{loss1}, \t L_p:{L_p} \t L_g:{L_g} \t total_loss:{loss}')
        # TODO: 使用本地原型进行推理 (√)
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(lp1, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        ####
        ctx.ys_feature.append(f1.detach().cpu())
        ####
    def _hook_on_fit_end_agg_local_proto(self, ctx):
        self.get_aggprotos()
        protos = ctx.agg_protos_label
        setattr(ctx, "agg_protos", protos)

    def _hook_on_fit_start_init_additionaly(self, ctx):
        ctx.agg_protos_label = CtxVar(dict(), LIFECYCLE.ROUTINE) # 每次本地训练之前，初始化agg_protos_label为空字典
        ctx.ys_feature = CtxVar([], LIFECYCLE.ROUTINE) #保存每个样本的representation，基于local prototype 计算acc时会用到

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_protos

    @torch.no_grad()
    def get_aggprotos(self):
        # TODO 确保evaluate阶段不会进入这个函数
        ctx = self.ctx
        reps_dict = {}
        ctx.model.eval()
        if self.debug:
            for batch_idx, (images, label_g) in enumerate(ctx.data['train']):
                images, labels = images.to(self.device), label_g.to(self.device)
                with torch.no_grad():
                    for i in range(len(ctx.backbone_list)):
                        backbone = ctx.backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)
                _, features = ctx.model(reps)
                for i in range(len(labels)):
                    if labels[i].item() in reps_dict:
                        reps_dict[labels[i].item()].append(features[i, :])
                    else:
                        reps_dict[labels[i].item()] = [features[i, :]]
        else:
            #####################################################################################
            for batch_idx, (images, labels) in enumerate(ctx.data['train']):
                images, labels = images.to(ctx.device), labels.to(ctx.device)
                _, features = ctx.model(images)
                features = F.normalize(features, dim=1)
                for i in range(len(labels)):
                    if labels[i].item() in reps_dict:
                        reps_dict[labels[i].item()].append(features[i, :])
                    else:
                        reps_dict[labels[i].item()] = [features[i, :]]

        ctx.agg_protos_label = agg_local_proto(reps_dict)


def call_my_torch_trainer(trainer_type):
    if trainer_type == 'fedpcl_cv_trainer':
        trainer_builder = FedPCL_CV_Trainer
        return trainer_builder


register_trainer('fedpcl_cv_trainer', call_my_torch_trainer)
