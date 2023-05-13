from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.model_heterogeneity.methods.FedHeNN.cka_utils_torch import linear_CKA
from federatedscope.register import register_trainer
import torch

class FedHeNN_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedHeNN_Trainer, self).__init__(model, data, device, config,
                                              only_for_eval, monitor)
    def _hook_on_batch_forward(self, ctx):
        RAD_dataloader = ctx.RAD_dataloader
        global_K = ctx.global_K
        eta = 1.0  # TODO:写进超参数;找到正确的计算方法

        # 1. the loss of proximal term
        # get reprensentation matrix from RAD
        for x, label in RAD_dataloader:
            _, intermediate_out = ctx.model(x.to(ctx.device))  # TODO: 这里的x变量是否需要释放GPU？

        # get K_i
        kernel_matric = torch.matmul(intermediate_out, torch.transpose(intermediate_out, 0, 1))
        # get distance between local and global kernel matrices
        # note: gloab_k加上.deatach()很重要，否则反向传播会报计算图已经被释放的错误
        proximal_loss = linear_CKA(kernel_matric, global_K.detach().to(ctx.device))  # TODO: global_K 是否需要释放GPU？

        # 2. the loss of downstream task.
        x, labels = [_.to(ctx.device) for _ in ctx.data_batch]
        pred, inter_out = ctx.model(x)
        if len(labels.size()) == 0:
            labels = labels.unsqueeze(0)
        pred_loss = ctx.criterion(pred, labels)

        loss = pred_loss + proximal_loss * eta

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)


def call_my_trainer(trainer_type):
    if trainer_type == 'fedhenn_trainer':
        trainer_builder = FedHeNN_Trainer
        return trainer_builder


register_trainer('fedhenn_trainer', call_my_trainer)
