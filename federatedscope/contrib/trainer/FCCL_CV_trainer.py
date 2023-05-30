from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.register import register_trainer
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class FCCL_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FCCL_Trainer, self).__init__(model, data, device, config,
                                              only_for_eval, monitor)
        self.criterionKL = nn.KLDivLoss(reduction='batchmean')  # 计算KL散度损失 指定损失减少方式为对所有样本损失平均

    def _hook_on_batch_forward(self, ctx):
        device = ctx.device
        model = ctx.model.to(device)
        inter_model = ctx.inter_model.to(device)
        pre_model = ctx.pre_model.to(device)

        self.criterionKL.to(device)

        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        outputs = model(x)  # 私有数据的Zi
        logsoft_outputs = F.log_softmax(outputs, dim=1)  # 进行对数softmax操作
        with torch.no_grad():  # 计算操作不会被记录梯度
            inter_soft_outpus = F.softmax(inter_model(x), dim=1)
            pre_soft_outpus = F.softmax(pre_model(x), dim=1)
        inter_loss = self.criterionKL(logsoft_outputs, inter_soft_outpus)  # 公式4 inter损失
        pre_loss = self.criterionKL(logsoft_outputs, pre_soft_outpus)  # 公式5 intra损失
        loss_hard = ctx.criterion(outputs, label)  # 交叉熵损失
        #TODO：为啥后两个loss不用*λloc
        loss = loss_hard + inter_loss + pre_loss  # 公式7 总损失

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(outputs, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)


def call_my_trainer(trainer_type):
    if trainer_type == 'fccl_trainer':
        trainer_builder = FCCL_Trainer
        return trainer_builder


register_trainer('fccl_trainer', call_my_trainer)
