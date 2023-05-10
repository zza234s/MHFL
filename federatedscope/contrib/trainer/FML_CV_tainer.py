from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.utils import calculate_batch_epoch_num
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.optimizer import wrap_regularized_optimizer
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.core.auxiliaries.model_builder import get_model
from typing import Type
import torch
import torch.nn as nn
import copy
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Build your trainer here.
"""
    基于 Federated Mutual Learning （FML） 源代码以及FS中内置的trainer_Ditto代码实现的Trainer；
    FML源代码链接：https://github.com/ZJU-DAI/Federated-Mutual-Learning
    trainer_Ditto位置： core/trainers/trainer_Ditto.py
"""

class FML_CV_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FML_CV_Trainer, self).__init__(model, data, device, config,
                                             only_for_eval, monitor)

        self.ctx.local_model = copy.deepcopy(self.ctx.model)
        self.ctx.model = get_model(model_config=config.fml.meme_model,local_data=data)  # the personalized model  #修改要聚合的模型为meme model
        # self.ctx.models = [self.ctx.local_model, self.ctx.model] #TODO: 感觉不需要定义models，需要确认定义/不定义 的影响

        self.register_hook_in_train(new_hook=self._hook_on_fit_start_clean,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_eval(new_hook=self._hook_on_fit_start_clean,
                                    trigger='on_fit_start',
                                    insert_pos=-1)

        # TODO: 注册_hook_on_batch_end_flop_count
        # TODO: 弄懂是否需要注册 _hook_on_fit_end_calibrate

        self.register_hook_in_train(new_hook=self._hook_on_fit_end_free_cuda,
                                    trigger="on_fit_end",
                                    insert_pos=-1)

        self.register_hook_in_eval(new_hook=self._hook_on_fit_end_free_cuda,
                                   trigger="on_fit_end",
                                   insert_pos=-1)

        self.KL_Loss = nn.KLDivLoss(reduction='batchmean')
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.alpha = self._cfg.fml.alpha
        self.beta = self._cfg.fml.beta

    def _hook_on_batch_forward(self, ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]

        # get both output and loss of mutual learning
        output_local = ctx.local_model(x)
        output_meme = ctx.model(x)

        ce_local = ctx.criterion(output_local, label)
        kl_local = self.KL_Loss(self.LogSoftmax(output_local), self.Softmax(output_meme.detach()))

        ce_meme = ctx.criterion(output_meme, label)
        kl_meme = self.KL_Loss(self.LogSoftmax(output_meme), self.Softmax(output_local.detach()))

        loss_local = self.alpha * ce_local + (1 - self.alpha) * kl_local
        loss_meme = self.beta * ce_meme + (1 - self.beta) * kl_meme

        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        # 个性化模型的结果用来计算指标
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(output_local, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_meme, LIFECYCLE.BATCH)  # meme 模型作为全局模型需要联邦更新
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

        # 记录local模型的loss
        ctx.loss_batch_local = CtxVar(loss_local, LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.local_optimizer.zero_grad()

        ctx.loss_batch.backward()
        ctx.loss_batch_local.backward()

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
            torch.nn.utils.clip_grad_norm_(ctx.local_model.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step()
        ctx.local_optimizer.step()

        # TODO: 将调度器也编写进代码（）
        # if ctx.scheduler is not None:
        #     ctx.scheduler.step()

    def _hook_on_fit_start_clean(self,ctx):
        # Set optimizer for local model additionally.
        ctx.local_model.to(ctx.device)
        if ctx.cur_mode in [MODE.TRAIN]:
            ctx.local_model.train()
            ctx.local_optimizer = get_optimizer(ctx.local_model,
                                            **ctx.cfg.train.optimizer)
        elif ctx.cur_mode in [MODE.VAL]:
            ctx.local_model.eval()

    def _hook_on_fit_end_free_cuda(self,ctx):
        # ctx.meme_model.to(torch.device("cpu"))
        ctx.local_model.to(torch.device("cpu"))

def call_my_trainer(trainer_type):
    if trainer_type == 'fml_cv_trainer':
        trainer_builder = FML_CV_Trainer
        return trainer_builder


register_trainer('fml_cv_trainer', call_my_trainer)
