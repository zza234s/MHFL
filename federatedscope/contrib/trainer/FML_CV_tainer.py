from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.utils import calculate_batch_epoch_num
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.optimizer import wrap_regularized_optimizer
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict
from typing import Type
import torch
import torch.nn as nn
import copy
import logging
import numpy as np
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
        '''
        self.model --> personalized local model
        self.meme_model --> meme model
        '''
        # self.ctx.local_model = self.ctx.model # the personalized model
        self.ctx.meme_model = get_model(model_config=config.fml.meme_model, local_data=data)  # 修改要聚合的模型为meme model
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

        self.register_hook_in_eval(new_hook=self._hook_on_fit_end_free_cuda,
                                   trigger="on_fit_end",
                                   insert_pos=-1)

        self.KL_Loss = nn.KLDivLoss(reduction='batchmean')
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.alpha = self._cfg.fml.alpha
        self.beta = self._cfg.fml.beta

    def get_model_para(self):
        """
        重写get_model_para(), 使其从返回ctx.model的参数变为返回ctx.meme_model的参数
        trainer.train()会调用该函数，以获得更新好的本地模型
        """
        if self.cfg.federate.process_num > 1:
            return self._param_filter(self.ctx.meme_model.state_dict())
        else:
            return self._param_filter(
                self.ctx.meme_model.state_dict() if self.cfg.federate.
                share_local_model else self.ctx.meme_model.cpu().state_dict())

    def update(self, model_parameters, strict=False):
        """
            Called by the FL client to update the model parameters
            修改被更新的模型为self.ctx.meme_model 而不是原来的self.ctx.model
            框架源代码的逻辑为：client接收server下发的全局模型权重，然后基于接收到的权重更新本地模型(self.ctx.model)
            但是FML中，全局更新的是meme model
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(self.ctx.meme_model.state_dict().copy(),
                                        self._param_filter(model_parameters))
        self.ctx.meme_model.load_state_dict(merged_param, strict=strict)


    def _hook_on_batch_forward(self, ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]

        # get both output and loss of mutual learning
        output_local = ctx.model(x)
        output_meme = ctx.meme_model(x)

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

        # 记录meme模型的logits
        ctx.meme_ys_prob.append(output_meme.clone().detach().cpu().numpy())
    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.meme_optimizer.zero_grad()

        ctx.loss_batch.backward()  # 等价于 loss_meme.backward()
        ctx.loss_batch_local.backward()

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
            torch.nn.utils.clip_grad_norm_(ctx.meme_model.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step()  # 更新local_model
        ctx.meme_optimizer.step()  # 更新meme_model

        # TODO: 将调度器也编写进代码（）
        # if ctx.scheduler is not None:
        #     ctx.scheduler.step()

    def _hook_on_fit_start_clean(self, ctx):
        #保存meme_model的输出结果用以验证
        ctx.meme_ys_prob = CtxVar([], LIFECYCLE.ROUTINE)


        # Set optimizer for meme model additionally.
        ctx.meme_model.to(ctx.device)
        if ctx.cur_mode in [MODE.TRAIN]:
            ctx.meme_model.train()
            ctx.meme_optimizer = get_optimizer(ctx.meme_model,
                                               **ctx.cfg.train.optimizer)  # TODO:假设memo model 的optimizer和local model的一样
        elif ctx.cur_mode in [MODE.VAL]:
            ctx.meme_model.eval()

    def _hook_on_fit_end_free_cuda(self, ctx):
        ctx.meme_model.to(torch.device("cpu"))
        ctx.meme_ys_prob = CtxVar(np.concatenate(ctx.meme_ys_prob), LIFECYCLE.ROUTINE)
        y_true =ctx.ys_true
        y_prob = ctx.meme_ys_prob
        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=-1)
        if y_prob.ndim == 2:
            y_prob = np.expand_dims(y_prob, axis=-1)

        # if len(y_prob.shape) > len(y_true.shape):
        y_pred = np.argmax(y_prob, axis=1)

        acc=eval_acc(y_true,y_pred)
        logger.info(f'meme_model acc :{acc}')
        # ctx.local_model.to(torch.device("cpu"))

def eval_acc(y_true, y_pred, **kwargs):
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)


def call_my_trainer(trainer_type):
    if trainer_type == 'fml_cv_trainer':
        trainer_builder = FML_CV_Trainer
        return trainer_builder


register_trainer('fml_cv_trainer', call_my_trainer)
