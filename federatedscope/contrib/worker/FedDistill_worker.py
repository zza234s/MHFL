import copy
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
import time
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.contrib.model.FedGH_FC import FedGH_FC

import torch
import torch.nn as nn
from collections import defaultdict

logger = logging.getLogger(__name__)


class FedDistill_Server(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(FedDistill_Server, self).__init__(ID, state, config, data, model, client_num,
                                                total_round_num, device, strategy, **kwargs)
        self.device = device
        self.task = config.MHFL.task
        self.global_logit_type = config.FedDistill.global_logit_type

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        # TODO: 需要完善当采样率不等于0时的实现
        min_received_num = len(self.comm_manager.get_neighbors().keys())

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True

        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                #################################################################
                local_logits_dict = dict()
                msg_list = self.msg_buffer['train'][self.state]
                for client_id, values in msg_list.items():
                    local_logits_dict[client_id] = values[1]
                avg_global_logits = self.logits_aggregation(local_logits_dict)
                #################################################################

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval(avg_global_logits)

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    if self.global_logit_type==0:
                        self._broadcast_custom_message(msg_type='avg_global_logits', content=avg_global_logits)
                    elif self.global_logit_type==1:
                        for client_id, logits in avg_global_logits.items():
                            self.comm_manager.send(
                                Message(msg_type='avg_global_logits',
                                        sender=self.ID,
                                        receiver=client_id,
                                        state=min(self.state, self.total_round_num),
                                        timestamp=self.cur_timestamp,
                                        content=logits
                                        )
                            )
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval(avg_global_logits)

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def logits_aggregation(self, local_logits_list):
        if self.global_logit_type == 0:
            agg_logits = defaultdict(list)
            avg_global_logits = dict()
            for idx in local_logits_list:
                local_logits = local_logits_list[idx]
                for label in local_logits.keys():
                    agg_logits[label].append(local_logits[label])
            for [label, logits_list] in agg_logits.items():
                if len(logits_list) > 1:
                    logit = 0 * logits_list[0].data
                    for i in logits_list:
                        logit += i.data
                    avg_global_logits[label] = logit / len(logits_list)
                else:
                    avg_global_logits[label] = logits_list[0].data
        elif self.global_logit_type == 1:
            agg_logits = dict()
            avg_global_logits = dict()
            for local_logits in local_logits_list.values():
                for label, logits in local_logits.items():
                    agg_logits.setdefault(label, 0 * logits.data)
                    agg_logits[label] += logits

            for idx in local_logits_list:
                avg_global_logits[idx] = dict()
                local_logits = local_logits_list[idx]
                for label in agg_logits.keys():
                    avg_global_logits[idx][label] = agg_logits[label]
                    avg_global_logits[idx][label] -= local_logits[label]
        return avg_global_logits

    def _broadcast_custom_message(self, msg_type, content,
                                  sample_client_num=-1,
                                  filter_unseen_clients=True):
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=content))

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def eval(self, avg_global_logits):
        if self.global_logit_type == 0:
            self._broadcast_custom_message(msg_type='evaluate', content=avg_global_logits,
                                           filter_unseen_clients=False)
        elif self.global_logit_type == 1:
            rnd = self.state - 1
            for client_id, logits in avg_global_logits.items():
                self.comm_manager.send(
                    Message(msg_type='evaluate',
                            sender=self.ID,
                            receiver=client_id,
                            state=rnd,
                            timestamp=self.cur_timestamp,
                            content=logits
                            )
                )


class FedDistill_Client(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(FedDistill_Client, self).__init__(ID, server_id, state, config, data, model, device,
                                                strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.global_protos = []
        self.trainer.ctx.client_ID = self.ID
        self.register_handlers('avg_global_logits',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para'])

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        if message.msg_type == 'avg_global_logits':
            self.trainer.update(content, strict=True)
        self.state = round
        self.trainer.ctx.cur_state = round
        st_time=time.time()
        sample_size, model_para, results, agg_logits = self.trainer.train()
        logger.info(f"client#{self.ID},local training时间开销{time.time() - st_time}s")
        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)

        if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
            self._monitor.save_formatted_results(train_log_res,
                                                 save_file_name="")

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, agg_logits)))


def call_my_worker(method):
    if method == 'feddistill':
        worker_builder = {'client': FedDistill_Client, 'server': FedDistill_Server}
        return worker_builder


register_worker('feddistill', call_my_worker)
