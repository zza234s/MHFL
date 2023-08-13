import copy

from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging

from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.contrib.model.FedGH_FC import FedGH_FC

import torch
import torch.nn as nn
logger = logging.getLogger(__name__)


class FedGH_Server(Server):
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
        super(FedGH_Server, self).__init__(ID, state, config, data, model, client_num,
                                            total_round_num, device, strategy, **kwargs)
        self.received_protos_dict = dict()
        self.device = device
        self.net_FC = FedGH_FC(config.model.feature_dim, config.model.num_classes).to(device)
        self.criteria = nn.CrossEntropyLoss()
        self.models[0] = copy.deepcopy(self.net_FC)
        self.model = copy.deepcopy(self.net_FC)
        self.optimizer = get_optimizer(model=self.net_FC, **config.train.optimizer)
        # self.optimizer = get_optimizer(model=self.net_FC, **config.FedGH.server_optimizer)
    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        #TODO: 需要完善当采样率不等于0时的实现
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
                # update global protos
                #################################################################
                local_protos_dict = dict()
                msg_list = self.msg_buffer['train'][self.state]
                for client_id, values in msg_list.items():
                    local_protos_dict[client_id] = values[1]

                self.global_header_training(local_protos_dict)
                global_header_para = copy.deepcopy(self.net_FC.state_dict())
                #################################################################

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self._broadcast_custom_message(msg_type='evaluate', content=global_header_para, filter_unseen_clients=False)
                    # self.eval()

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
                    self._broadcast_custom_message(msg_type='model_para',content=global_header_para)

                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self._broadcast_custom_message(msg_type='evaluate', content=global_header_para, filter_unseen_clients=False)
                    # self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def global_header_training(self, local_protos_dict):
        total_loss = 0.0
        total_samples = 0

        for client_id, local_protos in local_protos_dict.items():

            for cls, rep in local_protos.items():
                self.optimizer.zero_grad()
                rep=rep.unsqueeze(0)
                pred_server = self.net_FC(rep)
                loss = self.criteria(pred_server.view(1, -1), torch.tensor(cls).view(1).to(self.device))
                loss.backward()

                total_loss += loss.item()
                total_samples += 1

                torch.nn.utils.clip_grad_norm_(self.net_FC.parameters(), 50)
                self.optimizer.step()

        if total_samples > 0:
            mean_loss = total_loss / total_samples
            logger.info(f"round:{self.state} \t global head mean loss: {mean_loss}")

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





class FedGH_client(Client):
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
        super(FedGH_client, self).__init__(ID, server_id, state, config, data, model, device,
                                             strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.global_protos = []
        self.trainer.ctx.client_ID = self.ID
        self.register_handlers('global_proto',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para'])


    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        self.trainer.update(content,strict=True)
        self.state = round
        self.trainer.ctx.cur_state = round
        sample_size, model_para, results, agg_protos = self.trainer.train()

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
                    content=(sample_size, agg_protos)))

def call_my_worker(method):
    if method == 'fedgh':
        worker_builder = {'client': FedGH_client, 'server': FedGH_Server}
        return worker_builder


register_worker('fedgh', call_my_worker)
