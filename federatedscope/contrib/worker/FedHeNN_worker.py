import copy
import pickle, sys
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
import numpy as np
from federatedscope.model_heterogeneity.methods.FedHeNN.cka_utils import *
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict

import numpy as np

logger = logging.getLogger(__name__)


def get_representation_matrices(RAD_dataloader, model):
    for x, label in RAD_dataloader:
        pred, intermediate_out = model(x)
    return intermediate_out


# Build your worker here.
class FedHeNN_server(Server):
    # TODO: 伪代码中，每一轮的学习率eta=n_0 * f(current_round),但暂未找到关于f()的实现方式的描述
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
                 unseen_clients_id=None,
                 **kwargs):
        super(FedHeNN_server, self).__init__(ID, state, config, data, model, client_num, total_round_num,
                                             device, strategy, unseen_clients_id, **kwargs)
        self.personalized_models = dict()  # key:client_ID, values: torch.model
        self.representation_matrix = dict()
        self.kernal_matrix = dict()

        # self.register_handlers('initial_personalized_model', self.callback_funcs_for_initial_personalized_model,
        #                        ['model_para'])

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):

        min_received_num = len(self.comm_manager.get_neighbors().keys())
        move_on_flag = True

        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process

                # update global protos
                #################################################################
                RAD_dataloader = self.generated_RAD()
                global_K = self.get_global_K(RAD_dataloader)
                local_protos_list = dict()
                msg_list = self.msg_buffer['train'][self.state]
                aggregated_num = len(msg_list)
                for key, values in msg_list.items():
                    local_protos_list[key] = values[1]
                global_protos = self._proto_aggregation(local_protos_list)
                #################################################################

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

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
                    self._start_new_training_round(global_protos)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def generated_RAD(self):
        """
        原文中,在每个通讯轮，服务器从每个client的本地数据集中采样一个样本来构成RAD；我们通过在服务器上设置数据集直接进行采样来模拟该过程
        Returns: a list that containing L tensor, where L is the size of RAD.
        """
        train_dataset = self.data.train_data

        # 定义要采样的样本数量
        sample_size = self._cfg.federate.client_num

        # 获取数据集的长度
        dataset_length = len(train_dataset)

        # 使用 random_split 函数进行采样
        sampler_indices = torch.randperm(dataset_length)[:sample_size]
        sampled_dataset = torch.utils.data.Subset(train_dataset, sampler_indices)

        # 使用 DataLoader 加载采样的数据集
        RAD_dataloader = DataLoader(sampled_dataset, batch_size=dataset_length)

        return RAD_dataloader

    def get_global_K(self, RAD_dataloader):
        """
        这个函数计算$\bar{K}(t-1)=\sum_{j=1}^{N}w_{j}k_{j}$:即全局的表征距离矩阵
        输入生成的RAD以及每个client上传的模型(保存在server类中)
        """
        # TODO: 当client采样率不等于1.0时的实现？
        kernel_matrices = []
        for client_id, model in self.personalized_models.items():
            r_m = get_representation_matrices(RAD_dataloader, model)
            kernel_matric = torch.matmul(r_m, torch.transpose(r_m, 0, 1))
            kernel_matrices.append(kernel_matric)
        stack_tensor = torch.stack(kernel_matrices)
        global_K_values = torch.mean(stack_tensor,
                                     dim=0)  # 每个client的kernel_matric求平均：FedHeNN, each entry of the weight vector for aggregating the representations w is set to 1/N
        return global_K_values

    def _start_new_training_round(self, global_protos):
        self._broadcast_custom_message(msg_type='global_proto', content=global_protos)

    def eval(self):
        self._broadcast_custom_message(msg_type='evaluate', content=None, filter_unseen_clients=False)

    def callback_funcs_for_join_in(self, message: Message):
        """
            额外增加处理每个client个性化模型的内容
        """
        self.join_in_client_num += 1
        sender = message.sender
        address = message.content[0]

        #####################################################################################
        personalized_model = message.content[1]  # get personalized model
        self.personalized_models[sender] = personalized_model
        ####################################################################################

        if int(sender) == -1:  # assign number to client
            sender = self.join_in_client_num
            self.comm_manager.add_neighbors(neighbor_id=sender,
                                            address=address)
            self.comm_manager.send(
                Message(msg_type='assign_client_id',
                        sender=self.ID,
                        receiver=[sender],
                        state=self.state,
                        timestamp=self.cur_timestamp,
                        content=str(sender)))
        else:
            self.comm_manager.add_neighbors(neighbor_id=sender,
                                            address=address)

        self.trigger_for_start()

    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        额外向clients 发送全局K 以及RAD_loader
        """
        if self.check_client_join_in():

            if self.sampler is None:
                self.sampler = get_sampler(
                    sample_strategy=self._cfg.federate.sampler,
                    client_num=self.client_num
                )

            RAD_dataloader = self.generated_RAD()
            global_K_prev = self.get_global_K(RAD_dataloader)

            #TODO: 检查下直接把server端生成的dataloader传给clients会不会产生奇怪的问题
            self._broadcast_custom_message(msg_type='model_para',
                                           content=[global_K_prev,RAD_dataloader])

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))

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


class FedHeNN_client(Client):
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
        super(FedHeNN_client, self).__init__(ID, server_id, state, config, data, model, device,
                                             strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.global_protos = []
        self.register_handlers('global_proto',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para'])
        # self.register_handlers('ask_initial_model_parameter',
        #                        self.callback_funcs_for_ask_initial_model_parameter,
        #                        ['initial_model'])

    def join_in(self):
        """
        To send ``join_in`` message to the server for joining in the FL course.
        额外发送本地的个性化模型至client端
        """
        local_init_model = copy.deepcopy(self.model.cpu())
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=[self.local_address, local_init_model]))

    # def callback_funcs_for_ask_initial_model_parameter(self, message: Message):
    #     """
    #     在FedHeNN中，server端需要拥有每个client的本地模型及权重
    #     本函数在FL初始阶段调用，目的在于将client的本地模型发送至server
    #     """
    #     round = message.state
    #     self.state = round
    #     local_init_model = copy.deepcopy(self.model.cpu())
    #     sender = message.sender
    #     logger.info(f'client #{self.ID} send initial personalized model to the serve #{sender}r')
    #     self.comm_manager.send(
    #         Message(msg_type='initial_personalized_model',
    #                 sender=self.ID,
    #                 receiver=[sender],
    #                 state=self.state,
    #                 content=(local_init_model)))


def call_my_worker(method):
    if method == 'fedhenn':
        worker_builder = {'client': FedHeNN_client,
                          'server': FedHeNN_server
                          }
        return worker_builder


register_worker('fedhenn', call_my_worker)
