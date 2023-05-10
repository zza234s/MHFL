import pickle, sys

from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
import numpy as np
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict

logger = logging.getLogger(__name__)


# Build your worker here.


class FedHeNN_server(Server):
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
        #
        RAD = []
        train_data = self.data.train_data
        idx_list = np.random.sample(range(0, len(train_data)), self._cfg.federate.client_num)
        for idx in idx_list:
            RAD.append(train_data[idx][0])
        return RAD

    def _start_new_training_round(self, global_protos):
        self._broadcast_custom_message(msg_type='global_proto', content=global_protos)

    def eval(self):
        self._broadcast_custom_message(msg_type='evaluate', content=None, filter_unseen_clients=False)

    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """
        if self.check_client_join_in():
            model_size = sys.getsizeof(pickle.dumps(self.models[0])) / 1024.0 * 8.
            client_resource = [
                model_size / float(x['communication']) +
                float(x['computation']) / 1000.
                for x in self.client_resource_info
            ] if self.client_resource_info is not None else None

        if self.sampler is None:
            self.sampler = get_sampler(
                sample_strategy=self._cfg.federate.sampler,
                client_num=self.client_num,
                client_info=client_resource)

        # start feature engineering
        self.trigger_for_feat_engr(
            self._broadcast_custom_message, {
                'msg_type': 'model_para',
                'content': self.RAD, #向每个client广播RAD, 即论文Algorithm 2中的 X
                'sample_client_num': self.sample_client_num
            })

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


class FedHeNNClient(Client):
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
        super(FedHeNNClient, self).__init__(ID, server_id, state, config, data, model, device,
                                            strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.global_protos = []
        self.register_handlers('global_proto',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para'])

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        self.trainer.update(content)
        self.state = round

        sample_size, model_para, results, agg_protos = self.trainer.train()

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, agg_protos)))
