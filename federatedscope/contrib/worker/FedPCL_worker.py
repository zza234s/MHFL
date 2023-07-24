from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
from federatedscope.contrib.model.FedPCL_resnet18 import resnet18
import logging
import copy
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict

logger = logging.getLogger(__name__)


class FedPCL_Server(Server):
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
        super(FedPCL_Server, self).__init__(ID, state, config, data, model, client_num,
                                            total_round_num, device, strategy, **kwargs)
        self.received_protos_dict = dict()

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
                    #################################################################
                    # update global protos
                    local_protos_dict = self.received_protos_dict
                    global_avg_protos = self._proto_aggregation(local_protos_dict)
                    global_protos = copy.deepcopy(local_protos_dict)
                    self._broadcast_custom_message(msg_type='global_proto',
                                                   content=[global_avg_protos, global_protos])
                    self.broadcast_model_para(msg_type='model_para',
                                              sample_client_num=self.sample_client_num)
                    #################################################################
                    # self._start_new_training_round(global_protos)
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

    def _proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = proto / len(proto_list)
            else:
                agg_protos_label[label] = proto_list[0].data

        return agg_protos_label

    # def _start_new_training_round(self, global_protos):
    #     self._broadcast_custom_message(msg_type='global_proto',content=global_protos)

    def eval(self):
        self._broadcast_custom_message(msg_type='evaluate', content=None, filter_unseen_clients=False)

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

    def callback_funcs_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, which triggers \
        ``check_and_move_on`` (perform aggregation when enough feedback has \
        been received). This handling function is widely used in various FL \
        courses.
        Arguments:
            message: The received message.
        """
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')

        # update the currency timestamp according to the received message
        assert timestamp >= self.cur_timestamp  # for test
        self.cur_timestamp = timestamp

        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.msg_buffer['train'][round][sender] = content[:2]
            ####################################################
            self.received_protos_dict[sender] = content[2]
            ####################################################
        elif round >= self.state - self.staleness_toleration:
            # Save the staled messages
            #########################################
            # self.staled_msg_buffer.append((round, sender, content))
            self.staled_msg_buffer.append((round, sender, content[:2]))
            ##########################################
        else:
            # Drop the out-of-date messages
            logger.info(f'Drop a out-of-date message from round #{round}')
            self.dropout_num += 1

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(content)

        move_on_flag = self.check_and_move_on()
        if self._cfg.asyn.use and self._cfg.asyn.broadcast_manner == \
                'after_receiving':
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=1)

        return move_on_flag


class FedPCL_Client(Client):
    # TODO: test (use global proto)
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
        super(FedPCL_Client, self).__init__(ID, server_id, state, config, data, model, device,
                                            strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.global_protos = []
        self.trainer.ctx.client_ID = self.ID
        self.register_handlers('global_proto',
                               self.callback_funcs_for_global_proto,
                               ['model_para'])
        if config.fedpcl.debug:
            # load backbone
            resnet_quickdraw = resnet18(pretrained=True, ds='quickdraw',
                                        pretrain_weight_dir=config.model.fedpcl.model_weight_dir)
            self.trainer.ctx.backbone_list = [resnet_quickdraw]
            for backbone in self.trainer.ctx.backbone_list:
                backbone.to(config.device)
                backbone.eval()
    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.state = round
        # self.trainer.update(content, strict=self._cfg.federate.share_local_model)
        self.trainer.ctx.cur_state = self.state

        sample_size, model_para, results, agg_protos = self.trainer.train()
        model_para = copy.deepcopy(model_para)

        logger.info(
            self._monitor.format_eval_res(results, rnd=self.state,
                                          role='Client #{}'.format(self.ID),
                                          return_raw=True
                                          )
        )
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para, agg_protos)))

    def callback_funcs_for_global_proto(self, message: Message):
        # logger.info(f'client #{self.ID}, round:{state}: 接收到全局原型')
        self.trainer.ctx.global_avg_protos = message.content[0]
        self.trainer.ctx.global_protos = message.content[1]


def call_my_worker(method):
    if method == 'fedpcl':
        worker_builder = {'client': FedPCL_Client, 'server': FedPCL_Server}
        return worker_builder


register_worker('fedpcl', call_my_worker)
