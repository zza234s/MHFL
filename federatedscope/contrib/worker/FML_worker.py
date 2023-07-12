# from federatedscope.register import register_worker
# from federatedscope.core.workers import Server, Client
# from federatedscope.core.message import Message
# import logging
# import copy
# from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
#     Timeout, merge_param_dict
# logger = logging.getLogger(__name__)
#
#
# class FMLServer(Server):
#
#     def check_and_move_on(self,
#                           check_eval_result=False,
#                           min_received_num=None):
#         #TODO: 需要完善当采样率不等于0时的实现
#         min_received_num = len(self.comm_manager.get_neighbors().keys())
#
#         if check_eval_result and self._cfg.federate.mode.lower(
#         ) == "standalone":
#             # in evaluation stage and standalone simulation mode, we assume
#             # strong synchronization that receives responses from all clients
#             min_received_num = len(self.comm_manager.get_neighbors().keys())
#
#         move_on_flag = True
#
#         # round or finishing the evaluation
#         if self.check_buffer(self.state, min_received_num, check_eval_result):
#             if not check_eval_result:
#                 # Receiving enough feedback in the training process
#                 # update the shared meme model
#                 #################################################################
#                 local_protos_list = dict()
#                 msg_list = self.msg_buffer['train'][self.state]
#                 aggregated_num = len(msg_list)
#                 for key, values in msg_list.items():
#                     local_protos_list[key] = values[1]
#                 global_protos = self._proto_aggregation(local_protos_list)
#                 #################################################################
#
#                 self.state += 1
#                 if self.state % self._cfg.eval.freq == 0 and self.state != \
#                         self.total_round_num:
#                     #  Evaluate
#                     logger.info(f'Server: Starting evaluation at the end '
#                                 f'of round {self.state - 1}.')
#                     self.eval()
#
#                 if self.state < self.total_round_num:
#                     # Move to next round of training
#                     logger.info(
#                         f'----------- Starting a new training round (Round '
#                         f'#{self.state}) -------------')
#                     # Clean the msg_buffer
#                     self.msg_buffer['train'][self.state - 1].clear()
#                     self.msg_buffer['train'][self.state] = dict()
#                     self.staled_msg_buffer.clear()
#                     # Start a new training round
#                     self._start_new_training_round(global_protos)
#                 else:
#                     # Final Evaluate
#                     logger.info('Server: Training is finished! Starting '
#                                 'evaluation.')
#                     self.eval()
#
#             else:
#                 # Receiving enough feedback in the evaluation process
#                 self._merge_and_format_eval_results()
#                 if self.state >= self.total_round_num:
#                     self.is_finish = True
#
#         else:
#             move_on_flag = False
#
#         return move_on_flag
#
#     def _proto_aggregation(self, local_protos_list):
#         agg_protos_label = dict()
#         for idx in local_protos_list:
#             local_protos = local_protos_list[idx]
#             for label in local_protos.keys():
#                 if label in agg_protos_label:
#                     agg_protos_label[label].append(local_protos[label])
#                 else:
#                     agg_protos_label[label] = [local_protos[label]]
#
#         for [label, proto_list] in agg_protos_label.items():
#             if len(proto_list) > 1:
#                 proto = 0 * proto_list[0].data
#                 for i in proto_list:
#                     proto += i.data
#                 agg_protos_label[label] = [proto / len(proto_list)]
#             else:
#                 agg_protos_label[label] = [proto_list[0].data]
#
#         return agg_protos_label
#
#     def _start_new_training_round(self, global_protos):
#         self._broadcast_custom_message(msg_type='global_proto',content=global_protos)
#
#     def eval(self):
#         self._broadcast_custom_message(msg_type='evaluate',content=None, filter_unseen_clients=False)
#
#     def _broadcast_custom_message(self, msg_type, content,
#                                  sample_client_num=-1,
#                                  filter_unseen_clients=True):
#         if filter_unseen_clients:
#             # to filter out the unseen clients when sampling
#             self.sampler.change_state(self.unseen_clients_id, 'unseen')
#
#         if sample_client_num > 0:
#             receiver = self.sampler.sample(size=sample_client_num)
#         else:
#             # broadcast to all clients
#             receiver = list(self.comm_manager.neighbors.keys())
#             if msg_type == 'model_para':
#                 self.sampler.change_state(receiver, 'working')
#
#         rnd = self.state - 1 if msg_type == 'evaluate' else self.state
#
#         self.comm_manager.send(
#             Message(msg_type=msg_type,
#                     sender=self.ID,
#                     receiver=receiver,
#                     state=min(rnd, self.total_round_num),
#                     timestamp=self.cur_timestamp,
#                     content=content))
#
#         if filter_unseen_clients:
#             # restore the state of the unseen clients within sampler
#             self.sampler.change_state(self.unseen_clients_id, 'seen')
#
#
# class FMLClient(Client):
#     #TODO: test (use global proto)
#     def __init__(self,
#                  ID=-1,
#                  server_id=None,
#                  state=-1,
#                  config=None,
#                  data=None,
#                  model=None,
#                  device='cpu',
#                  strategy=None,
#                  is_unseen_client=False,
#                  *args,
#                  **kwargs):
#         super(FMLClient, self).__init__(ID, server_id, state, config, data, model, device,
#                                              strategy, is_unseen_client, *args, **kwargs)
#
#
#     def callback_funcs_for_model_para(self, message: Message):
#         round = message.state
#         sender = message.sender
#         timestamp = message.timestamp
#         content = message.content
#
#         # When clients share the local model, we must set strict=True to
#         # ensure all the model params (which might be updated by other
#         # clients in the previous local training process) are overwritten
#         # and synchronized with the received model
#         if self._cfg.federate.process_num > 1:
#             for k, v in content.items():
#                 content[k] = v.to(self.device)
#         self.trainer.update(content,
#                             strict=self._cfg.federate.share_local_model)
#         self.state = round
#         skip_train_isolated_or_global_mode = \
#             self.early_stopper.early_stopped and \
#             self._cfg.federate.method in ["local", "global"]
#         if self.is_unseen_client or skip_train_isolated_or_global_mode:
#             # for these cases (1) unseen client (2) isolated_global_mode,
#             # we do not local train and upload local model
#             sample_size, model_para_all, results = \
#                 0, self.trainer.get_model_para(), {}
#             if skip_train_isolated_or_global_mode:
#                 logger.info(
#                     f"[Local/Global mode] Client #{self.ID} has been "
#                     f"early stopped, we will skip the local training")
#                 self._monitor.local_converged()
#         else:
#             if self.early_stopper.early_stopped and \
#                     self._monitor.local_convergence_round == 0:
#                 logger.info(
#                     f"[Normal FL Mode] Client #{self.ID} has been locally "
#                     f"early stopped. "
#                     f"The next FL update may result in negative effect")
#                 self._monitor.local_converged()
#             sample_size, model_para_all, results = self.trainer.train()
#             if self._cfg.federate.share_local_model and not \
#                     self._cfg.federate.online_aggr:
#                 model_para_all = copy.deepcopy(model_para_all)
#             train_log_res = self._monitor.format_eval_res(
#                 results,
#                 rnd=self.state,
#                 role='Client #{}'.format(self.ID),
#                 return_raw=True)
#             logger.info(train_log_res)
#             if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
#                 self._monitor.save_formatted_results(train_log_res,
#                                                      save_file_name="")
#
#         # Return the feedbacks to the server after local update
#         shared_model_para = model_para_all
#         self.comm_manager.send(
#             Message(msg_type='model_para',
#                     sender=self.ID,
#                     receiver=[sender],
#                     state=self.state,
#                     timestamp=self._gen_timestamp(
#                         init_timestamp=timestamp,
#                         instance_number=sample_size),
#                     content=(sample_size, shared_model_para)))
#
#
# def call_my_worker(method):
#     if method == 'FML':
#         worker_builder = {'client': FMLClient, 'server': FMLServer}
#         return worker_builder
#
#
# register_worker('FML', call_my_worker)
