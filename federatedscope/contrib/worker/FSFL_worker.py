from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler, Subset
# from torch.utils.data.dataset import Subset
from federatedscope.contrib.common_utils import get_public_dataset, EarlyStopMonitor, train_CV, eval_CV
from federatedscope.contrib.model.FSFL_DomainIdentifier_CV import DomainIdentifier
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.model_heterogeneity.methods.FSFL.fsfl_utils import DomainDataset, divide_dataset_epoch
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict,param2tensor
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Build your worker here.
class FSFL_Server(Server):
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
        super(FSFL_Server, self).__init__(ID, state, config, data, model, client_num, total_round_num,
                                          device, strategy, unseen_clients_id, **kwargs)

        self.DI_model = DomainIdentifier()
        r"""
        FSFL+ consists of three of training stages.
        Stage1: local pre-train for the private model using public and private dataset .
        Stage2: rounds 0 to DI_epochs, federated training for domain identifier.
        Stage3: -> 2 * fedgen_epoch + total_round_num: federated training
        for GraphSAGE Classifier
        """

    def check_and_move_on(self, check_eval_result=False):
        client_IDs = [i for i in range(1, self.client_num + 1)]

        if check_eval_result:
            # all clients are participating in evaluation
            minimal_number = self.client_num
        else:
            # sampled clients are participating in training
            minimal_number = self.sample_client_num

        move_on_flag = True  # To record whether moving to a new training
        if self.check_buffer(self.state, minimal_number,
                             check_eval_result) and self.state < self._cfg.fsfl.domain_identifier_epochs:
            "Domain identitor weights aggregating procedure"
            aggregated_num = self._perform_federated_aggregation_for_DI()  # 完成DI的聚合
            self.state += 1
            self.broadcast_model_para(msg_type='DI_model_para', model_name='domain_identitor')

        if self.check_buffer(
                self.state, self.client_num
        ) and self.state == self._cfg.fedsageplus.fedgen_epoch:
            self.state += 1
            # Setup Clf_trainer for each client
            self.comm_manager.send(
                Message(msg_type='setup',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state))

        if self.check_buffer(self.state, minimal_number, check_eval_result
                             ) and self.state >= self._cfg.fedsageplus.fedgen_epoch:

            if not check_eval_result:  # in the training process
                aggregated_num = self._perform_federated_aggregation()
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
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:  # in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True
        else:
            move_on_flag = False
        return move_on_flag

    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """
        if self.check_client_join_in():
            # get sampler
            if self.sampler is None:
                self.sampler = get_sampler(
                    sample_strategy=self._cfg.federate.sampler,
                    client_num=self.client_num,
                    client_info=None)  # TODO: client_resource改成None了，需要研究其含义

            # TODO:考虑仅向部分客户端广播的情况
            self.comm_manager.send(
                Message(msg_type='local_pre_training',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state))

    def _perform_federated_aggregation_for_DI(self):
        """
        Perform federated aggregation and update the domain identifier
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]

        model = self.DI_model
        aggregator = self.aggregators[0]
        msg_list = list()
        staleness = list()

        for client_id in train_msg_buffer.keys():
            msg_list.append(train_msg_buffer[client_id])

            # The staleness of the messages in train_msg_buffer
            # should be 0
            staleness.append((client_id, 0))

        # Trigger the monitor here (for training)
        self._monitor.calc_model_metric(model.state_dict(),
                                        msg_list,
                                        rnd=self.state)  # TODO:待弄懂
        # Aggregate
        aggregated_num = len(msg_list)
        agg_info = {
            'client_feedback': msg_list,
            'recover_fun': self.recover_fun,
            'staleness': staleness,
        }
        # logger.info(f'The staleness is {staleness}')
        result = aggregator.aggregate(agg_info)
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(model.state_dict().copy(), result)
        model.load_state_dict(merged_param, strict=False)

        return aggregated_num

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True,
                             model_name=None):
        """
            重写broadcast_model_para
        """
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

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        ########################################################################
        elif model_name is not None:
            if model_name in 'domain_identitor':
                model_para = {} if skip_broadcast else self.DI_model.state_dict()
        ########################################################################
        else:
            model_para = {} if skip_broadcast else self.models[0].state_dict()

        # quantization
        if msg_type == 'model_para' and not skip_broadcast and \
                self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_quantization
            nbits = self._cfg.quantization.nbits
            if self.model_num > 1:
                model_para = [
                    symmetric_uniform_quantization(x, nbits)
                    for x in model_para
                ]
            else:
                model_para = symmetric_uniform_quantization(model_para, nbits)

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class FSFL_Client(Client):
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
        super(FSFL_Client, self).__init__(ID, server_id, state, config, data, model, device,
                                          strategy, is_unseen_client, *args, **kwargs)
        self.register_handlers('local_pre_training',
                               self.callback_funcs_for_local_pre_training,
                               ['model_para', 'ss_model_para'])
        self.register_handlers('DI_model_para',
                               self.callback_funcs_for_DI_model_para, ['model_para'])

        self.task = self._cfg.MHFL.task
        # DomianIdentifier相关变量:

        self.DI_epoch = 0  # 从0开始计数
        self.DI_model = DomainIdentifier()
        self.num_DI_epoch = self._cfg.fsfl.domain_identifier_epochs

        # TODO: 生成domain_alignment所需的dataset
        pulic_dataset_name = self._cfg.MHFL.public_dataset
        self.public_train_data, _ = get_public_dataset(pulic_dataset_name)
        private_length = len(self.data.train_data)
        self.DI_dict_epoch = divide_dataset_epoch(dataset=self.public_train_data, epochs=self.num_DI_epoch,
                                                  num_samples_per_epoch=private_length)

    def callback_funcs_for_local_pre_training(self, message: Message):
        round = message.state
        content = message.content
        self.state = round

        cfg_MHFL = self._cfg.MHFL
        device = self._cfg.device
        epochs = self._cfg.fsfl.pre_training_epochs

        dataset = cfg_MHFL.public_dataset
        task = cfg_MHFL.task
        model_dir = cfg_MHFL.model_weight_dir
        train_batch_size = cfg_MHFL.public_train.batch_size
        test_batch_size = cfg_MHFL.public_train.batch_size

        if task == 'CV':
            train_data, test_data = get_public_dataset(dataset)
            train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

        # train on public dataset
        model = self.model.to(device)
        optimizer = get_optimizer(model=model, **cfg_MHFL.public_train.optimizer)
        criterion = nn.NLLLoss().to(device)

        if not os.path.exists(model_dir):
            logger.info(f'文件夹{model_dir} 不存在，创建！')
            os.mkdir(model_dir)

        logger.info(f'Client#{self.ID}: training on th public dataset {dataset}')
        early_stopper = EarlyStopMonitor(higher_better=False)
        for epoch in tqdm(range(1)):
            train_loss = train_CV(model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
                                  device=device, client_id=self.ID, epoch=epoch)
            val_loss, acc = eval_CV(model=model, criterion=criterion, test_loader=test_loader, device=device,
                                    client_id=self.ID, epoch=epoch)
            if early_stopper.early_stop_check(val_loss):
                logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'the best epoch is {early_stopper.best_epoch}')
                torch.save(model.state_dict(), f'{model_dir}/{dataset}_model_{self.ID}.pt')
                logger.info(
                    f'client{self.ID}#存储在公共数据集上预训练的模型至./model_weight/{dataset}_model_{self.ID}.pt')  # TODO:提示变成英语
                break

        early_stopper.reset()  # 重置early_stopper的成员变量
        # TODO: 释放GPU？
        # TODO:  Train to convergence on femnist
        for epoch in tqdm(range(epochs)):
            sample_size, model_para, results = self.trainer.train()  # TODO:模型的output_channels明显大于 public_dataset的output_channels
            eval_metrics = self.trainer.evaluate(target_data_split_name='val')
            logger.info(f'client:{self.ID} private dataset pre training')
            logger.info(f"epoch:{epoch}\t"
                        f"train_loss:{results['train_loss']} \t train_acc:{results['train_acc']}\t "
                        f"val_loss:{eval_metrics['val_avg_loss']}\t val_acc:{eval_metrics['val_acc']} ")
            if early_stopper.early_stop_check(eval_metrics['val_avg_loss']):
                logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'the best epoch is {early_stopper.best_epoch}')
                torch.save(model.state_dict(), f'{model_dir}/{dataset}_private_model_{self.ID}.pt')
                logger.info(
                    f'client{self.ID}#存储在公共数据集上预训练的模型至./model_weight/{dataset}_private_model_{self.ID}.pt')  # TODO:提示变成英语
                break

        self.DI_training()

    def DI_training(self):
        """
            Train the Domain Identifier locally and send its updated weights to the server vis a message.
        """
        self.DI_model = self.DI_model.to(self.device)
        epoch = self.state
        self.DI_epoch += 1
        public_indices = list(self.DI_dict_epoch[epoch])
        batch_size = self._cfg.fsfl.domain_identifier_batch_size
        traindataset = DomainDataset(publicadataset=Subset(self.public_train_data, public_indices),
                                     privatedataset=self.data.train_data,
                                     localindex=self.ID,
                                     step1=True)
        trainloader = DataLoader(traindataset, batch_size=batch_size, num_workers=0, shuffle=True)

        "定义loss, optimizer"
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model=self.DI_model, **self._cfg.fsfl.DI_optimizer)

        """
        Start training
        """
        batch_loss = []
        model = self.model.to(self.device)
        for batch_idx, (images, domain_labels) in enumerate(tqdm(trainloader)):
            images, domain_labels = images.to(self.device), domain_labels.to(self.device)
            optimizer.zero_grad()
            temp_outputs = model(images, True)
            domain_outputs = self.DI_model(temp_outputs, self.ID)
            loss = criterion(domain_outputs, domain_labels)
            loss.backward()
            optimizer.step()
            logger.info(f'Gan Step1 on Model {self.ID} DI Epoch: {self.DI_epoch} Loss: {loss}')
            batch_loss.append(loss)
            # TODO:下面两行代码在变成pytorch时是否需要特殊处理？
            # weights = mindspore.ParameterTuple(optimizer.parameters)
            # grad = ops.GradOperation(get_by_list=True)

        """
        Send weights to the server
        """
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[0],
                    state=self.state,
                    content=(len(traindataset), self.DI_model.cpu().state_dict())))

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
    def callback_funcs_for_DI_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        model_parameters = message.content

        #update local weights
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(self.DI_model.state_dict().copy(),
                                        model_parameters)
        self.DI_model.load_state_dict(merged_param, strict=True)

        self.state = round

        self.DI_training()


def call_my_worker(method):
    if method == 'fsfl':
        worker_builder = {'client': FSFL_Client, 'server': FSFL_Server}
        return worker_builder


register_worker('fsfl', call_my_worker)
