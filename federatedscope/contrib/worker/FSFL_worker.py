from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
from federatedscope.contrib.common_utils import get_public_dataset, EarlyStopMonitor, train_CV, eval_CV
from federatedscope.contrib.model.FSFL_DomainIdentifier_CV import DomainIdentifier
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict, param2tensor

from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler, Subset
from tqdm import tqdm

import torch
import torch.nn as nn
import os
import logging

logger = logging.getLogger(__name__)

from federatedscope.model_heterogeneity.methods.FSFL.fsfl_utils import DomainDataset, divide_dataset_epoch, \
    DigestDataset


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
        r"""
            FSFL+ consists of three of training stages.
            Stage1: local pre-train for the private model using public and private dataset .
            Stage2: rounds 0 to DI_epochs, federated training for domain identifier.
            Stage3: -> 2 * fedgen_epoch + total_round_num: federated training
            for GraphSAGE Classifier
        """
        self.FSFL_cfg = config.fsfl

        self.DI_model = DomainIdentifier()

        self.DI_epochs = self.FSFL_cfg.domain_identifier_epochs
        self.collaborative_epoch = self.FSFL_cfg.collaborative_epoch
        self.fed_epoch = self.DI_epochs + self.collaborative_epoch

        self.register_handlers('finish_LEA',
                               self.callback_funcs_model_para)
        self.register_handlers('logits',
                               self.callback_funcs_model_para)

    def check_and_move_on(self, check_eval_result=False):
        # 与源代码一样：假定所有客户端都参与训练，不考虑每轮只采样部分客户端的情况
        move_on_flag = True  # To record whether moving to a new training
        if self.check_buffer(self.state, self.client_num,
                             check_eval_result) and self.state < self.DI_epochs:
            "Domain identitor weights aggregating procedure"
            aggregated_num = self._perform_federated_aggregation_for_DI()  # 完成DI的聚合
            self.state += 1
            self.broadcast_model_para(msg_type='DI_model_para', model_name='domain_identitor')

        if self.check_buffer(self.state, self.client_num) and self.state == self.DI_epochs:
            # Setup for model agnostic federated learning
            self.state += 1
            dataset_name = self._cfg.MHFL.public_dataset
            public_train_data, _ = get_public_dataset(dataset_name)
            MAFL_epoch_dict = divide_dataset_epoch(dataset=public_train_data, epochs=self.FSFL_cfg.collaborative_epoch,
                                                   num_samples_per_epoch=5000)
            self.comm_manager.send(
                Message(msg_type='setup',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state,
                        content=MAFL_epoch_dict
                        )
            )
        if self.check_buffer(self.state,
                             self.client_num) and self.state > self.DI_epochs and self.state < self.fed_epoch:
            # start collaborative training for model agnostic federated learning
            logist_dict = self.msg_buffer['train'][self.state]
            stacked_tensors = torch.stack(list(logist_dict.values()))
            avg_logits = stacked_tensors.mean(dim=0)
            self.comm_manager.send(
                Message(msg_type='avg_logits',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state,
                        content=avg_logits))
            self.state += 1

        if self.check_buffer(self.state, self.client_num) and self.state >= self.fed_epoch:
            if not check_eval_result:  # in the training process
                # Final Evaluate
                logger.info('Server: Training is finished! Starting evaluation.')
                self.comm_manager.send(
                    Message(msg_type='evaluate',
                            sender=self.ID,
                            receiver=list(self.comm_manager.neighbors.keys()),
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=None))
            else:  # in the evaluation process
                self._merge_and_format_eval_results()
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
        # 注册hook
        self.register_handlers('local_pre_training',
                               self.callback_funcs_for_local_pre_training,
                               ['model_para', 'ss_model_para'])
        self.register_handlers('DI_model_para',
                               self.callback_funcs_for_DI_model_para, ['model_para'])
        self.register_handlers('setup',
                               self.callback_funcs_for_setup, ['logits'])
        self.register_handlers('avg_logits', self.callback_funcs_avg_logits, ['logits'])
        self.task = self._cfg.MHFL.task

        # public dataset
        pulic_dataset_name = self._cfg.MHFL.public_dataset
        self.public_train_data, _ = get_public_dataset(pulic_dataset_name)
        private_length = len(self.data.train_data)

        # DomianIdentifier相关变量:
        self.DI_model = DomainIdentifier()
        self.DI_batch_size = self._cfg.fsfl.domain_identifier_batch_size
        self.num_DI_epoch = self._cfg.fsfl.domain_identifier_epochs
        self.DI_dict_epoch = divide_dataset_epoch(dataset=self.public_train_data, epochs=self.num_DI_epoch,
                                                  num_samples_per_epoch=private_length)

        # MAFL-模型异构联邦学习用到的成员变量
        self.MAFL_epoch_cnt = 0
        self.L1loss = nn.L1Loss(reduction='mean')

    def callback_funcs_for_local_pre_training(self, message: Message):
        round = message.state
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
        self.model.to(device)
        optimizer = get_optimizer(model=self.model, **cfg_MHFL.public_train.optimizer)
        criterion = nn.NLLLoss().to(device)

        if cfg_MHFL.save_pretraining_model and not os.path.exists(model_dir):
            logger.info(f'create {model_dir} folder！')
            os.mkdir(model_dir)

        logger.info(f'Client#{self.ID}: training on th public dataset {dataset}')
        early_stopper = EarlyStopMonitor(higher_better=False)
        for epoch in tqdm(range(1)):
            train_loss = train_CV(model=self.model, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
                                  device=device, client_id=self.ID, epoch=epoch)
            val_loss, acc = eval_CV(model=self.model, criterion=criterion, test_loader=test_loader, device=device,
                                    client_id=self.ID, epoch=epoch)

            if early_stopper.early_stop_check(val_loss):
                if cfg_MHFL.save_pretraining_model:
                    torch.save(self.model.state_dict(), f'{model_dir}/{dataset}_model_{self.ID}.pt')
                    logger.info(f'No improvment over {early_stopper.max_round} epochs, stop training')
                    logger.info(
                        f'client#{self.ID}： save pre-training model to ./model_weight/{dataset}_model_{self.ID}.pt')
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
                torch.save(self.model.state_dict(), f'{model_dir}/{dataset}_private_model_{self.ID}.pt')
                logger.info(
                    f'client{self.ID}#存储在公共数据集上预训练的模型至./model_weight/{dataset}_private_model_{self.ID}.pt')  # TODO:提示变成英语
                break

        self.DI_training()

    def DI_training_step_two(self):
        gan_local_epochs = self._cfg.fsfl.gan_local_epochs
        self.model = self.model.to(self.device)
        self.DI_model = self.DI_model.to(self.device)
        for epoch in range(gan_local_epochs):
            local_losses = []
            "Create dataset"
            public_indices = list(self.DI_dict_epoch[epoch])
            traindataset = DomainDataset(publicadataset=Subset(self.public_train_data, public_indices),
                                         privatedataset=self.data.train_data,
                                         localindex=self.ID,
                                         step1=False)  # stel1为True代表公共数据集标签为0，私有数据集标签为client ID；为False则相反
            trainloader = DataLoader(traindataset, batch_size=self.DI_batch_size, num_workers=0, shuffle=True)

            "Define loss function and optimizer"
            criterion = nn.CrossEntropyLoss()  # TODO:确保和源代码损失一致
            optimizer = get_optimizer(model=self.model, **self._cfg.fsfl.DI_optimizer_step_2)

            # criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            # optimizer = nn.Adam(params=model.trainable_params(), learning_rate=0.0001, weight_decay=1e-4)

            """
            Start training
            """
            batch_loss = []
            self.DI_model.train()
            for batch_idx, (images, domain_labels) in enumerate(tqdm(trainloader)):
                images, domain_labels = images.to(self.device), domain_labels.to(self.device)
                optimizer.zero_grad()
                temp_outputs = self.model(images, True)  # TODO: self.model 和 self.ctx.model里是一个东西吗，会同步吗？
                domain_outputs = self.DI_model(temp_outputs, self.ID)
                loss = criterion(domain_outputs, domain_labels)
                loss.backward()
                optimizer.step()
                logger.info(f'Gan Step2 on Model {self.ID} Train Epoch: {epoch} Loss: {loss}')
                batch_loss.append(loss)
        logger.info(f'client:{self.ID} DI training step2 finish')

    def DI_training(self):
        """
            Train the Domain Identifier locally and send its updated weights to the server vis a message.
        """
        self.DI_model = self.DI_model.to(self.device)
        epoch = self.state
        public_indices = list(self.DI_dict_epoch[epoch])
        batch_size = self._cfg.fsfl.domain_identifier_batch_size
        traindataset = DomainDataset(publicadataset=Subset(self.public_train_data, public_indices),
                                     privatedataset=self.data.train_data,
                                     localindex=self.ID,
                                     step1=True)
        trainloader = DataLoader(traindataset, batch_size=batch_size, num_workers=0, shuffle=True)

        "定义loss, optimizer"
        criterion = nn.CrossEntropyLoss()  # TODO:确保和源代码损失一致
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
            logger.info(f'Gan Step1 on Model {self.ID} DI Epoch: {epoch} Loss: {loss}')
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

    def callback_funcs_for_DI_model_para(self, message: Message):
        round = message.state
        model_parameters = message.content

        # update local weights
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(self.DI_model.state_dict().copy(),
                                        model_parameters)
        self.DI_model.load_state_dict(merged_param, strict=True)

        self.state = round
        if self.state < self.num_DI_epoch:
            self.DI_training()
        else:
            self.DI_training_step_two()
            # 发送message给server，告诉他Latent Embedding Adaptation的步骤已经结束了
            self.comm_manager.send(
                Message(msg_type='finish_LEA',
                        sender=self.ID,
                        receiver=[0],
                        state=self.state)
            )

    def callback_funcs_for_setup(self, message: Message):
        round, sender, MAFL_epoch_dict = message.state, message.sender, message.content
        self.state = round
        self.MAFL_epoch_dict = MAFL_epoch_dict
        logits = self.calculate_logits(self.MAFL_epoch_cnt)  # 这里self.MAFL_epoch应为0

        self.comm_manager.send(
            Message(msg_type='logits',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=logits)
        )

    def callback_funcs_avg_logits(self, message: Message):
        round, sender, avg_logits = message.state, message.sender, message.content
        epoch = self.MAFL_epoch_cnt

        # 取出之前计算logits的数据集
        original_dataset = Subset(self.public_train_data, list(self.MAFL_epoch_dict[epoch]))
        traindataset = DigestDataset(original_dataset, new_labels=avg_logits)
        trainloader = DataLoader(traindataset, batch_size=self._cfg.fsfl.MAFL_batch_size,
                                 shuffle=False)
        self.model.to(self.device)
        optimizer = get_optimizer(model=self.model, **self._cfg.train.optimizer)

        # TODO: 验证每一个baatch拿到的数据是否和 self.calculate_logits的一致
        for batch_idx, (images, labels) in enumerate(tqdm(trainloader)):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.L1loss(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                logger.info(
                    'Collaborative traing : Local Model {} Train Epoch: {} Loss: {}'.format(self.ID, epoch + 1, loss))

        # 计算新一轮的logits
        self.MAFL_epoch_cnt += 1
        logits = self.calculate_logits(self.MAFL_epoch_cnt)
        self.comm_manager.send(
            Message(msg_type='logits',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=logits)
        )

    @torch.no_grad()
    def calculate_logits(self, epoch):
        self.model.to(self.device)

        train_dataset = Subset(self.public_train_data, list(self.MAFL_epoch_dict[epoch]))
        trainloader = DataLoader(train_dataset, batch_size=self._cfg.fsfl.MAFL_batch_size,
                                 shuffle=False)  # 为了对齐每个client算出来logits，设置shuffle为False

        logist_list = []
        for batch_idx, (images, _) in enumerate(tqdm(trainloader)):
            if batch_idx==1:
                self.temp= images.copy()
            images = images.to(self.device)
            outputs = self.model(images)
            logist_list.append(outputs.cpu())
        logists = torch.cat(logist_list)

        # 释放GPU
        self.model.to('cpu')

        return logists


def call_my_worker(method):
    if method == 'fsfl':
        worker_builder = {'client': FSFL_Client, 'server': FSFL_Server}
        return worker_builder


register_worker('fsfl', call_my_worker)
