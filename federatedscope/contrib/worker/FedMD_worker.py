import copy
import os
import torch
from torch import nn
from federatedscope.register import register_worker, logger
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
from federatedscope.contrib.common_utils import get_public_dataset, EarlyStopMonitor, train_CV, eval_CV, \
    divide_dataset_epoch
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler, Subset
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

class FedMD_server(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=2,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):
        super(FedMD_server, self).__init__(ID, state, config, data, model, client_num, total_round_num,
                                           device, strategy, unseen_clients_id, **kwargs)
        public_dataset_name = config.MHFL.public_dataset
        public_train, _ = get_public_dataset(public_dataset_name)

        self.selected_sample_per_epochs = divide_dataset_epoch(dataset=public_train,
                                                               epochs=config.federate.total_round_num,
                                                               num_samples_per_epoch=config.fedmd.public_subset_size)

    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """
        if self.check_client_join_in():
            # broadcast the "local_pre_training" message tell clients to pre-train.
            self.comm_manager.send(
                Message(msg_type='local_pre_training',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state,
                        content=self.selected_sample_per_epochs))

    def check_and_move_on(self, check_eval_result=False):
        min_received_num = len(self.comm_manager.get_neighbors().keys())
        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                #################################################################
                logger.info(f'Server: starting aggregation.')
                clients_uploaded_logits = self.msg_buffer['train'][self.state]
                stacked_tensor = torch.stack(list(clients_uploaded_logits.values()))
                aggregate_logits = torch.sum(stacked_tensor, dim=0) / self.client_num
                logger.info(f' Global epoch {self.state} \t server: aggregation completed')

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
                    self.comm_manager.send(
                        Message(msg_type='updated_consensus',
                                sender=self.ID,
                                receiver=list(self.comm_manager.neighbors.keys()),
                                state=self.state,
                                content=aggregate_logits.cpu()))
                    # self._start_new_training_round(aggregated_num)
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

class FedMD_client(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cuda',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(FedMD_client, self).__init__(ID, server_id, state, config, data, model, device,
                                           strategy, is_unseen_client, *args, **kwargs)
        self.register_handlers('local_pre_training',
                               self.callback_funcs_for_local_pre_training,
                               ['model_para'])
        self.register_handlers('updated_consensus', self.callback_funcs_for_updated_consensus,
                               ['class_scores'])

        # self.device = config.device
        self.cfg_MHFL = config.MHFL
        self.public_dataset_name = config.MHFL.public_dataset
        self.task = config.MHFL.task
        self.model_weight_dir = config.MHFL.model_weight_dir
        self.local_update_steps = config.train.local_update_steps

        # TODO: FedMD的cfg和MHFL的cfg有重叠部分，需要统一
        # Local pretraining related settings
        self.rePretrain = config.fedmd.pre_training.rePretrain
        self.public_epochs = config.fedmd.pre_training.public_epochs
        self.private_epochs = config.fedmd.pre_training.private_epochs
        self.public_batch_size = config.fedmd.pre_training.public_batch_size
        self.private_batch_size = config.fedmd.pre_training.private_batch_size
        if self.cfg_MHFL.save_model and not os.path.exists(self.model_weight_dir):
            os.mkdir(self.model_weight_dir)  # 生成保存预训练模型权重所需的文件夹

        # load public dataset for pretraining
        self.pub_train_dataset, self.pub_test_dataset = get_public_dataset(self.public_dataset_name)
        self.pub_train_loader = DataLoader(self.pub_train_dataset, batch_size=self.public_batch_size,
                                           shuffle=True, num_workers=4)
        self.pub_test_loader = DataLoader(self.pub_test_dataset, batch_size=self.public_batch_size, shuffle=False,
                                          num_workers=4)

        # Federated training related settings
        self.fed_batch_size = config.dataloader.batch_size

        # For Digest
        self.digest_epochs = config.fedmd.digest_epochs

        # define additional local trainer for pretraining
        local_cfg = config.clone()
        local_cfg.defrost()
        local_cfg.train.local_update_steps = 1
        local_cfg.freeze()
        self.trainer_local_pretrain = get_trainer(model=model,
                                                  data=data,
                                                  device=device,
                                                  config=local_cfg,
                                                  is_attacker=self.is_attacker,
                                                  monitor=self._monitor)

    # pre_train
    def callback_funcs_for_local_pre_training(self, message: Message):
        round = message.state
        self.state = round
        self.selected_sample_per_epochs = message.content

        model_file = os.path.join(self.model_weight_dir,
                                  'FedMD_' + self.public_dataset_name + '_client_' + str(self.ID) + '.pth')
        if os.path.exists(model_file) and not self.rePretrain:
            # 如果已经存在预训练好的模型权重并且不要求重新预训练
            self.model.load_state_dict(torch.load(model_file, self.device))
            eval_metrics = self.trainer.evaluate(target_data_split_name='test')
            logger.info(
                f"Load the pretrained model weight."
                f"The accuracy of the pretrained model on the local test dataset is {eval_metrics['test_acc']}")
        else:
            logger.info(f'Client#{self.ID}: training on th public dataset {self.public_dataset_name}')
            self.model.to(self.device)
            self._pretrain_on_public_datset()
            self._pretrain_on_private_datset(model_file)

        # send class_scores (i.e., logits) on the public dataset to server
        train_dataset = Subset(self.pub_train_dataset, list(self.selected_sample_per_epochs[0]))
        train_loader = DataLoader(train_dataset, batch_size=self.fed_batch_size, shuffle=False)
        class_scores = self.get_class_scores(train_loader)
        self.pre_train_loader = train_loader

        self.comm_manager.send(
            Message(
                msg_type='model_para',
                sender=self.ID,
                receiver=[0],
                state=self.state,
                content=class_scores.clone().detach()
            )
        )

    def _pretrain_on_public_datset(self):
        self.model.to(self.device)
        optimizer = get_optimizer(model=self.model, **self.cfg_MHFL.public_train.optimizer)
        criterion = nn.CrossEntropyLoss()
        train_loader, test_loader = self.pub_train_loader, self.pub_test_loader
        early_stopper = EarlyStopMonitor(max_round=self._cfg.early_stop.patience, higher_better=True)
        for epoch in range(self.public_epochs):
            train_loss = train_CV(model=self.model, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
                                  device=self.device, client_id=self.ID, epoch=epoch)
            if epoch % 10 == 0:
                test_loss, test_acc = eval_CV(model=self.model, criterion=criterion, test_loader=test_loader,
                                              device=self.device,
                                              client_id=self.ID, epoch=epoch)

                if early_stopper.early_stop_check(test_acc):
                    logger.info(f'client#{self.ID}: No improvment over {early_stopper.max_round} epochs.'
                                f'Stop pretraining on public dataset')
                    break
        self.model.to('cpu')

    def _pretrain_on_private_datset(self, model_file=None):
        logger.info(f'Client#{self.ID}: train to convergence on the private datasets')
        early_stopper = EarlyStopMonitor(max_round=self._cfg.early_stop.patience, higher_better=True)
        for epoch in range(self.private_epochs):
            sample_size, model_para, results = self.trainer_local_pretrain.train()
            eval_metrics = self.trainer_local_pretrain.evaluate(target_data_split_name='test')
            logger.info(
                f"epoch:{epoch}\t"
                f"train_acc:{results['train_acc']}\t "
                f"test_acc:{eval_metrics['test_acc']} ")

            early_stop_now, update_best_this_round = early_stopper.early_stop_check(eval_metrics['test_acc'])

            if update_best_this_round and model_file is not None:
                torch.save(self.model.state_dict(), model_file)
                logger.info(
                    f'client#{self.ID}: save the pre-trained model weight with the test_acc {early_stopper.last_best}')
            if early_stop_now:
                logger.info(f'No improvment over {early_stopper.max_round} epochs, stop training')
                logger.info(f"the best epoch is {early_stopper.best_epoch},test_acc: {early_stopper.last_best}")
                break

    @torch.no_grad()
    def get_class_scores(self, trainloader):
        temp_tensor_list = []
        self.model.to(self.device)
        self.model.eval()
        for batch_idx, (images, _) in enumerate(trainloader):
            images = images.to(self.device)
            outputs = self.model(images)
            temp_tensor_list.append(outputs)
        class_scores = torch.cat(temp_tensor_list)
        return class_scores

    # deal with updated consensus from server: digest, revist and send new updated_consensus
    def callback_funcs_for_updated_consensus(self, message: Message):
        self.state = message.state
        avg_logits = message.content

        # Digest: Each party trains its model fk to approach the consensus ˜f on the public dataset D0
        sliced_avg_logits = avg_logits.split(self.fed_batch_size, dim=0)
        optimizer = get_optimizer(model=self.model, **self._cfg.train.optimizer)
        criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.digest_epochs):
            train_loss = 0
            for batch_idx, (images, _) in enumerate(self.pre_train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)
                label = sliced_avg_logits[batch_idx].to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss = train_loss / len(self.pre_train_loader.dataset)
            logger.info(f'Client#{self.ID}\t federated epoch:{self.state}\t Local epoch:{epoch}\t Loss: {train_loss}')

        # Revisit: Each party trains its model fk on its own private data for a few epochs.
        sample_size, model_para, results = self.trainer.train()
        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)

        # get class_scores (i.e., logits) and send it to the server
        new_train_dataset = Subset(self.pub_train_dataset, list(self.selected_sample_per_epochs[self.state]))
        train_loader = DataLoader(new_train_dataset, batch_size=self.fed_batch_size, shuffle=False)
        class_scores = self.get_class_scores(train_loader)
        self.pre_train_loader = train_loader

        self.comm_manager.send(
            Message(
                msg_type='model_para',
                sender=self.ID,
                receiver=[0],
                state=self.state,
                content=class_scores
            )
        )


def call_my_worker(method):
    if method == 'fedmd':
        worker_builder = {'client': FedMD_client, 'server': FedMD_server}
        return worker_builder


register_worker('fedmd', call_my_worker)
