from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
from torch.utils.data import Dataset, DataLoader
from federatedscope.contrib.common_utils import get_public_dataset, EarlyStopMonitor, train_CV, eval_CV
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
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

    def callback_funcs_for_local_pre_training(self, message: Message):
        round = message.state
        content = message.content
        cfg_MHFL = self._cfg.MHFL
        device = self._cfg.device

        dataset = cfg_MHFL.public_dataset
        task = cfg_MHFL.task
        # epochs = cfg_MHFL.public_train.epochs
        epochs = 1
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
        for epoch in tqdm(range(epochs)):
            train_loss = train_CV(model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
                                  device=device, client_id=self.ID, epoch=epoch)
            val_loss, acc = eval_CV(model=model, criterion=criterion, test_loader=test_loader, device=device,
                                    client_id=self.ID, epoch=epoch)
            logger.info(
                f'client{self.ID}#存储在公共数据集上预训练的模型至./model_weight/{dataset}_model_{self.ID}.pt')  # TODO:提示变成英语
            torch.save(model.state_dict(), f'{model_dir}/{dataset}_model_{self.ID}.pt')



            # if early_stopper.early_stop_check(val_loss):
            #     logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            #     logger.info(f'the best epoch is {early_stopper.best_epoch}')
            #     torch.save(model.state_dict(), f'{model_dir}/{dataset}_model_{self.ID}.pt')
            #     logger.info(
            #         f'client{self.ID}#存储在公共数据集上预训练的模型至./model_weight/{dataset}_model_{self.ID}.pt')  # TODO:提示变成英语
            #     break

        # TODO: 释放GPU？
        # TODO:  Train to convergence on femnist
        for epoch in tqdm(range(epochs)):
            sample_size, model_para, results = self.trainer.train()

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


def call_my_worker(method):
    if method == 'fsfl':
        worker_builder = {'client': FSFL_Client, 'server': FSFL_Server}
        return worker_builder


register_worker('fsfl', call_my_worker)
