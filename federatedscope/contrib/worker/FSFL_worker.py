from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
from torch.utils.data import Dataset, DataLoader
from federatedscope.contrib.common_utils import get_public_dataset
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
import torch
import torch.nn as nn
import numpy as np
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
                    client_info=None) #TODO: client_resource改成None了，需要研究其含义

            #TODO:考虑仅向部分客户端广播的情况
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
        epochs = cfg_MHFL.public_train.epochs
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

        logger.info(f'Client#{self.ID}: training on th public dataset {dataset}')
        for epoch in range(epochs):
            train_batch_losses = []
            model.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)#TODO:模型输出维度和标签对不上
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    logger.info(f'Local Model {self.ID}\t Train Epoch: {epoch + 1} \t Loss: {loss.item()}')
                train_batch_losses.append(loss.item())

            #评估阶段
            model.eval()
            val_loss = 0
            gt_labels = []
            pred_labels = []
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, 1)
                    gt_labels.append(labels.cpu().data.numpy())
                    pred_labels.append(preds.cpu().data.numpy())
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
            val_loss = val_loss / len(test_loader.dataset)
            gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels) #TODO:查看federateddscope里的计算精度的代码
            acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
            logger.info(f'Epoch: {epoch} \tValidation Loss: {val_loss}, Accuracy: {acc}')


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
