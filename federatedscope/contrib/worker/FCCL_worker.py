from federatedscope.core.message import Message
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import numpy as np
import torch.nn as nn
import copy
import torch.optim as optim
import torch
import logging
import os

from torch.utils.data import DataLoader,SubsetRandomSampler
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm


from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client


from federatedscope.model_heterogeneity.methods.FCCL.datasets.cifar100 import MyCifar100
# from federatedscope.model_heterogeneity.methods.FCCL.datasets import get_prive_dataset, get_public_dataset
# from federatedscope.model_heterogeneity.methods.FCCL.utils.conf import data_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Build your worker here.
class FCCLServer(Server):
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
        super(FCCLServer, self).__init__(ID, state, config, data, model, client_num, total_round_num,
                                             device, strategy, unseen_clients_id, **kwargs)
        self.client_models = dict()  # 用于放客户端模型
        self.selected_transform = None
        self.train_dataset = None
        # 选transform
        if self._cfg.MHFL.pub_aug == 'weak':
            self.selected_transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4802, 0.4480, 0.3975),
                                      (0.2770, 0.2691, 0.2821))])
        elif self._cfg.MHFL.pub_aug == 'strong':
            self.selected_transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomApply([
                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                 ], p=0.8),
                 transforms.RandomGrayscale(p=0.2),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4802, 0.4480, 0.3975),
                                      (0.2770, 0.2691, 0.2821))])

        # 训练的数据集
        if self._cfg.MHFL.public_dataset == 'MyCifar100':
            self.train_dataset = MyCifar100(self._cfg.MHFL.public_path, train=True, transform=self.selected_transform, download=True)
        # 获取dataloaders
        # 随机采样
        n_train = len(self.train_dataset)
        idxs = np.random.permutation((n_train))  # 生成一个长度为n_train的随机排列
        if self._cfg.MHFL.public_len != None:
            idxs = idxs[0:self._cfg.MHFL.public_len]  # 选取前public_scale的数
        train_sampler = SubsetRandomSampler(idxs)  # 对idxs进行子集采样
        # 使用采样器在DataLoader中指定子集的采样方式
        # TODO:num_workers与源代码不一样（源代码为4不好跑）
        self.train_loader = DataLoader(self.train_dataset, batch_size=self._cfg.MHFL.public_train.batch_size, sampler=train_sampler,
                                  num_workers=0)


    #每个客户端join时触发
    def callback_funcs_for_join_in(self, message: Message):
        """
            额外增加处理每个client模型的内容
        """
        self.join_in_client_num += 1
        sender = message.sender
        address = message.content[0]

        #####################################################################################
        # 获取客户端传来的模型
        client = message.sender
        client_model = message.content[0]
        self.client_models[client] = client_model
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

    #所有客户端join in之后触发，只触发一次
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
            ####################################################################################
            # 首先在服务器计算Mi和loss
            self.calculate_logits_output()
            #TODO:lr没改
            #self._cfg.MHFL.public_train.optimizer.lr *= (1 -  / self._cfg.MHFL.public_train.epochs * 0.9)
            self.send_per_client_message(msg_type='model_para')
            ####################################################################################

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))


    #服务器向客户端传各自的模型
    def send_per_client_message(self, msg_type,
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

        for client_id,client_model in self.client_models.items():
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=client_id,
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=client_model.state_dict()))

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    #TODO：是否要复制一些父类有的代码
    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        min_received_num = len(self.comm_manager.get_neighbors().keys())
        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training

        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                #################################################################
                # update model parameters
                for model_idx, model in self.client_models.items():
                    model.load_state_dict(self.msg_buffer['train'][self.state][model_idx][1])
                self.calculate_logits_output()

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
                    self.send_per_client_message(msg_type='model_para')

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

    # 在公共数据集上训练获取output Zi
    def calculate_logits_output(self):
        device = self._cfg.device
        lr = self._cfg.MHFL.public_train.optimizer.lr

        #服务器端根据output计算Mi矩阵和loss
        for batch_idx, (images, _) in enumerate(self.train_loader):
            # 获取output Zi
            linear_output_list = dict()  # TODO:这俩是不是应该在外面啊
            linear_output_target_list = dict()
            images = images.to(device)
            for model_idx, model in self.client_models.items():
                model.to(device)  #
                model.train()
                linear_output = model(images)
                linear_output_target_list[model_idx] = linear_output.clone().detach()
                linear_output_list[model_idx] = linear_output

            #计算Mi矩阵和损失
            for model_idx, model in self.client_models.items():
                optimizer = optim.Adam(model.parameters(), lr=lr)
                linear_output_target_avg_list = []
                for k,val in linear_output_target_list.items():
                    linear_output_target_avg_list.append(val)
                linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)
                linear_output = linear_output_list[model_idx]

                # 公式1 计算Mi矩阵（公式分母对不上）
                z_1_bn = (linear_output - linear_output.mean(0)) / linear_output.std(0)
                z_2_bn = (linear_output_target_avg - linear_output_target_avg.mean(0)) / linear_output_target_avg.std(0)
                c = z_1_bn.T @ z_2_bn
                c.div_(len(images))

                #TODO：服务器loss的计算会受optimizer上的影响吗
                #公式2 根据Mi计算loss
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = self._off_diagonal(c).add_(1).pow_(2).sum()  # _off_diagonal()自定义函数将非对角线数字放在数组中
                optimizer.zero_grad()
                col_loss = on_diag + self._cfg.fccl.off_diag_weight * off_diag  # 公式2
                if batch_idx == len(self.train_loader) - 2:
                    print('Communcation: ' + str(self.state) + ' Net: ' + str(model_idx) + 'Col: ' + str(
                        col_loss.item()))
                col_loss.backward()
                optimizer.step()


    #一个在计算loss时会用到的函数
    def _off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




class FCCLClient(Client):
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
        super(FCCLClient, self).__init__(ID, server_id, state, config, data, model, device,
                                             strategy, is_unseen_client, *args, **kwargs)
        self.inter_model = None
        self.pre_model = None
        # self.pretrain == True

    def join_in(self):
        """
        To send ``join_in`` message to the server for joining in the FL course.
        额外发送本地的个性化模型至client端
        """
        #预训练
        self._pretrain_nets()
        local_init_model = copy.deepcopy(self.model.cpu())
        self.inter_model = copy.deepcopy(self.model)
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=[local_init_model]))


    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        content = message.content
        self.state = round
        #根据服务器传过来的参数更新本地模型
        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)
        #把worker中的值赋给trainer
        self.trainer.ctx.inter_model = self.inter_model
        self.trainer.ctx.pre_model = self.pre_model
        sample_size, model_para, results = self.trainer.train()
        #TODO:worker中的model是否和trainer中一样
        self.inter_model.load_state_dict(self.model.state_dict())

        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size,model_para)))

    def _pretrain_nets(self):
        self.pre_model = copy.deepcopy(self.model)
        pretrain_path = os.path.join('./pretrain/', 'low_10_CNN_256')
        ckpt_files = os.path.join(pretrain_path, str(self.ID)+'.ckpt')

        if not os.path.exists(pretrain_path):
            os.makedirs(pretrain_path)
        else:
            if not os.path.exists(ckpt_files):
                self._pretrain_net(50)
                save_path = os.path.join(pretrain_path,str(self.ID)+'.ckpt')
                torch.save(self.pre_model.state_dict(),save_path)
            else:
                save_path = os.path.join(pretrain_path, str(self.ID) + '.ckpt')
                self.pre_model.load_state_dict(torch.load(save_path,self.device))
                self._evaluate_net()
        self.model.load_state_dict(torch.load(ckpt_files,self.device))

    def _pretrain_net(self,epoch):
        device = self._cfg.device
        model = self.pre_model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # scheduler = CosineLRScheduler(optimizer, t_initial=epoch, decay_rate=1., lr_min=1e-6)
        scheduler = CosineLRScheduler(optimizer, t_initial=epoch, lr_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)
        iterator = tqdm(range(epoch))
        for epoch_index in iterator:
            for batch_idx, (images, labels) in enumerate(self.data['train']):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (self.ID,loss)
                optimizer.step()
            # if epoch_index %10 ==0:
            acc = self._evaluate_net()
            scheduler.step(epoch_index)
                # if acc >80:
                #     break

    def _evaluate_net(self):
        device = self._cfg.device
        model = self.pre_model.to(device)
        dl = self.data['test']
        status = model.training  #TODO:没有training这个变量 Oh好像有的
        model.eval()
        correct, total,top1,top5 = 0.0, 0.0,0.0,0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)

        top1acc= round(100 * top1 / total,2)
        top5acc= round(100 * top5 / total,2)
        print('The '+str(self.ID)+'participant top1acc:'+str(top1acc)+'_top5acc:'+str(top5acc))
        model.train(status)
        return top1acc



def call_my_worker(method):
    if method == 'fccl':
        worker_builder = {'client': FCCLClient, 'server': FCCLServer}
        return worker_builder


register_worker('fccl', call_my_worker)
