from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.contrib.common_utils import Ensemble,KLDiv
from federatedscope.contrib.model.Generator import DENSE_Generator,AdvSynthesizer
import logging
import torch
import os
import copy
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict

logger = logging.getLogger(__name__)


class DENSE_Server(Server):
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
        super(DENSE_Server, self).__init__(ID, state, config, data, model, client_num, total_round_num,
                                           device, strategy, unseen_clients_id, **kwargs)
        self.local_models = dict()  # key:client_ID, values: torch.model
        self.global_model = get_model(model_config=config.model, local_data=data)
        self.nz = config.DENSE.nz
        self.nc = 3 if "CIFAR" in config.data.type or config.data.type == "svhn" else 1

    def callback_funcs_for_join_in(self, message: Message):
        """
            额外增加处理每个client个性化模型的内容
        """
        self.join_in_client_num += 1
        sender = message.sender
        address = message.content[0]
        #####################################################################################
        local_model = message.content[1]
        self.local_models[sender] = local_model
        ####################################################################################

        self.comm_manager.add_neighbors(neighbor_id=sender,
                                        address=address)

        self.start_global_distillation()

    def start_global_distillation(self):
        model_list = list(self.local_models.values())  # TODO: 按client_id 对字典排序
        ensemble_model = Ensemble(model_list)
        global_model=self.global_model

        # data generator
        nz = self._cfg.DENSE.nz
        nc = 3 if "CIFAR" in self._cfg.data.type or self._cfg.data.type == "svhn" else 1
        img_size = 32 if "CIFAR" in self._cfg.data.type or self._cfg.data.type == "svhn" else 28

        generator = DENSE_Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
        cur_ep = 0  # TODO: 源代码中为 args.cur_ep=0
        img_size2 = (3, 32, 32) if "CIFAR" in self._cfg.data.type or self._cfg.data.type == "svhn" else (1, 28, 28)
        num_class = 100 if self._cfg.data.type == "CIFAR100@torchvision" else 10
        synthesizer = AdvSynthesizer(ensemble_model, model_list, global_model, generator,
                                     nz=nz, num_classes=num_class, img_size=img_size2,
                                     iterations=self._cfg.DENSE.g_steps, lr_g=self._cfg.DENSE.lr_g,
                                     synthesis_batch_size=self._cfg.DENSE.synthesis_batch_size,
                                     sample_batch_size=self._cfg.DENSE.sample_batch_size,#todo：是否可以用cfg.dataloader的batch_size来替换这里的batchsize?
                                     adv=self._cfg.DENSE.adv, bn=self._cfg.DENSE.bn, oh=self._cfg.DENSE.oh,
                                     save_dir=self._cfg.DENSE.save_dir, dataset=self._cfg.data.type)#todo: 这里的data.type是类似“CIFAR100@torchvision”这种格式，可能不满足函数要求
        # criterion = KLDiv(T=self._cfg..T)


class DENSE_Client(Client):
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
        super(DENSE_Client, self).__init__(ID, server_id, state, config, data, model, device,
                                           strategy, is_unseen_client, *args, **kwargs)
        self.model_weight_dir = config.MHFL.model_weight_dir
        self.dataset_name = config.data.type
        if not os.path.exists(self.model_weight_dir):
            os.mkdir(self.model_weight_dir)  # 生成保存预训练模型权重所需的文件夹

    def join_in(self):
        """
        额外发送预训练好的本地的个模型至Server
        """
        self.local_pre_training()
        local_init_model = copy.deepcopy(self.model.cpu())
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=[self.local_address, local_init_model]))

    def local_pre_training(self):
        logger.info(f'\tClient #{self.ID} pre-train start...')
        save_path = os.path.join(self.model_weight_dir,
                                 'DENSE_' + self.dataset_name + '_client_' + str(self.ID) + '.pth')

        if os.path.exists(save_path) and not self._cfg.MHFL.rePretrain:
            self.model.load_state_dict(torch.load(save_path, self.device))
            eval_metrics = self.trainer.evaluate(target_data_split_name='test')

            logger.info(
                f"Client # {self.ID} load the pretrained model weight."
                f"The accuracy of the pretrained model on the local test dataset is {eval_metrics['test_acc']}")
        else:
            for i in range(self._cfg.DENSE.pretrain_epoch):
                num_samples_train, _, results = self.trainer.train()
                if i % self._cfg.eval.freq == 0:
                    eval_metrics = self.trainer.evaluate(target_data_split_name='test')

                    logger.info(f"Client #{self.ID} local pre-train @Epoch {i}."
                                f" train_acc:{results['train_acc']}\t "
                                f" test_acc:{eval_metrics['test_acc']} ")

            logger.info(f"Client #{self.ID} pre-train finish. Save the model weight file")
            torch.save(self.model.state_dict(), save_path)


def call_my_worker(method):
    if method == 'dense':
        worker_builder = {'client': DENSE_Client, 'server': DENSE_Server}
        return worker_builder


register_worker('DENSE', call_my_worker)
