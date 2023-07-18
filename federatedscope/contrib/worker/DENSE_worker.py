from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.contrib.common_utils import Ensemble, KLDiv, save_checkpoint,test
from federatedscope.contrib.model.Generator import DENSE_Generator, AdvSynthesizer
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.contrib.model.DENSE_resnet import resnet18
import logging
import torch
import os
import copy
import sys
from tqdm import tqdm
import pickle
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
        self.global_model = get_model(model_config=config.model, local_data=data) #TODO: global model换成resnet18和源码统一
        self.nz = config.DENSE.nz
        self.test_loader =self.data['test']
        self.nc = 3 if "CIFAR" in config.data.type or config.data.type == "svhn" else 1
        self.other = config.data.type #TODO: 源代码中other的含义还是再要确认一下
        self.model_weight_dir = os.path.join(config.MHFL.model_weight_dir, 'df_ckpt')

        if not os.path.exists(self.model_weight_dir):
            os.mkdir(self.model_weight_dir)  # 生成保存预训练模型权重所需的文件夹
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

        self.trigger_for_start()


    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """
        if self.check_client_join_in():
            logger.info(
                '----------- Starting Global Distillation -------------'.
                format(self.state))
            self.start_global_distillation()
    def start_global_distillation(self):
        model_list = list(self.local_models.values())  # TODO: 按client_id 对字典排序
        for model in model_list:
            model.to(self.device)
        ensemble_model = Ensemble(model_list)
        global_model = resnet18(num_classes=10).to(self.device)

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
                                     sample_batch_size=self._cfg.DENSE.sample_batch_size,
                                     # todo：是否可以用cfg.dataloader的batch_size来替换这里的batchsize?
                                     adv=self._cfg.DENSE.adv, bn=self._cfg.DENSE.bn, oh=self._cfg.DENSE.oh,
                                     save_dir=self._cfg.DENSE.save_dir,
                                     dataset=self._cfg.data.type)  # todo: 这里的data.type是类似“CIFAR100@torchvision”这种格式，可能不满足函数要求
        criterion = KLDiv(T=self._cfg.DENSE.T)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self._cfg.train.optimizer.lr,
                                    momentum=0.9)  # todo:是否要用其他变量开控制这里的学习率?
        global_model.train()
        distill_acc = []
        bst_acc=-1
        for epoch in tqdm(range(self._cfg.federate.total_round_num)):  # todo:是否要用其他变量来控制这里的epoch总数?
            # 1. Data synthesis
            synthesizer.gen_data(cur_ep)  # g_steps
            cur_ep += 1
            kd_train(synthesizer, [global_model, ensemble_model], criterion, optimizer)  # # kd_steps
            acc, test_loss = test(global_model, self.test_loader,self.device)
            distill_acc.append(acc)
            is_best = acc > bst_acc
            bst_acc = max(acc, bst_acc)
            _best_ckpt = f"{self.model_weight_dir}/{self.other}.pth"
            print("best acc:{}".format(bst_acc))
            save_checkpoint({
                'state_dict': global_model.state_dict(),
                'best_acc': float(bst_acc),
            }, is_best, _best_ckpt)
            # wandb.log({'accuracy': acc})

        # wandb.log({"global_accuracy" : wandb.plot.line_series(
        #     xs=[ i for i in range(args.epochs) ],
        #     ys=distill_acc,
        #     keys="DENSE",
        #     title="Accuacy of DENSE")})
        # np.save("distill_acc_{}.npy".format(args.dataset), np.array(distill_acc))


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


def kd_train(synthesizer, model, criterion, optimizer):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/heter_fl.py
    """
    student, teacher = model
    student.train()
    teacher.eval()
    description = "loss={:.4f} acc={:.2f}%"
    total_loss = 0.0
    correct = 0.0
    with tqdm(synthesizer.get_data()) as epochs:
        for idx, (images) in enumerate(epochs):
            optimizer.zero_grad()
            images = images.cuda()
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(synthesizer.data_loader.dataset) * 100

            epochs.set_description(description.format(avg_loss, acc))


def call_my_worker(method):
    if method == 'dense':
        worker_builder = {'client': DENSE_Client, 'server': DENSE_Server}
        return worker_builder


register_worker('DENSE', call_my_worker)
