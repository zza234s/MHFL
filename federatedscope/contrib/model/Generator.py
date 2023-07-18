import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from kornia import augmentation
from torchvision import transforms
from tqdm import tqdm
import torchvision.utils as vutils
class DENSE_Generator(nn.Module):
    '''
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/models/generator.py
    '''
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(DENSE_Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class AdvSynthesizer():
    def __init__(self, teacher, model_list, student, generator, nz, num_classes, img_size,
                 iterations, lr_g,
                 synthesis_batch_size, sample_batch_size,
                 adv, bn, oh, save_dir, dataset):
        super(AdvSynthesizer, self).__init__()
        self.student = student
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.save_dir = save_dir
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.teacher = teacher
        self.dataset = dataset

        self.generator = generator.cuda().train()
        self.model_list = model_list

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
            ]),
        ])
        # =======================
        if not ("cifar" in dataset):
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

        # datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        # if len(datasets) != 0:
        #     self.data_loader = torch.utils.data.DataLoader(
        #         datasets, batch_size=self.sample_batch_size, shuffle=True,
        #         num_workers=4, pin_memory=True, )

    def gen_data(self, cur_ep):
        self.synthesize(self.teacher, cur_ep)

    def get_data(self):
        datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader

    def synthesize(self, net, cur_ep):
        net.eval()
        best_cost = 1e6
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()  #
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        targets = targets.sort()[0]
        targets = targets.cuda()
        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                     betas=[0.5, 0.999])#TODO:未懂
        hooks = []
        #############################################
        dim_in = 500 if "cifar100" == self.dataset else 50
        net = Ensemble_A(self.model_list)
        net.eval()
        # net_mlp = MLP(dim_in).cuda()
        # net_mlp.train()
        # optimizer_mlp = torch.optim.SGD(net_mlp.parameters(), lr=0.01,
        #                                 momentum=0.9)
        #############################################
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(DeepInversionHook(m))#TODO:未懂

        with tqdm(total=self.iterations) as t:
            for it in range(self.iterations):
                optimizer.zero_grad()
                # optimizer_mlp.zero_grad()
                inputs = self.generator(z)  # bs,nz #zhl:返回（256,3,32,32）
                global_view, _ = self.aug(inputs)  # crop and normalize #zhl:返回（256,3,32,32）
                #############################################
                # Gate
                t_out = net(global_view)
                # data_ensm = net(global_view)
                # t_out = net_mlp(data_ensm)
                #############################################
                # t_out = net(global_view)
                loss_bn = sum([h.r_feature for h in hooks])  # bn层loss
                loss_oh = F.cross_entropy(t_out, targets)  # ce_loss
                # if cur_ep <= 20:
                #     adv = 1
                # elif cur_ep <= 50:
                #     adv = 10
                # elif cur_ep <= 100:
                #     adv = 20
                # elif cur_ep <= 150:
                #     adv = 30
                # else:
                #     adv = 50
                # self.adv = adv
                s_out = self.student(global_view)
                mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
                    1) * mask).mean()  # decision adversarial distillation

                loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
                # loss = loss_inv
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

                loss.backward()
                optimizer.step()
                # optimizer_mlp.step()
                t.set_description('iters:{}, loss:{}'.format(it, loss.item()))
            vutils.save_image(best_inputs.clone(), '1.png', normalize=True, scale_each=True, nrow=10)

        # save best inputs and reset data iter
        self.data_pool.add(best_inputs)  # 生成了一个batch的数据

