from federatedscope.core.configs.config import CN
from federatedscope.core.configs.yacs_config import Argument
from federatedscope.register import register_config
import logging

logger = logging.getLogger(__name__)


def extend_model_heterogeneous_cfg(cfg):
    '''模型异构联邦学习用到的通用参数'''
    # MHFL: Model Heterogeneous Federated Learning
    cfg.MHFL = CN()
    cfg.MHFL.task = 'CV'  # choice:['CV','NLP']

    cfg.MHFL.public_train = CN()  # 在公共数据集上训练相关的参数
    cfg.MHFL.public_dataset = 'mnist'
    cfg.MHFL.public_path = './data'
    cfg.MHFL.public_train.batch_size = 128  # 训练、测试公共数据集的batch_size
    cfg.MHFL.public_train.epochs = 40
    cfg.MHFL.public_len = 5000  # weak or strong
    cfg.MHFL.model_weight_dir = './contrib/model_weight'

    # public training optimizer相关
    cfg.MHFL.public_train.optimizer = CN()
    cfg.MHFL.public_train.optimizer.type = 'Adam'
    cfg.MHFL.public_train.optimizer.lr = 0.001
    cfg.MHFL.public_train.optimizer.weight_decay = 0.
    # cfg.MHFL.public_train.optimizer.momentum = 1e-4

    cfg.model.filter_channels = [64, 64, 64]

    # Pretrain related option
    cfg.MHFL.pre_training = CN()
    cfg.MHFL.pre_training.public_epochs = 1
    cfg.MHFL.pre_training.private_epochs = 1
    cfg.MHFL.pre_training.public_batch_size = 256
    cfg.MHFL.pre_training.private_batch_size = 256
    cfg.MHFL.pre_training.rePretrain = True
    cfg.MHFL.pre_training.save_model = True  # 是否保存预训练模型

    # dataset realated option
    cfg.data.local_eval_whole_test_dataset = False
    cfg.result_floder = 'model_heterogeneity/result/csv'
    cfg.exp_name = 'test'

    # other option
    cfg.model.return_proto = False # 用来控制是否在前向传播时输出每个输入对应的表征。FedProto,FedPCL,FedheNN等需要用到
    cfg.train.optimizer.momentum = 0.9
    cfg.model.num_classes = 10  # FederatedScope中原代码中没有这个变量，但是这个变量在创建模型时很常用，故添加
    cfg.show_label_distribution = False  # 可视化相关参数
    cfg.show_client_best_individual = True

    '''benchmark中各方法所需的参数'''
    # ---------------------------------------------------------------------- #
    # FedMD: Heterogenous Federated Learning via Model Distillation
    # ---------------------------------------------------------------------- #
    cfg.fedmd = CN()

    # Pre-training steps before starting federated communication
    cfg.fedmd.pre_training = CN()
    cfg.fedmd.pre_training.public_epochs = 1
    cfg.fedmd.pre_training.private_epochs = 1
    cfg.fedmd.pre_training.public_batch_size = 256
    cfg.fedmd.pre_training.private_batch_size = 256
    cfg.fedmd.pre_training.rePretrain = True

    # Communication step
    cfg.fedmd.public_subset_size = 5000

    # Digest step
    cfg.fedmd.digest_epochs = 1
    # Revisit step
    cfg.fedmd.revisit_epochs = 1

    # ---------------------------------------------------------------------- #
    # FedProto related options
    # ---------------------------------------------------------------------- #
    cfg.fedproto = CN()
    cfg.fedproto.proto_weight = 1.0  # weight of proto loss

    # Model related options
    cfg.model.stride = [1, 4]
    cfg.model.fedproto_femnist_channel_temp = 18
    cfg.model.pretrain_resnet = False

    # data related options
    cfg.fedproto.iid = False
    cfg.fedproto.unequal = False
    cfg.fedproto.ways = 5
    cfg.fedproto.stdev = 2
    cfg.fedproto.shots = 100
    cfg.fedproto.train_shots_max = 110
    cfg.fedproto.test_shots = 15

    # other options
    cfg.fedproto.show_verbose = False # Weather display verbose loss information
    # ---------------------------------------------------------------------- #
    # (FedPCL) Federated Learning from Pre-Trained Models: A Contrastive Learning Approach
    # ---------------------------------------------------------------------- #
    cfg.fedpcl = CN()
    # for debug
    cfg.model.fedpcl = CN()
    cfg.model.fedpcl.model_weight_dir = './contrib/model_weight'
    cfg.model.fedpcl.input_size = 512
    cfg.model.fedpcl.output_dim = 256

    cfg.fedpcl.debug = False

    cfg.fedpcl.show_verbose = False  # Weather display verbose loss information

    # ---------------------------------------------------------------------- #
    # (FML) Federated Mutual Learning related options
    # ---------------------------------------------------------------------- #
    cfg.fml = CN()
    cfg.fml.alpha = 0.5
    cfg.fml.beta = 0.5
    cfg.model.T = 5  # 临时变量

    # Model related options
    cfg.fml.meme_model = CN()
    cfg.fml.meme_model.type = 'CNN'
    cfg.fml.meme_model.hidden = 256
    cfg.fml.meme_model.dropout = 0.5
    cfg.fml.meme_model.in_channels = 0
    cfg.fml.meme_model.out_channels = 1
    cfg.fml.meme_model.layer = 2
    cfg.fml.meme_model.T = 5  # TODO: 临时变量

    # ---------------------------------------------------------------------- #
    # FedHeNN related options
    # ---------------------------------------------------------------------- #
    cfg.fedhenn = CN()
    cfg.fedhenn.eta = 0.001  # weight of proto loss

    # ---------------------------------------------------------------------- #
    # (FSFL) Few-Shot Model Agnostic Federated Learning
    # ---------------------------------------------------------------------- #
    cfg.fsfl = CN()

    # dataset realated option to verify the correctness of the reproduction
    cfg.fsfl.public_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 参考源代码
    cfg.fsfl.private_classes = [10, 11, 12, 13, 14, 15]  # 参考源代码
    cfg.fsfl.N_samples_per_class = 12

    # Latent Embedding Adaptation
    # Step1: domain identifier realated option
    cfg.fsfl.domain_identifier_epochs = 4
    cfg.fsfl.domain_identifier_batch_size = 30
    cfg.fsfl.DI_optimizer = CN()
    cfg.fsfl.DI_optimizer.type = 'Adam'
    cfg.fsfl.DI_optimizer.lr = 0.001  # 参考源代码
    cfg.fsfl.DI_optimizer.weight_decay = 1e-4  # 参考源代码

    # Step2: local gan training related option
    cfg.fsfl.gan_local_epochs = 4  # 参考源代码
    cfg.fsfl.DI_optimizer_step_2 = CN()
    cfg.fsfl.DI_optimizer_step_2.type = 'Adam'
    cfg.fsfl.DI_optimizer_step_2.lr = 0.0001  # 参考源代码
    cfg.fsfl.DI_optimizer_step_2.weight_decay = 1e-4  # 参考源代码

    # model agnostic federated learning related option
    cfg.fsfl.collaborative_epoch = 1  # 参考源代码
    cfg.fsfl.collaborative_num_samples_epochs = 5000
    cfg.fsfl.MAFL_batch_size = 256  # 参考源代码

    # model related options
    cfg.model.fsfl_cnn_layer1_out_channels = 128
    cfg.model.fsfl_cnn_layer2_out_channels = 512

    # ---------------------------------------------------------------------- #
    # FCCL related options
    # ---------------------------------------------------------------------- #
    cfg.fccl = CN()
    cfg.fccl.structure = 'homogeneity'
    cfg.fccl.beta = 0.1
    cfg.fccl.off_diag_weight = 0.0051
    cfg.fccl.pretrain_epoch = 50
    cfg.fccl.pretrain_path = 'low_5_CNN_alpha100'
    cfg.fccl.loss_dual_weight = 1
    cfg.fccl.pub_aug = 'weak'

    # ---------------------------------------------------------------------- #
    # DENSE: Data-Free One-Shot Federated Learning
    # ---------------------------------------------------------------------- #
    cfg.DENSE = CN()
    cfg.DENSE.pretrain_epoch = 300
    cfg.DENSE.model_heterogeneous = True
    cfg.DENSE.nz = 256  # number of total iterations in each epoch
    cfg.DENSE.g_steps = 256  # number of iterations for generation
    cfg.DENSE.lr_g = 1e-3  # initial learning rate for generation
    cfg.DENSE.synthesis_batch_size = 256
    cfg.DENSE.sample_batch_size = 256
    cfg.DENSE.adv = 0  # scaling factor for adv loss
    cfg.DENSE.bn = 0  # scaling factor for BN regularization
    cfg.DENSE.oh = 0  # scaling factor for one hot loss (cross entropy)
    cfg.DENSE.act = 0  # scaling factor for activation loss used in DAFL
    cfg.DENSE.save_dir = './contrib/synthesis'
    cfg.DENSE.T = 1.0

    # ----------------------------------wait to delete----------------------- #
    cfg.MHFL.save_pretraining_model = True  # 是否保存预训练模型

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_mdfh_cfg)


def assert_mdfh_cfg(cfg):
    if cfg.model.num_classes != cfg.model.out_channels:
        cfg.model.num_classes = cfg.model.out_channels
        logger.warning(f"The values of cfg.model.num_classes and cfg.model.out_channels must be equal."
                       f" Now the cfg.model.num_classes is {cfg.model.num_classes},"
                       f" but cfg.model.out_channels is {cfg.model.out_channels}")
        logger.warning(f"now we set cfg.model.num_classes as cfg.model.out_channels={cfg.model.out_channels}")


register_config("model_heterogeneity", extend_model_heterogeneous_cfg)
