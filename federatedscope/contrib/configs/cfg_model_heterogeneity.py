from federatedscope.core.configs.config import CN
from federatedscope.core.configs.yacs_config import Argument
from federatedscope.register import register_config


def extend_model_heterogeneous_cfg(cfg):
    '''模型异构联邦学习用到的通用参数'''
    # MHFL: model_heterogeneous federated learning
    cfg.MHFL = CN()
    cfg.MHFL.task = 'CV'  # choice:['CV','NLP']

    cfg.MHFL.save_pretraining_model = False #是否保存预训练模型

    cfg.MHFL.public_train = CN()  # 在公共数据集上训练相关的参数
    cfg.MHFL.public_dataset = 'mnist'
    cfg.MHFL.public_path = './data'
    cfg.MHFL.public_train.batch_size = 128  # 训练、测试公共数据集的batch_size
    cfg.MHFL.public_train.epochs = 40
    cfg.MHFL.public_len = 5000
    cfg.MHFL.pub_aug = 'weak' #weak or strong
    cfg.MHFL.model_weight_dir = './contrib/model_weight'

    # public training optimizer相关
    cfg.MHFL.public_train.optimizer = CN()
    cfg.MHFL.public_train.optimizer.type = 'Adam'
    cfg.MHFL.public_train.optimizer.lr = 0.001
    cfg.MHFL.public_train.optimizer.weight_decay = 0.
    # cfg.MHFL.public_train.optimizer.momentum = 1e-4

    cfg.model.filter_channels = [64,64,64]

    # 数据集相关参数
    cfg.data.local_eval_whole_test_dataset = False

    cfg.csv_name ='./'


    '''benchmark中各方法所需的参数'''
    # ---------------------------------------------------------------------- #
    # fedproto related options
    # ---------------------------------------------------------------------- #
    cfg.fedproto = CN()
    cfg.fedproto.proto_weight = 1.0 #weight of proto loss

    # Model related options
    cfg.model.stride = [1, 4]
    cfg.model.fedproto_femnist_channel_temp = 18

    # ---------------------------------------------------------------------- #
    # FML related options
    # ---------------------------------------------------------------------- #
    cfg.fml = CN()
    cfg.fml.alpha = 0.5
    cfg.fml.beta = 0.5

    # Model related options
    cfg.fml.meme_model = CN()
    cfg.fml.meme_model.type = 'CNN'
    cfg.fml.meme_model.hidden = 256
    cfg.fml.meme_model.dropout = 0.5
    cfg.fml.meme_model.in_channels = 0
    cfg.fml.meme_model.out_channels = 1
    cfg.fml.meme_model.layer = 2

    # ---------------------------------------------------------------------- #
    # FedHeNN related options
    # ---------------------------------------------------------------------- #
    cfg.fedhenn = CN()
    cfg.fedhenn.n_0 = 0.01

    # ---------------------------------------------------------------------- #
    # FSFL related options
    # ---------------------------------------------------------------------- #
    cfg.fsfl = CN()

    # 本地模型预训练相关
    cfg.fsfl.pre_training_epochs = 40

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
    cfg.fsfl.collaborative_epoch= 1 #参考源代码
    cfg.fsfl.collaborative_num_samples_epochs = 5000
    cfg.fsfl.MAFL_batch_size = 256 #参考源代码


    # model related options
    cfg.model.fsfl_cnn_layer1_out_channels = 128
    cfg.model.fsfl_cnn_layer2_out_channels = 512


    # ---------------------------------------------------------------------- #
    # Fccl related options
    # ---------------------------------------------------------------------- #
    cfg.fccl = CN()
    cfg.fccl.structure = 'homogeneity'
    cfg.fccl.beta = 0.1
    cfg.fccl.off_diag_weight = 0.0051


register_config("model_heterogeneity", extend_model_heterogeneous_cfg)
