from federatedscope.core.configs.config import CN
from federatedscope.core.configs.yacs_config import Argument
from federatedscope.register import register_config


def extend_model_heterogeneous_cfg(cfg):
    '''模型异构联邦学习用到的通用参数'''
    # MHFL: model_heterogeneous federated learning
    cfg.MHFL = CN()
    cfg.MHFL.task = 'CV'  # choice:['CV','NLP']

    cfg.MHFL.public_train = CN()  # 在公共数据集上训练相关的参数
    cfg.MHFL.public_dataset = 'mnist'
    cfg.MHFL.model_weight_dir = './contrib/model_weight'


    cfg.MHFL.public_train.batch_size = 128  # 训练、测试公共数据集的batch_size
    cfg.MHFL.public_train.epochs = 40
    #public training optimizer相关
    cfg.MHFL.public_train.optimizer = CN()
    cfg.MHFL.public_train.optimizer.type ='Adam'
    cfg.MHFL.public_train.optimizer.lr = 0.001
    cfg.MHFL.public_train.optimizer.weight_decay = 0.
    # cfg.MHFL.public_train.optimizer.momentum = 1e-4


    '''benchmark中各方法所需的参数'''
    # ---------------------------------------------------------------------- #
    # fedproto related options
    # ---------------------------------------------------------------------- #
    cfg.fedproto = CN()

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

    # model related options
    cfg.model.fsfl_cnn_layer1_out_channels = 128
    cfg.model.fsfl_cnn_layer2_out_channels = 512


register_config("model_heterogeneity", extend_model_heterogeneous_cfg)
