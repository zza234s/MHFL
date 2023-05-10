from federatedscope.core.configs.config import CN
from federatedscope.core.configs.yacs_config import Argument
from federatedscope.register import register_config

def extend_Model_heterogeneity_cfg(cfg):
    cfg.client_cfg_file = './per_client.yaml'
    # ---------------------------------------------------------------------- #
    # fedproto related options
    # ---------------------------------------------------------------------- #
    cfg.fedproto = CN()

    # Model related options
    cfg.model.stride = [1,4]
    cfg.model.fedproto_femnist_channel_temp =18

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


register_config("model_heterogeneity", extend_Model_heterogeneity_cfg)
