from .base_logger import BaseLogger
from .wandb_logger import WandbLogger


def build_logger(cfg):
    if cfg.type == 'wandb_logger':
        _cfg = cfg.copy(del_type=True)
        logger = WandbLogger(**_cfg)
    else:
        logger = BaseLogger()
    return logger