import copy
from .base_runner import BaseRunner


def build_runner(cfg, model, optimizer):
    if cfg.type == 'base_runner':
        _cfg = cfg.copy()
        del _cfg['type']
        runner = BaseRunner(model, optimizer, **_cfg)
    else:
        raise TypeError('')
    return runner
