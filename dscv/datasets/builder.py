import torch

from .pipelines.builder import build_pipeline
from dscv.utils import Registry, build_from_cfg


DATASETS = Registry('datasets')
WRAPPERS = Registry('wrappers')


def build_dataset(cfg):
    _cfg = cfg.copy()
    _cfg.pipeline = build_pipeline(_cfg.pipeline)
    if isinstance(cfg, dict):
        dataset = build_from_cfg(_cfg, DATASETS)
        if hasattr(_cfg, 'wrapper'):
            wrapper_cfg = _cfg.copy()
            wrapper_cfg.type = _cfg.wrapper
            wrapper_cfg.preprocess = build_pipeline(_cfg.get('preprocess'))
            dataset = build_from_cfg(wrapper_cfg, WRAPPERS, dataset=dataset)
    else:
        raise TypeError('cfg must be a dict, but got {}'.format(type(_cfg)))

    return dataset


def build_dataloader(dataset, **kwargs):
    data_loader = torch.utils.data.DataLoader(
        dataset, **kwargs
    )
    return data_loader
