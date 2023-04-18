import os
import torch

import dscv.datasets
from .pipelines.builder import build_pipeline


def build_from_cfg(cfg, obj_cls, default_args=None):
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    obj_type = args.pop('type')

    return obj_cls(**args)


def build_dataset(cfg):
    _cfg = cfg.copy(del_type=True)
    _cfg.pipeline = build_pipeline(_cfg.pipeline)
    if cfg.type == 'ImageNetDataset':
        dataset = dscv.datasets.ImageNetDataset(**_cfg)
    else:
        raise TypeError('')

    if cfg.get('wrapper'):
        if cfg.wrapper == 'AugMixDataset':
            _cfg.preprocess = build_pipeline(_cfg.get('preprocess'))
            dataset = dscv.datasets.AugMixDataset(dataset, **_cfg)

    return dataset


def build_dataloader(dataset, **kwargs):
    data_loader = torch.utils.data.DataLoader(
        dataset, **kwargs
    )
    return data_loader
