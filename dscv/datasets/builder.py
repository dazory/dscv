import os
import torch

import dscv.datasets


def build_from_cfg(cfg, obj_cls, default_args=None):
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    obj_type = args.pop('type')

    return obj_cls(**args)


def build_dataset(cfg):
    if cfg.type == 'ImageNetDataset':
        dataset = build_from_cfg(cfg, dscv.datasets.ImageNetDataset)
    elif cfg.type == 'CustomDataset':
        dataset = build_from_cfg(cfg, dscv.datasets.CustomDataset)
    else:
        raise TypeError('')
    return dataset


def build_dataloader(dataset, **kwargs):
    data_loader = torch.utils.data.DataLoader(
        dataset, **kwargs
    )
    return data_loader
