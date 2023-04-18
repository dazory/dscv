import torch


def build_optimizer(model, cfg):
    if cfg.type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            cfg.lr,
            momentum=cfg.get('momentum'),
            weight_decay=cfg.get('weight_decay'))
    else:
        raise TypeError('')
    return optimizer
