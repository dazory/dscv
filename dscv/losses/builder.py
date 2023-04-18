from dscv.utils import Registry, build_from_cfg


LOSS = Registry('loss')


def build_loss(cfg):
    if isinstance(cfg, list):
        loss = []
        for _cfg in cfg:
            loss.append(build_loss(_cfg))
        return loss
    elif isinstance(cfg, dict):
        loss = build_from_cfg(cfg, LOSS)
        return loss
    else:
        raise TypeError('cfg must be a dict or list, but got {}'.format(type(cfg)))