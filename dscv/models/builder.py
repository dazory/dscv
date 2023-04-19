import warnings

from dscv.utils import Registry, build_from_cfg


MODELS = Registry('models')


def build_model(cfg, train_cfg=None, test_cfg=None):
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '

    if isinstance(cfg, dict):
        model = build_from_cfg(cfg, MODELS)
    else:
        raise TypeError('cfg must be a dict, but got {}'.format(type(cfg)))

    model.train_cfg = train_cfg
    model.test_cfg = test_cfg

    return model
