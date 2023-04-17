import warnings

import torch.nn
import torch.backends.cudnn as cudnn
from torchvision import models


def build_model(cfg, train_cfg=None, test_cfg=None):
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    if cfg.type in models.__dict__:
        model = models.__dict__[cfg.type](pretrained=cfg.get('pre_trained'))
        model.train_cfg = train_cfg
        model.test_cfg = test_cfg
    else:
        raise TypeError('')

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    return model
