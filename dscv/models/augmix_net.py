import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from dscv.losses.builder import build_loss
from dscv.models.builder import MODELS


@MODELS.register_module
class AugMixNet(nn.Module):
    def __init__(self, model, loss, num_inputs=3):
        super(AugMixNet, self).__init__()
        self.model = self._build_model(model)
        self.criterions = build_loss(loss)

        self.num_inputs = num_inputs

    def forward(self, images, labels, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(images, labels, **kwargs)
        elif mode == 'val':
            return self.forward_val(images, labels, **kwargs)
        else:
            raise NotImplementedError

    def forward_train(self, images, labels, **kwargs):
        self.model.train()
        outputs = dict()

        # forward
        if isinstance(images, tuple) or isinstance(images, list):
            images = torch.cat(images, dim=0)
        logits = self.model(images)

        # compute loss
        losses = self.loss(logits, labels)
        outputs.update(losses)

        return outputs

    @torch.no_grad()
    def forward_val(self, images, labels, **kwargs):
        self.model.eval()
        outputs = dict()

        # forward
        logits = self.model(images)

        # compute loss
        losses = self.loss(logits, labels)
        outputs.update(losses)

        return outputs

    def loss(self, logits, labels):
        logit_clean = torch.split(logits, logits.size(0) // self.num_inputs)[0]
        clean_loss = self.criterions[0](logit_clean, labels)
        dg_loss = self.criterions[1](logits)

        losses = dict()
        losses.update(clean_loss)
        losses.update(dg_loss)

        return losses

    def _build_model(self, cfg):
        return models.__dict__[cfg.type](pretrained=cfg.get('pre_trained'))
