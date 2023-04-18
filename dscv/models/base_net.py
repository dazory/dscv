import torch
import torch.nn as nn
import torch.nn.functional as F

from dscv.models.builder import MODELS
from dscv.losses.builder import build_loss


@MODELS.register_module
class BaseNet(nn.Module):
    def __init__(self, model, loss):
        super(BaseNet, self).__init__()
        self.model = model
        self.criterions = build_loss(loss)

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
        losses = dict()
        for criterion in self.criterions:
            losses.update(criterion(logits, labels))
        return losses
