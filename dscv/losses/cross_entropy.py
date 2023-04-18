import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSS


@LOSS.register_module
class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 name='ce_loss',
                 weight=1.0,
                 ):
        super(CrossEntropyLoss, self).__init__()
        self.name = name
        self.weight = weight

    def forward(self, logits, labels, *args, **kwargs):
        loss = F.cross_entropy(logits, labels)
        return {self.name: self.weight * loss}
