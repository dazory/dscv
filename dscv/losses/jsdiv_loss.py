import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSS


@LOSS.register_module
class JSDivLoss(nn.Module):
    def __init__(self,
                 name='jsdiv_loss',
                 weight=1.0,
                 num_inputs=1,
                 ):
        super(JSDivLoss, self).__init__()
        self.name = name
        self.weight = weight
        assert num_inputs > 0
        self.num_inputs = num_inputs

    def forward(self, logits, *args, **kwargs):
        if self.num_inputs > 1:
            logits_list = torch.split(logits, logits.size(0) // self.num_inputs)
        else:
            logits_list = [logits]

        p_list = [F.softmax(logits, dim=1) for logits in logits_list]
        p_mixture = torch.clamp(sum(p_list) / self.num_inputs, min=1e-7, max=1.0).log()
        kldiv_loss_list = [F.kl_div(p_mixture, p, reduction='batchmean') for p in p_list]
        loss = torch.mean(torch.stack(kldiv_loss_list))

        return {self.name: self.weight * loss}