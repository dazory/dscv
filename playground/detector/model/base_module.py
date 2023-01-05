import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def forward(self):
        pass
