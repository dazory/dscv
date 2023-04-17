import torch
import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
