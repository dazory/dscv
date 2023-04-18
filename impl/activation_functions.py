import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax(x):
    '''
    Same with `torch.nn.functional.softmax(x)`
        F.softmax(x_i) = exp(x_i) / sum_j^{D}(exp(x_j))
    Input:
        x: (N, D), where N is the number of samples and D is the dimension of each sample
    Output: F.sofmtax(x) = (N, D)
    '''
    # Maps the input to the range (-inf, 0], which makes the maximum value become 1 (e.g. exp(0) = 1).
    maxes = torch.max(x, dim=1, keepdim=True)[0] # max function outputs a tuple (max, argmax).
    x = x - maxes

    # Averages the exponentials of the input along the D-dimension.
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)

    # Outputs the result.
    return exp_x / sum_exp_x

