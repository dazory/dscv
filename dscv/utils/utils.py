import torch
import shutil

import numpy as np


def get_dir_and_file_name(full_path):
    '''
    Examples:
        '/ws/data', None = get_dir_and_file_name('/ws/data')
        '/ws/data', 'img.png' = get_dir_and_file_name('/ws/data/img.png')
        'data', None = get_dir_and_file_name('data')
        None, 'img.png' = get_dir_and_file_name('img.png')
    '''
    name_dir = full_path.split('/')
    name_file = full_path.split('.')

    file = name_dir[-1] if len(name_file) > 1 else None
    dir = full_path if file is None else '/'.join(name_dir[:-1])
    if len(name_dir) == 1 and (not file is None):
        dir = None

    return dir, file


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:min(k, maxk)].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
