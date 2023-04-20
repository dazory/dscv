import os
import time
import torch

from .builder import RUNNERS

from dscv.loggers.builder import build_logger
from dscv.utils.dist_utils import get_dist_info


def parse_losses(outputs, device='cuda'):
    loss = torch.tensor(0.0).to(device)
    for key, value in outputs.items():
        if 'loss' in key:
            loss += torch.mean(value)
    return loss


@RUNNERS.register_module
class BaseRunner:
    def __init__(self,
                 model,
                 optimizer,
                 epochs=None,
                 work_dir=None,
                 modes=None,
                 logger=None,
                 distributed=False,
                 print_freq=100,
                 ):
        super(BaseRunner, self).__init__()
        self.model = model
        self.optimizer = optimizer
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.epochs = epochs

        self.distributed = distributed
        self.device = torch.cuda.current_device()
        self.rank, _ = get_dist_info()

        self.work_dir = work_dir
        self.modes = ['train'] if modes is None else modes
        assert isinstance(self.modes, list)
        self.logger = build_logger(logger)
        self.print_freq = print_freq

        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._best_score = 0

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.data_loader = data_loader
        self.logger.before_train_epoch()
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        end = time.time()
        for i, data_batch in enumerate(self.data_loader):
            data_time = time.time() - end
            self._inner_iter = i
            self.logger.before_train_iter()
            self.adjust_lr()

            # forward
            outputs = self.model(*data_batch, mode='train', **kwargs)
            self.log_vars(outputs)

            # compute loss
            loss = parse_losses(outputs, device=self.device)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time = time.time() - end
            if self.rank == 0 and i % self.print_freq == 0:
                print(
                    'Epoch [{}][{}/{}]: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f}'.format(
                        self._epoch, i, len(data_loader), data_time, batch_time, loss.item()))

            self.logger.after_train_iter()
            self._iter += 1
        self.logger.after_train_epoch()
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.data_loader = data_loader
        self.logger.before_val_epoch()
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        outputs = dict()
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.logger.before_val_iter()

            _outputs = self.model(*data_batch, mode='val', **kwargs)
            self._update_outputs(outputs, _outputs)
            self.logger.after_val_iter()

        self.log_vars(outputs)
        self.logger.after_val_epoch()

    def run(self, data_loaders, start_epoch=0, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
        """
        assert isinstance(data_loaders, dict)

        self.logger.before_run()
        for epoch in range(start_epoch, self.epochs):
            if 'train' in self.modes:
                if self.distributed:
                    data_loaders['train'].sampler.set_epoch(epoch)
                self.train(data_loaders['train'], **kwargs)
            if 'val' in self.modes:
                self.val(data_loaders['val'], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.logger.after_run()

    def log_vars(self, outputs, prefix_key=None):
        def _mean(value):
            if torch.is_tensor(value):
                return torch.mean(value)
            elif isinstance(value, list):
                return torch.tensor(value).mean()
            else:
                return value

        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                name = key if prefix_key is None else f"{prefix_key}_{key}"
                self.logger.add_data(name, torch.mean(value).item())
            elif isinstance(value, list):
                _value = _mean(value)
                name = key if prefix_key is None else f"{prefix_key}_{key}"
                self.logger.add_data(name, _value.item())
            elif isinstance(value, dict):
                name = key if prefix_key is None else f"{prefix_key}_{key}"
                self.log_vars(value, prefix_key=name)
            else:
                name = key if prefix_key is None else f"{prefix_key}_{key}"
                self.logger.add_data(name, value)

    def save_checkpoint(self, score):
        if self._best_score <= score:
            checkpoint = {'epoch': self._epoch,
                          'model': self.model,
                          'state_dict': self.model.state_dict(),
                          'score': score,
                          'optimizer': self.optimizer.state_dict(), }
            ckpt_name = f"best.pth"
            ckpt_path = os.path.join(self.work_dir, ckpt_name)
            torch.save(checkpoint, ckpt_path)

    @staticmethod
    def _update_outputs(outputs, _outputs):
        for key, value in _outputs.items():
            if key not in outputs:
                outputs[key] = []
            outputs[key].append(value)

    def adjust_lr(self):
        b = self.data_loader.batch_size / 256.
        k = self.epochs // 3
        if self._epoch < k:
            m = 1
        elif self._epoch < 2 * k:
            m = 0.1
        else:
            m = 0.01
        lr = self.lr * m * b
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
