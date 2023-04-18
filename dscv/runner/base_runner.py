import os
import time
import torch
from dscv.loggers.builder import build_logger


def parse_losses(outputs, device='cuda'):
    loss = torch.tensor(0.0).to(device)
    for key, value in outputs.items():
        if 'loss' in key:
            loss += value
    return loss


class BaseRunner:
    def __init__(self,
                 model,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 epochs=None,
                 evaluate=True,
                 device='cuda'):
        super(BaseRunner, self).__init__()
        self.model = model.to(device)

        self.optimizer = optimizer
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']

        self.work_dir = work_dir
        self.logger = build_logger(logger)
        self.epochs = epochs
        self.evaluate = evaluate
        self.device = device

        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._best_score = 0

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.data_loader = data_loader
        self.logger.before_train_epoch()
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.logger.before_train_iter()
            self.adjust_lr()

            # forward
            outputs = self.model(*data_batch, mode='train', **kwargs)
            self.log_vars(outputs)

            # compute loss
            loss = parse_losses(outputs, device=self.device)

            # backward
            loss.backward()
            self.optimizer.step()

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
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.logger.before_val_iter()

            outputs = self.model(*data_batch, mode='val', **kwargs)
            self.log_vars(outputs)
            self.logger.after_val_iter()

        self.logger.after_val_epoch()

    def run(self, data_loaders, start_epoch=0, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
        """
        assert isinstance(data_loaders, list)

        self.logger.before_run()
        for epoch in range(start_epoch, self.epochs):
            self.train(data_loaders[0], **kwargs)
            if self.evaluate:
                self.val(data_loaders[1], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.logger.after_run()

    def log_vars(self, outputs):
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                self.logger.add_data(key, value.item())
            elif isinstance(value, list):
                _value = torch.tensor(0.0)
                for v in value:
                    _value += v
                self.logger.add_data(key, _value.item())
            elif isinstance(value, dict):
                self.log_vars(value)
            else:
                self.logger.add_data(key, value)

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
