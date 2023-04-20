import time
import torch

from .base_runner import BaseRunner
from .builder import RUNNERS


@RUNNERS.register_module
class RunnerC(BaseRunner):
    def __init__(self,
                 model,
                 optimizer,
                 *args, **kwargs
                 ):
        super().__init__(model, optimizer, *args, **kwargs)
        self.model = model
        self.optimizer = optimizer
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']


    def run(self, data_loaders, start_epoch=0, **kwargs):
        assert isinstance(data_loaders, dict)

        self.logger.before_run()
        for epoch in range(start_epoch, self.epochs):
            if 'train' in self.modes:
                if self.distributed:
                    data_loaders['train'].sampler.set_epoch(epoch)
                self.train(data_loaders['train'], **kwargs)
            if 'val' in self.modes:
                self.val(data_loaders['val'], **kwargs)
        if 'val_c' in self.modes:
            self.val_c(data_loaders['val_c'], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.logger.after_run()

    @torch.no_grad()
    def val_c(self, data_loader, **kwargs):
        self.model.eval()
        self.data_loader = data_loader
        self.logger.log_dict(prefix='val_c')  # before_val_epoch()
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        corruptions = data_loader.dataset.corruptions
        severities = data_loader.dataset.severities
        mce_results = dict()
        for corruption in corruptions:
            mce_results[corruption] = []
            for severity in severities:
                data_loader.dataset.set_mode(corruption, severity)
                outputs = dict()
                for i, data_batch in enumerate(data_loader):
                    self._inner_iter = i
                    _outputs = self.model(*data_batch, mode='val', **kwargs)
                    self._update_outputs(outputs, _outputs)
                mce_results[corruption].append(sum(outputs['acc1']) / len(outputs['acc1']))
            if self.rank == 0:
                average_mce = sum(mce_results[corruption]) / len(mce_results[corruption])
                _acc1s =', '.join([f'{acc1:.3f}' for acc1 in mce_results[corruption]])
                print(f"[{corruption}] acc1: {average_mce:.3f} = [{_acc1s}]")

        # log
        mce_log_data = [sum(mce_results[corruption]) / len(mce_results[corruption]) for corruption in corruptions]
        self.logger.add_data('acc1', [mce_log_data], columns=corruptions, type='table',
                             x_label='corruption', y_label='acc1', plot_type='line')
        self.logger.log_dict(prefix='val_c')
        if self.rank == 0:
            print(f"total MEC(acc1) = {sum(mce_log_data) / len(mce_log_data):.3f}")
