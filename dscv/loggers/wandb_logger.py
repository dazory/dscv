import functools
from typing import Callable
import pandas as pd

from .utils import master_only


def wandb_used(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.use_wandb:
            return func(*args, **kwargs)

    return wrapper


class WandbLogger():
    def __init__(self,
                 init_kwargs=None,
                 train_epoch_interval=10,
                 train_iter_interval=100,
                 val_epoch_interval=1,
                 val_iter_interval=10,
                 use_wandb=False):
        super(WandbLogger, self).__init__()
        self.wandb = None
        self.init_kwargs = init_kwargs
        self.use_wandb = use_wandb

        self.data = dict()

        self.train_epoch = 0
        self.train_iter = 0
        self.val_epoch = 0
        self.val_iter = 0
        self.train_epoch_interval = train_epoch_interval
        self.train_iter_interval = train_iter_interval
        self.val_epoch_interval = val_epoch_interval
        self.val_iter_interval = val_iter_interval

        self.import_wandb()

    @wandb_used
    def add_data(self, key, value, type=None, **kwargs):
        if type == 'image':
            self.data[key] = self.wandb.Image(value)
        elif type == 'table':
            self.add_table(key, value, **kwargs)
        else:
            self.data[key] = value

    @wandb_used
    def add_table(self, key, value, columns, x_label='x', y_label='y', plot_type=None):
        assert isinstance(value, list) and isinstance(columns, list)
        assert len(value[0]) == len(columns)
        # table = self.wandb.Table(data=value, columns=columns)
        table = pd.DataFrame(value, columns=columns)
        if plot_type == 'line':
            self.data[key] = self.wandb.plot.line(table, x_label, y_label, title=key)
        else:
            self.data[key] = table

    def clear_data(self):
        self.data = dict()

    @master_only
    def log(self, key, value):
        self.wandb.log({key, value})

    @master_only
    def log_dict(self, prefix='default'):
        for key, value in self.data.items():
            self.wandb.log({f"{prefix}/{key}": value})

    @wandb_used
    @master_only
    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    # run
    @wandb_used
    @master_only
    def before_run(self):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()

    @wandb_used
    @master_only
    def after_run(self):
        self.wandb.finish()

    # epoch
    @wandb_used
    @master_only
    def before_train_epoch(self):
        if self.train_epoch % self.train_epoch_interval == 0:
            self.log_dict('train')
        self.clear_data()

    @wandb_used
    @master_only
    def after_train_epoch(self):
        if self.train_epoch % self.train_epoch_interval == 0:
            self.log_dict('train')
        self.clear_data()
        self.train_epoch += 1

    @wandb_used
    @master_only
    def before_val_epoch(self):
        if self.val_epoch % self.val_epoch_interval == 0:
            self.log_dict('val')
        self.clear_data()

    @wandb_used
    @master_only
    def after_val_epoch(self):
        if self.val_epoch % self.val_epoch_interval == 0:
            self.log_dict('val')
        self.clear_data()
        self.val_epoch += 1

    # iter
    @wandb_used
    @master_only
    def before_train_iter(self):
        if self.train_iter % self.train_iter_interval == 0:
            self.log_dict('train')
        self.clear_data()

    @wandb_used
    @master_only
    def after_train_iter(self):
        if self.train_iter % self.train_iter_interval == 0:
            self.log_dict('train')
        self.clear_data()
        self.train_iter += 1

    @wandb_used
    @master_only
    def before_val_iter(self):
        if self.val_iter % self.val_iter_interval == 0:
            self.log_dict('val')
        self.clear_data()

    @wandb_used
    @master_only
    def after_val_iter(self):
        if self.val_iter % self.val_iter_interval == 0:
            self.log_dict('val')
        self.clear_data()
        self.val_iter += 1