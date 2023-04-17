import functools
from typing import Callable


def logger_used(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.use_logger:
            return func(*args, **kwargs)
    return wrapper


class BaseLogger():
    def __init__(self,
                 train_epoch_interval=10,
                 train_iter_interval=100,
                 val_epoch_interval=1,
                 val_iter_interval=10,
                 use_logger=False):
        super(BaseLogger, self).__init__()
        self.use_logger = use_logger

        self.train_epoch_interval = train_epoch_interval
        self.train_iter_interval = train_iter_interval
        self.val_epoch_interval = val_epoch_interval
        self.val_iter_interval = val_iter_interval

        self.train_epoch = 0
        self.train_iter = 0
        self.val_epoch = 0
        self.val_iter = 0

        self.data = dict()

    # write
    def add_data(self, key, value=None):
        if value is None:
            assert isinstance(key, dict), 'value is needed.'
            self.data.update(key) # key is data
        else:
            self.data[key] = value

    def clear_data(self):
        self.data = dict()

    # read & log
    def log(self, key, value):
        pass

    def log_dict(self, prefix=''):
        if prefix != '': prefix = f"{prefix}/"
        for key, value in self.data.items():
            self.log(f'{prefix}{key}', value)

    # run
    @logger_used
    def before_run(self):
        pass

    @logger_used
    def after_run(self):
        pass

    # epoch
    @logger_used
    def before_train_epoch(self):
        if self.train_epoch % self.train_epoch_interval == 0:
            self.log_dict('train')
        self.clear_data()

    @logger_used
    def after_train_epoch(self):
        if self.train_epoch % self.train_epoch_interval == 0:
            self.log_dict('train')
        self.clear_data()
        self.train_epoch += 1

    @logger_used
    def after_val_epoch(self):
        if self.val_epoch % self.val_epoch_interval == 0:
            self.log_dict('val')
        self.clear_data()
        self.val_epoch += 1

    @logger_used
    def before_val_epoch(self):
        if self.val_epoch % self.val_epoch_interval == 0:
            self.log_dict('val')
        self.clear_data()

    # iter
    @logger_used
    def before_train_iter(self):
        if self.train_iter % self.train_iter_interval == 0:
            self.log_dict('train')
        self.clear_data()

    @logger_used
    def after_train_iter(self):
        if self.train_iter % self.train_iter_interval == 0:
            self.log_dict('train')
        self.clear_data()
        self.train_iter += 1

    @logger_used
    def before_val_iter(self):
        if self.val_iter % self.val_iter_interval == 0:
            self.log_dict('val')
        self.clear_data()

    @logger_used
    def after_val_iter(self):
        if self.val_iter % self.val_iter_interval == 0:
            self.log_dict('val')
        self.clear_data()
        self.val_iter += 1
