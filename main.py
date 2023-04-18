import argparse
import os
import torch
import numpy as np

from dscv.datasets.builder import build_dataset, build_dataloader
from dscv.models.builder import build_model
from dscv.optimizer.builder import build_optimizer
from dscv.runner.builder import build_runner
from dscv.utils.config import Config

parser = argparse.ArgumentParser(description='Trains an ImageNet Classifier')
parser.add_argument('cfg', type=str)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# GUIDE: Build configurations
cfg = Config()
cfg.load(args.cfg)
if args.debug:
    cfg.data.num_workers = 0
    cfg.data.batch_size = 3
    if cfg.runner.get('logger'):
        if cfg.runner.logger.type == 'wandb_logger':
            cfg.runner.logger.use_wandb=False

# GUIDE: Build dataset
train_dataset = build_dataset(cfg.data.train)
val_dataset = build_dataset(cfg.data.val)
train_loader = build_dataloader(train_dataset,
                                batch_size=cfg.data.batch_size,
                                num_workers=cfg.data.num_workers,
                                shuffle=True)
val_loader = build_dataloader(val_dataset,
                              batch_size=cfg.data.batch_size,
                              num_workers=cfg.data.num_workers,
                              shuffle=False)
data_loaders = [train_loader, val_loader]

# GUIDE: Build model
model = build_model(cfg.model,
                    train_cfg=cfg.get('train_cfg'),
                    test_cfg=cfg.get('test_cfg'))
# model.init_weights()

# GUIDE: Build optimizer
optimizer = build_optimizer(model, cfg.optimizer)

start_epoch = 0
if cfg.runner.get('resume'):
    if os.path.isfile(cfg.runner.resume):
        checkpoint = torch.load(cfg.runner.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

# GUIDE: Logger
if cfg.runner.get('save'):
    if not os.path.exists(cfg.runner.save):
        os.makedirs(cfg.runner.save)

# GUIDE: Train
torch.manual_seed(1)
np.random.seed(1)

runner = build_runner(cfg.runner, model, optimizer)
runner.run(data_loaders, start_epoch=start_epoch)
