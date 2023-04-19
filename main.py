import argparse
import os
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import distributed as dist
from functools import partial

from dscv.datasets.builder import build_dataset, build_dataloader
from dscv.models.builder import build_model
from dscv.optimizer.builder import build_optimizer
from dscv.runner.builder import build_runner
from dscv.utils.config import Config
from dscv.utils.dist_utils import get_dist_info, worker_init_fn

parser = argparse.ArgumentParser(description='Trains an ImageNet Classifier')
parser.add_argument('cfg', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--local_rank', type=int, default=0)
group_gpus = parser.add_mutually_exclusive_group()
group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
group_gpus.add_argument(
    '--gpu-ids',
    type=int,
    nargs='+',
    help='ids of gpus to use '
    '(only applicable to non-distributed training)')
parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(0)


def main():
    # GUIDE: Build configurations
    cfg = Config()
    cfg.load(args.cfg)
    if args.debug:
        cfg.data.num_workers = 0
        cfg.data.batch_size = 3
        if cfg.runner.get('logger'):
            if cfg.runner.logger.type == 'wandb_logger':
                cfg.runner.logger.use_wandb=False

    if cfg.get('seed') is None:
        cfg.seed = args.seed

    # GUIDE: Use multiple GPUs
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
        rank = 0
    elif args.launcher == 'pytorch':
        distributed = True
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(**cfg.dist_params)

        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    else:
        raise TypeError('Unknown launcher type: {}'.format(args.launcher))
    print(f"distributed: {distributed} (rank: {rank}, gpu_ids: {cfg.gpu_ids})")

    main_worker(rank, len(cfg.gpu_ids), cfg, distributed=distributed)


def main_worker(rank, num_gpus, cfg, distributed=False):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # GUIDE: Build dataset
    samples_per_gpu = cfg.data.batch_size // len(cfg.gpu_ids)
    workers_per_gpu = cfg.data.num_workers // len(cfg.gpu_ids)

    if distributed:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = cfg.data.batch_size
        num_workers = cfg.data.num_workers

    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=num_gpus, rank=rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=num_gpus, rank=rank, shuffle=False)
    else:
        train_sampler, val_sampler = None, None

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=cfg.seed)
    loader_cfg = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=init_fn, pin_memory=False)
    train_loader = build_dataloader(train_dataset, **loader_cfg, sampler=train_sampler, drop_last=True)
    val_loader = build_dataloader(val_dataset, **loader_cfg, sampler=val_sampler)
    data_loaders = [train_loader, val_loader]

    # GUIDE: Build model
    model = build_model(cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))

    device = torch.cuda.current_device()
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        model = torch.nn.DataParallel(model).to(device)

    # GUIDE: Build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    start_epoch = 0
    if cfg.runner.get('resume'):
        if os.path.isfile(cfg.runner.resume):
            print("=> loading checkpoint '{}'".format(cfg.runner.resume))
            if cfg.gpu is None:
                checkpoint = torch.load(cfg.runner.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(cfg.runner.resume, map_location=loc)

            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    cudnn.benchmark = True

    # GUIDE: Logger
    if cfg.runner.get('save'):
        if not os.path.exists(cfg.runner.save):
            os.makedirs(cfg.runner.save)

    # GUIDE: Train
    runner = build_runner(cfg.runner, model, optimizer, distributed=distributed)
    runner.run(data_loaders, start_epoch=start_epoch)

if __name__ == '__main__':
    main()