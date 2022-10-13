from __future__ import print_function

import argparse
import os
import time
import torch
import math

import numpy as np
import pandas as pd

from dscv.datasets.utils import get_dataset, get_classes
from dscv.utils.wandb_logger import WandbLogger
from dscv.utils.utils import accuracy, get_lr, save_checkpoint, CORRUPTIONS


# TODO: Set the default arguments
parser = argparse.ArgumentParser(description='default run')
# Dataset
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--data_dir', type=str, default='/ws/data')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='learning_rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--batch-size', '-b', type=int, default=128)
# Acceleration
parser.add_argument('--num-workers', type=int, default=4, help='Number of pre-fetching threads.')
# Logging
parser.add_argument('--wandb', '-wb', action='store_true', default=False, dest='wandb')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--device', default='cuda', type=str)

def main():
    args = parser.parse_args()
    torch.manual_seed(1)
    np.random.seed(1)

    #########################
    ### Initialize logger ###
    #########################
    # TODO: fill the name
    wandb_config = dict(project='Classification', entity='kaist-url-ai28', name='')
    wandblogger = WandbLogger(wandb_config,
                              train_epoch_interval=1, train_iter_interval=100,
                              val_epoch_interval=1, val_iter_interval=10,
                              use_wandb=args.wandb)
    wandblogger.before_run()

    ###############
    ### Dataset ###
    ###############
    train_dataset, val_dataset = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    classes = get_classes(args)

    #############
    ### Model ###
    #############
    model = None.to(args.device) # TODO

    criterion = None # TODO

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda step: get_lr(step, args.epochs * len(train_loader),
                                                                                1, 1e-6 / args.learning_rate))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #############
    ### Train ###
    #############
    for epoch in range(args.start_epoch, args.epochs):

        wandblogger.before_train_epoch()

        train(train_loader, model, criterion, optimizer, scheduler, epoch, args, wandblogger, type='train')

        evaluate(val_loader, model, criterion, optimizer, scheduler, epoch, args, wandblogger, type='val')

        save_checkpoint({
            'state_dict': model.state_dict(),
            'criterion': criterion,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

    test_c(val_loader, model, args, wandblogger, type='test_c')


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, wandblogger, type='train'):

    wandblogger.before_train_epoch()

    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):

        data_time = time.time() - end

        wandblogger.before_train_iter()

        optimizer.zero_grad()

        images = images.to(args.device)
        target = targets.to(args.device)

        outputs = model(images)

        # TODO: Design your loss function
        loss = criterion(outputs)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # TODO: Get logits from outputs
        logits = outputs['logits']
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        wandblogger.add_data(f'{type}/data_time', data_time)
        wandblogger.add_data(f'{type}/loss', loss)
        wandblogger.add_data(f'{type}/acc1', acc1)
        wandblogger.add_data(f'{type}/acc5', acc5)
        wandblogger.after_train_iter()

        end = time.time()

    wandblogger.after_train_epoch()


def evaluate(val_loader, model, criterion, optimizer, scheduler, epoch, args, wandblogger, type='val'):

    wandblogger.before_val_epoch()

    model.eval()

    total_loss, total_acc1, total_acc5 = 0., 0., 0.
    confusion_matrix = torch.zeros(10, 10)

    with torch.no_grad():

        end = time.time()

        for i, (images, targets) in enumerate(val_loader):

            data_time = time.time() - end

            wandblogger.before_val_iter()

            images = images.to(args.device)
            target = targets.to(args.device)

            outputs = model(images)

            # TODO: Design your loss function
            loss = criterion(outputs)
            total_loss += loss

            # TODO: Get logits from outputs
            logits = outputs['logits']
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            total_acc1 += acc1
            total_acc5 += acc5

            wandblogger.add_data(f'{type}/data_time', data_time)
            wandblogger.after_val_iter()

            # TODO: Replace 'logits' to valid value.
            pred = logits.data.max(1)[1]
            for t, p in zip(targets.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            end = time.time()

        data_size = len(val_loader.dataset)
        wandblogger.add_data(f'{type}/loss', total_loss / data_size)
        wandblogger.add_data(f'{type}/acc1', total_acc1 / data_size)
        wandblogger.add_data(f'{type}/acc5', total_acc5 / data_size)
        wandblogger.add_data(f'{type}/confusion_matrix', confusion_matrix)
        wandblogger.after_val_epoch()


def test(test_loader, model, criterion, args):

    model.eval()

    total_acc1, total_acc5 = 0., 0.
    confusion_matrix = torch.zeros(10, 10)
    outputs = dict()

    with torch.no_grad():

        end = time.time()

        for i, (images, targets) in enumerate(test_loader):

            data_time = time.time() - end

            images = images.to(args.device)
            target = targets.to(args.device)

            outputs = model(images)

            # TODO: Get logits from outputs
            logits = outputs['logits']
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            total_acc1 += acc1
            total_acc5 += acc5

            # TODO: Replace 'logits' to valid value.
            pred = logits.data.max(1)[1]
            for t, p in zip(targets.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            end = time.time()

        data_size = len(test_loader.dataset)
        outputs.update({'acc1': total_acc1 / data_size,
                        'acc5': total_acc5 / data_size,
                        'confusion_matrix': confusion_matrix})

    return outputs


def test_c(val_loader, model, args, wandblogger, type='test_c'):

    results_table = pd.DataFrame(columns=CORRUPTIONS, index=['acc1', 'acc5'])

    for corruption in CORRUPTIONS:
        val_loader.dataset.data = np.load(f'{val_loader.dataset.data_c_dir}' + corruption + '.npy')
        val_loader.dataset.targets = torch.LongTensor(np.load(val_loader.dataset.data_c_dir + 'labels.npy'))
        test_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers)

        outputs = test(test_loader, model, args)

        results_table[corruption]['acc1'] = outputs['acc1'] * 100
        results_table[corruption]['acc5'] = outputs['acc5'] * 100

    results_table = wandblogger.wandb.Table(data=results_table)
    wandblogger.add_data(f"{type}/results", results_table)

    mean_accs = results_table.mean(axis=1, skipna=True)
    wandblogger.add_data(f"{type}/acc1", mean_accs.acc1)
    wandblogger.add_data(f"{type}/acc1", mean_accs.acc5)
    wandblogger.log_all_data()


if __name__ == '__main__':
    main()
