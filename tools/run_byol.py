'''
@misc{grill2020bootstrap,
    title = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
    author = {Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
    year = {2020},
    eprint = {2006.07733},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
'''
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

from thirdparty.byol_pytorch.byol_pytorch import BYOL
from thirdparty.cifar10_models.resnet import resnet50


parser = argparse.ArgumentParser(description='run BYOL')
# Dataset
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--data_dir', type=str, default='/ws/data')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='learning_rate')
parser.add_argument('--batch-size', '-b', type=int, default=128)
# Acceleration
parser.add_argument('--num-workers', type=int, default=2, help='Number of pre-fetching threads.')
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
    wandb_config = dict(project='Classification', entity='kaist-url-ai28', name='cifar10_byol_original')
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
    resnet = resnet50(pretrained=True)
    model = BYOL(
        resnet,
        image_size=32,
        hidden_layer='avgpool',
        projection_size=len(classes),
        projection_hidden_size=256,
        moving_average_decay=0.99,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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

        train(train_loader, model, optimizer, epoch, args, wandblogger, type='train')

        evaluate(val_loader, model, epoch, args, wandblogger, type='val')

        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

    test_c(val_loader, model, args, wandblogger, type='test_c')

    print("Done!")


def train(train_loader, model, optimizer, epoch, args, wandblogger, type='train'):

    wandblogger.before_train_epoch()
    print(f"Train start!")

    model.train()

    total_loss, total_acc1, total_acc5 = 0., 0., 0.
    end = time.time()
    for i, (images, targets) in enumerate(train_loader):

        data_time = time.time() - end

        wandblogger.before_train_iter()

        optimizer.zero_grad()

        images = images.to(args.device)
        targets = targets.to(args.device)

        outputs = model(images)

        loss = outputs['loss']

        loss.backward()
        optimizer.step()

        logits_one = outputs['online_pred_one']
        logits_two = outputs['online_pred_two']
        acc1_one, acc5_one = accuracy(logits_one, targets, topk=(1, 5))
        acc1_two, acc5_two = accuracy(logits_two, targets, topk=(1, 5))
        acc1 = (acc1_one + acc1_two) / 2.
        acc5 = (acc5_one + acc5_two) / 2.

        wandblogger.add_data(f'{type}/data_time', data_time)
        wandblogger.add_data(f'{type}/loss', loss)
        wandblogger.add_data(f'{type}/acc1', acc1)
        wandblogger.add_data(f'{type}/acc5', acc5)
        wandblogger.after_train_iter()

        total_loss += loss
        total_acc1 += acc1
        total_acc5 += acc5

        end = time.time()

    wandblogger.after_train_epoch()

    data_size = len(train_loader.dataset)
    print(f"[{epoch}/{args.epochs}] loss: {total_loss/data_size:.2f}  acc1: {total_acc1/data_size:.2f}  acc5: {total_acc5/data_size:.2f}")


def evaluate(val_loader, model, epoch, args, wandblogger, type='val'):

    wandblogger.before_val_epoch()
    print(f"Evaluation start!")

    model.eval()

    total_acc1, total_acc5 = 0., 0.
    confusion_matrix = torch.zeros(10, 10)

    with torch.no_grad():

        end = time.time()

        for i, (images, targets) in enumerate(val_loader):

            data_time = time.time() - end

            wandblogger.before_val_iter()

            images = images.to(args.device)
            targets = targets.to(args.device)

            outputs = model.evaluate(images)

            logits = outputs['online_pred']
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            total_acc1 += acc1
            total_acc5 += acc5

            wandblogger.add_data(f'{type}/data_time', data_time)
            wandblogger.after_val_iter()

            pred = logits.data.max(1)[1]
            for t, p in zip(targets.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            end = time.time()

        data_size = len(val_loader.dataset)
        wandblogger.add_data(f'{type}/acc1', total_acc1 / data_size)
        wandblogger.add_data(f'{type}/acc5', total_acc5 / data_size)
        wandblogger.add_data(f'{type}/confusion_matrix', confusion_matrix)
        wandblogger.after_val_epoch()
        print(f"[{epoch}/{args.epochs}] acc1: {total_acc1/data_size:.2f}  acc5: {total_acc5/data_size:.2f}")


def test(test_loader, model, args):

    model.eval()

    total_acc1, total_acc5 = 0., 0.
    confusion_matrix = torch.zeros(10, 10)
    outputs = dict()

    with torch.no_grad():

        end = time.time()

        for i, (images, targets) in enumerate(test_loader):

            data_time = time.time() - end

            images = images.to(args.device)
            targets = targets.to(args.device)

            outputs = model.evaluate(images)

            logits = outputs['online_pred']
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            total_acc1 += acc1
            total_acc5 += acc5

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

    print("Test start!")

    results_table = pd.DataFrame(columns=CORRUPTIONS, index=['acc1', 'acc5'])

    cnt = 0
    for corruption in CORRUPTIONS:
        print(f"[{cnt}/{len(CORRUPTIONS)}] ", end='')
        val_loader.dataset.data = np.load(f'{val_loader.dataset.data_c_dir}' + corruption + '.npy')
        val_loader.dataset.targets = torch.LongTensor(np.load(val_loader.dataset.data_c_dir + 'labels.npy'))
        test_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers)

        outputs = test(test_loader, model, args)

        results_table[corruption]['acc1'] = outputs['acc1'] * 100
        results_table[corruption]['acc5'] = outputs['acc5'] * 100
        cnt += 1
        print(f"acc1: {outputs['acc1']:.2f}  acc5: {outputs['acc5']:.2f}")

    results_table = wandblogger.wandb.Table(data=results_table)
    wandblogger.add_data(f"{type}/results", results_table)

    mean_accs = results_table.mean(axis=1, skipna=True)
    wandblogger.add_data(f"{type}/acc1", mean_accs.acc1)
    wandblogger.add_data(f"{type}/acc1", mean_accs.acc5)
    wandblogger.log_all_data()


if __name__ == '__main__':
    main()
