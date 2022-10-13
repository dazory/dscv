import os

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets


def get_dataset(args):
    if args.dataset == 'cifar10':
        mean, std = [0.5] * 3, [0.5] * 3
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std),])
        val_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

        data_dir = f'{args.data_dir}/cifar'
        data_c_dir = f'{data_dir}/CIFAR-10-C/'

        train_dataset = datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR10(data_dir, train=False, transform=val_transform, download=True)
        val_dataset.data_c_dir = data_c_dir

    elif args.dataset == 'cifar100':
        mean, std = [0.5] * 3, [0.5] * 3
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std),])
        val_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

        data_dir = f'{args.data_dir}/cifar'
        data_c_dir = f'{data_dir}/CIFAR-10-C'

        train_dataset = datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR100(data_dir, train=False, transform=val_transform, download=True)
        val_dataset.data_c_dir = data_c_dir

    elif args.dataset == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std),])
        val_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

        data_dir = f'{args.data_dir}/imagenet'
        data_c_dir = f'{data_dir}/imagenet-c'

        train_dataset = datasets.ImageFolder(f'{data_dir}/train', train_transform)
        val_dataset = datasets.ImageFolder(f'{data_dir}/val', val_transform)
        val_dataset.data_c_dir = data_c_dir

    else:
        raise TypeError(f"dataset must be in ['cifar10', 'cifar100', 'imagenet'],"
                        f"but got {args.dataset}.")

    return train_dataset, val_dataset


def get_classes(args):
    if args.dataset == 'cifar10':
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == 'cifar100':
        # TODO
        raise NotImplementedError('not implemented for cifar100')
    elif args.dataset == 'imagenet':
        # TODO
        raise NotImplementedError('not implemented for imagenet')
    else:
        raise TypeError(f"dataset must be in ['cifar10', 'cifar100', 'imagenet'],"
                        f"but got {args.dataset}.")
    return classes

