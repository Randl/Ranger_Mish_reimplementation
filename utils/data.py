import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def woof_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.35, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def get_transform_imagenet(augment=True, input_size=224):
    normalize = __imagenet_stats
    scale_size = int(input_size / 0.875)
    if augment:
        return woof_preproccess(input_size=input_size, normalize=normalize)
    else:
        return scale_crop(input_size=input_size, scale_size=scale_size, normalize=normalize)


def get_transform():
    return transforms.ToTensor()


def get_loaders_imagenet(dataroot, val_batch_size, train_batch_size, input_size, workers, num_nodes, local_rank):
    # TODO: pin-memory currently broken for distributed
    pin_memory = False
    # TODO: datasets.ImageNet
    val_data = datasets.ImageFolder(root=os.path.join(dataroot, 'val'),
                                    transform=get_transform_imagenet(False, input_size))
    val_sampler = DistributedSampler(val_data, num_nodes, local_rank)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, sampler=val_sampler,
                                             num_workers=workers, pin_memory=pin_memory)

    train_data = datasets.ImageFolder(root=os.path.join(dataroot, 'train'),
                                      transform=get_transform_imagenet(input_size=input_size))
    train_sampler = DistributedSampler(train_data, num_nodes, local_rank)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, sampler=train_sampler,
                                               num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_loaders_cifar(dataroot, val_batch_size, train_batch_size, input_size,
                      workers, num_nodes, local_rank, dataset='CIFAR10'):
    # TODO: pin-memory currently broken for distributed
    pin_memory = False
    val_data = datasets.CIFAR10(root=dataroot, train=False, transform=get_transform(),
                                download=True) if dataset == 'CIFAR10' else \
        datasets.CIFAR100(root=dataroot, train=False, transform=get_transform(), download=True)

    val_sampler = DistributedSampler(val_data, num_nodes, local_rank)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, sampler=val_sampler,
                                             num_workers=workers, pin_memory=pin_memory)

    train_data = datasets.CIFAR10(root=dataroot, train=True, transform=get_transform(),
                                  download=True) if dataset == 'CIFAR10' else \
        datasets.CIFAR100(root=dataroot, train=True, transform=get_transform(), download=True)
    train_sampler = DistributedSampler(train_data, num_nodes, local_rank)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, sampler=train_sampler,
                                               num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_loaders_mnist(dataroot, val_batch_size, train_batch_size, input_size,
                      workers, num_nodes, local_rank):
    # TODO: pin-memory currently broken for distributed
    pin_memory = False
    val_data = datasets.MNIST(root=dataroot, train=False, transform=get_transform(), download=True)

    val_sampler = DistributedSampler(val_data, num_nodes, local_rank)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, sampler=val_sampler,
                                             num_workers=workers, pin_memory=pin_memory)

    train_data = datasets.MNIST(root=dataroot, train=True, transform=get_transform())
    train_sampler = DistributedSampler(train_data, num_nodes, local_rank)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, sampler=train_sampler,
                                               num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_loaders(dataroot, val_batch_size, train_batch_size, input_size,
                workers, num_nodes, local_rank, dataset='ImageNet'):
    if dataset == 'ImageNet':
        return get_loaders_imagenet(dataroot, val_batch_size, train_batch_size,
                                    input_size, workers, num_nodes, local_rank)
    elif dataset == 'CIFAR10':
        return get_loaders_cifar(dataroot, val_batch_size, train_batch_size,
                                 input_size, workers, num_nodes, local_rank, dataset)
    elif dataset == 'CIFAR100':
        return get_loaders_cifar(dataroot, val_batch_size, train_batch_size,
                                 input_size, workers, num_nodes, local_rank, dataset)
    elif dataset == 'MNIST':
        return get_loaders_mnist(dataroot, val_batch_size, train_batch_size,
                                 input_size, workers, num_nodes, local_rank)
    else:
        raise ValueError("Wrong dataset!")
