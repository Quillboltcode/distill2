import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch


from utils.policy import CIFAR10Policy, SVHNPolicy, ImageNetPolicy
from utils.cutout import Cutout

FER_SIZE = 48
CIFAR_SIZE = 32



data_transforms_CIFAR = {
    'train': transforms.Compose([
        transforms.RandomCrop(CIFAR_SIZE, padding=4, fill=128),
        transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'val': transforms.Compose([
        transforms.RandomCrop(CIFAR_SIZE, padding=4, fill=128),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'test': transforms.Compose([
        transforms.RandomCrop(CIFAR_SIZE, padding=4, fill=128),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
}

mean = 0
std = 255

data_transforms_FER = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(FER_SIZE, scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((FER_SIZE, FER_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((FER_SIZE, FER_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
}

