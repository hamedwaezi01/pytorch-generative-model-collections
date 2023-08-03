import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == "emnist":
        data_loader = DataLoader(
            datasets.EMNIST('data/emnist', train=True, download=True, transform=transform, split="bymerge"),
            batch_size=batch_size, shuffle=True)
    elif dataset == "emnist-cgan-aug":
        with open("data/emnist/augmented/cgan.pkl", "rb") as f:
            data_loader = DataLoader(
                torch.load(f)
            )
    elif dataset == "emnist-acgan-aug":
        with open("data/emnist/augmented/acgan.pkl", "rb") as f:
            data_loader = DataLoader(
                torch.load(f)
            )
    return data_loader