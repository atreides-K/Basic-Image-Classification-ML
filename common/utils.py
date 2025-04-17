import torch
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_data(
        dataset_name: str,
        batch_size: int = 32,
        transform: transforms = transforms.ToTensor(),
) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == 'mnist':
        train_data = datasets.MNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'fashion_mnist':
        train_data = datasets.FashionMNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        train_data = datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform)
    else:
        raise ValueError('Dataset not supported')

    # for some reason num_workers>0 pauses training after every epoch, so not
    # using it and keeping it at 0 (default value). For more discussion visit:
    # https://discuss.pytorch.org/t/dataloader-with-num-workers-1-hangs-every-epoch/20323/18
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader

def save_training_plot(name, tplt):
    try:
        # Convert lists to numpy arrays before saving
        tplot_np = {key: np.array(value) for key, value in tplt.items()}
        
        np.savez(name, **tplot_np)
        print("History saved using NumPy.")
    except Exception as e:
        print(f"Error saving history with NumPy: {e}")