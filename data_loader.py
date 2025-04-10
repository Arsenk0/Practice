# data_loader.py

import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # стандартна нормалізація
    ])


class CustomImageDataset(ImageFolder):
    def __init__(self, root_dir, transform=None):
        super().__init__(root=root_dir, transform=transform)


def get_dataloaders(train_dir, val_dir, batch_size=32):
    transform = get_transforms()
    train_dataset = CustomImageDataset(train_dir, transform=transform)
    val_dataset = CustomImageDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
