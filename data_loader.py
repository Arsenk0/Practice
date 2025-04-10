# data_loader.py

import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Зміна розміру зображень
        transforms.ToTensor(),  # Перетворення в тензор
        transforms.Normalize(  # Нормалізація для ImageNet моделей
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class CustomImageDataset(datasets.ImageFolder):
    def __init__(self, root_dir, transform=None):
        super().__init__(root=root_dir, transform=transform)


def get_dataloaders(train_dir, test_dir, batch_size=32):
    transform = get_transforms()

    # Перевірка, чи існують каталоги для даних
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory {train_dir} does not exist!")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory {test_dir} does not exist!")

    # Завантаження тренувальних та тестових даних
    train_dataset = CustomImageDataset(train_dir, transform=transform)
    test_dataset = CustomImageDataset(test_dir, transform=transform)

    # Перевірка чи є зображення в даних
    if len(train_dataset) == 0:
        raise ValueError(f"No images found in the training directory: {train_dir}")
    if len(test_dataset) == 0:
        raise ValueError(f"No images found in the test directory: {test_dir}")

    # Завантаження даних в DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
