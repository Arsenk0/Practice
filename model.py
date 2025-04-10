# model.py

import torch
import torch.nn as nn
from torchvision import models

class ArtifactDetector(nn.Module):
    def __init__(self):
        super(ArtifactDetector, self).__init__()
        # Завантажуємо попередньо навчений ResNet18
        self.resnet = models.resnet18(pretrained=True)
        # Замінюємо останній шар для бінарної класифікації
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x
