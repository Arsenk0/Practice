import torch
import torch.nn as nn
from torchvision import models


class ArtifactDetector(nn.Module):
    def __init__(self):
        super(ArtifactDetector, self).__init__()
        # Завантажуємо ResNet18 без останнього шару
        self.resnet = models.resnet18(pretrained=True)

        # Заміняємо останній шар на 1 вихідний нейрон для бінарної класифікації
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

        # Sigmoid для бінарної класифікації
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)
