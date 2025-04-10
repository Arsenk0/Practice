# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from model import ArtifactDetector
from data_loader import get_dataloaders
import os

def train_model(train_loader, val_loader, epochs=10, lr=0.001, device='cuda'):
    # ініціалізуємо модель та переміщаємо її на пристрій
    model = ArtifactDetector().to(device)
    criterion = torch.nn.BCELoss()  # функція втрат для бінарної класифікації
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0  # для збереження найкращої моделі за F1-мітрикою
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)  # отримуємо прогноз моделі
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Виведення статистики на кожній епосі
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # Валідація на кожній епосі
        val_f1 = validate_model(val_loader, model, device)
        print(f"Validation F1: {val_f1}")

        # Зберігаємо модель, якщо F1 покращилась
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print("Модель збережена!")

def validate_model(val_loader, model, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images).squeeze(1)
            preds = (outputs > 0.5).float()  # поріг для класифікації

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Обчислюємо F1-мікро
    return f1_score(all_labels, all_preds, average='micro')

def main():
    # Шляхи до даних
    train_dir = './train'  # Шлях до папки з тренувальними зображеннями
    val_dir = './val'      # Шлях до папки з валідаційними зображеннями

    # Завантаження даних
    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size=32)

    # Використовуємо GPU або CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Навчання моделі
    train_model(train_loader, val_loader, epochs=10, lr=0.001, device=device)

if __name__ == '__main__':
    main()
