import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
DATASET_DIR = "/home/viraj/Robocon/Team-Vulcans-Robocon-2026/teams/team-2_vision/dataset/final"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------
# Dataset Loader
# ---------------------------------------
class KFSDataset(Dataset):
    def __init__(self, files, labels, augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment

        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.HueSaturationValue(20, 30, 20, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, p=0.5),
            A.RandomShadow(p=0.3),
            A.RandomFog(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ]) if augment else A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(image=img)["image"]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


# ---------------------------------------
# Load Files
# ---------------------------------------
real_files = glob(os.path.join(DATASET_DIR, "real", "*.jpg"))
fake_files = glob(os.path.join(DATASET_DIR, "fake", "*.jpg"))

files = real_files + fake_files
labels = [1] * len(real_files) + [0] * len(fake_files)

train_f, val_f, train_l, val_l = train_test_split(files, labels, test_size=0.2, shuffle=True)

train_ds = KFSDataset(train_f, train_l, augment=True)
val_ds = KFSDataset(val_f, val_l, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# ---------------------------------------
# Model
# ---------------------------------------
print("🚀 Loading MobileNetV3 Small (ImageNet pretrained)...")

model = mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = nn.Linear(1024, 1)

model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()


# ---------------------------------------
# Training Loop
# ---------------------------------------
def train_one_epoch():
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, desc="Training"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(imgs)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate():
    model.eval()
    total_acc = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            output = model(imgs)
            preds = (torch.sigmoid(output) > 0.5).float()
            total_acc += (preds == labels).float().mean().item()

    return total_acc / len(val_loader)


# ---------------------------------------
# Run Training
# ---------------------------------------
print("\n🔥 Starting Training on", DEVICE)
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch()
    val_acc = validate()
    print(f"\n📌 Epoch {epoch}/{EPOCHS} — Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}\n")

    torch.save(model.state_dict(), f"kfs_mobilenetv3_epoch{epoch}.pth")

print("✅ Training complete!")
