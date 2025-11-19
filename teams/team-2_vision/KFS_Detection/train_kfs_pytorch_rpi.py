#!/usr/bin/env python3
"""
KFS Binary Classifier - Pure PyTorch Version
Optimized for Raspberry Pi 5 deployment using TorchScript.

Features:
- Recursive dataloading (Real/ScrollXX/* and Fake/ScrollXX/*)
- Strong augmentations to prevent overfitting
- Training stabilizers (Cosine LR schedule, shuffling, augmentation)
- TensorBoard support
- Pure PyTorch TorchScript export for Raspberry Pi
"""

import os
import argparse
from pathlib import Path
import random
from glob import glob

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from PIL import Image

# --------------------------
# Dataset class
# --------------------------
class KFSDataset(Dataset):
    def __init__(self, files, labels, img_size):
        self.files = files
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),

            # Strong augmentations
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# --------------------------
# Model
# --------------------------
def create_model():
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Sequential(
        nn.Linear(model.classifier[3].in_features, 1),
        nn.Sigmoid()
    )
    return model

# --------------------------
# Training loop
# --------------------------
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()

    acc = correct / len(loader.dataset)
    return running_loss / len(loader), acc

# --------------------------
# Validation loop
# --------------------------
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs).squeeze()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()

    acc = correct / len(loader.dataset)
    return running_loss / len(loader), acc

# --------------------------
# TorchScript Export for Raspberry Pi
# --------------------------
def export_torchscript(model, img_size):
    model.eval()
    example = torch.randn(1, 3, img_size, img_size)
    traced = torch.jit.trace(model.cpu(), example)
    traced.save("model_rpi5.pt")
    print("[OK] TorchScript model saved: model_rpi5.pt (Use on Raspberry Pi)")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root path")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    DATASET = Path(args.data)

    # --------------------------
    # Recursive dataset scanning
    # --------------------------
    real_dir = DATASET / "Real"
    fake_dir = DATASET / "Fake"

    real_files = sorted([
        f for f in glob(str(real_dir / "**" / "*"), recursive=True)
        if f.lower().endswith(("jpg", "jpeg", "png"))
    ])

    fake_files = sorted([
        f for f in glob(str(fake_dir / "**" / "*"), recursive=True)
        if f.lower().endswith(("jpg", "jpeg", "png"))
    ])

    print(f"[INFO] Found REAL images: {len(real_files)}")
    print(f"[INFO] Found FAKE images: {len(fake_files)}")

    files = real_files + fake_files
    labels = [1] * len(real_files) + [0] * len(fake_files)

    if len(files) == 0:
        print("ERROR: Dataset empty. Check folder names: Real/ Fake/")
        exit()

    # Train/validation split
    combined = list(zip(files, labels))
    random.shuffle(combined)
    files, labels = zip(*combined)

    split = int(0.85 * len(files))
    train_files, val_files = files[:split], files[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    print(f"[INFO] Train size: {len(train_files)}  |  Val size: {len(val_files)}")

    # --------------------------
    # Datasets + Loaders
    # --------------------------
    train_ds = KFSDataset(train_files, train_labels, args.img_size)
    val_ds = KFSDataset(val_files, val_labels, args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --------------------------
    # Model setup
    # --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = create_model().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter("runs/kfs_training")

    # --------------------------
    # Training loop
    # --------------------------
    best_acc = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

        print(f"Epoch {epoch+1}/{args.epochs}  | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("[OK] Best model updated")

    print("\nTraining complete. Best val accuracy:", best_acc)

    # --------------------------
    # Export TorchScript for Raspberry Pi
    # --------------------------
    export_torchscript(model, args.img_size)

    print("\nAll done!")
