#!/usr/bin/env python3
"""
train_kfs_pytorch_with_augmentation.py

KFS binary classifier (Real vs Fake) using:
- MobileNetV3-Large (pretrained on ImageNet)
- torchvision-only augmentations
- Mixed precision (AMP) when CUDA is available
- Optional MixUp (enabled here with alpha=0.2)
- EarlyStopping on validation AUC
- ReduceLROnPlateau scheduler on validation AUC
- TensorBoard logging
- TorchScript export for Raspberry Pi-style deployment
"""

import time
import math
import random
from glob import glob
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

import torchvision.transforms as T
from torchvision.models import mobilenet_v3_large


# --------------------------
# Utilities
# --------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_autocast_ctx(device: str):
    """Return an appropriate autocast co4ntext manager depending on device."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda")
    else:
        # CPU autocast (mainly for API consistency)
        try:
            return torch.cpu.amp.autocast()
        except Exception:
            from contextlib import nullcontext
            return nullcontext()


# --------------------------
# Dataset (torchvision transforms)
# --------------------------
class KFS_TorchDataset(Dataset):
    def __init__(self, files, labels, img_size=224, augment=False):
        self.files = list(files)
        self.labels = list(labels)
        self.img_size = img_size
        self.augment = augment

        # Train transforms (simple, robust torchvision pipeline)
        self.train_transform = T.Compose([
            T.Resize(int(img_size * 1.15)),  # small upsample for cropping
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)],
                p=0.6,
            ),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))],
                p=0.2,
            ),
            T.ToTensor(),  # must be before RandomErasing
            T.RandomErasing(p=0.1, scale=(0.02, 0.2)),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # Validation transforms: deterministic, no strong augments
        self.val_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        label = float(self.labels[idx])

        # Read with cv2 (BGR) -> convert to RGB -> PIL Image
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            img = Image.open(p).convert("RGB")
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)

        if self.augment:
            img_t = self.train_transform(img)
        else:
            img_t = self.val_transform(img)

        return img_t, torch.tensor(label, dtype=torch.float32)


# --------------------------
# MixUp
# --------------------------
def mixup_data(x, y, alpha=0.4):
    """
    x: images [B, C, H, W]
    y: labels [B, 1]
    """
    if alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# --------------------------
# Training / Validation
# --------------------------
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    mixup_alpha=0.0,
    writer=None,
    epoch=0,
):
    model.train()
    running_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Train E{epoch}")
    autocast_ctx = get_autocast_ctx(device)

    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)  # [B, 1]

        optimizer.zero_grad()

        if mixup_alpha > 0:
            mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
            with autocast_ctx:
                outputs = model(mixed_imgs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            with autocast_ctx:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix({'loss': running_loss / n})

    return running_loss / max(1, n)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds = []
    trues = []
    pbar = tqdm(loader, desc="Val")
    autocast_ctx = get_autocast_ctx(device)

    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        with autocast_ctx:
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds.extend(probs.tolist())
        trues.extend(labels.cpu().numpy().flatten().tolist())

    preds = np.array(preds)
    trues = np.array(trues)
    pred_labels = (preds > 0.5).astype(int)
    acc = (pred_labels == trues).mean()
    try:
        auc = roc_auc_score(trues, preds)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(trues, pred_labels)
    return acc, auc, cm, preds, trues


# --------------------------
# TorchScript export
# --------------------------
def export_torchscript(model, img_size, out_path="kfs_mobilenetv3_large_rpi.pt"):
    model.eval()
    cpu_model = model.cpu()
    example = torch.randn(1, 3, img_size, img_size)
    traced = torch.jit.trace(cpu_model, example)
    traced.save(out_path)
    print(f"[OK] TorchScript saved -> {out_path}")


# --------------------------
# Model wrapper
# --------------------------
class MobileNetWrapper(nn.Module):
    """Wraps MobileNetV3-Large backbone and adds a single-logit head."""
    def __init__(self, backbone: nn.Module, feat_channels: int):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feat_channels, 1)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.pool(x)          # [B, C, 1, 1]
        x = torch.flatten(x, 1)   # [B, C]
        x = self.fc(x)            # [B, 1]
        return x


# --------------------------
# Main (hardcoded config)
# --------------------------
def main():
    # ===== Hardcoded config =====
    data_root = Path(
        "D:/Robotics Club/Robocon2026/Team-Vulcans-Robocon-2026/teams/team-2_vision/DatasetIRL"
    )
    img_size = 224
    batch_size = 16
    epochs = 30
    lr = 1e-4
    mixup_alpha = 0.2       # set to 0.0 to disable MixUp
    num_workers = 4
    patience = 6            # early stopping patience (epochs)
    seed = 42
    save_dir = Path("outputs_mobilenetv3")
    # ============================

    seed_everything(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    real_dir = data_root / "Real"
    fake_dir = data_root / "Fake"

    real_files = sorted([
        p for p in glob(str(real_dir / "**" / "*"), recursive=True)
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    fake_files = sorted([
        p for p in glob(str(fake_dir / "**" / "*"), recursive=True)
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"[INFO] Real images: {len(real_files)}  |  Fake images: {len(fake_files)}")
    files = real_files + fake_files
    labels = [1] * len(real_files) + [0] * len(fake_files)

    if len(files) == 0:
        print("ERROR: No images found. Check Real/ and Fake/ folders.")
        return

    # Stratified split (preserve class ratio)
    train_files, val_files, train_labels, val_labels = train_test_split(
        files,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=seed,
    )
    print(f"[INFO] Train: {len(train_files)}  Val: {len(val_files)}")

    train_ds = KFS_TorchDataset(train_files, train_labels, img_size=img_size, augment=True)
    val_ds = KFS_TorchDataset(val_files, val_labels, img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    # Load backbone
    backbone = mobilenet_v3_large(weights="IMAGENET1K_V1")
    backbone.eval()

    # Infer feature channels by passing a dummy through backbone.features
    with torch.no_grad():
        dummy = torch.randn(1, 3, img_size, img_size)
        feat = backbone.features(dummy)
        feat_dim = feat.shape[1]
    print("[INFO] Detected backbone feature channels:", feat_dim)

    model = MobileNetWrapper(backbone, feat_dim).to(device)

    # Loss / optimizer / scheduler
    pos_weight = torch.tensor(
        (len(labels) - sum(labels)) / (sum(labels) + 1e-6)
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    scaler = torch.amp.GradScaler(enabled=(device == "cuda")) if torch.cuda.is_available() else None

    writer = SummaryWriter(log_dir=str(save_dir / "runs"))
    best_auc = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            mixup_alpha=mixup_alpha,
            writer=writer,
            epoch=epoch,
        )
        val_acc, val_auc, val_cm, val_preds, val_trues = validate(
            model,
            val_loader,
            device,
        )

        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} — "
            f"train_loss: {train_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"val_auc: {val_auc:.4f} | "
            f"time: {epoch_time:.1f}s"
        )
        print("Confusion Matrix:\n", val_cm)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        if not math.isnan(val_auc):
            writer.add_scalar("val/auc", val_auc, epoch)

        scheduler.step(val_auc if not math.isnan(val_auc) else 0.0)

        # Checkpointing by AUC
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            ckpt_path = save_dir / f"best_mobilenetv3_large_epoch{epoch}.pth"
            torch.save(model.state_dict(), str(ckpt_path))
            print("[OK] Saved best checkpoint ->", ckpt_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[STOP] Early stopping triggered (no improvement for {patience} epochs).")
            break

    # Load best if exists
    ckpt_candidates = sorted(save_dir.glob("best_mobilenetv3_large_epoch*.pth"))
    if ckpt_candidates:
        best_ckpt = ckpt_candidates[-1]
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("[INFO] Loaded best checkpoint:", best_ckpt)

    # Export TorchScript
    export_torchscript(
        model,
        img_size,
        out_path=str(save_dir / "kfs_mobilenetv3_large_rpi.pt"),
    )

    writer.close()
    print("[DONE] Training & export complete. Outputs in:", save_dir)


if __name__ == "__main__":
    main()
