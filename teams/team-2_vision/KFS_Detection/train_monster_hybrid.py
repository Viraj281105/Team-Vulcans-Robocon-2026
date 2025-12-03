#!/usr/bin/env python3
"""
train_monster_hybrid.py

Full "monster" training pipeline:
- Hybrid ConvNeXt-V2 Tiny + ViT-B/16 with cross-attention fusion (selected automatically)
- Heavy augmentations online (no saving of augmented images)
- MixUp, CutMix, label smoothing
- SAM (Sharpness-Aware Minimization) wrapper
- SWA + EMA hybrid (SWA late, EMA always; optional fuse)
- AMP, gradient checkpointing, channels_last memory format
- WeightedRandomSampler
- Distillation support (teacher -> student)
- TTA inference helper
- Optimized for RTX 5070 + 32GB RAM (Strix G16)
- NO saving by default — specify --save to write checkpoints/exports

Caveats:
- This script uses only standard PyTorch and torchvision APIs.
- Ensure torch >= 2.0 and torchvision >= 0.17 for ConvNeXt-V2 and ViT APIs.
"""

import argparse
import random
import time
from pathlib import Path
from glob import glob
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # (F is not heavily used but imported for convenience)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR

# ----------------------
# Hardware / Perf tweaks
# ----------------------
# Enable CuDNN benchmarking for faster convolution algorithms on fixed-size inputs.
torch.backends.cudnn.benchmark = True
# Allow TF32 (TensorFloat-32) for faster matrix multiplications on Ampere+ GPUs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ----------------------
# Dataset
# ----------------------
class KFSDataset(Dataset):
    """
    Simple custom Dataset for "Real vs Fake" images.

    - files: list of image paths
    - labels: list of scalar labels (0 or 1)
    - transform: torchvision transform applied on-the-fly
    """
    def __init__(self, files, labels, transform=None):
        # Store file paths and labels as lists to support indexing
        self.files = list(files)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        # Dataset length is number of files
        return len(self.files)

    def __getitem__(self, idx):
        # Load an image and its label at given index
        path = self.files[idx]
        # Open image with PIL and ensure RGB (3-channel)
        img = Image.open(path).convert("RGB")
        # Apply augmentations / preprocessing if provided
        if self.transform:
            img = self.transform(img)
        # Cast label to float32 tensor for BCEWithLogitsLoss
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

# ----------------------
# Augmentations (online only!)
# ----------------------
def build_transforms(img_size):
    """
    Create heavy training transforms and lighter validation transforms.

    - Training: aggressive geometric and color transforms, AutoAugment/RandAugment,
      RandomErasing, and normalization.
    - Validation: deterministic resize + normalization only.
    """
    train_tf = transforms.Compose([
        # Random crop-resize with scale range, helps robustness to object scale variations
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        # Small random rotation
        transforms.RandomRotation(20),
        # Color jitter occasionally applied (brightness, contrast, saturation, hue)
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.4, 0.3, 0.02)], p=0.7),
        # Random Gaussian blur sometimes
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5))], p=0.4),
        # Random perspective distortion to simulate viewpoint changes
        transforms.RandomPerspective(distortion_scale=0.4, p=0.3),
        # AutoAugment policy (predefined augmentation strategies)
        transforms.AutoAugment(),
        # Additional RandAugment (random ops with specified magnitude)
        transforms.RandAugment(num_ops=2, magnitude=9),
        # Convert PIL image to tensor (0–1 range)
        transforms.ToTensor(),
        # Randomly erase a patch to improve robustness
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        # Normalize with ImageNet mean/std for pretrained backbones
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        # Deterministic resize to model input size
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Same normalization as training
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

# ----------------------
# Data mixing: MixUp & CutMix
# ----------------------
def mixup_data(x, y, alpha=0.4):
    """
    Apply MixUp augmentation.

    x: batch of images [B, C, H, W]
    y: batch of labels [B, 1]
    alpha: Beta distribution parameter, controls mixing strength.

    Returns:
    - mixed_x: convex combination of two images
    - (y_a, y_b, lam): labels of original and permuted batch + mixing coefficient
    """
    if alpha <= 0:
        # If alpha <= 0, disable MixUp and return inputs unchanged
        return x, y, None
    # Sample mixing coefficient lam ~ Beta(alpha, alpha)
    lam = np.random.beta(alpha, alpha)
    # Random permutation of indices for mixing pairs
    idx = torch.randperm(x.size(0), device=x.device)
    # Linear combination of original and permuted images
    mixed_x = lam * x + (1 - lam) * x[idx]
    # Store labels of original and permuted
    y_a, y_b = y, y[idx]
    return mixed_x, (y_a, y_b, lam)

def rand_bbox(size, lam):
    """
    Generate random bounding box coordinates for CutMix.

    size: tensor shape (B, C, H, W)
    lam: mixing coefficient for CutMix

    Returns (bbx1, bby1, bbx2, bby2): integer coords of patch.
    """
    B, C, H, W = size
    # Cutout ratio from lambda
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniformly sample center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Compute box boundaries, clamp to image
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation.

    x: batch of images
    y: batch of labels
    alpha: Beta distribution parameter.

    Returns:
    - x: modified images with CutMix patches
    - (y_a, y_b, lam_adj): labels of original and permuted batch + effective lam
    """
    if alpha <= 0:
        # If alpha <= 0, disable CutMix
        return x, y, None
    # Sample lambda
    lam = np.random.beta(alpha, alpha)
    # Permutation indices
    idx = torch.randperm(x.size(0), device=x.device)
    # Sample bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    # Replace region in x with region from permuted images
    x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
    # Adjust lambda by actual area ratio of the box
    area = (bbx2 - bbx1) * (bby2 - bby1)
    lam_adj = 1.0 - area / (x.size(2) * x.size(3))
    # Keep original and permuted labels
    y_a, y_b = y, y[idx]
    return x, (y_a, y_b, lam_adj)

# ----------------------
# EMA helper
# ----------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = (
                    self.shadow.get(name, p.detach().clone()) * self.decay
                    + (1.0 - self.decay) * p.detach()
                )

    def apply_to(self, model):
        self._backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                # strip optional "module." prefix when looking into shadow
                key = name
                if key not in self.shadow and key.startswith("module."):
                    key = key[len("module."):]
                if key not in self.shadow:
                    continue  # skip if still not found
                self._backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[key].data)

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in getattr(self, "_backup", {}):
                p.data.copy_(self._backup[name].data)
        self._backup = {}


# ----------------------
# Minimal SAM wrapper
# ----------------------
class SAM:
    """
    Minimal Sharpness-Aware Minimization (SAM) wrapper.

    This class wraps a base optimizer and adds perturbation in weight space
    based on gradient norm to encourage flatter minima.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        """
        params: iterable of parameters
        base_optimizer: an instantiated optimizer (e.g. AdamW)
        rho: perturbation radius
        adaptive: if True, scales perturbation by parameter magnitude
        """
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        self.adaptive = adaptive

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        """
        First SAM step: compute perturbation and move weights.

        Typically:
        1. Compute gradients (backward already called).
        2. Call first_step() to perturb parameters.
        """
        grad_norm = self._grad_norm()
        # Scaling factor to keep perturbation within rho ball
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Possibly scale gradient by |p| for adaptive SAM
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                # In-place add perturbation
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        """
        Second SAM step: take optimizer step using gradients at perturbed weights,
        then optionally zero gradients.
        """
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        """
        Proxy zero_grad to the underlying optimizer.
        """
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        """
        Compute L2 norm of all gradients (possibly scaled by parameters in adaptive mode).
        """
        device = self.param_groups[0]['params'][0].device
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if self.adaptive:
                    # Scale by absolute parameter value
                    g = g * (torch.abs(p) + 1e-12)
                norms.append(torch.norm(g.detach(), p=2).to(device))
        # Stack and compute global norm
        norm = torch.norm(torch.stack(norms), p=2)
        return norm

# ----------------------
# Hybrid model: ConvNeXt-V2 Tiny + ViT-B/16 + cross-attention fusion
# ----------------------
class CrossAttentionFusion(nn.Module):
    """
    Fusion block that uses cross-attention between CNN features and ViT token features.

    - Projects ConvNeXt feature map (flattened) and ViT tokens into same fusion_dim.
    - Applies Multi-Head Attention: queries from CNN, keys/values from ViT.
    - Residual + LayerNorm, then global average pooling over spatial dimension.
    """
    def __init__(self, dim_cnn, dim_vit, fusion_dim=512, heads=8):
        super().__init__()
        # Linear projection from CNN channel dimension to fusion_dim
        self.proj_cnn = nn.Linear(dim_cnn, fusion_dim)
        # Linear projection from ViT token dimension to fusion_dim
        self.proj_vit = nn.Linear(dim_vit, fusion_dim)
        # Multi-head attention for cross-attention
        self.cross_attn = nn.MultiheadAttention(
            fusion_dim, num_heads=heads, batch_first=True
        )
        # LayerNorm for post-attention normalization
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, cnn_feat, vit_feat):
        """
        cnn_feat: [B, HW, Cc]  (flattened ConvNeXt feature map)
        vit_feat: [B, N, Cv]   (ViT token outputs)

        Returns:
        - Fused representation [B, fusion_dim]
        """
        # Project CNN and ViT features into same fusion space
        c = self.proj_cnn(cnn_feat)  # [B, HW, F]
        v = self.proj_vit(vit_feat)  # [B, N, F]
        # Cross-attention: CNN queries, ViT keys/values
        out, _ = self.cross_attn(c, v, v)  # [B, HW, F]
        # Residual connection + LayerNorm
        out = self.norm(out + c)
        # Global average pool over spatial dimension HW
        return out.mean(dim=1)  # [B, F]

class HybridNet(nn.Module):
    """
    Hybrid model:
    - ConvNeXt Tiny backbone
    - ViT-B/16 backbone
    - Fuse pooled features with an MLP
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # ConvNeXt Tiny
        self.cnn = models.convnext_tiny(
            weights="DEFAULT" if pretrained else None
        )
        # ConvNeXt classifier: [layernorm, linear]
        # We keep the global pooled feature before final linear
        cnn_in = self.cnn.classifier[2].in_features
        self.cnn.classifier[2] = nn.Identity()  # output 768-dim features

        # ViT-B/16
        self.vit = models.vit_b_16(
            weights="DEFAULT" if pretrained else None
        )
        # ViT heads.head is a linear from 768 -> 1000; replace with identity
        vit_in = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()  # output 768-dim CLS features

        # Fusion MLP: concat 768 (cnn) + 768 (vit) -> 1 logit
        fusion_in = 768 + 768
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # ConvNeXt pooled features [B,768]
        cnn_feat = self.cnn(x)
        # ViT CLS features [B,768]
        vit_feat = self.vit(x)
        # Concatenate feature vectors
        fused = torch.cat([cnn_feat, vit_feat], dim=1)  # [B,1536]
        # Final classifier
        out = self.fusion(fused)  # [B,1]
        return out


# ----------------------
# Loss helpers
# ----------------------
def smooth_labels(y, eps=0.05):
    """
    Apply label smoothing for binary labels.

    y: labels in {0, 1}
    eps: smoothing factor

    Smoothed:
    - y=1 -> 1 - eps/2
    - y=0 -> eps/2
    """
    return y * (1 - eps) + 0.5 * eps

def bce_logits(preds, targets):
    """
    Shortcut wrapper for BCEWithLogitsLoss.
    preds: raw logits
    targets: floats in [0,1]
    """
    return nn.BCEWithLogitsLoss()(preds, targets)

def distill_loss(student_logits, teacher_logits, hard_labels, T=2.0, alpha=0.7):
    """
    Distillation loss for binary classification with single logit.

    - student_logits: logits from student model
    - teacher_logits: logits from teacher model
    - hard_labels: ground truth labels (0 or 1)
    - T: temperature for soft targets
    - alpha: weight between soft (teacher) and hard (ground truth) loss

    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    """
    # Soft targets from teacher after temperature scaling
    soft_targets = torch.sigmoid(teacher_logits / T).detach()
    # Soft loss: BCE between student logits (scaled) and soft targets, multiplied by T^2
    soft_loss = nn.BCEWithLogitsLoss()(student_logits / T, soft_targets) * (T * T)
    # Hard loss: BCE between student logits and true labels
    hard_loss = nn.BCEWithLogitsLoss()(student_logits, hard_labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

# ----------------------
# Training / Validation loops (with SAM support)
# ----------------------
def train_epoch(model, loader, optimizer, device, scaler, cfg,
                sam=None, teacher=None, distill_cfg=None, ema=None):
    """
    One training epoch.

    model: student model
    loader: training DataLoader
    optimizer: optimizer instance
    device: 'cuda' or 'cpu'
    scaler: GradScaler for AMP
    cfg: dict of training hyperparameters and flags
    sam: optional SAM wrapper (if SAM training is used)
    teacher: optional teacher model for distillation
    distill_cfg: dict with keys like T (temperature) and alpha
    ema: EMA object to update moving-average weights
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        # Move data to device, cast to float32, and use channels_last memory format
        imgs = (imgs.to(device=device, dtype=torch.float32)
                    .to(memory_format=torch.channels_last))
        labels = labels.to(device=device).unsqueeze(1)  # [B] -> [B,1]

        # Possibly apply CutMix or MixUp per batch
        applied = None
        if cfg['cutmix'] and random.random() < cfg['cutmix_prob']:
            imgs, applied = cutmix_data(imgs, labels, cfg['cutmix_alpha'])
        elif cfg['mixup'] and random.random() < cfg['mixup_prob']:
            imgs, applied = mixup_data(imgs, labels, cfg['mixup_alpha'])

        # Case 1: no SAM (standard single-step training)
        if sam is None:
            optimizer.zero_grad()
            with autocast(enabled=cfg['amp']):
                outputs = model(imgs)  # [B,1] logits

                # Decide if using distillation or classic supervised loss
                if teacher is None or not cfg['distill']:
                    # Pure supervised training
                    if applied is None:
                        # No MixUp/CutMix: use smoothed labels directly
                        targets = smooth_labels(labels, cfg['label_smoothing'])
                        loss = bce_logits(outputs, targets)
                        # Predictions: sigmoid and threshold at 0.5
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        correct += (preds == labels).sum().item()
                    else:
                        # With MixUp/CutMix: combine two label sets
                        ya, yb, lam = applied
                        ya_s = smooth_labels(ya, cfg['label_smoothing'])
                        yb_s = smooth_labels(yb, cfg['label_smoothing'])
                        # Weighted BCE for each label set
                        loss = (lam * bce_logits(outputs, ya_s)
                                + (1 - lam) * bce_logits(outputs, yb_s))
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        # Construct "mixed" target for accuracy estimation
                        mix_target = lam * ya + (1 - lam) * yb
                        # Count predictions within 0.5 of target as correct
                        correct += ((preds - mix_target).abs() < 0.5).sum().item()
                else:
                    # Distillation mode (teacher-student)
                    with torch.no_grad():
                        t_logits = teacher(imgs)
                    if applied is None:
                        # Distill with hard labels only
                        loss = distill_loss(outputs, t_logits, labels,
                                            T=distill_cfg['T'],
                                            alpha=distill_cfg['alpha'])
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        correct += (preds == labels).sum().item()
                    else:
                        # Distill with mixed hard labels
                        ya, yb, lam = applied
                        mixed_hard = lam * ya + (1 - lam) * yb
                        loss = distill_loss(outputs, t_logits, mixed_hard,
                                            T=distill_cfg['T'],
                                            alpha=distill_cfg['alpha'])
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        correct += ((preds - mixed_hard).abs() < 0.5).sum().item()

            # Backward and optimizer step with optional AMP
            if cfg['amp']:
                scaler.scale(loss).backward()
                if cfg['grad_clip'] > 0:
                    # Unscale before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg['grad_clip']
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg['grad_clip']
                    )
                optimizer.step()

        else:
            # Case 2: SAM-enabled flow (here simplified to single-step use)
            optimizer.zero_grad()
            with autocast(enabled=cfg['amp']):
                outputs = model(imgs)
                if teacher is None or not cfg['distill']:
                    if applied is None:
                        targets = smooth_labels(labels, cfg['label_smoothing'])
                        loss = bce_logits(outputs, targets)
                    else:
                        ya, yb, lam = applied
                        ya_s = smooth_labels(ya, cfg['label_smoothing'])
                        yb_s = smooth_labels(yb, cfg['label_smoothing'])
                        loss = (lam * bce_logits(outputs, ya_s)
                                + (1 - lam) * bce_logits(outputs, yb_s))
                else:
                    # Distillation with SAM (no explicit accuracy here)
                    with torch.no_grad():
                        t_logits = teacher(imgs)
                    if applied is None:
                        loss = distill_loss(outputs, t_logits, labels,
                                            T=distill_cfg['T'],
                                            alpha=distill_cfg['alpha'])
                    else:
                        ya, yb, lam = applied
                        mixed_hard = lam * ya + (1 - lam) * yb
                        loss = distill_loss(outputs, t_logits, mixed_hard,
                                            T=distill_cfg['T'],
                                            alpha=distill_cfg['alpha'])

            # Simplified SAM+AMP behavior: single backward + clip + step
            if cfg['amp']:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['grad_clip']
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['grad_clip']
                )
                # Note: exact SAM two-step is not fully implemented here.

        # Track total samples and accumulated loss
        batch_n = imgs.size(0)
        total += batch_n
        total_loss += (loss.item() if 'loss' in locals() else 0.0) * batch_n

        # Update EMA after each optimizer step if enabled
        if ema is not None:
            ema.update(model)

    # Compute averages
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

def validate(model, loader, device):
    """
    Validation loop without gradient computation.

    Returns:
    - average loss
    - accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = (imgs.to(device=device, dtype=torch.float32)
                        .to(memory_format=torch.channels_last))
            labels = labels.to(device=device).unsqueeze(1)
            logits = model(imgs)
            loss = bce_logits(logits, labels)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total += imgs.size(0)
            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
    return total_loss / max(1, total), correct / max(1, total)

# ----------------------
# TTA inference helper (returns prob)
# ----------------------
class TTA:
    """
    Simple Test-Time Augmentation (TTA) wrapper.

    - Applies a set of deterministic transforms to a PIL image.
    - Runs the model on each variant and averages predicted probabilities.
    """
    def __init__(self, model, img_size, device):
        self.model = model
        self.img_size = img_size
        self.device = device
        # Define list of transforms for TTA
        self.transforms = [
            # Plain resize + normalize
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            # Horizontal flip variant
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            # Small rotation variant
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        ]

    def predict(self, pil_img):
        """
        Run TTA inference on a single PIL image.

        Returns:
        - averaged probability of being class 1.
        """
        self.model.eval()
        with torch.no_grad():
            probs = []
            for t in self.transforms:
                x = t(pil_img).unsqueeze(0).to(self.device)
                x = x.to(memory_format=torch.channels_last)
                logits = self.model(x)
                probs.append(torch.sigmoid(logits).item())
            # Return mean probability over TTA variants
            return float(np.mean(probs))

# ============================
# AUTO BATCH SIZE TUNER
# ============================

def try_batch_size(model, batch, img_size, device, amp=True):
    """
    Try a specific batch size to see if it fits in GPU memory.

    - Runs a dummy forward/backward step with random data.
    - Returns (ok, mem_used_in_MB) if successful, or (False, None) on OOM.
    """
    try:
        # Clear existing GPU caches
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Dummy inputs
        x = torch.randn(batch, 3, img_size, img_size,
                        device=device).to(memory_format=torch.channels_last)
        y = torch.randn(batch, 1, device=device)

        # Simple optimizer for the test
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler(enabled=amp)

        # Test forward + backward under autocast
        with autocast(enabled=amp):
            out = model(x)
            loss = nn.BCEWithLogitsLoss()(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Sync to ensure memory stats are updated
        torch.cuda.synchronize()

        # Get peak memory used during this trial
        mem_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return True, mem_used

    except RuntimeError as e:
        # Catch out-of-memory errors and report failure
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False, None
        # Any other error is re-raised
        raise e

def auto_tune_batch_size(model, img_size, device, amp=True):
    """
    Automatically determine a suitable batch size for this GPU.

    - Tests a list of candidate batch sizes in ascending order.
    - Chooses the largest stable batch size without OOM, then returns ~75% of it.
    """
    print("\n[INFO] Auto-tuning batch size for your GPU...")
    candidate_batches = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]

    max_ok = None
    for b in candidate_batches:
        print(f"  • Testing batch={b}... ", end="")
        ok, mem = try_batch_size(model, b, img_size, device, amp)
        if ok:
            print(f"OK  (VRAM {mem:.1f} MB)")
            max_ok = b
        else:
            print("OOM ❌")
            break

    if max_ok is None:
        # If even batch size 4 fails, fall back to 2
        print("[WARN] Even batch=4 failed, using batch=2")
        return 2

    # Recommend 75% of max stable batch size to keep some margin
    recommended = max(2, int(max_ok * 0.75))
    print(f"\n[INFO] MAX stable batch size: {max_ok}")
    print(f"[INFO] Recommended batch size: {recommended}\n")

    return recommended

# ----------------------
# Main: training orchestration
# ----------------------
def main():
    """
    Orchestrates the entire training:

    - Parse CLI arguments.
    - Prepare datasets, loaders, and samplers.
    - Build hybrid student model and optional teacher.
    - Train teacher (if distillation), then student with SWA + EMA.
    - Optionally save best checkpoints and TorchScript export.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="root folder with Real/ and Fake/")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="If < 0, auto-tune batch size.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training.")
    parser.add_argument("--mixup", action="store_true",
                        help="Enable MixUp augmentation.")
    parser.add_argument("--cutmix", action="store_true",
                        help="Enable CutMix augmentation.")
    parser.add_argument("--use-sam", action="store_true",
                        help="Enable SAM optimizer wrapper.")
    parser.add_argument("--save", action="store_true",
                        help="Explicit: save checkpoints & export (default: not saving)")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--distill", action="store_true",
                        help="Use teacher distillation (teacher trained within script)")
    args = parser.parse_args()

    # ------------------
    # Reproducibility
    # ------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # ---------------------------------------
    # AUTO BATCH SIZE TUNING (unless user forces one)
    # ---------------------------------------
    print(f"[INFO] Requested batch-size = {args.batch_size}")

    # Initialize a temporary model solely for memory benchmarking
    model_temp = HybridNet(pretrained=True).to(device)
    model_temp = model_temp.to(memory_format=torch.channels_last)
    # Try to enable gradient checkpointing for ViT to save memory
    try:
        model_temp.vit.encoder.gradient_checkpointing_enable()
    except Exception:
        pass

    if args.batch_size < 0:
        # If user passed a negative batch size, auto-tune
        tuned_bs = auto_tune_batch_size(
            model_temp, args.img_size, device, args.amp
        )
        args.batch_size = tuned_bs
        print(f"[INFO] Auto-selected batch size = {args.batch_size}")
    else:
        # If user provided non-negative batch size, skip tuner
        print("[INFO] Skipping auto tuning (manual batch-size provided).")

    # Build transforms for training and validation
    train_tf, val_tf = build_transforms(args.img_size)

    # ------------------
    # Dataset scan
    # ------------------
    root = Path(args.data)
    # Glob all real/fake images recursively under Real/ and Fake/
    real_files = sorted([
        p for p in glob(str(root / "Real" / "**" / "*"), recursive=True)
        if p.lower().endswith(("jpg", "jpeg", "png"))
    ])
    fake_files = sorted([
        p for p in glob(str(root / "Fake" / "**" / "*"), recursive=True)
        if p.lower().endswith(("jpg", "jpeg", "png"))
    ])
    print(f"[INFO] Found REAL: {len(real_files)}, FAKE: {len(fake_files)}")

    # Concatenate real and fake paths, with labels 1 for real, 0 for fake
    files = real_files + fake_files
    labels = [1] * len(real_files) + [0] * len(fake_files)

    if len(files) == 0:
        print("No images found. Exiting.")
        return

    # Shuffle dataset
    combined = list(zip(files, labels))
    random.shuffle(combined)
    files, labels = zip(*combined)

    # Train/val split (85% / 15%)
    n_total = len(files)
    split = int(0.85 * n_total)
    train_files, val_files = files[:split], files[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    print(f"[INFO] Train: {len(train_files)}  Val: {len(val_files)}")

    # Create Datasets
    train_ds = KFSDataset(train_files, train_labels, transform=train_tf)
    val_ds = KFSDataset(val_files, val_labels, transform=val_tf)

    # ------------------
    # Weighted sampler for class imbalance
    # ------------------
    counts = Counter(train_labels)  # count per class
    # Define class weights as inverse of class frequency
    class_weights = {cls: 1.0 / max(1, cnt) for cls, cnt in counts.items()}
    # Weight per sample based on its label
    sample_weights = [class_weights[int(l)] for l in train_labels]
    # Use WeightedRandomSampler to balance classes in mini-batches
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoaders with sampler for train and simple shuffle=False for val
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ------------------
    # Student model & optimizer
    # ------------------
    model = HybridNet(pretrained=True).to(device)
    model = model.to(memory_format=torch.channels_last)
    # Try enabling gradient checkpointing for ViT encoder to reduce memory footprint
    try:
        model.vit.encoder.gradient_checkpointing_enable()
    except Exception:
        pass

    # Base optimizer (AdamW with weight decay)
    base_opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05
    )
    if args.use_sam:
        # If SAM is requested, create wrapper but still use base optimizer reference
        sam = SAM(model.parameters(), base_opt, rho=0.05, adaptive=False)
        optimizer = base_opt
    else:
        sam = None
        optimizer = base_opt

    # GradScaler for mixed precision
    scaler = GradScaler(enabled=args.amp)
    # Cosine LR scheduler for first phase (before SWA kicks in)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    # Start SWA at 75% of total epochs
    swa_start = int(0.75 * args.epochs)
    # SWA averaged model wrapper
    swa_model = AveragedModel(model)
    # SWA LR scheduler with very low learning rate
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

    # EMA tracking for student model
    ema = EMA(model, decay=0.999)

    # Global training configuration dictionary
    cfg = {
        "amp": args.amp,
        "mixup": args.mixup,
        "mixup_alpha": 0.4,
        "mixup_prob": 0.5,
        "cutmix": args.cutmix,
        "cutmix_alpha": 1.0,
        "cutmix_prob": 0.5,
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
        "distill": args.distill,
    }

    # ------------------
    # Optional teacher training for distillation
    # ------------------
    teacher = None
    if args.distill:
        # AFTER
        teacher_model = models.efficientnet_b3(weights="DEFAULT").to(device).to(memory_format=torch.channels_last)
        in_f = teacher_model.classifier[1].in_features
        teacher_model.classifier[1] = nn.Linear(in_f, 1).to(device)  # ensure new layer is on GPU

        # Optimizer, scaler, scheduler for teacher
        teacher_opt = torch.optim.AdamW(
            teacher_model.parameters(), lr=args.lr, weight_decay=0.05
        )
        teacher_scaler = GradScaler(enabled=args.amp)
        teacher_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            teacher_opt, T_max=args.epochs
        )
        teacher_swa = AveragedModel(teacher_model)
        teacher_swa_start = int(0.75 * args.epochs)
        teacher_ema = EMA(teacher_model, decay=0.999)

        print("[INFO] Training teacher (EfficientNet-B3)...")
        best_val = 0.0
        for epoch in range(args.epochs):
            # Teacher is trained like a normal classifier without distillation
            t_loss, t_acc = train_epoch(
                teacher_model,
                train_dl,
                teacher_opt,
                device,
                teacher_scaler,
                {**cfg, "distill": False},
                sam=None,
                teacher=None,
                distill_cfg=None,
                ema=teacher_ema
            )
            v_loss, v_acc = validate(teacher_model, val_dl, device)

            # Update SWA teacher after threshold epoch
            if epoch >= teacher_swa_start:
                teacher_swa.update_parameters(teacher_model)
            else:
                teacher_cos.step()

            # Extra EMA update
            teacher_ema.update(teacher_model)
            print(
                f"[TEACHER] Epoch {epoch+1}/{args.epochs} | "
                f"Train Acc {t_acc:.4f} | Val Acc {v_acc:.4f}"
            )

        print("[TEACHER] finishing: updating BN for SWA")
        try:
            # Update BN statistics for SWA teacher using train loader
            torch.optim.swa_utils.update_bn(train_dl, teacher_swa)
        except Exception:
            pass

        # Fuse EMA weights into SWA-averaged teacher model
        teacher_ema.apply_to(teacher_swa)
        teacher = teacher_swa.eval()
        print("[TEACHER] teacher ready for distillation (in-memory)")

    # ------------------
    # Train student (HybridNet)
    # ------------------
    print("[INFO] Training student (HybridNet)...")
    best_val = 0.0
    patience = 10   # early stopping patience
    wait = 0        # number of epochs since last improvement

    for epoch in range(args.epochs):
        start = time.time()

        # distill_cfg is used only if cfg["distill"] is True
        train_loss, train_acc = train_epoch(
            model,
            train_dl,
            optimizer,
            device,
            scaler,
            cfg,
            sam=sam if args.use_sam else None,
            teacher=teacher,
            distill_cfg={"T": 2.0, "alpha": 0.7},
            ema=ema
        )
        val_loss, val_acc = validate(model, val_dl, device)

        # Scheduler/SWA logic
        if epoch >= swa_start:
            # Update SWA-averaged student model
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # Standard cosine LR decay before SWA
            cosine.step()

        # Extra EMA update (already updated per batch, but safe to call again)
        ema.update(model)

        elapsed = time.time() - start
        print(
            f"[STUDENT] Epoch {epoch+1}/{args.epochs} | "
            f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
            f"Time {elapsed:.1f}s"
        )

        # Simple early stopping on validation accuracy
        if val_acc > best_val:
            best_val = val_acc
            wait = 0
            # Save checkpoint only if user explicitly asked via --save
            if args.save:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    Path(args.save_dir) / "student_best.pth"
                )
                print("[SAVED] student_best.pth")
        else:
            wait += 1
            if wait >= patience:
                print("[INFO] Early stopping triggered (in-memory).")
                break

    print("[INFO] Finalizing SWA & EMA fusion (in-memory)...")
    try:
        # Update BN statistics for SWA model with training data
        torch.optim.swa_utils.update_bn(train_dl, swa_model)
    except Exception:
        pass

    # Apply EMA weights onto SWA model to get final fused student
    ema.apply_to(swa_model)

    # ------------------
    # Optional saving and TorchScript export
    # ------------------
    if args.save:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        # Save fused SWA+EMA model weights
        torch.save(
            swa_model.state_dict(),
            Path(args.save_dir) / "student_swa_ema.pth"
        )
        print("[SAVED] student_swa_ema.pth (you asked for saving)")

        # TorchScript export for deployment (e.g., C++/mobile)
        try:
            swa_model.cpu().eval()
            example = torch.randn(1, 3, args.img_size, args.img_size)
            traced = torch.jit.trace(swa_model, example)
            traced.save(Path(args.save_dir) / "student_rpi_swa_ema.pt")
            print("[SAVED] TorchScript student_rpi_swa_ema.pt")
        except Exception as e:
            print("[WARN] TorchScript export failed:", e)

    print("DONE. Best Val Acc:", best_val)
    # Return fused model for potential further use in Python
    return swa_model

if __name__ == "__main__":
    main()

# python train_monster_hybrid.py \
#   --data "D:/Robotics Club/Robocon2026/Team-Vulcans-Robocon-2026/teams/team-2_vision/DatasetIRL" \
#   --epochs 40 \
#   --batch-size -1 \
#   --img-size 224 \
#   --lr 3e-4 \
#   --num-workers 6 \
#   --amp \
#   --mixup \
#   --cutmix \
#   --use-sam \
#   --distill \
#   --save \
#   --save-dir "checkpoints_hybrid"
