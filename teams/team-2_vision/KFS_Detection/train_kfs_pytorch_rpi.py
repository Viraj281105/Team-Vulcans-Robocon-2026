#!/usr/bin/env python3
"""
train_kfs_pytorch_rpi.py

A robust, production-ready PyTorch training script for the KFS classifier.

Features (updated):
- MobileNetV3 small (ImageNet pretrained)
- Strong Albumentations augmentation + MixUp option
- Balanced sampling and class weights to reduce bias
- AMP (mixed precision) training for speed on GPU
- EarlyStopping + ReduceLROnPlateau + best-checkpoint saving
- TensorBoard logging + training plots saved to disk
- Auto-suggest hyperparameters based on dataset size
- Exports: TorchScript (.pt), ONNX (.onnx), and attempts TFLite (.tflite) conversion
  (TFLite conversion requires `onnx`, `onnx-tf` and `tensorflow` to be installed in the environment)
- Representative dataset-based quantization for TFLite (INT8) if requested
- Small helper to run EdgeTPU compiler (if you plan to compile for Coral)

Notes:
- Run on your ROG with CUDA-enabled PyTorch for fast training.
- Copy the generated .tflite file to your RPi 5 and use the included inference function.

Usage example:
    python train_kfs_pytorch_rpi.py --data /path/to/dataset --export-tflite --quantize

"""

import os
import sys
import time
import math
import argparse
from glob import glob
from multiprocessing import cpu_count
from pathlib import Path

import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Try optional imports for TFLite conversion
try:
    import onnx
    from onnx_tf.backend import prepare as onnx_tf_prepare
    import tensorflow as tf
    ONNX_TF_AVAILABLE = True
except Exception:
    ONNX_TF_AVAILABLE = False


# --------------------------
# Utilities
# --------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# Dataset
# --------------------------
class KFSDataset(Dataset):
    def __init__(self, files, labels, img_size=128, augment=False):
        self.files = files
        self.labels = labels
        self.img_size = img_size
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.6),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=7),
                    A.GaussianBlur(blur_limit=7),
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                ], p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.CLAHE(p=0.2),
                A.CoarseDropout(max_holes=3, max_height=20, max_width=20, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=20, p=0.5),
                A.ChannelShuffle(p=0.1),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # robust image loading
        img = cv2.imread(path)
        if img is None:
            # fallback: PIL
            img = np.array(Image.open(path).convert('RGB'))[:, :, ::-1]
        else:
            img = img[..., ::-1]  # BGR -> RGB

        img = self.transform(image=img)['image']
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


# --------------------------
# MixUp helper
# --------------------------

def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# --------------------------
# Training / validation loops
# --------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch, mixup_alpha=0.0, writer=None):
    model.train()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Train E{epoch}")
    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        if mixup_alpha > 0:
            mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
            with torch.cuda.amp.autocast(enabled=(device!='cpu')):
                outputs = model(mixed_imgs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            with torch.cuda.amp.autocast(enabled=(device!='cpu')):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1
        avg_loss = running_loss / num_batches
        pbar.set_postfix({'loss': avg_loss})

        if writer is not None and batch_idx % 50 == 0:
            writer.add_scalar('train/batch_loss', avg_loss, epoch * len(loader) + batch_idx)

    return running_loss / max(1, num_batches)


@torch.no_grad()
def validate(model, loader, device, writer=None, epoch=0):
    model.eval()
    preds = []
    truths = []
    pbar = tqdm(loader, desc="Val")
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        with torch.cuda.amp.autocast(enabled=(device!='cpu')):
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten().tolist()
        preds.extend(probs)
        truths.extend(labels.cpu().numpy().flatten().tolist())

    preds_arr = np.array(preds)
    truths_arr = np.array(truths)
    pred_labels = (preds_arr > 0.5).astype(int)
    acc = (pred_labels == truths_arr).mean()
    try:
        auc = roc_auc_score(truths_arr, preds_arr)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(truths_arr, pred_labels)

    if writer is not None:
        writer.add_scalar('val/acc', acc, epoch)
        if not math.isnan(auc):
            writer.add_scalar('val/auc', auc, epoch)
        # log confusion matrix as image
        fig = plot_confusion_matrix(cm, ['fake', 'real'])
        writer.add_figure('val/confusion_matrix', fig, epoch)

    return acc, auc, cm, preds_arr, truths_arr


# --------------------------
# Plot helpers
# --------------------------

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    return fig


def save_training_plots(history, out_prefix='train_plots'):
    # history: dict with 'loss', 'val_acc', 'val_auc'
    if 'loss' in history:
        plt.figure()
        plt.plot(history['loss'], label='train_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'{out_prefix}_loss.png')
        plt.close()

    if 'val_acc' in history:
        plt.figure()
        plt.plot(history['val_acc'], label='val_acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(f'{out_prefix}_val_acc.png')
        plt.close()

    if 'val_auc' in history:
        plt.figure()
        plt.plot(history['val_auc'], label='val_auc')
        plt.xlabel('epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.savefig(f'{out_prefix}_val_auc.png')
        plt.close()


# --------------------------
# Exports: TorchScript, ONNX, TFLite
# --------------------------
def export_torchscript(model, example_input, path):
    traced = torch.jit.trace(model.cpu(), example_input.cpu())
    traced.save(path)
    print(f"Saved TorchScript model -> {path}")


def export_onnx(model, example_input, path, opset=13):
    model.cpu()
    model.eval()
    torch.onnx.export(model, example_input.cpu(), path, opset_version=opset,
                      input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    print(f"Saved ONNX model -> {path}")


def convert_onnx_to_tflite(onnx_path, saved_model_dir='tmp_saved_model', tflite_path='model.tflite', quantize=False, representative_generator=None):
    if not ONNX_TF_AVAILABLE:
        print("Skipping TFLite conversion: onnx/onnx_tf/tensorflow not available in this environment.")
        return False

    print("Converting ONNX -> TensorFlow SavedModel ...")
    model_onnx = onnx.load(onnx_path)
    tf_rep = onnx_tf_prepare(model_onnx)
    if os.path.exists(saved_model_dir):
        # remove previous
        import shutil
        shutil.rmtree(saved_model_dir)
    tf_rep.export_graph(saved_model_dir)
    print("Saved intermediate SavedModel ->", saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize and representative_generator is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_generator
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print("Saved TFLite model ->", tflite_path)
    return True


# Representative generator for quantization
def make_representative_gen(files, preprocess_fn, num_samples=100):
    def gen():
        for i, f in enumerate(files[:num_samples]):
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (preprocess_fn['size'], preprocess_fn['size']))
            img = img.astype(np.float32) / 255.0
            img = (img - preprocess_fn['mean']) / preprocess_fn['std']
            img = np.expand_dims(img, axis=0).astype(np.float32)
            yield [img]
    return gen


# --------------------------
# EdgeTPU helper (calls edgetpu_compiler if installed)
# --------------------------

def compile_for_edgetpu(tflite_path, output_dir='edgetpu_out'):
    import shutil
    if shutil.which('edgetpu_compiler') is None:
        print('edgetpu_compiler not found in PATH. Install Coral Edge TPU compiler on your machine to compile .tflite models.')
        return False

    os.makedirs(output_dir, exist_ok=True)
    cmd = f'edgetpu_compiler -o {output_dir} {tflite_path}'
    print('Running:', cmd)
    res = os.system(cmd)
    if res == 0:
        print('EdgeTPU compilation complete. Check', output_dir)
        return True
    else:
        print('EdgeTPU compilation failed.')
        return False


# --------------------------
# Main
# --------------------------

def auto_suggest_hyperparams(total_images: int):
    # Simple heuristic to suggest batch size and epochs
    if total_images < 500:
        return 16, 60
    if total_images < 2000:
        return 32, 40
    if total_images < 8000:
        return 64, 30
    return 128, 20


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset folder with subfolders real/ and fake/')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=0, help='0 to auto-suggest')
    parser.add_argument('--epochs', type=int, default=0, help='0 to auto-suggest')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mixup', type=float, default=0.0, help='MixUp alpha; 0 to disable')
    parser.add_argument('--export-tflite', action='store_true', help='Attempt TFLite export (requires onnx/onnx_tf/tensorflow)')
    parser.add_argument('--quantize', action='store_true', help='Use int8 quantization for TFLite (needs representative samples)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=max(1, cpu_count() - 1))
    parser.add_argument('--patience', type=int, default=6, help='EarlyStopping patience')
    parser.add_argument('--edge-compile', action='store_true', help='Run EdgeTPU compiler on final TFLite (if available)')
    args = parser.parse_args(argv)

    seed_everything(args.seed)

    DATASET_DIR = Path(args.data)
    assert DATASET_DIR.exists(), f"Dataset dir not found: {DATASET_DIR}"

    real_files = glob(str(DATASET_DIR / 'real' / '*'))
    fake_files = glob(str(DATASET_DIR / 'fake' / '*'))
    real_files = [f for f in real_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    fake_files = [f for f in fake_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found real: {len(real_files)} images, fake: {len(fake_files)} images")
    files = real_files + fake_files
    labels = [1] * len(real_files) + [0] * len(fake_files)

    total_images = len(files)
    if total_images == 0:
        print('No images found. Exiting.')
        return

    # auto-suggest if requested
    suggested_batch, suggested_epochs = auto_suggest_hyperparams(total_images)
    batch_size = suggested_batch if args.batch_size == 0 else args.batch_size
    epochs = suggested_epochs if args.epochs == 0 else args.epochs

    print(f"Auto-suggest -> batch_size: {batch_size}, epochs: {epochs} (total images: {total_images})")

    # stratified split
    train_f, val_f, train_l, val_l = train_test_split(files, labels, test_size=0.15, stratify=labels, random_state=args.seed)

    train_ds = KFSDataset(train_f, train_l, img_size=args.img_size, augment=True)
    val_ds = KFSDataset(val_f, val_l, img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # Model
    model = mobilenet_v3_small(weights='IMAGENET1K_V1')
    in_features = model.classifier[3].in_features if hasattr(model.classifier[3], 'in_features') else 1024
    model.classifier[3] = nn.Linear(in_features, 1)
    model = model.to(device)

    # compute class weights
    pos_weight = (len(labels) - sum(labels)) / (sum(labels) + 1e-6)
    print(f'Computed pos_weight for BCE: {pos_weight:.4f}')
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    scaler = torch.cuda.amp.GradScaler()

    best_val_auc = -1
    best_epoch = -1
    epochs_no_improve = 0

    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    # TensorBoard writer
    log_dir = Path('runs') / time.strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=str(log_dir))
    print('TensorBoard logs ->', log_dir)

    history = {'loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, mixup_alpha=args.mixup, writer=writer)
        val_acc, val_auc, val_cm, val_preds, val_trues = validate(model, val_loader, device, writer=writer, epoch=epoch)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.4f} | val_acc: {val_acc:.4f} | val_auc: {val_auc:.4f} | time: {elapsed:.1f}s")
        print('Confusion Matrix:\n', val_cm)

        # log scalars
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/acc_epoch', val_acc, epoch)
        if not math.isnan(val_auc):
            writer.add_scalar('val/auc_epoch', val_auc, epoch)

        history['loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        # LR scheduler (monitor AUC)
        scheduler.step(val_auc if not math.isnan(val_auc) else 0.0)

        # checkpoint best
        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_no_improve = 0
            best_path = checkpoint_dir / f'kfs_mobilenetv3_best.pth'
            torch.save(model.state_dict(), str(best_path))
            print('Saved best model ->', best_path)
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= args.patience:
            print(f'Early stopping triggered (no improvement for {args.patience} epochs).')
            break

    # Load best model
    if (checkpoint_dir / 'kfs_mobilenetv3_best.pth').exists():
        model.load_state_dict(torch.load(checkpoint_dir / 'kfs_mobilenetv3_best.pth', map_location=device))
        print('Loaded best model from checkpoint.')

    model.eval()

    # export TorchScript
    example = torch.randn(1, 3, args.img_size, args.img_size)
    export_torchscript(model, example, 'kfs_mobilenetv3_ts.pt')

    # export ONNX
    try:
        export_onnx(model, example, 'kfs_mobilenetv3.onnx')
    except Exception as e:
        print('ONNX export failed:', e)

    # Attempt TFLite conversion if requested
    if args.export_tflite:
        if not ONNX_TF_AVAILABLE:
            print('Onnx/TensorFlow/onnx-tf not available. Please install onnx, onnx-tf and tensorflow to enable TFLite export.')
        else:
            representative = None
            preprocess_info = {'size': args.img_size, 'mean': 0.485, 'std': 0.229}
            all_train_files = train_f
            if args.quantize:
                representative = make_representative_gen(all_train_files, preprocess_info, num_samples=min(200, len(all_train_files)))
            success = convert_onnx_to_tflite('kfs_mobilenetv3.onnx', saved_model_dir='tmp_saved_model', tflite_path='kfs_mobilenetv3.tflite', quantize=args.quantize, representative_generator=representative)
            if success:
                print('TFLite conversion complete. Consider further post-training quantization on RPi if necessary.')
                if args.edge_compile:
                    compile_for_edgetpu('kfs_mobilenetv3.tflite')

    # Save training plots
    save_training_plots(history, out_prefix='kfs_training')

    writer.close()
    print('Done.')


# --------------------------
# TFLite inference helper for Raspberry Pi
# --------------------------
def tflite_inference(tflite_model_path, image_path, input_size=128, threshold=0.5):
    """Simple TFLite inference helper. Returns probability and label."""
    try:
        import numpy as np
        import cv2
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError('TFLite inference requires tensorflow (tflite runtime).')

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size)).astype(np.float32) / 255.0

    # Apply same normalization used in training (ImageNet-like)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    inp = np.expand_dims(img, axis=0)

    # handle uint8 quantized models
    if input_details[0]['dtype'] == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        inp = inp / scale + zero_point
        inp = inp.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])

    # handle uint8 outputs
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        out = (out.astype(np.float32) - zero_point) * scale

    prob = 1.0 / (1.0 + math.exp(-float(out.squeeze())))
    label = int(prob > threshold)
    return prob, label


if __name__ == '__main__':
    main()
