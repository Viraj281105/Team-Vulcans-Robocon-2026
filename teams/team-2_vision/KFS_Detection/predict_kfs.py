import os
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------------------------------------
# PATHS
# ------------------------------------------------
MODEL_PATH = r"/home/meherdeep/rpi-github/Team-Vulcans-Robocon-2026/teams/team-2_vision/KFS_Detection/outputs/kfs_mobilenetv3_large_rpi.pt"
DATASET_PATH = r"/home/meherdeep/rpi-github/Team-Vulcans-Robocon-2026/teams/team-2_vision/dataset"

IMG_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------------------------------------
# Albumentations Preprocess
# ------------------------------------------------
preprocess = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ------------------------------------------------
# Load Model (TorchScript)
# ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
model.to(device)

# ------------------------------------------------
# Utilities
# ------------------------------------------------
def list_images_recursive(folder):
    """Return all image file paths in folder & subfolders."""
    folder = Path(folder)
    return [str(p) for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def pick_random_samples(num=5):
    """Pick random samples from real and fake folders recursively."""
    real_dir = os.path.join(DATASET_PATH, "Real")
    fake_dir = os.path.join(DATASET_PATH, "Fake")

    real_imgs = list_images_recursive(real_dir)
    fake_imgs = list_images_recursive(fake_dir)

    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)

    return real_imgs[:num], fake_imgs[:num]

def preprocess_image(path):
    """Load + preprocess an image."""
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    tensor = preprocess(image=img_np)["image"].unsqueeze(0)
    return tensor.to(device)

def predict(path):
    """Return (label, score)"""
    tensor = preprocess_image(path)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)   # single logit output
        real_prob = probs.item()        # probability of REAL class

    label = "REAL" if real_prob > 0.5 else "FAKE"
    return label, real_prob

# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":
    real_samples, fake_samples = pick_random_samples()

    print("\n===== REAL SAMPLES =====")
    for path in real_samples:
        label, score = predict(path)
        print(f"{os.path.basename(path)} → {label}  (RealProb={score:.4f})")

    print("\n===== FAKE SAMPLES =====")
    for path in fake_samples:
        label, score = predict(path)
        print(f"{os.path.basename(path)} → {label}  (RealProb={score:.4f})")
