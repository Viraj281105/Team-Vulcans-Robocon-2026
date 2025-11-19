import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ------------------------------
# PATHS
# ------------------------------
MODEL_PATH = r"D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\KFS_Detection\outputs\kfs_mobilenetv3_large_rpi.pt"
DATASET_PATH = r"D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\DatasetIRL"

IMG_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------------------
# Albumentations preprocessing
# ------------------------------
preprocess = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ------------------------------
# Load model
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
model.to(device)

# ------------------------------
# Utilities
# ------------------------------
def list_images_recursive(folder):
    folder = Path(folder)
    return [str(p) for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    tensor = preprocess(image=img_np)["image"].unsqueeze(0)
    return tensor.to(device)

def predict(path):
    tensor = preprocess_image(path)
    with torch.no_grad():
        output = model(tensor)
        if output.numel() == 1:
            real_prob = torch.sigmoid(output).item()  # probability of REAL
            fake_prob = 1 - real_prob
        else:
            probs = torch.softmax(output, dim=1)
            fake_prob = probs[0][0].item()  # class 0 → FAKE
    label = "FAKE" if fake_prob > 0.5 else "REAL"
    return label, fake_prob

# ------------------------------
# Batch predict & metrics
# ------------------------------
if __name__ == "__main__":
    results = []
    y_true = []
    y_pred = []

    for class_name in ["real", "fake"]:
        folder = os.path.join(DATASET_PATH, class_name)
        images = list_images_recursive(folder)
        print(f"\n[INFO] Predicting {len(images)} images from '{class_name}'...")

        for path in images:
            label, score = predict(path)
            print(f"{class_name.upper()}: {os.path.basename(path)} → {label} (FakeProb={score:.4f})")
            
            results.append({
                "folder": class_name,
                "image": os.path.basename(path),
                "label": label,
                "fake_prob": score
            })
            
            # ground truth vs predicted
            y_true.append(class_name.upper())
            y_pred.append(label)

    # ------------------------------
    # Confusion Matrix & Metrics
    # ------------------------------
    print("\n[INFO] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))

    cm = confusion_matrix(y_true, y_pred, labels=["REAL", "FAKE"])
    print("[INFO] Confusion Matrix:")
    print(cm)

    acc = accuracy_score(y_true, y_pred)
    print(f"[INFO] Overall Accuracy: {acc:.4f}")

    # Average fake probabilities
    real_probs = [r["fake_prob"] for r in results if r["folder"] == "real"]
    fake_probs = [r["fake_prob"] for r in results if r["folder"] == "fake"]
    print(f"[INFO] Average FakeProb for REAL images: {np.mean(real_probs):.4f}")
    print(f"[INFO] Average FakeProb for FAKE images: {np.mean(fake_probs):.4f}")

    # Optional: save to CSV
    import csv
    csv_path = os.path.join("outputs", "batch_predictions.csv")
    os.makedirs("outputs", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "image", "label", "fake_prob"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[INFO] Batch predictions saved to {csv_path}")
