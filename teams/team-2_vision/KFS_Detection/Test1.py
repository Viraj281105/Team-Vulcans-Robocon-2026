#!/usr/bin/env python3
"""
Integrated Scroll/Cube Detection and KFS Classification with ROI Cropping (50% Outset)
USB Camera + PyTorch Classification Pipeline
"""

import time
import cv2 as cv
import numpy as np

# --- PyTorch Imports ---
import torch
from torchvision import transforms
import sys

# -------------------- CONFIGURATION --------------------
# Camera Settings
CAMERA_INDEX = 0       # USB camera -> /dev/video0
W, H = 640, 480
FPS = 20

# PyTorch Model Settings
MODEL_PATH = "/home/meherdeep/rpi-github/Team-Vulcans-Robocon-2026/teams/team-2_vision/KFS_Detection/outputs/kfs_mobilenetv3_large_rpi.pt"
MODEL_IMG_SIZE = (128, 128)
KFS_THRESHOLD = 0
DEVICE = torch.device("cpu")

# Color Detection Parameters
lower_blue = np.array([0, 50, 50])
upper_blue = np.array([25, 255, 255])
MIN_CONTOUR_AREA = 500

# ROI buffer
OUTSET_PERCENT = 0.50

# -------------------- INITIALIZATION --------------------

# PyTorch Transformation
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(MODEL_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("🔄 Loading PyTorch model...")
try:
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    model.to(DEVICE)
    print("✅ Model Ready.")
except Exception as e:
    print(f"❌ Model load error: {e}")
    sys.exit()

# --- Initialize USB Camera ---
print("📸 Initializing USB Camera...")
cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv.CAP_PROP_FPS, FPS)
time.sleep(1)

if not cap.isOpened():
    print("❌ ERROR: Camera not accessible.")
    sys.exit()

print("🚀 System Ready. Press 'q' to exit.")

# -------------------- MAIN LOOP --------------------

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame not received. Retrying...")
            continue

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        color_mask = cv.inRange(hsv_frame, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel)
        cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(cleaned_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        display_text = "NO SCROLL/CUBE"
        display_color = (255, 255, 0)

        if contours:
            largest = max(contours, key=cv.contourArea)
            area = cv.contourArea(largest)

            if area > MIN_CONTOUR_AREA:
                peri = cv.arcLength(largest, True)
                approx = cv.approxPolyDP(largest, 0.04 * peri, True)

                if len(approx) == 4:
                    x, y, w, h = cv.boundingRect(approx)

                    # Apply ROI buffer
                    bx = int(w * OUTSET_PERCENT * 0.5)
                    by = int(h * OUTSET_PERCENT * 0.5)

                    x1 = max(0, x - bx)
                    y1 = max(0, y - by)
                    x2 = min(W, x + w + bx)
                    y2 = min(H, y + h + by)

                    roi = frame[y1:y2, x1:x2]

                    if roi.size > 0:
                        try:
                            img_tensor = preprocess(roi).unsqueeze(0).to(DEVICE)

                            with torch.no_grad():
                                pred = model(img_tensor).item()

                            label = "REAL" if pred >= KFS_THRESHOLD else "FAKE"
                            display_color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
                            display_text = f"SCROLL Exists | {label} ({pred:.2f})"

                            cv.rectangle(frame, (x, y), (x + w, y + h), display_color, 3)
                            cv.rectangle(frame, (x1, y1), (x2, y2), display_color, 1)

                        except Exception as e:
                            display_text = "CLASSIFICATION ERROR"
                            display_color = (0, 165, 255)
                            print(e)

        cv.putText(frame, display_text, (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)
        cv.imshow("KFS Detector (USB Camera)", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    cap.release()
    cv.destroyAllWindows()
    print("🔚 Shutdown complete.")
