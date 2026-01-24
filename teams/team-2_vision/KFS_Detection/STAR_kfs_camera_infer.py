#!/usr/bin/env python3
"""
KFS Detection – Jetson Runtime
USB Camera + TorchScript Inference + ROI Cropping + Stability Smoothing
Runs inside NVIDIA L4T Docker container.
"""

import time
import cv2
import numpy as np
from collections import deque

import torch
from torchvision import transforms

# -------------------- CONFIG --------------------

# Camera
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
TARGET_FPS = 30

# Model
MODEL_PATH = "/workspace/Robocon/kfs_mobilenetv3_large_rpi.pt"
MODEL_IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.50

# ROI
MIN_CONTOUR_AREA = 500
OUTSET_PERCENT = 0.50

# Color detection (HSV)
LOWER_BLUE = np.array([0, 50, 50])
UPPER_BLUE = np.array([25, 255, 255])

# Stability smoothing
SMOOTH_WINDOW = 5
pred_buffer = deque(maxlen=SMOOTH_WINDOW)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- PREPROCESS --------------------

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(MODEL_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------- LOAD MODEL --------------------

print("🔄 Loading TorchScript model...")
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval().to(DEVICE)
print("✅ Model loaded on:", DEVICE)

# Warmup (important for Jetson latency)
dummy = torch.zeros(1, 3, *MODEL_IMG_SIZE).to(DEVICE)
with torch.no_grad():
    for _ in range(3):
        model(dummy)

# -------------------- CAMERA INIT --------------------

print("📸 Initializing camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
time.sleep(1)

if not cap.isOpened():
    raise RuntimeError("❌ Camera not accessible")

print("🚀 System Ready. Press 'q' to exit.")

# -------------------- FPS TRACKER --------------------

fps_counter = 0
fps_timer = time.time()
fps_display = 0

# -------------------- MAIN LOOP --------------------

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame drop")
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display_text = "NO TARGET"
        display_color = (255, 255, 0)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest)

                # ROI buffer
                bx = int(w * OUTSET_PERCENT * 0.5)
                by = int(h * OUTSET_PERCENT * 0.5)

                x1 = max(0, x - bx)
                y1 = max(0, y - by)
                x2 = min(FRAME_W, x + w + bx)
                y2 = min(FRAME_H, y + h + by)

                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    try:
                        img = preprocess(roi).unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            score = torch.sigmoid(model(img)).item()

                        pred_buffer.append(score)
                        smooth_score = sum(pred_buffer) / len(pred_buffer)

                        label = "REAL" if smooth_score >= CONF_THRESHOLD else "FAKE"
                        display_color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
                        display_text = f"{label} | {smooth_score:.2f}"

                        cv2.rectangle(frame, (x, y), (x + w, y + h), display_color, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 1)

                    except Exception as e:
                        display_text = "INFERENCE ERROR"
                        display_color = (0, 165, 255)
                        print("⚠", e)

        # FPS calculation
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        cv2.putText(frame, f"{display_text} | FPS: {fps_display}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, display_color, 2)

        cv2.imshow("KFS Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("🔚 Shutdown complete")
