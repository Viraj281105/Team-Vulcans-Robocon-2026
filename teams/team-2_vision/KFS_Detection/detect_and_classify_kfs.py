#!/usr/bin/env python3
"""
KFS Detection + Classification
Team Vulcans - ABU Robocon 2026
Detects Kung Fu Scroll (cube) and classifies it as Real or Fake.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# ==== CONFIG ====
MODEL_PATH = "/home/meherdeep/rpi-github/Team-Vulcans-Robocon-2026/teams/team-2_vision/KFS_Detection/kfs_classifier.h5"
IMG_SIZE = (224, 224)  # match your CNN training input size
CONFIDENCE_THRESHOLD = 0.5  # scroll mask threshold

# ==== LOAD MODEL ====
print("🔄 Loading classification model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ==== HELPER FUNCTIONS ====
def preprocess_image(img):
    """Resize + normalize image for CNN input."""
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=0)

def classify_scroll(crop):
    """Predict if the detected scroll face is real or fake."""
    pred = model.predict(preprocess_image(crop))[0][0]
    label = "REAL ✅" if pred >= 0.5 else "FAKE ❌"
    conf = pred if pred >= 0.5 else 1 - pred
    return label, float(conf)

# ==== MAIN DETECTION ====
def detect_scroll_cube(frame):
    """Detect cube-like scroll using color segmentation + contours."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue and Red color ranges (tuned for paper + arena)
    blue_lower = np.array([90, 80, 40])
    blue_upper = np.array([130, 255, 255])
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])

    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask = cv2.bitwise_or(mask_blue, mask_red)
    mask = cv2.medianBlur(mask, 7)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        aspect_ratio = max(rect[1]) / (min(rect[1]) + 1e-5)

        if 0.5 < aspect_ratio < 2.0:  # near-square
            detected_boxes.append(box)

    return detected_boxes, mask

# ==== LIVE CAMERA ====
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not detected.")
        return

    print("🎥 Starting KFS detection & classification... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, mask = detect_scroll_cube(frame)
        display = frame.copy()

        for box in boxes:
            x, y, w, h = cv2.boundingRect(box)
            crop = frame[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            label, conf = classify_scroll(crop)
            color = (0, 255, 0) if "REAL" in label else (0, 0, 255)

            cv2.drawContours(display, [box], 0, color, 2)
            cv2.putText(display, f"{label} ({conf:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("KFS Detection + Classification", display)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==== ENTRY ====
if __name__ == "__main__":
    main()
