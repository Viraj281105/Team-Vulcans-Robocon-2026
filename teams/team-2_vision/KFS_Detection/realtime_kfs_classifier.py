#!/usr/bin/env python3
"""
Real-time KFS Classifier (Transfer-Learning Fine-Tuned Model)
--------------------------------------------------------------
- Uses your fine-tuned MobileNetV3 classifier.
- Works with any webcam or PiCam.
- Displays prediction ("REAL" / "FAKE") and confidence.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# --- Config ---
MODEL_PATH = "/home/viraj/Robocon/Team-Vulcans-Robocon-2026/teams/team-2_vision/kfs_classifier_v4_transfer.keras"
IMG_SIZE = (128, 128)
THRESHOLD = 0.5  # Adjust this if the model outputs uncertain scores (try 0.45 or 0.55)

# --- Load Model ---
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# --- Camera Setup ---
cap = cv2.VideoCapture(0)  # 0 for default webcam; use PiCamera2 for Raspberry Pi

if not cap.isOpened():
    raise IOError("❌ Cannot open camera")

print("🎥 Starting real-time classification... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured. Retrying...")
        continue

    # Flip horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)

    # Preprocess frame for model
    img = cv2.resize(frame, IMG_SIZE)
    img_array = np.expand_dims(img.astype("float32"), axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = float(model.predict(img_array, verbose=0)[0][0])
    label = "REAL" if pred >= THRESHOLD else "FAKE"
    confidence = pred if label == "REAL" else 1 - pred

    # Display
    color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
    text = f"{label} ({confidence:.2f})"

    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3, cv2.LINE_AA)
    cv2.imshow("KFS Classifier - Live", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting classifier.")
