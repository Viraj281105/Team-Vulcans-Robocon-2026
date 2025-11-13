#!/usr/bin/env python3
"""
Real-time KFS Classifier using Pi Camera (CSI)
----------------------------------------------
- Works with libcamera + Picamera2.
- Displays "REAL"/"FAKE" predictions in real time.
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# --- Config ---
MODEL_PATH = "/home/meherdeep/rpi-github/Team-Vulcans-Robocon-2026/teams/team-2_vision/kfs_classifier_v4_transfer.keras"
IMG_SIZE = (128, 128)
THRESHOLD = 0.5

# --- Load model ---
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# --- Initialize Pi Camera ---
print("📸 Initializing Pi Camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
print("🎥 Camera started! Press 'q' to quit.")

while True:
    # Capture a frame as a NumPy array
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)  # Mirror horizontally

    # Preprocess for model
    img = cv2.resize(frame, IMG_SIZE)
    img_array = np.expand_dims(img.astype("float32"), axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = float(model.predict(img_array, verbose=0)[0][0])
    label = "REAL" if pred >= THRESHOLD else "FAKE"
    confidence = pred if label == "REAL" else 1 - pred

    # Display results
    color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3, cv2.LINE_AA)

    cv2.imshow("KFS Classifier - Live (PiCam)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
print("👋 Exiting classifier.")
