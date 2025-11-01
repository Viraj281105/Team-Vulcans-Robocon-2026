#!/usr/bin/env python3
"""
Scroll Cube Detection and KFS Face Capture
Team Vulcans - Robocon 2026

Detects red or blue KFS cubes (scrolls) from live video feed,
filters out false positives (faces, random bright objects),
and saves detected front faces for later character classification.
"""

import cv2
import numpy as np
import os
from datetime import datetime

# ======= CONFIG =======
output_dir = "/home/viraj/Robocon/Team-Vulcans-Robocon-2026/teams/team-2_vision/captured_faces"
os.makedirs(output_dir, exist_ok=True)
# ======================

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found or frame read failed.")
        break

    # Slight blur to reduce noise
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    # ======= COLOR MASKS (Red + Blue, strong colors only) =======
    blue_mask = cv2.inRange(hsv, (90, 100, 80), (130, 255, 255))
    red_mask1 = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 120, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    mask = cv2.bitwise_or(red_mask, blue_mask)

    # ======= MORPH CLEANUP =======
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ======= INTENSITY CHECK (Remove weak areas) =======
    mask_mean = cv2.mean(mask)[0]
    if mask_mean < 10:
        mask[:] = 0

    # ======= FIND CONTOURS =======
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000 or area > 200000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if aspect < 0.7 or aspect > 1.3:
            continue

        # Ignore regions near top of frame (faces/hands)
        if y < frame.shape[0] * 0.25:
            continue

        # Solidity filter (avoid soft blobs)
        hull = cv2.convexHull(cnt)
        solidity = float(area) / cv2.contourArea(hull)
        if solidity < 0.8:
            continue

        # ======= DRAW DETECTION =======
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Scroll Detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ======= SAVE CROP =======
        face_crop = frame[y:y + h, x:x + w]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{output_dir}/scroll_{timestamp}.jpg"
        cv2.imwrite(filename, face_crop)
        print(f"[INFO] Saved: {filename}")

    # ======= SHOW OUTPUT =======
    cv2.imshow("Mask", mask)
    cv2.imshow("Scroll Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
