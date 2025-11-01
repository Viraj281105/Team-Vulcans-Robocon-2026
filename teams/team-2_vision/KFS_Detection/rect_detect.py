#!/usr/bin/env python3
"""
rect_detect.py — KFS Cube Face Capture
--------------------------------------
Detects the Kung Fu Scroll (cube) from live camera feed,
isolates its largest visible face, and saves that face as a
flattened image (for dataset building).

Press 'q' to exit.
"""

import cv2
import numpy as np
import os
import time

# Create output directory
output_dir = "/home/viraj/Robocon/Team-Vulcans-Robocon-2026/teams/team-2_vision/captured_faces"
os.makedirs(output_dir, exist_ok=True)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

# Timer for saving images
last_save_time = 0
save_interval = 2  # seconds
img_counter = 1

print("📷 Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Resize for performance
    frame = cv2.resize(frame, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for blue and red
    blue_lower = np.array([90, 80, 50])
    blue_upper = np.array([130, 255, 255])
    red_lower1 = np.array([0, 80, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 80, 50])
    red_upper2 = np.array([180, 255, 255])

    # Masks
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_blue, mask_red)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Contour detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > best_area:
                best_area = area
                best_cnt = approx

    # If a good cube face found
    if best_cnt is not None:
        cv2.drawContours(frame, [best_cnt], -1, (0, 255, 0), 2)

        # Sort points (top-left, top-right, bottom-right, bottom-left)
        pts = best_cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Compute width & height for warp
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxWidth = int(max(widthA, widthB))
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Warp perspective
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

        # Show warped face
        cv2.imshow("Detected Face", warp)

        # Save image every few seconds
        current_time = time.time()
        if current_time - last_save_time > save_interval:
            filename = os.path.join(output_dir, f"kfs_{img_counter:03d}.jpg")
            cv2.imwrite(filename, warp)
            print(f"💾 Saved: {filename}")
            img_counter += 1
            last_save_time = current_time

    # Show main window
    cv2.imshow("KFS Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
