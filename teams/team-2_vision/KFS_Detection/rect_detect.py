#!/usr/bin/env python3
import cv2 as cv
import numpy as np

# -------------------------------
# CONFIGURATION
# -------------------------------
MIN_AREA = 4000          # ignore small objects
MAX_AREA = 250000        # ignore huge ones
ASPECT_TOL = 0.4         # tolerance for aspect ratio check
DEBUG = True              # set False for headless use

# HSV color ranges for blue & red scrolls
COLOR_RANGES = {
    "blue": [(100, 80, 80), (130, 255, 255)],   # HSV range for blue
    "red1": [(0, 100, 80), (10, 255, 255)],     # lower red
    "red2": [(160, 100, 80), (179, 255, 255)]   # upper red
}

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def color_mask(img_hsv):
    """Combine red and blue masks"""
    mask_blue = cv.inRange(img_hsv, np.array(COLOR_RANGES["blue"][0]), np.array(COLOR_RANGES["blue"][1]))
    mask_red1 = cv.inRange(img_hsv, np.array(COLOR_RANGES["red1"][0]), np.array(COLOR_RANGES["red1"][1]))
    mask_red2 = cv.inRange(img_hsv, np.array(COLOR_RANGES["red2"][0]), np.array(COLOR_RANGES["red2"][1]))
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    combined = cv.bitwise_or(mask_blue, mask_red)
    return combined

def detect_scrolls(frame):
    """Detect red/blue rectangular scrolls even with tilt"""
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = color_mask(hsv)

    # Cleanup
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detected = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        rect = cv.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)
        if 0.5 - ASPECT_TOL < aspect_ratio < 2.0 + ASPECT_TOL:  # roughly rectangular
            box = cv.boxPoints(rect)
            box = np.intp(box)

            # Color confidence: mean mask value inside box
            sub_mask = np.zeros_like(mask)
            cv.drawContours(sub_mask, [box], 0, 255, -1)
            color_score = cv.mean(mask, mask=sub_mask)[0]
            if color_score < 50:
                continue

            detected.append(box)
            cv.drawContours(frame, [box], 0, (0, 255, 0), 3)
            cv.putText(frame, "KFS Detected", (int(x - w/2), int(y - h/2 - 10)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, len(detected)

# -------------------------------
# MAIN LOOP
# -------------------------------
def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("🎥 Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed, count = detect_scrolls(frame)

        if DEBUG:
            cv.imshow("KFS Detection", processed)
            cv.imshow("Mask", color_mask(cv.cvtColor(frame, cv.COLOR_BGR2HSV)))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
