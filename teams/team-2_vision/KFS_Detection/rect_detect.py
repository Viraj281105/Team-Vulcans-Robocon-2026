#!/usr/bin/env python3
"""
rect_detect.py
Detects Blue or Red Kung Fu Scrolls (KFS) and extracts a flat ROI
for further processing (e.g., character classification).

Usage:
    python3 rect_detect.py
Press 'q' to quit.
"""

import cv2 as cv
import numpy as np

def detect_scroll(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # --- Color ranges for Blue and Red ---
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])

    lower_red1 = np.array([0, 100, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 70])
    upper_red2 = np.array([180, 255, 255])

    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv.inRange(hsv, lower_red1, upper_red1) | cv.inRange(hsv, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask_blue, mask_red)

    # --- Morphological filtering ---
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)

    # --- Contour detection ---
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, None

    # --- Find largest contour ---
    contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(contour)
    if area < 5000:  # too small, ignore
        return frame, None

    # --- Approximate polygon ---
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)

    warped = None
    if len(approx) == 4:
        # Draw contour
        cv.drawContours(frame, [approx], -1, (0, 255, 0), 3)

        # Order points for perspective transform
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

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

        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(frame, M, (maxWidth, maxHeight))

        cv.putText(frame, "Scroll Detected", (20, 40), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2, cv.LINE_AA)
    else:
        cv.drawContours(frame, [contour], -1, (0, 0, 255), 2)

    return frame, warped


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, roi = detect_scroll(frame)

        cv.imshow("Scroll Detection", result)
        if roi is not None:
            cv.imshow("Warped Scroll ROI", roi)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
