#!/usr/bin/env python3
"""
KFS Detection – Universal Runtime (ROBUST GUI + FIXED PREDICTIONS)
Works on:
 - Windows (GUI enabled + HSV Tuner)
 - Jetson Nano Docker (headless)
USB Camera + TorchScript Inference + ROI Cropping + Stability Smoothing
"""

import platform
import time
import cv2
import numpy as np
from collections import deque

import torch
from torchvision import transforms

# -------------------- PLATFORM DETECTION --------------------

IS_WINDOWS = platform.system().lower().startswith("win")
IS_LINUX = platform.system().lower().startswith("linux")

# -------------------- CONFIG --------------------

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
TARGET_FPS = 30

# Auto model path selection
if IS_WINDOWS:
    MODEL_PATH = r"D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\KFS_Detection\outputs\kfs_mobilenetv3_large_rpi.pt"
else:
    MODEL_PATH = "/workspace/Robocon/kfs_mobilenetv3_large_rpi.pt"

MODEL_IMG_SIZE = (224, 224)

# 🔥 Tuned threshold based on actual score distribution
# REAL scores: 0.10-0.50, FAKE scores: 0.50-0.80
CONF_THRESHOLD = 0.50  # Lower = classify as REAL, Higher = classify as FAKE

MIN_CONTOUR_AREA = 500

# 🔥 REMOVED outer box expansion - allows partial detection
OUTSET_PERCENT = 0.0  # Set to 0 to disable outer box

# ✅ CORRECTED HSV blue range for actual BLUE objects
LOWER_BLUE = np.array([100, 100, 50])   # Blue hue range
UPPER_BLUE = np.array([130, 255, 255])  # Covers cyan to deep blue

# 🔥 Larger smoothing window for more stability
SMOOTH_WINDOW = 10  # Increased from 5 to 10 for less flickering
pred_buffer = deque(maxlen=SMOOTH_WINDOW)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable HSV tuner on Windows (set to False to disable)
ENABLE_HSV_TUNER = True

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

print("[INFO] Platform:", platform.system())
print("[INFO] Loading model from:", MODEL_PATH)

model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval().to(DEVICE)

print("[OK] Model loaded on:", DEVICE)

# Warmup
dummy = torch.zeros(1, 3, *MODEL_IMG_SIZE).to(DEVICE)
with torch.no_grad():
    for _ in range(3):
        model(dummy)

# -------------------- CAMERA INIT --------------------

print("[INFO] Initializing camera...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # DirectShow backend for Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
time.sleep(1)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# -------------------- GUI MODE --------------------

GUI_AVAILABLE = IS_WINDOWS
if GUI_AVAILABLE:
    print("[INFO] GUI mode enabled. Initializing OpenCV windows...")
else:
    print("[INFO] Headless mode on Jetson.")

# -------------------- HSV TUNER SETUP --------------------

hsv_tuner_ready = False

def init_hsv_tuner():
    """Initialize HSV tuner window with trackbars"""
    global hsv_tuner_ready
    try:
        window_name = "HSV Tuner"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 400, 250)
        
        cv2.createTrackbar("H_min", window_name, 100, 180, lambda x: None)
        cv2.createTrackbar("H_max", window_name, 130, 180, lambda x: None)
        cv2.createTrackbar("S_min", window_name, 100, 255, lambda x: None)
        cv2.createTrackbar("S_max", window_name, 255, 255, lambda x: None)
        cv2.createTrackbar("V_min", window_name, 50, 255, lambda x: None)
        cv2.createTrackbar("V_max", window_name, 255, 255, lambda x: None)
        
        # Create blank image for tuner window
        blank = np.zeros((250, 400, 3), dtype=np.uint8)
        cv2.putText(blank, "Adjust sliders to isolate", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(blank, "blue KFS cube", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(window_name, blank)
        
        hsv_tuner_ready = True
        print("[OK] HSV Tuner initialized")
        return True
    except Exception as e:
        print(f"[ERROR] Could not initialize HSV tuner: {e}")
        return False

def get_hsv_range():
    """Get current HSV range from trackbars or use defaults"""
    if hsv_tuner_ready:
        try:
            h_min = cv2.getTrackbarPos("H_min", "HSV Tuner")
            h_max = cv2.getTrackbarPos("H_max", "HSV Tuner")
            s_min = cv2.getTrackbarPos("S_min", "HSV Tuner")
            s_max = cv2.getTrackbarPos("S_max", "HSV Tuner")
            v_min = cv2.getTrackbarPos("V_min", "HSV Tuner")
            v_max = cv2.getTrackbarPos("V_max", "HSV Tuner")
            return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])
        except:
            pass
    return LOWER_BLUE, UPPER_BLUE

def mouse_hsv_callback(event, x, y, flags, param):
    """Click on frame to print HSV values"""
    if event == cv2.EVENT_LBUTTONDOWN and param is not None:
        hsv_frame = param
        if 0 <= y < hsv_frame.shape[0] and 0 <= x < hsv_frame.shape[1]:
            h, s, v = hsv_frame[y, x]
            print(f"[HSV] Pixel ({x},{y}): H={h} S={s} V={v}")

# -------------------- FPS --------------------

fps_counter = 0
fps_timer = time.time()
fps_display = 0
last_log_time = time.time()

# 🔥 Hysteresis to prevent flickering
last_label = "NONE"
HYSTERESIS_MARGIN = 0.05  # Require 5% score change to switch labels

print("[INFO] System running. Press 'q' to quit.")
if IS_WINDOWS:
    print("[INFO] Click on camera view to see HSV values.")

# -------------------- MAIN LOOP --------------------

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame")
            time.sleep(0.1)
            continue

        frame_count += 1
        
        # Initialize GUI windows after getting first valid frame
        if GUI_AVAILABLE and frame_count == 5:
            try:
                cv2.namedWindow("KFS Detector", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("KFS Detector", 800, 600)
                cv2.namedWindow("Blue Mask", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Blue Mask", 640, 480)
                
                if ENABLE_HSV_TUNER:
                    init_hsv_tuner()
                    
                print("[OK] GUI windows initialized")
            except Exception as e:
                print(f"[ERROR] GUI initialization failed: {e}")
                GUI_AVAILABLE = False

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get HSV range (from tuner or defaults)
        lower_blue, upper_blue = get_hsv_range()
        
        # Create mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_label = "NONE"
        smooth_score = 0.0

        # Reset buffer when nothing visible
        if not contours:
            pred_buffer.clear()
            last_label = "NONE"  # Reset hysteresis state

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest)

                # 🔥 Use tight bounding box (no expansion) to allow partial detection
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(FRAME_W, x + w)
                y2 = min(FRAME_H, y + h)

                roi = frame[y1:y2, x1:x2]
                roi_area = roi.shape[0] * roi.shape[1]

                # 🔥 Lower ROI size requirement for partial detection
                if roi.size > 0 and roi_area > 2000:
                    # Preprocess and infer
                    img = preprocess(roi).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        score = torch.sigmoid(model(img)).item()

                    # 🔥 INVERT PREDICTION (model trained with swapped labels)
                    score = 1.0 - score

                    pred_buffer.append(score)
                    smooth_score = sum(pred_buffer) / len(pred_buffer)

                    # 🔥 Hysteresis: prevent flickering near threshold
                    if last_label == "REAL":
                        # If currently REAL, need score > threshold + margin to switch to FAKE
                        detected_label = "FAKE" if smooth_score > (CONF_THRESHOLD + HYSTERESIS_MARGIN) else "REAL"
                    elif last_label == "FAKE":
                        # If currently FAKE, need score < threshold - margin to switch to REAL
                        detected_label = "REAL" if smooth_score < (CONF_THRESHOLD - HYSTERESIS_MARGIN) else "FAKE"
                    else:
                        # First detection, use standard threshold
                        detected_label = "REAL" if smooth_score < CONF_THRESHOLD else "FAKE"
                    
                    last_label = detected_label

                    # Draw only the tight bounding box (no outer box)
                    if GUI_AVAILABLE:
                        color = (0, 255, 0) if detected_label == "REAL" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        # Console status log
        if time.time() - last_log_time >= 1.0:
            print(f"[STATUS] FPS={fps_display} | Target={detected_label} | Score={smooth_score:.2f}")
            last_log_time = time.time()

        # Display GUI (Windows only)
        if GUI_AVAILABLE and frame_count >= 5:
            try:
                # Add overlay text
                cv2.putText(
                    frame,
                    f"{detected_label} | Confidence: {smooth_score:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"FPS: {fps_display}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Show current HSV range
                cv2.putText(
                    frame,
                    f"HSV: H[{lower_blue[0]}-{upper_blue[0]}] S[{lower_blue[1]}-{upper_blue[1]}] V[{lower_blue[2]}-{upper_blue[2]}]",
                    (20, FRAME_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
                
                # Set mouse callback
                cv2.setMouseCallback("KFS Detector", mouse_hsv_callback, hsv)
                
                # Show windows
                cv2.imshow("KFS Detector", frame)
                cv2.imshow("Blue Mask", mask)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quit command received")
                    break
                elif key == ord('r'):
                    print("[INFO] Resetting prediction buffer")
                    pred_buffer.clear()
                    
            except Exception as e:
                print(f"[ERROR] GUI display error: {e}")
                GUI_AVAILABLE = False

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("[INFO] Shutting down...")
    cap.release()
    if GUI_AVAILABLE:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    print("[INFO] Shutdown complete")