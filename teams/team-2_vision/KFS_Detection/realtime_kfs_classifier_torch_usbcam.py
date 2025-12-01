#!/usr/bin/env python3
"""
Integrated Scroll/Cube Detection and KFS Classification with ROI Cropping (50% Outset)
--------------------------------------------------------------------------------
- Stage 1: Fast detection of the scroll/cube using HSV/Contour analysis.
- Stage 2: Crop the detected area + 50% buffer and classify REAL/FAKE using a PyTorch model.
"""
import time
import cv2 as cv
import numpy as np

# --- PyTorch Imports ---
import torch
from torchvision import transforms 
import sys 

# --- 1. CONFIGURATION ---
# Camera Settings
CAMERA_INDEX = 0       # USB camera -> /dev/video0
W, H = 640, 480
FPS = 20


# PyTorch Model Settings
MODEL_PATH = "/home/meherdeep/rpi-github/Team-Vulcans-Robocon-2026/teams/team-2_vision/KFS_Detection/outputs/kfs_mobilenetv3_large_rpi.pt"
MODEL_IMG_SIZE = (128, 128) # Input size required by your PyTorch model
KFS_THRESHOLD = 0        # REAL if pred >= 0.5, FAKE otherwise
DEVICE = torch.device("cpu")

# Detection Settings 
lower_blue = np.array([0, 50, 50])      # Your chosen HSV range
upper_blue = np.array([25, 255, 255])
MIN_CONTOUR_AREA = 500 # Minimum pixel area for a detected shape

# --- MODIFIED: ROI Cropping Buffer ---
OUTSET_PERCENT = 0.30 # Changed from 0.30 to 0.50 (50% buffer)
# --- END MODIFIED ---

# --- 2. INITIALIZATION ---

# --- PyTorch Preprocessing Pipeline ---
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(MODEL_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load PyTorch model ---
print("🔄 Loading PyTorch model...")
try:
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    model.to(DEVICE)
    print("✅ PyTorch Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

# --- Initialize Pi Camera ---
print("📸 Initializing USB Camera...")
cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv.CAP_PROP_FPS, FPS)
time.sleep(1)


# Manually set the camera controls
if not cap.isOpened():
    print("❌ ERROR: Camera not accessible.")
    sys.exit()


print("Starting integrated detection and classification with ROI 50% crop. Press 'q' to exit.")

# --- 3. MAIN LOOP ---
try:
    while True:
        ret, frame = cap.read()
        
        # --- Stage 1: Scroll/Cube Detection ---
        hsv_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV) 
        color_mask = cv.inRange(hsv_frame, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8) 
        cleaned_mask = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel)
        cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(cleaned_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        scroll_detected = False
        display_text = "NO SCROLL/CUBE"
        display_color = (255, 255, 0) # Cyan for not detected

        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            area = cv.contourArea(largest_contour)

            if area > MIN_CONTOUR_AREA:
                perimeter = cv.arcLength(largest_contour, True)
                approx = cv.approxPolyDP(largest_contour, 0.04 * perimeter, True)

                if len(approx) == 4:
                    x, y, w, h = cv.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    
                    if 0.5 < aspect_ratio < 2.0: 
                        scroll_detected = True
                        
                        # --- Stage 2: KFS Classification with 50% Outset Cropping ---
                        
                        # Calculate the buffer (outset) in pixels (Half the total 50% for each side)
                        buffer_x = int(w * OUTSET_PERCENT * 0.5) 
                        buffer_y = int(h * OUTSET_PERCENT * 0.5)
                        
                        # Define new padded coordinates, clamped to frame edges
                        x1 = max(0, x - buffer_x)
                        y1 = max(0, y - buffer_y)
                        x2 = min(W, x + w + buffer_x)
                        y2 = min(H, y + h + buffer_y)
                        
                        # Crop the ROI from the original frame
                        roi_frame = frame[y1:y2, x1:x2]
                        
                        if roi_frame.size > 0:
                            # 1. Preprocess the cropped ROI
                            try:
                                img_tensor = preprocess(roi_frame) 
                                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                                
                                # 2. Predict REAL/FAKE
                                with torch.no_grad():
                                    output = model(img_tensor)
                                    pred = output.item()
                                
                                # 3. Process prediction and set display text
                                label = "REAL" if pred >= KFS_THRESHOLD else "FAKE"
                                confidence = pred if label == "REAL" else pred
                                
                                display_color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
                                display_text = f"SCROLL Exists! | {label} ({confidence:.2f})"
                                
                                # Draw detection box with classification color
                                cv.rectangle(frame, (x, y), (x + w, y + h), display_color, 3)
                                # Optionally draw the extended ROI border
                                cv.rectangle(frame, (x1, y1), (x2, y2), display_color, 1)

                            except Exception as e:
                                print(f"Error during classification: {e}")
                                display_text = "SCROLL Exists! | CLASSIFICATION ERROR"
                                display_color = (0, 165, 255) # Orange
                        
        # --- Display the results ---
        # Show the primary status (Existence & Classification) prominently
        cv.putText(frame, display_text, (30, 50),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, display_color, 2)
        
        cv.imshow("Integrated KFS Detector (Live)", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"An error occurred: {e}")
    
finally:
    # --- Cleanup ---
    picam2.stop()
    cv.destroyAllWindows()
    print("Camera and windows closed. Exiting.")
