import time
import cv2 as cv
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

# --- Optimized Settings for Raspberry Pi 3B+ ---
# Use a lower resolution to significantly speed up processing
W, H = 640, 480
FPS = 20 # Target FPS

picam2 = Picamera2()
# Use a video configuration that the Pi can handle efficiently
config = picam2.create_video_configuration(main={"size": (W, H), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2) # Allow camera to warm up

#-----Manually set the camera controls-----
picam2.set_controls({"AwbEnable":False, "AwbMode": controls.AwbModeEnum.Fluorescent}) 

# --- HSV Color Range for the Blue Scroll Box ---
lower_blue = np.array([0, 50, 50])
upper_blue = np.array([25, 255, 255])
# -------------------------------------------------------------------

print("Starting lightweight scroll detection. Press 'q' to exit.")
try:
    while True:
        frame = picam2.capture_array("main")

        # Step 1: Isolate the Blue Color
        hsv_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
        color_mask = cv.inRange(hsv_frame, lower_blue, upper_blue)

        # Step 2: Clean Up the Mask (use a slightly smaller kernel for performance)
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel)

        # Step 3: Find Shapes in the Cleaned Mask
        contours, _ = cv.findContours(cleaned_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Step 4: Verify and Lock On to the Target
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            area = cv.contourArea(largest_contour)

            # Adjust minimum area for the lower resolution
            if area > 500:
                perimeter = cv.arcLength(largest_contour, True)
                approx = cv.approxPolyDP(largest_contour, 0.04 * perimeter, True)

                if len(approx) == 4:
                    x, y, w, h = cv.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    
                    if 0.85 < aspect_ratio < 1.15:
                        # --- TARGET LOCKED ---
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv.putText(frame, "LOCKED", (x, y - 10),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Display the results ---
        # Displaying the window uses resources. For pure performance on the robot,
        # you might comment this line out during the actual competition.
        cv.imshow("Scroll Detector (Lightweight)", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    picam2.stop()
    cv.destroyAllWindows()
    print("Camera and windows closed.")