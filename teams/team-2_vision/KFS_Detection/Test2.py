import cv2

# USB camera index (usually 0)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Optional settings for clarity
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Camera not detected.")
    exit()

print("📷 Camera streaming... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("⚠ No frame captured.")
        continue

    cv2.imshow("USB Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🔚 Camera closed.")
