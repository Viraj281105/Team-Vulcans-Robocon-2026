import cv2 as cv
import numpy as np

# Load the image you provided to find the color range
# Make sure the image file 'image_f192fa.jpg' is in the same folder
image = cv.imread('../Test_Images/Image1.jpeg')
if image is None:
    print("Error: Could not load image. Make sure 'image_f192fa.jpg' is in the correct path.")
    exit()

# Resize for easier display
image = cv.resize(image, (640, 480))
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

def nothing(x):
    pass

# Create a window for the trackbars
cv.namedWindow('HSV Tuner')
cv.createTrackbar('H Lower', 'HSV Tuner', 0, 179, nothing)
cv.createTrackbar('S Lower', 'HSV Tuner', 0, 255, nothing)
cv.createTrackbar('V Lower', 'HSV Tuner', 0, 255, nothing)
cv.createTrackbar('H Upper', 'HSV Tuner', 179, 179, nothing)
cv.createTrackbar('S Upper', 'HSV Tuner', 255, 255, nothing)
cv.createTrackbar('V Upper', 'HSV Tuner', 255, 255, nothing)

# Set some initial values that might be close for blue
cv.setTrackbarPos('H Lower', 'HSV Tuner', 90)
cv.setTrackbarPos('S Lower', 'HSV Tuner', 100)
cv.setTrackbarPos('V Lower', 'HSV Tuner', 100)
cv.setTrackbarPos('H Upper', 'HSV Tuner', 130)
cv.setTrackbarPos('S Upper', 'HSV Tuner', 255)
cv.setTrackbarPos('V Upper', 'HSV Tuner', 255)


while True:
    h_l = cv.getTrackbarPos('H Lower', 'HSV Tuner')
    s_l = cv.getTrackbarPos('S Lower', 'HSV Tuner')
    v_l = cv.getTrackbarPos('V Lower', 'HSV Tuner')
    h_u = cv.getTrackbarPos('H Upper', 'HSV Tuner')
    s_u = cv.getTrackbarPos('S Upper', 'HSV Tuner')
    v_u = cv.getTrackbarPos('V Upper', 'HSV Tuner')

    lower_bound = np.array([h_l, s_l, v_l])
    upper_bound = np.array([h_u, s_u, v_u])

    mask = cv.inRange(hsv_image, lower_bound, upper_bound)
    result = cv.bitwise_and(image, image, mask=mask)

    cv.imshow('Original Image', image)
    cv.imshow('Mask', mask)
    cv.imshow('Result', result)

    if cv.waitKey(1) & 0xFF == ord('q'):    
        break

cv.destroyAllWindows()
print(f"Your final HSV range is: \nlower_blue = np.array([{h_l}, {s_l}, {v_l}])\nupper_blue = np.array([{h_u}, {s_u}, {v_u}])")