import cv2
import numpy as np
import pyautogui

# set color ranges for classification
lower_color1 = np.array([0, 100, 100])   # red-like color
upper_color1 = np.array([10, 255, 255])

lower_color2 = np.array([50, 100, 100])  # green-like color
upper_color2 = np.array([70, 255, 255])

# classifies the color of the object
def classify_color(frame):
    # converts the image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # creates masks for the colors
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)

    # detects if any color is present
    if np.any(mask1):
        return "Color 1 Detected"
    elif np.any(mask2):
        return "Color 2 Detected"
    else:
        return "No Object"

# function to perform scrolling based on color classification
def perform_action_based_on_color(result):
    if result == "Color 1 Detected":
        pyautogui.scroll(100)  # Scroll up
        print("Scrolling up")
    elif result == "Color 2 Detected":
        pyautogui.scroll(-100)  # Scroll down
        print("Scrolling down")
    else:
        print("No action")

# start capturing video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # read frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # classifies color from the frame
    color_result = classify_color(frame)

    # performs action based on color detection
    perform_action_based_on_color(color_result)

    # displays the frame with classification result
    cv2.putText(frame, color_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Color Classification Feed', frame)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()