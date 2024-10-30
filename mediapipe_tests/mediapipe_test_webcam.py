# mediapipe_test_webcam.py
# CS 5100 - Fall 2024
# Final Project: Testing Hand Landmark Detection with webcam feed

# Import OpenCV library
import cv2

# Import mediapipe dependencies
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions 

import numpy as np

# Define model path
model_path = 'hand_landmarker.task'

# Set up options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)                           # Maybe switch this to LIVE_STREAM mode

# Define opencv constants
RED   = (0, 0, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
FONT  = cv2.FONT_HERSHEY_COMPLEX_SMALL
RESIZE_W = 1280 
RESIZE_H = 720

# Define a helper function to plot key points on hand
def plot_points(img, landmarker_result):
    for landmarks in landmarker_result.hand_landmarks:
        img_H, img_W = img.shape[:2]

        # Define hand path to trace (point indices)
        line_pairs = [[0, 1], [1, 2], [2, 3], [3, 4],
                    [0, 5], [5, 6], [6, 7], [7, 8],
                    [5, 9], [9, 10], [10, 11], [11, 12],
                    [9, 13], [13, 14], [14, 15], [15, 16],
                    [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
                    ]

        # Plot connecting lines between the points
        for s_pt, e_pt in line_pairs:
            s_x = int(landmarks[s_pt].x * img_W)
            s_y = int(landmarks[s_pt].y * img_H)
            e_x = int(landmarks[e_pt].x * img_W)
            e_y = int(landmarks[e_pt].y * img_H)

            cv2.line(img, (s_x, s_y), (e_x, e_y), GREEN, 1)

        # Plot 21 key hand landmarks
        for i, landmark in enumerate(landmarks):
            # Make sure to un-normalize points
            x = int(landmark.x * img_W)
            y = int(landmark.y * img_H)

            cv2.circle(img, (x, y), 5, RED, -1)
            cv2.putText(img, str(i), (x - 20, y), FONT, 0.6, WHITE, 1)

    return img


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    print(f"Hands detected: {len(hand_landmarks_list)}")
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    1, GREEN, 1, cv2.LINE_AA)

    return annotated_image

# Open the webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_H)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not captured correctly, break the loop
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to mp_image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    with HandLandmarker.create_from_options(options) as landmarker:
        hand_landmarker_result = landmarker.detect(mp_image)

        #processed = plot_points(frame, hand_landmarker_result)
        processed = draw_landmarks_on_image(frame, hand_landmarker_result)

        # Display the resulting frame in a window
        cv2.imshow('Processed Video Feed', processed)

    # Press 'q' to exit the video feed window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()