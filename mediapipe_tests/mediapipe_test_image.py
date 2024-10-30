# mediapipe_test_image.py
# CS 5100 - Fall 2024
# Final Project: Testing Hand Landmark Detection with single image

# Import packages
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

# Define model path
model_path = 'hand_landmarker.task'

# Define opencv constants
RED   = (0, 0, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
FONT  = cv2.FONT_HERSHEY_COMPLEX_SMALL
RESIZE_W = 640
RESIZE_H = 480


# Set up options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)


# Define a helper function to plot key points on hand
def plot_points(img, landmarker_result):
  landmarks = hand_landmarker_result.hand_landmarks[0]
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
    print(landmark)

    # Make sure to un-normalize points
    x = int(landmark.x * img_W)
    y = int(landmark.y * img_H)

    cv2.circle(img, (x, y), 5, RED, -1)
    cv2.putText(img, str(i), (x - 20, y), FONT, 0.6, WHITE, 1)

  return img

with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  img_filepath = 'jacob_twohands.jpg'

  # Load the input image from an image file.
  mp_image = mp.Image.create_from_file(img_filepath)

  # Or from a numpy array
  hand_img = cv2.imread(img_filepath)
  hand_img = cv2.resize(hand_img, (RESIZE_W, RESIZE_H))
  img_H, img_W = hand_img.shape[:2]
  # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=hand_img)

  # Perform hand landmarks detection on the provided single image.
  # The hand landmarker must be created with the image mode.
  hand_landmarker_result = landmarker.detect(mp_image)

  print(np.array(mp_image))
  #cv2.imshow("Hand Image", hand_img)

  processed = plot_points(hand_img, hand_landmarker_result)

  cv2.imshow("Processed Hand Image", processed)
  # print(key_pts)

  cv2.waitKey(0)