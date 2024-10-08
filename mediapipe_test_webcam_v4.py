# mediapipe_test_webcam.py
# CS 5100 - Fall 2024
# Final Project: Testing Hand Landmark Detection with webcam feed
#
# v4 Update: Add FPS display and 3D scatter plot of hand coordinates
# Used some code found at https://github.com/kinivi/hand-gesture-recognition-mediapipe
#
# FPS at 1280x720: ~15
# FPS at 640x480: ~31


# Import OpenCV library
import cv2

# Import mediapipe dependencies
import mediapipe as mp
import numpy as np
import time

# Import plotting libraries
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import custom libraries
from cvfpscalc import CvFpsCalc

# Helpful mediapipe shortcuts
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define opencv constants (BGR format)
RED   = (0, 0, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
FONT  = cv2.FONT_HERSHEY_COMPLEX_SMALL
RESIZE_W = 1280 
RESIZE_H = 720

# Define MediaPipe color palette in RGB-255 format (for use in 3D scatter plot
PINKY_BLUE    = (21, 101, 192)
RING_GREEN    = (48, 255, 48)
MIDDLE_YELLOW = (255, 204, 0)
INDEX_PURPLE  = (128, 64, 128)
THUMB_TAN     = (255, 219, 180)
PALM_RED      = (255, 48, 48)

# Custom drawing function
def draw_landmarks_on_image(image, detection_result):
    def calc_bounding_rect(image, landmarks):
        """
        Helper function to calculate bounding box rectangle of hand
        """
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    if detection_result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp.solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

            # Get handedness
            handedness = results.multi_handedness[i]

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = image.shape
            x_coordinates = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))]
            y_coordinates = [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - 15

            # Draw hand bounding box
            buffer_px = 10
            bbox = calc_bounding_rect(image, hand_landmarks)
            cv2.rectangle(image, (bbox[0] - buffer_px, bbox[1] - buffer_px), (bbox[2] + buffer_px, bbox[3] + buffer_px), BLACK, 3)

            # Draw handedness (left or right hand) on the image.
            cv2.putText(image, f"{handedness.classification[0].label}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        1, RED, 1, cv2.LINE_AA)
            

    return image



# Define helper function for plotting
def plot_hand_3D(detection_result):
    # Extract XYZ hand coordinates
    all_x = []
    all_y = []
    all_z = []
    all_colors = []
    if detection_result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp.solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            all_x.extend([hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))])
            all_y.extend([hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))])
            all_z.extend([hand_landmarks.landmark[i].z for i in range(len(hand_landmarks.landmark))])
    
            # Set scatter point colors
            colors = [(0, 0, 0) for i in range(21)]
            for i in range(21):
                if (i in [18, 19, 20]):
                    # Pinky
                    colors[i] = np.array(PINKY_BLUE) / 255
                elif (i in [14, 15, 16]):
                    # Ring finger
                    colors[i] = np.array(RING_GREEN) / 255
                elif (i in [10, 11, 12]):
                    # Middle finger
                    colors[i] = np.array(MIDDLE_YELLOW) / 255
                elif (i in [6, 7, 8]):
                    # Index finer
                    colors[i] = np.array(INDEX_PURPLE) / 255
                elif (i in [1, 2, 3, 4]):
                    # Thumb 
                    colors[i] = np.array(THUMB_TAN) / 255
                else:
                    # Palm
                    colors[i] = np.array(PALM_RED) / 255

            # Append colors to all_colors array
            all_colors.extend(colors)


    # Update XYZ data in figure
    src._offsets3d = (all_x, all_y, all_z)
    src.set_color(all_colors)
    fig.canvas.draw()
    fig.canvas.flush_events()



# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_H)

# Set up empty 3D scatter plot for hands
plt.ion()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
src = ax.scatter(np.random.random(10), np.random.random(10), np.random.random(10))
title = ax.set_title('3d Hand Coordinates')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([-0.3, 0.3])

# Set 3D viewpoint
# To mimic camera perception, use the line below
#ax.view_init(elev=-90, azim=-90, roll=0)
# To visualize hand depth (z axis)
ax.view_init(elev=0, azim=0, roll=0)
plt.show(block=False)




with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Define FPS counter object
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
    
        # Get FPS
        fps = cvFpsCalc.get()

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed = draw_landmarks_on_image(image, results)

        # Draw FPS (TODO: Put in a draw_info function)
        processed = cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

        # Create 3D scatterplot of 21 key hand points (if detected)
        plot_hand_3D(results)
            
        # Showed processed frame
        cv2.imshow('Processed MediaPipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()