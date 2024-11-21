# inference.py
# CS 5100 - Fall 2024
# Final Project: Testing Hand Landmark Detection with webcam feed
#
# This script performs the inference using the trained model 
#
# FPS at 1280x720: XX
# FPS at 640x480: XX

# Import OpenCV library
import cv2

# Import mediapipe dependencies
import mediapipe as mp
import numpy as np
import time

# Import plotting libraries
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model

import tensorflow as tf
print(tf.__version__)


# Import custom libraries
import sys
sys.path.append("..")
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

# Define MediaPipe color palette in RGB-255 format (for use in 3D scatter plot)
PINKY_BLUE    = (21, 101, 192)
RING_GREEN    = (48, 255, 48)
MIDDLE_YELLOW = (255, 204, 0)
INDEX_PURPLE  = (128, 64, 128)
THUMB_TAN     = (255, 219, 180)
PALM_RED      = (255, 48, 48)

# Define model parameters
GESTURE_TIME   = 3                                  # [s], time to collect all frames
FRAMES_PER_SEQ = 10
FRAME_DELAY    = GESTURE_TIME / FRAMES_PER_SEQ      # [s]
last_save_time = 0
buffer_idx     = 0
gesture_idx    = 0
still_thres    = 0.15                               # Normalized distances - May need to adjust depending on FRAME_DELAY
buffer_counter = 1
reached_first_still = False
LOGGING        = True                               # If logging is enabled, store each gesture to enumerated buffer .npy file in data_collection/data folder

# Load LSTM model for gesture classification
#lstm_model = load_model('nn_weights/lstm_2class_20241114_better.h5')
lstm_model = load_model('nn_weights/lstm_2class_20241119_test.h5')

# Define an array to store the hand key point sequence (gesture buffer)
buffer_seq    = np.zeros((1, 10, 21, 3))        # Continuously grabbing hand landmark points
gesture_seq   = np.zeros((1, 10, 21, 3))        # Gesture sequence that is fed to LSTM model (only begins to get filled once the first still hand frame has been reached)
idx_seq       = np.zeros((1, 10))
gesture_label = ""  # Variable to store current gesture label

# Function to detect if hand is still
def is_still():
    # Calculate difference between previous point (for now) - This needs to be scaled with distance (z coord)
    # TODO: Maybe use Euclidean (2-norm) distance instead
    global buffer_idx
    global buffer_seq

    #gesture_seq_xyz = np.transpose(gesture_seq.reshape(1, FRAMES_PER_SEQ, 3, 21), (0, 1, 3, 2))                 # Shape: [1, 10, 21, 3]
    # print(f"Gesture orig xyz: {gesture_seq[0, 0, [0, 21, 42]]}, new xyz: {gesture_seq_xyz[0, 0, 0, :]}")
    #total_diff = np.mean(np.divide(abs(gesture_seq[0, buffer_idx, 0] - gesture_seq[0, buffer_idx - 1, 1]), gesture_seq[0, buffer_idx - 1, 2]))
    total_diff = np.sum(np.sqrt(np.sum((buffer_seq[0, buffer_idx - 1, :, 0:2] - buffer_seq[0, buffer_idx - 2, :, 0:2])**2, axis=1)))
    #print(f"Pixel diff: {total_diff}")
    return total_diff < still_thres

# Function to process landmarks into the required buffer format
def update_buffers(gesture_seq, landmarks):
    """
    Update the gesture buffer with the latest hand landmarks.
    The buffer shape is (1, 10, 21, 3)
    """
    global buffer_idx
    global gesture_idx
    global gesture_label
    global reached_first_still

    hand_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    # Tried creating hand_data array similar to data_collection_FULL, but no difference
    # gesture_seq = np.roll(gesture_seq, shift=-1, axis=1)  # Shift buffer
    # gesture_seq[0, -1, :] = hand_data                       # Add new frame to the end of the buffer

    # Log all hand coordinate data at the specified intervals for still hand 
    print(f"Recording buffer index {buffer_idx}")
    buffer_seq[0, buffer_idx, :, :] = hand_data
    buffer_idx = (buffer_idx + 1) % FRAMES_PER_SEQ       

    if (gesture_idx != FRAMES_PER_SEQ - 1 and reached_first_still):
        # Make sure to only fill frames in the buffer AFTER first still has been reached
        print(f"Reached first still, gesture_idx {gesture_idx}")
        gesture_seq[0, gesture_idx, :, :] = hand_data  # Add new frame to the buffer (NO forward filling)
        gesture_idx = (gesture_idx + 1) % FRAMES_PER_SEQ
    elif (gesture_idx == FRAMES_PER_SEQ - 1 and reached_first_still):
        # Fill the buffer with the current hand 
        print("Gesture sequence full")
        gesture_seq[0, gesture_idx:, :, :] = hand_data  # Add new frame to the buffer
        gesture_label = predict_gesture(gesture_seq)
        gesture_idx = (gesture_idx + 1) % FRAMES_PER_SEQ

        # OR

        # # Shift the buffer 1 index to the left and add current hand points to the last position in buffer
        # gesture_seq = np.roll(gesture_seq, shift=-1, axis=1)    # Shift buffer
        # gesture_seq[0, -1, :] = hand_data                       # Add new frame to the end of the buffer

    return gesture_seq

# Function to predict gesture based on the current buffers
def predict_gesture(gesture_seq):
    """
    Predicts the gesture using the LSTM model.
    Returns a string: "Zoom In" or "Zoom Out" based on the predicted class.

    NOTE: This function assumes gesture_seq is full
    """
    # global gesture_label
    global buffer_counter

    if (LOGGING):
        np.save(f"data_collection/data/gesture_{buffer_counter:03d}.npy", gesture_seq)
        buffer_counter += 1
        
    output_prob = lstm_model.predict(gesture_seq, verbose=0)

    # Try only performing prediction on full sequence - Did not seem to work
    # if (gesture_idx == 9):
    #     print("Full seq")

    #     if (LOGGING):
    #         np.save(f"data_collection/data/gesture_{buffer_counter:03d}.npy", gesture_seq)
    #         buffer_counter += 1
            
    #     output_prob = lstm_model.predict(gesture_seq, verbose=0)
    # else:
    #     if (gesture_idx == 0):
    #         print("New seq")
    #     return "N/A"

    print(f"Output probabilities: {output_prob}")
    return "Zoom In" if output_prob[0, 0] > output_prob[0, 1] else "Zoom Out"

# Custom drawing function
def draw_results(image, detection_result, gesture_label):
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
        for i, hand_landmarks in enumerate(detection_result.multi_hand_landmarks):
            mp.solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

            # Get handedness
            handedness = detection_result.multi_handedness[i]

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
    
    # Display the predicted gesture label in the top-right corner of the image
    cv2.putText(image, f"Gesture: {gesture_label}",
                (RESIZE_W - 300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, GREEN, 2, cv2.LINE_AA)

    return image

# Open the webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_H)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

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

        # I have an idea
        # Perhaps the motion of the hand entering the frame is confusing the LSTM
        # Using the "still" flag, wait until first "still" state has been reached then start collecting

        # If hands are detected, process landmarks
        current_time = time.time()
        if results.multi_hand_landmarks and (current_time - last_save_time >= FRAME_DELAY):
            #print("grabbed coordinates")
            last_save_time = current_time
            gesture_seq = update_buffers(gesture_seq, results.multi_hand_landmarks[0])
            
            # Check if hand is still or not
            #gesture_label = predict_gesture(gesture_seq)
            if (is_still()):
                gesture_label = "Still"
                if (not reached_first_still):
                    #print("Reached first still hand")
                    reached_first_still = True
        elif (not results.multi_hand_landmarks):
            #print("no hands detected")
            gesture_seq = np.zeros((1, 10, 21, 3))      # Clear the gesture sequence buffer
            gesture_label = "N/A"                       # Remove the gesture label
            reached_first_still = False                 # Reset reached last first still flag
            gesture_idx = 0                             # Reset gesture_idx


        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed = draw_results(image, results, gesture_label)

        # Draw FPS
        processed = cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

        # Show processed frame
        cv2.imshow('Processed MediaPipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
