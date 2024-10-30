# data_collection_FULL.py
# CS 5100 - Fall 2024
# Final Project: Testing Hand Landmark Detection with webcam feed
#
# This program is based off of data_collection_key_pts.py, but extends the functionality 
# for saving full hand subimages and adding timing delays

# Import OpenCV library
import cv2

# Import mediapipe dependencies
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

# Import custom libraries
import sys
sys.path.append("..")
from cvfpscalc import CvFpsCalc

# Helpful mediapipe shortcuts
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Define opencv constants (BGR format)
RED      = (0, 0, 255)
RED_RGB  = (255, 0, 0)
WHITE    = (255, 255, 255)
GREEN    = (0, 255, 0)
BLACK    = (0, 0, 0)
FONT     = cv2.FONT_HERSHEY_COMPLEX_SMALL
RESIZE_W = 1280 
RESIZE_H = 720


# Define MediaPipe color palette in RGB-255 format (for use in 3D scatter plot
PINKY_BLUE    = (21, 101, 192)
RING_GREEN    = (48, 255, 48)
MIDDLE_YELLOW = (255, 204, 0)
INDEX_PURPLE  = (128, 64, 128)
THUMB_TAN     = (255, 219, 180)
PALM_RED      = (255, 48, 48)

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


# Custom drawing function
def draw_landmarks_on_image(image, detection_result):
    global last_save_time

    if detection_result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(detection_result.multi_hand_landmarks):
            imagePreLandMark = image.copy()
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
            buffer_px = 20
            bbox = calc_bounding_rect(image, hand_landmarks)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1 - buffer_px, y1 - buffer_px), (x2 + buffer_px, y2 + buffer_px), BLACK, 3)

            # Draw handedness (left or right hand) on the image.
            cv2.putText(image, f"{handedness.classification[0].label}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        1, RED, 1, cv2.LINE_AA)
            

    return image



# Set up some variables for collection state
seq_id         = 0
frame_id       = 0
is_collecting  = False
prev_hand      = None
last_save_time = 0
N_LANDMARKS    = 21
N_COORDS       = 3
N_GESTURES     = 6
GESTURE_TIME   = 3                                  # [s], time to collect all frames
FRAMES_PER_SEQ = 10
FRAME_DELAY    = GESTURE_TIME / FRAMES_PER_SEQ      # [s]
IMG_SIZE       = 128

def draw_logging_status(image, results):
    """
    Helper function to display data collection/logging status on frame

    NOTE: For now, capture each subsequent frame until `frames_per_seq` is reached 
    TODO: Add delay and specify collection time [S]
    """
    # Determine if a hand was detected from results variable 
    # ** NOTE: This way, if tracking is lost on the hand, the sequence logging will automatically stop
    # TODO: Make sure to also check the handedness of the detected hand (data collection assumes only ONE hand is performing the gesture - otherwise disable logging)
    global is_collecting
    global prev_hand
    global frame_id
    global gesture

    hand_detected = True if results.multi_hand_landmarks else False
    if (hand_detected):
        curr_hand = results.multi_handedness[0].classification[0].label
        #print(f"prev_hand: {prev_hand}, curr_hand: {curr_hand}")
        if (not is_collecting):
            prev_hand = curr_hand

    # Create is_collecting string
    if (is_collecting and hand_detected and curr_hand == prev_hand):
        is_collecting_str = "YES"
    else:
        is_collecting_str = "NO"
        is_collecting = False
        frame_id = 0


    # Define color
    is_collecting_str_color = GREEN if is_collecting else RED_RGB

    # Place text in upper-right corner of the frame
    image = cv2.putText(image, f"Collecting Data?: {is_collecting_str} (Seq {seq_id})", (RESIZE_W - 500, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, is_collecting_str_color, 2, cv2.LINE_AA)
    
    # Print current gesture right below it
    image = cv2.putText(image, f"Gesture: {gesture}", (RESIZE_W - 175, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, is_collecting_str_color, 2, cv2.LINE_AA)
    
    return image


def perform_logging(img_pre_landmark, results):
    # Place the 21 key frames into the proper frame_id index in X_data
    global frame_id
    global seq_id
    global is_collecting
    global last_save_time

    # First, Check if data collection is finished
    if (seq_id >= n_seq):
        return

    current_time = time.time()
    if is_collecting and (current_time - last_save_time >= FRAME_DELAY) and results.multi_hand_landmarks and frame_id < FRAMES_PER_SEQ:
        print(f"Saving seq {seq_id} frame {frame_id}")

        hand_landmarks = results.multi_hand_landmarks[0]
        x_coordinates  = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))]
        y_coordinates  = [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]
        z_coordinates  = [hand_landmarks.landmark[i].z for i in range(len(hand_landmarks.landmark))]
        full_xyz       = np.array([x_coordinates, y_coordinates, z_coordinates]).reshape(N_LANDMARKS, N_COORDS)

        # Save hand landmark point data
        X_data[seq_id, :, :, frame_id] = full_xyz
        if (np.sum(Y_data[seq_id]) < 1):
            # Make sure not to add additionals ones in one-hot vector when hand tracking is lost
            Y_data[seq_id, gesture] = 1

        # Save subimage hand data
        current_time = time.time()
        last_save_time = current_time
        buffer_px = 20
        bbox = calc_bounding_rect(img_pre_landmark, hand_landmarks)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1 - buffer_px, y1 - buffer_px), (x2 + buffer_px, y2 + buffer_px), BLACK, 3)
        hand_region = img_pre_landmark[y1:y2, x1:x2]
        resized_image = cv2.resize(hand_region, (IMG_SIZE, IMG_SIZE))
        img_data[seq_id, :, :, :, frame_id] = resized_image

        frame_id += 1
        last_save_time = time.time()
    elif (frame_id >= FRAMES_PER_SEQ):
        # Frame sequence for given observation have been filled, so disable data collection for the current observation
        is_collecting = False
        seq_id += 1
        frame_id = 0


# Define parameters 
n_seq = 2
gesture = 0                 # TODO: Allow this to be changed during run-time
gesture_map = {0: "ScrollUp", 1: "ScrollDown", 2: "ZoomIn", 3: "ZoomOut", 4: "AppSwitchLeft", 5: "AppSwitchRight"}


# Create training data
X_data    = np.zeros((n_seq, N_LANDMARKS, N_COORDS, FRAMES_PER_SEQ))
Y_data    = np.zeros((n_seq, N_GESTURES))
img_data  = np.zeros((n_seq, IMG_SIZE, IMG_SIZE, 3, FRAMES_PER_SEQ))


# Open the webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_H)


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
        image_pre_landmark = image.copy()

        # Draw FPS (TODO: Put in a draw_info function)
        processed = cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

        # Draw logging status text in upper right corner of frame
        processed = draw_logging_status(processed, results)
        perform_logging(image_pre_landmark, results)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed = draw_landmarks_on_image(image, results)
            
        # Showed processed frame
        cv2.imshow('Processed MediaPipe Feed', image)

        # Check if data collection is finished
        if (seq_id >= n_seq):
            print(f"All {n_seq} sequences have been collected! Exiting...")
            break

        # Check for space key to collect data
        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord(' '):
            is_collecting = not is_collecting

        # Check for W & S to chnage logged gesture
        if key_pressed == ord('w'):
            # Increment logged gesture
            gesture = min(gesture + 1, N_GESTURES - 1)

        if key_pressed == ord('s'):
            # Decement logged gesture 
            gesture = max(gesture - 1, 0)

        # Check for 'q' key to kill program
        if key_pressed & 0xFF == ord('q'):
            break


# Print data
print(f"X_data shape: {X_data.shape}")
print(f"Y_data shape: {Y_data.shape}")
print(f"Image data shape: {img_data.shape}")
#print(f"Y_data:\n{Y_data}")

# Save data to file
datetime_str = datetime.now().strftime('%Y%m%d_%H%M')  
np.save(f"data/X_data_{datetime_str}.npy", X_data)
np.save(f"data/y_data_{datetime_str}.npy", Y_data)
np.save(f"data/img_data_{datetime_str}.npy", img_data)

# Release camera and close all windows before quitting
cap.release()
cv2.destroyAllWindows()