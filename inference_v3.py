# Import OpenCV library
import cv2

# Import mediapipe dependencies
import mediapipe as mp
import numpy as np
import time

# Import pyautogui
import pyautogui
import platform

# Import plotting libraries
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


# Detect the operating system
is_mac = platform.system() == "Darwin"
is_windows = platform.system() == "Windows"

# Define model parameters
GESTURE_TIME   = 3                                  # [s], time to collect all frames
FRAMES_PER_SEQ = 10
FRAME_DELAY    = GESTURE_TIME / FRAMES_PER_SEQ      # [s]
last_save_time = 0
buffer_idx     = 0
gesture_idx    = 0
still_thres    = 0.20                               # Normalized distances - May need to adjust depending on FRAME_DELAY
buffer_counter = 1
reached_first_still = False
LOGGING        = False                               # If logging is enabled, store each gesture to enumerated buffer .npy file in data_collection/data folder


gesture_map = {0: "ScrollUp", 1: "ScrollDown", 2: "ZoomIn", 3: "ZoomOut", 4: "AppSwitchLeft", 5: "AppSwitchRight"}
#flag to track if a gesture has already been predicted (this is for the freeze reset logic)
predicted = False
# Load LSTM model for gesture classification
lstm_model = load_model('nn_weights/lstm_6class_20241127_test2.h5')

# Define an array to store the hand key point sequence (gesture buffer)
buffer_seq    = np.zeros((1, 10, 21, 3))        # Continuously grabbing hand landmark points
gesture_seq   = np.zeros((1, 10, 21, 3))        # Gesture sequence that is fed to LSTM model (only begins to get filled once the first still hand frame has been reached)
gesture_label = ""  # Variable to store current gesture label

# Function to detect if hand is still
def is_still():
    global buffer_idx
    total_diff = np.sum(np.sqrt(np.sum((buffer_seq[0, buffer_idx - 1] - buffer_seq[0, buffer_idx - 2]) ** 2, axis=1)))
    print(f"Total diff for stillness: {total_diff}")
    return total_diff < still_thres


# Function to process landmarks into the required buffer format
def update_buffers(gesture_seq, landmarks):
    global buffer_idx
    global gesture_idx
    global gesture_label
    global reached_first_still
    global predicted

    hand_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    buffer_seq[0, buffer_idx, :, :] = hand_data
    buffer_idx = (buffer_idx + 1) % FRAMES_PER_SEQ       

    if reached_first_still and not predicted:
        print(f"Recording gesture index {gesture_idx}")
        gesture_seq = np.roll(gesture_seq, shift=-1, axis=1)
        gesture_seq[0, -1, :, :] = hand_data

        if np.all(gesture_seq[0, :, :, :] != 0):  # Buffer is full
            gesture_label = predict_gesture(gesture_seq)
            predicted = True  # Set flag to stop further predictions
            #print(f"Gesture predicted: {gesture_label}")
    return gesture_seq


# Function to predict gesture based on the current buffers
def predict_gesture(gesture_seq):
    global buffer_counter

    if LOGGING:
        np.save(f"data_collection/data/gesture_{buffer_counter:03d}.npy", gesture_seq)
        buffer_counter += 1

    output_prob = lstm_model.predict(gesture_seq, verbose=0)
    print(f"Output probabilities: {output_prob}")
    
    
    # output_label = "Zoom In" if output_prob[0, 0] > output_prob[0, 1] else "Zoom Out"
    # print(f"Predicted Gesture: {output_label}")
    # return output_label

    # Find the gesture with the highest probability
    predicted_class = np.argmax(output_prob)
    predicted_gesture = gesture_map.get(predicted_class, "Unknown")
    print(f"Predicted Gesture: {predicted_gesture}")
    return predicted_gesture


def action(predicted_gesture):
    """
    Recognizes gestures and maps them to computer commands.
    """
    print("Starting gesture recognition. Press 'q' to exit.")
    # Gesture-action mapping
    if is_mac:
        if (predicted_gesture == "ScrollUp"):
            pyautogui.scroll(300)
        elif (predicted_gesture == "ScrollDown"):
            pyautogui.scroll(-300)
        elif (predicted_gesture == "ZoomIn"):
            pyautogui.hotkey('command', '+')
        elif (predicted_gesture == "ZoomOut"):
            pyautogui.hotkey('command', '-')
        elif (predicted_gesture == "ScrollDown"):
            pyautogui.hotkey('command', '[')
        elif (predicted_gesture == "ScrollDown"):
            pyautogui.hotkey('command', ']')
    elif is_windows:
        if (predicted_gesture == "ScrollUp"):
            pyautogui.scroll(300)
        elif (predicted_gesture == "ScrollDown"):
            pyautogui.scroll(-300)
        elif (predicted_gesture == "ZoomIn"):
            pyautogui.hotkey('ctrl', '+')
        elif (predicted_gesture == "ZoomOut"):
            pyautogui.hotkey('ctrl', '-')
        elif (predicted_gesture == "AppSwitchLeft"):
            pyautogui.keyDown('alt')
            pyautogui.keyDown('tab')
            pyautogui.keyUp('tab')
            time.sleep(0.2)
            pyautogui.keyDown('left')
            pyautogui.keyUp('left')
            pyautogui.keyDown('left')
            pyautogui.keyUp('left')
            pyautogui.keyUp('alt')
        elif (predicted_gesture == "AppSwitchRight"):
            pyautogui.keyDown('alt')
            pyautogui.keyDown('tab')
            pyautogui.keyUp('tab')
            time.sleep(0.2)
            pyautogui.keyUp('alt')
    else:
        raise RuntimeError("Unsupported operating system. Only macOS and Windows are supported.")
    
    time.sleep(0.5)


# Function to reset everything once the hand becomes still
def reset_buffers():
    global gesture_seq, buffer_idx, gesture_idx, reached_first_still, predicted
    gesture_seq = np.zeros((1, 10, 21, 3))
    buffer_idx = 0
    gesture_idx = 0
    reached_first_still = False
    predicted = False
    #print("Buffers and flags reset. Ready for next gesture.")

# Custom drawing function
def draw_results(image, detection_result, gesture_label):
    def calc_bounding_rect(image, landmarks):
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

            handedness = detection_result.multi_handedness[i]
            height, width, _ = image.shape
            x_coordinates = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))]
            y_coordinates = [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - 15

            buffer_px = 10
            bbox = calc_bounding_rect(image, hand_landmarks)
            cv2.rectangle(image, (bbox[0] - buffer_px, bbox[1] - buffer_px), (bbox[2] + buffer_px, bbox[3] + buffer_px), BLACK, 3)
            cv2.putText(image, f"{handedness.classification[0].label}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        1, RED, 1, cv2.LINE_AA)

    cv2.putText(image, f"Gesture: {gesture_label}",
                (RESIZE_W - 300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, GREEN, 2, cv2.LINE_AA)
    return image

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_H)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Main loop for processing the webcam feed
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        fps = cvFpsCalc.get()
        image.flags.writeable = False
        image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
        results = hands.process(image)
        current_time = time.time()

        if results.multi_hand_landmarks and (current_time - last_save_time >= FRAME_DELAY):
            last_save_time = current_time
            gesture_seq = update_buffers(gesture_seq, results.multi_hand_landmarks[0])

            if is_still():
                if not reached_first_still:
                    reached_first_still = True
                    print("Hand is still. Starting gesture collection.")
                elif predicted:
                    predicted_gesture = predict_gesture(gesture_seq)
                    action(predicted_gesture)
                    reset_buffers()  # Reset buffers after performing action
            else:
                print("Hand is moving.")

        elif not results.multi_hand_landmarks:
            gesture_seq = np.roll(gesture_seq, shift=-1, axis=1)
            gesture_seq[0, -1, :, :] = np.zeros((21, 3))
            gesture_label = "N/A"
            reset_buffers()  # Reset everything if no hands are detected

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed = draw_results(image, results, gesture_label)
        processed = cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
        cv2.imshow('Processed MediaPipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
cap.release()
cv2.destroyAllWindows()