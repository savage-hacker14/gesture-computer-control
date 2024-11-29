import numpy as np
from tensorflow.keras.models import load_model
import pyautogui
import cv2
import mediapipe as mp
import time
from inference_v2 import predict_gesture

# Load the trained model
model = load_model("nn_weights/lstm_2class_20241127_test2.h5")  

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture-action mapping
gesture_actions = {
    0: lambda: pyautogui.scroll(100),  # Scroll up
    1: lambda: pyautogui.scroll(-100),  # Scroll down
    2: lambda: pyautogui.hotkey('ctrl', '+'),  # Zoom in
    3: lambda: pyautogui.hotkey('ctrl', '-'),  # Zoom out
    4: lambda: pyautogui.hotkey('alt', 'left'),  # Navigate back
    5: lambda: pyautogui.hotkey('alt', 'right')  # Navigate forward
}

keypoint_buffer = []

# Real-time gesture recognition and action execution
def recognize_and_act():
    global keypoint_buffer
    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process frame
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Inside the real-time loop:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = preprocess_keypoints(hand_landmarks)
                gesture_id = predict_gesture(keypoints)
        
            # Perform corresponding action
            action = gesture_actions.get(gesture_id)
            if action:
                action()
                time.sleep(1)
        
        # Display frame
        cv2.imshow("Gesture Recognition", frame)

        # Exit with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the gesture-to-command control
if __name__ == "__main__":
    recognize_and_act()
