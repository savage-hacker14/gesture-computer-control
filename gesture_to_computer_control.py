import numpy as np
from tensorflow.keras.models import load_model
import pyautogui
import cv2
import mediapipe as mp
import time

# Load the trained model
model = load_model("gesture_classification_model.h5")  # Replace with your model's file name

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

# Real-time gesture recognition and action execution
def recognize_and_act():
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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract key points
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                keypoints = keypoints.reshape(1, -1)  # Prepare input for the model

                # Predict gesture
                prediction = model.predict(keypoints)
                gesture_id = np.argmax(prediction)

                # Perform corresponding action
                action = gesture_actions.get(gesture_id)
                if action:
                    action()
                    time.sleep(1)  # Add delay to avoid repeated actions

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
