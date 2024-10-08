# mediapipe_test_webcam.py
# CS 5100 - Fall 2024
# Final Project: Testing Hand Landmark Detection with webcam feed
#
# v3 Update: Rather than using HandLandmarkerOptions, just use mp.solutions.hands.Hands

# Import OpenCV library
import cv2

# Import mediapipe dependencies
import mediapipe as mp
import numpy as np
import time

# Helpful mediapipe shortcuts
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define opencv constants
RED   = (0, 0, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
FONT  = cv2.FONT_HERSHEY_COMPLEX_SMALL
RESIZE_W = 1280 
RESIZE_H = 720


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



    # hand_landmarks_list = detection_result.hand_landmarks
    # handedness_list = detection_result.handedness
    # annotated_image = np.copy(rgb_image)

    # # Loop through the detected hands to visualize.
    # print(f"Hands detected: {len(hand_landmarks_list)}")
    # for idx in range(len(hand_landmarks_list)):
    #     hand_landmarks = hand_landmarks_list[idx]
    #     handedness = handedness_list[idx]

    #     # Draw the hand landmarks.
    #     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #     hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
    #     solutions.drawing_utils.draw_landmarks(
    #         annotated_image,
    #         hand_landmarks_proto,
    #         solutions.hands.HAND_CONNECTIONS,
    #         solutions.drawing_styles.get_default_hand_landmarks_style(),
    #         solutions.drawing_styles.get_default_hand_connections_style())

    #     # Get the top left corner of the detected hand's bounding box.
    #     height, width, _ = annotated_image.shape
    #     x_coordinates = [landmark.x for landmark in hand_landmarks]
    #     y_coordinates = [landmark.y for landmark in hand_landmarks]
    #     text_x = int(min(x_coordinates) * width)
    #     text_y = int(min(y_coordinates) * height) - 10

    #     # Draw handedness (left or right hand) on the image.
    #     cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #                 1, GREEN, 1, cv2.LINE_AA)

    # return annotated_image


# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_H)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed = draw_landmarks_on_image(image, results)
            
        cv2.imshow('Processed MediaPipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()