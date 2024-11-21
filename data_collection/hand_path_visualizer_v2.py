# hand_path_visualizer_v2.py
# CS 5100 - Fall 2024
# Final Project: Dynamic Hand Gesture Recognition for Remote Computer Control
#
# This program is a handy tool to visualize the 21 key points for each frame in a gesture sequence
# v2 update: Rather than a single 3D plot, plot the XY coordinates in a gif

# Import numpy for data loading
import numpy as np

# Import numpy for data loading
import matplotlib.pyplot as plt

# Import mediapipe for drawing hands
import mediapipe as mp

# Import opencv for drawing on frames
from PIL import Image
import cv2

# Import os for file handling
import os

# Define colors for plotting [RGB]
PINKY_BLUE    = (21, 101, 192)
RING_GREEN    = (48, 255, 48)
MIDDLE_YELLOW = (255, 204, 0)
INDEX_PURPLE  = (128, 64, 128)
THUMB_TAN     = (255, 219, 180)
PALM_RED      = (255, 48, 48)
BLACK         = (0, 0, 0)


# Load in desired image data array
x_data_filename = 'X_data_merged_v2'
seq_id   = 104
X_data   = np.load(f"data_full/{x_data_filename}.npy")

# Transpose data if necessary
#X_data = np.transpose(X_data, (0, 2, 3, 1))         # ONLY do this for buffer type
print(f"X data size: {X_data.shape}")

# Load the data from the data files into the correct format for plotting
gesture_coord_data = X_data[seq_id]

# Create background white frames
IMG_H = 720
IMG_W = 1280
FRAMES_PER_SEQ = 10
frames = np.ones((FRAMES_PER_SEQ, IMG_H, IMG_W, 3)) * 255

# Define helper function for drawing hand
def draw_hand_in_frame(frame, frame_num, coord_data):
    # First draw Frame number in bottom of image
    cv2.putText(frame, f"Frame {frame_num:02d}", (15, IMG_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLACK, 2, cv2.LINE_AA)

    # Then draw all points
    radius = 5
    thickness = -1                          # To fill the circle
    for i in range(coord_data.shape[0]):
        unnorm_coord = (int(coord_data[i, 0] * IMG_W), int(coord_data[i, 1] * IMG_H))
        if (i in [18, 19, 20]):
            # Pinky
            cv2.circle(frame, unnorm_coord, radius, PINKY_BLUE, thickness)
        elif (i in [14, 15, 16]):
            # Ring finger
            cv2.circle(frame, unnorm_coord, radius, RING_GREEN, thickness)
        elif (i in [10, 11, 12]):
            # Middle finger
            cv2.circle(frame, unnorm_coord, radius, MIDDLE_YELLOW, thickness)
        elif (i in [6, 7, 8]):
            # Index finer
            cv2.circle(frame, unnorm_coord, radius, INDEX_PURPLE, thickness)
        elif (i in [1, 2, 3, 4]):
            # Thumb 
            cv2.circle(frame, unnorm_coord, radius, THUMB_TAN, thickness)
        else:
            # Palm
            cv2.circle(frame, unnorm_coord, radius, PALM_RED, thickness)

    # Now draw the lines
    line_pairs = [[0, 1], [1, 2], [2, 3], [3, 4],
                  [0, 5], [5, 6], [6, 7], [7, 8],
                  [5, 9], [9, 10], [10, 11], [11, 12],
                  [9, 13], [13, 14], [14, 15], [15, 16],
                  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]]
    for s_pt, e_pt in line_pairs:
        s_x = int(coord_data[s_pt, 0] * IMG_W)
        s_y = int(coord_data[s_pt, 1] * IMG_H)
        e_x = int(coord_data[e_pt, 0] * IMG_W)
        e_y = int(coord_data[e_pt, 1] * IMG_H)

        # Pick color of line depending on whether end point z > start point z and vice verse
        # TODO: Make this based on palm points
        if (coord_data[e_pt, 2] > coord_data[s_pt, 2]):
            # Line moving forwards in z
            cv2.line(frame, (s_x, s_y), (e_x, e_y), PALM_RED, 1)
        else: 
            # Line moving backwards in z
            cv2.line(frame, (s_x, s_y), (e_x, e_y), BLACK, 1)

    return frame

# Plot hand coordinates on each frame
for i in range(FRAMES_PER_SEQ):
    coord_data = gesture_coord_data[:, :, i]
    frames[i, :, :, :] = draw_hand_in_frame(frames[i, :, :, :], i, coord_data)

# Convert frames to image object
print(f"Image shape: {frames[0, :, :, :].shape}")
frames_img = [Image.fromarray(frames[i, :, :, :].astype('uint8')) for i in range(FRAMES_PER_SEQ)]
print(f"Frame length: {len(frames_img)}")

# Now save to GIF
os.makedirs("gifs_pts/", exist_ok=True)
output_path = f"gifs_pts/{x_data_filename}_seq_{seq_id}.gif"
frames_img[0].save(output_path,
               save_all=True,
               append_images=frames_img[1:],
               duration=(3 / FRAMES_PER_SEQ)*500,          # Make these parameters later
               loop=0,
               format="GIF")
print(f"Saved GIF to {output_path}")