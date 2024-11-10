# hand_path_visualizer.py
# CS 5100 - Fall 2024
# Final Project: Dynamic Hand Gesture Recognition for Remote Computer Control
#
# This program is a handy tool to visualize the 21 key points for each frame in a gesture sequence

# Import numpy for data loading
import numpy as np

# Import numpy for data loading
import matplotlib.pyplot as plt

# Define colors for plotting [RGB]
PINKY_BLUE    = (21, 101, 192)
RING_GREEN    = (48, 255, 48)
MIDDLE_YELLOW = (255, 204, 0)
INDEX_PURPLE  = (128, 64, 128)
THUMB_TAN     = (255, 219, 180)
PALM_RED      = (255, 48, 48)

# Load in desired image data array
datetime = '20241110_1622'
img_filename = f'img_data_{datetime}'
x_data_filename = f'X_data_{datetime}'
seq_id   = 0
img_data = np.load(f"data/{img_filename}.npy")
X_data   = np.load(f"data/{x_data_filename}.npy")
print(f"Gesture frame size: {img_data.shape[1:]}")
print(f"X data size: {X_data.shape}")

# Load the data from the data files into the correct format for plotting
gesture_coord_data = X_data[seq_id]

# Create blank 3D figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
#ax.plot(x_coords, y_coords, z_coords, c=colors)
#ax.view_init(elev=0, azim=0, roll=0)
ax.view_init(elev=-90, azim=-90, roll=0)
# src.set_color(colors)
title = ax.set_title('3D Hand Coordinates')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Get the lines for each key point (21 lines, 10 points each)
for i in range(X_data.shape[1]):
    # Get line color
    if (i in [18, 19, 20]):
        # Pinky
        c = tuple(np.array(PINKY_BLUE) / 255)
    elif (i in [14, 15, 16]):
        # Ring finger
        c = tuple(np.array(RING_GREEN) / 255)
    elif (i in [10, 11, 12]):
        # Middle finger
        c = tuple(np.array(MIDDLE_YELLOW) / 255)
    elif (i in [6, 7, 8]):
        # Index finer
        c = tuple(np.array(INDEX_PURPLE) / 255)
    elif (i in [1, 2, 3, 4]):
        # Thumb 
        c = tuple(np.array(THUMB_TAN) / 255)
    else:
        # Palm
        c = tuple(np.array(PALM_RED) / 255)

    # Get correct line data
    x_coords = gesture_coord_data[i, 0, :]
    y_coords = gesture_coord_data[i, 1, :]
    z_coords = gesture_coord_data[i, 2, :]

    # Plot the line
    ax.plot(x_coords, y_coords, z_coords, label=f'keypt {i}', color=c)


# Show the figure
plt.show()