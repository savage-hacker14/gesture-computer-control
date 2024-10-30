# gesture_to_gif.py
# CS 5100 - Fall 2024
# Final Project: Dynamic Hand Gesture Recognition for Remote Computer Control
#
# This program is a handy tool to save a gesture sequence of n images into an animated gif

# Import PIL package
from PIL import Image
import cv2

# Import numpy for data loading
import numpy as np

# Import os for file handling
import os
import time

# Load in desired image data array
filename = 'img_data_20241030_0130'
seq_id   = 1
img_data = np.load(f"data/{filename}.npy")
print(f"Gesture frame size: {img_data.shape[1:]}")

# # Plot each image
# n_frames = img_data.shape[4]
# # frame = img_data[seq_id, :, :, :, 0].astype('uint8')
# # print(frame.shape)
# # cv2.imshow("Frames", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# for i in range(n_frames):
#     print(i)
#     frame = img_data[seq_id, :, :, :, i].astype('uint8')
#     cv2.imshow("Frames", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#     #time.sleep(2)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

# Convert each frame to a PIL Image
n_frames = img_data.shape[4]
frames   = [Image.fromarray(img_data[seq_id, :, :, :, i].astype('uint8')) for i in range(n_frames)]

# Save frames as an animated GIF
os.makedirs("gifs/", exist_ok=True)
output_path = f"gifs/{filename}_seq_{seq_id}.gif"
frames[0].save(output_path,
               save_all=True,
               append_images=frames[1:],
               duration=(3 / n_frames)*500,          # Make these parameters later
               loop=0,
               format="GIF")
print(f"Saved GIF to {output_path}")