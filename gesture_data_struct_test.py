# gesture_data_struct_test.py
# CS 5100 - Fall 2024
# Final Project: Testing Hand Landmark Detection with webcam feed
#
# This script test model inference when doing repeated features for empty data
#
# FPS at 1280x720: XX
# FPS at 640x480: XX

# Import numpy for data processing
import numpy as np

# Import load_model function
from tensorflow.keras.models import load_model

# Import matplotlib for probability visualization
import matplotlib.pyplot as plt

# Load a sample observation
x_data = np.load('data_collection/data_full/X_data_merged_jk_mb_yw.npy')
y_data = np.load('data_collection/data_full/Y_data_merged_jk_mb_yw.npy')
idx = np.where(y_data[:, 3] == 1)[0][0]
gesture = x_data[idx, :, :, :]
y_label = y_data[idx, :]
print(f"Gesture shape: {gesture.shape}")
print(f"Y true: {y_label}")
# print(f"Gesture xyz: {gesture[0, :, :]}, {gesture[0, :, :].shape}")
# gesture = gesture.reshape(10, 21, 3)
gesture = gesture.reshape(10, 21 * 3)            # NOTE: Reshaping makes the 63-long row in the form x,x,x,...,y,y,y,...,z,z,z,... NOT x,y,z,x,y,z,x,y,z....
#print(f"New gesture xyz: {gesture[:, 0:6]}")

# Run the model with 
buffer = np.zeros((1, 10, 63))

# Load model
lstm = load_model('nn_weights/lstm_2class_20241114_better.h5')
#print(lstm.summary())

# Empty buffer
output_prob = lstm.predict(buffer)
print(f"Empty buffer test: {output_prob}")

# Simple test
# test_buffer = np.zeros((10,))
# for i in range(10):
#     test_buffer[i:] = i + 1
#     print(f"Test buffer: {test_buffer}")

# Run 10 various inferences
all_probs = np.zeros((10, 2))
for i in range(10):
    # Places key points from frames [0, i] into buffer
    buffer[0, i:, :] = gesture[i]
    output_prob = lstm.predict(buffer)              # Inference time Jacob laptop: 23 ms (43 FPS)
    all_probs[i, :] = output_prob[0]
    print(f"output_prob for idxs 0-{i}, {i} repeated: {output_prob[0]}")


# Create a plot of the changing probabilities
plt.figure()
plt.grid(True)
plt.plot([i for i in range(10)], all_probs[:, 0], color='r', label='ZoomIn Prob')
plt.plot([i for i in range(10)], all_probs[:, 1], color='b', label='ZoomOut Prob')
plt.title('Class Probabilities with Changing Buffer - True class ZoomOut')
plt.xlabel('Indices 0-i, and i repeated to fill all 10 entries')
plt.ylabel('Class Probability')
plt.legend()
plt.show()

# NOTES:
# With zero input, model is biased and says 97% zoom out

