# gesture_data_struct_test.py
# CS 5100 - Fall 2024
# Final Project: Testing Hand Landmark Detection with webcam feed
#
# This script test model inference when doing repeated features for empty data
#
# Test with data from gesture buffer

# Import numpy for data processing
import numpy as np

# Import load_model function
from tensorflow.keras.models import load_model

# Import matplotlib for probability visualization
import matplotlib.pyplot as plt

# Load a sample 
gesture_num = 'gesture_005'
x_data = np.load(f'data_collection/data/{gesture_num}.npy')         # Buffer only contains ONE gesture
#y_data = np.array([[0, 0, 1, 0, 0, 0]])
#idx = np.where(y_data[:, 2] == 1)[0][0]
idx = 0
gesture = x_data[idx, :, :, :]
# y_label = y_data[idx, :]
# print(f"Y true: {y_label}")
print(f"Gesture shape: {gesture.shape}")

# # Run the model with 
# buffer = np.zeros((1, 10, 63))

# Load model
lstm = load_model('nn_weights/lstm_2class_20241121_test.h5')
#print(lstm.summary())

# Regular full buffer test
output_prob = lstm.predict(np.expand_dims(gesture, 0))
print(f"Regular buffer test: {output_prob}")

# # Simple test
# # test_buffer = np.zeros((10,))
# # for i in range(10):
# #     test_buffer[i:] = i + 1
# #     print(f"Test buffer: {test_buffer}")

# # Run 10 various inferences
# all_probs = np.zeros((10, 2))
# for i in range(10):
#     # Places key points from frames [0, i] into buffer
#     buffer[0, i:, :] = gesture[i]
#     output_prob = lstm.predict(buffer)              # Inference time Jacob laptop: 23 ms (43 FPS)
#     all_probs[i, :] = output_prob[0]
#     print(f"output_prob for idxs 0-{i}, {i} repeated: {output_prob[0]}")


# # Create a plot of the changing probabilities
# plt.figure()
# plt.grid(True)
# plt.plot([i for i in range(10)], all_probs[:, 0], color='r', label='ZoomIn Prob')
# plt.plot([i for i in range(10)], all_probs[:, 1], color='b', label='ZoomOut Prob')
# plt.title('Class Probabilities with Changing Buffer - True class ZoomOut')
# plt.xlabel('Indices 0-i, and i repeated to fill all 10 entries')
# plt.ylabel('Class Probability')
# plt.legend()
# plt.show()

# # NOTES:
# # With zero input, model is biased and says 97% zoom out

