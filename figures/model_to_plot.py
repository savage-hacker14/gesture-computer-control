# Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Paths to your models
model_6class_path = 'nn_weights/lstm_6class_20241127_test2.h5'
model_2class_path = 'nn_weights/lstm_2class_20241121_test.h5'

# Load the models
model_6class = load_model(model_6class_path)
model_2class = load_model(model_2class_path)

# Paths to save the PNG images
plot_6class_path = 'lstm_6class_model.png'
plot_2class_path = 'lstm_2class_model.png'

# Generate and save the model plots
plot_model(
    model_6class,
    to_file=plot_6class_path,
    show_shapes=True,  # Display the shapes of input and output tensors
    show_layer_names=True,  # Display layer names
    rankdir="TB",  # Top to Bottom layout
    dpi=200  # High resolution for the image
)

plot_model(
    model_2class,
    to_file=plot_2class_path,
    show_shapes=True,  # Display the shapes of input and output tensors
    show_layer_names=True,  # Display layer names
    rankdir="TB",  # Top to Bottom layout
    dpi=200  # High resolution for the image
)

print(f"Plots saved: {plot_6class_path}, {plot_2class_path}")