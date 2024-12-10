# train_2class_LSTM.py
# CS 5100 - Fall 2024
# Final Project: Training 2-class LSTM Only  (ZoomIn/ZoomOut gestures)
#
# This script performs the inference using the trained model 


import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Reshape, LSTM, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed (for evaluation purposes)
np.random.seed(123)

# data_loading_and_preprocessing
def load_and_preprocess_data():
    # loading data
    X_data = np.load('data_collection/data_full/X_data_merged_v2.npy')
    Y_data = np.load('data_collection/data_full/y_data_merged_v2.npy')
    
    # Make sure to only use the ZoomIn and ZoomOut columns in the dataset
    Y_data = Y_data[:, 2:4]

    # Reshape data for proper batching
    X_data_reshaped = np.transpose(X_data, (0, 3, 1, 2))
    
    return X_data_reshaped, Y_data

#model building module
def build_model(input_shape, num_classes, load_model_pth=None):
    if (not load_model_pth):
        inputs = Input(shape=input_shape)
        x = Reshape((10, 63))(inputs)
        x = LSTM(64, activation='tanh')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        # Load model from model path
        print("Loading pre-trained model...")
        model = load_model(load_model_pth)

    print(model.summary())
    return model

# training and validation module
def train_and_validate_model(model, features, labels):
    batch_size = 4
    epochs = 50
    history = model.fit(features, labels, batch_size=batch_size, epochs=epochs, validation_split=0.15)
    return history

# assessment module
def evaluate_model(model, features, labels):
    y_pred = model.predict(features)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(labels, axis=1)
    
    # confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    # classification report
    #class_names = ["ScrollUp", "ScrollDown", "ZoomIn", "ZoomOut", "AppSwitchLeft", "AppSwitchRight"]
    class_names = ["ZoomIn", "ZoomOut"]
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Create heapmap for confusion matrix
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names)
    #plt.title("2-Class LSTM Confusion Matrix: 36 Gestures")
    plt.xticks(rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true_classes, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Model saving function
def save_model(model, filename="gesture_recognition_model.h5"):
    model.save(filename)
    print(f"Model saved to {filename}")

# main
if __name__ == "__main__":
    # loading and preprocessing data
    combined_features, Y_data = load_and_preprocess_data()
    
    # Make sure to set aside a shuffled train and test set for evaluation
    combined_features_train, combined_features_test, Y_data_train, Y_data_test = train_test_split(combined_features, Y_data, test_size=0.20)
    print(f"Input train shape: {combined_features_train.shape}\nInput test shape: {combined_features_test.shape}")
    print(f"Train Frequences: {np.sum(Y_data_train, axis=0)}")

    # constructing the model
    model = build_model(combined_features_train.shape[1:], Y_data_train.shape[1])
    #model = build_model(combined_features_train.shape[1:], Y_data_train.shape[1], load_model_pth="nn_weights/lstm_2class_20241115_test.h5")
    
    # training and validating the model
    train_and_validate_model(model, combined_features_train, Y_data_train)

    # Save model
    save_model(model, "nn_weights/lstm_2class.h5")
    
    # evaluating the model
    evaluate_model(model, combined_features_test, Y_data_test)
