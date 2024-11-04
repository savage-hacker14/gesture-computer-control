# gesture_recognition_model.py
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, LSTM, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# feature extraction module
def extract_inception_features(img):
    inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    img = img_to_array(img)
    img = tf.image.resize(img, (299, 299))
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    features = inception_base.predict(img)
    return features

# data_loading_and_preprocessing
def load_and_preprocess_data():
    # loading data
    X_data = np.load('X_data.npy')
    sub_images = np.load('img_data.npy')
    Y_data = np.load('Y_data.npy')
    
    # resizing image for Inception V3
    sub_images_resized = np.array([tf.image.resize(img, (299, 299)) for img in sub_images.reshape(-1, 128, 128, 3)])
    
    # extract Inception V3 features
    inception_features = np.array([extract_inception_features(img) for img in sub_images_resized])
    inception_features = inception_features.reshape(X_data.shape[0], X_data.shape[1], -1)
    
    # merge key points and Inception features
    combined_features = np.concatenate((X_data, inception_features), axis=-1
    return combined_features, Y_data

#model building module
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = LSTM(128)(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# training and validation module
def train_and_validate_model(model, features, labels):
    batch_size = 32
    epochs = 20
    history = model.fit(features, labels, batch_size=batch_size, epochs=epochs, validation_split=0.2)
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
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # ROC  curve
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

# main
if __name__ == "__main__":
    # loading and preprocessing data
    combined_features, Y_data = load_and_preprocess_data()
    
    # constructing the model
    model = build_model(combined_features.shape[1:], Y_data.shape[1])
    
    # training and validating the model
    train_and_validate_model(model, combined_features, Y_data)
    
    # evaluating the model
    evaluate_model(model, combined_features, Y_data)
