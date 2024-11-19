# gesture_recognition_model.py
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
    #X_data = np.load('data_collection/data/x_data_jacob_gesture_2_3_combined.npy')
    X_data = np.load('data_collection/data_full/X_data_merged_jk_mb_yw.npy')
    # sub_images = np.load('data_collection\data\img_data_20241030_0130.npy')
    #Y_data = np.load('data_collection/data/y_data_jacob_gesture_2_3_combined.npy')
    Y_data = np.load('data_collection/data_full/Y_data_merged_jk_mb_yw.npy')
    #print(f"Y_data: \n{Y_data}")

    # Make sure to only use the ZoomIn and ZoomOut columns in the dataset
    Y_data = Y_data[:, 2:4]

    #n_seq, N_LANDMARKS, N_COORDS, FRAMES_PER_SEQ = X_data.shape
    #X_data_reshaped = X_data.reshape(n_seq, FRAMES_PER_SEQ, N_LANDMARKS * N_COORDS)
    X_data_reshaped = np.transpose(X_data, (0, 3, 1, 2))
    
    # resizing image for Inception V3
    # Make the array writable by copying it
    # sub_images_resized = np.array([tf.image.resize(img, (299, 299)).numpy() for img in sub_images.reshape(-1, 128, 128, 3)])
    # sub_images_resized = sub_images_resized.copy()  # Explicitly make the array writable

    # extract Inception V3 features
    # inception_features = np.array([extract_inception_features(img) for img in sub_images_resized])
    # inception_features = inception_features.reshape(X_data.shape[0], X_data.shape[1], -1)
    
    # merge key points and Inception features
    # combined_features = np.concatenate((X_data, inception_features), axis=-1)
    return X_data_reshaped, Y_data

#model building module
def build_model(input_shape, num_classes, load_model_pth=None):
    # LSTM 256 and Dense 128 (deeper model) yielded only 78% validation accuracy model
    if (not load_model_pth):
        inputs = Input(shape=input_shape)
        x = Reshape((10, 63))(inputs)
        x = LSTM(128, activation='tanh')(x)
        x = Dense(64, activation='sigmoid')(x)
        x = Dropout(0.4)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        #opt = Adam(learning_rate=0.0005, clipnorm=1.0)
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        # Load model from 
        print("Loading pre-trained model...")
        model = load_model(load_model_pth)

    print(model.summary())
    return model

# training and validation module
def train_and_validate_model(model, features, labels):
    batch_size = 4
    epochs = 80
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
    class_names = ["ScrollUp", "ScrollDown", "ZoomIn", "ZoomOut", "AppSwitchLeft", "AppSwitchRight"]
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names[2:4]))
    
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

# Model saving function
def save_model(model, filename="gesture_recognition_model.h5"):
    model.save(filename)
    print(f"Model saved to {filename}")

# main
if __name__ == "__main__":
    # loading and preprocessing data
    combined_features, Y_data = load_and_preprocess_data()
    
    # TODO: Make sure to set aside a shuffled train and test set for evaluation
    # n_total_obs = combined_features.shape[0]
    combined_features_train, combined_features_test, Y_data_train, Y_data_test = train_test_split(combined_features, Y_data, test_size=0.20)

    # shuffled_idxs = np.random.permutation(n_total_obs)
    # train
    # combined_features_train = combined_features[:int(shuffled_idxs]
    # combined_features_test = combined_features[shuffled_idxs:]
    # Y_data_train = Y_data[-test_idxs]
    # Y_data_test = Y_data[test_idxs]
    print(f"Input train shape: {combined_features_train.shape}\nInput test shape: {combined_features_test.shape}")
    print(f"Train Frequences: ZoomIn: {np.sum(Y_data_train[:, 0])}, ZoomOut: {np.sum(Y_data_train[:, 1])}")

    # constructing the model
    model = build_model(combined_features_train.shape[1:], Y_data_train.shape[1])
    #model = build_model(combined_features_train.shape[1:], Y_data_train.shape[1], load_model_pth="nn_weights/lstm_2class_20241115_test.h5")
    
    # training and validating the model
    train_and_validate_model(model, combined_features_train, Y_data_train)

    # Save model
    save_model(model, "nn_weights/lstm_2class_20241119_test.h5")
    
    # evaluating the model
    evaluate_model(model, combined_features_test, Y_data_test)
