import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from google.cloud import storage, bigquery
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --------------------------------------
# Step 1: GCP Initialization and Data Fetching
# --------------------------------------

# GCP Variables
BUCKET_NAME = 'image-angle-bucket'  # Replace with your GCS bucket name
BIGQUERY_TABLE = 'angle-query-table'  # Replace with your BigQuery table
IMG_SIZE = (128, 128)

def initialize_gcp_clients():
    """
    Initialize GCP clients for Cloud Storage and BigQuery.
    """
    print("Initializing GCP clients...")
    storage_client = storage.Client()
    bigquery_client = bigquery.Client()
    return storage_client, bigquery_client

def fetch_images_from_gcs(storage_client, bucket_name):
    """
    Simulate fetching image data from GCS.
    """
    print(f"Fetching images from Cloud Storage bucket: {bucket_name}")
    # Simulate fetching image paths
    num_images = 100
    images = [np.random.rand(*IMG_SIZE, 3) for _ in range(num_images)]
    print(f"Fetched {len(images)} images (simulated).")
    return np.array(images)

def fetch_angles_from_bigquery(bigquery_client, table):
    """
    Simulate fetching angle data from BigQuery.
    """
    print(f"Fetching angles from BigQuery table: {table}")
    # Simulate fetching angles
    num_angles = 100
    angles = np.random.randint(30, 121, size=(num_angles, 2))  # Two angles per entry (x, y)
    print(f"Fetched {len(angles)} angle entries (simulated).")
    return np.array(angles)

# Data Loading Pipeline
def load_data_from_gcp():
    """
    Load data (images and angles) from simulated GCP services.
    """
    storage_client, bigquery_client = initialize_gcp_clients()
    images = fetch_images_from_gcs(storage_client, BUCKET_NAME)
    angles = fetch_angles_from_bigquery(bigquery_client, BIGQUERY_TABLE)
    return images, angles

# --------------------------------------
# Step 2: Preprocessing and Augmentation
# --------------------------------------

def preprocess_images(images):
    """
    Preprocess images for input to the neural network.
    """
    processed_images = tf.image.resize(images, IMG_SIZE)
    processed_images = processed_images / 255.0  # Normalize to [0, 1]
    return processed_images

def augment_images(images):
    """
    Augment image data for training.
    """
    augmented_images = tf.image.random_flip_left_right(images)
    augmented_images = tf.image.random_brightness(augmented_images, max_delta=0.1)
    augmented_images = tf.image.random_contrast(augmented_images, lower=0.8, upper=1.2)
    return augmented_images

# --------------------------------------
# Step 3: Model Definition
# --------------------------------------

def build_model():
    """
    Build a convolutional neural network for angle prediction.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    l2_reg = 0.001

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dense(2, activation='linear')  # Predict two angles: x and y
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    model.summary()
    return model

# --------------------------------------
# Step 4: Training and Evaluation
# --------------------------------------

def train_and_evaluate(images, angles):
    """
    Train and evaluate the model.
    """
    # Preprocess and split data
    images = preprocess_images(images)
    X_train, X_temp, y_train, y_temp = train_test_split(images, angles, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build the model
    model = build_model()

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        verbose=1
    )

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Save the model
    model.save('gcp_angle_predictor.h5')
    print("Model saved as 'gcp_angle_predictor.h5'.")

    return model, history

# --------------------------------------
# Step 5: Main Execution
# --------------------------------------

if __name__ == "__main__":
    # Load data from GCP
    images, angles = load_data_from_gcp()

    # Train and evaluate the model
    model, history = train_and_evaluate(images, angles)

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
