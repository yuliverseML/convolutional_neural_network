import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ==================================================
# Data Loading and Preprocessing
# ==================================================

# Path to the dataset folder
base_folder = "NEU-DET/train/images"

# List of defect classes
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# Load images and labels
images = []
labels = []

# Iterate through each class folder
for class_name in classes:
    class_path = os.path.join(base_folder, class_name)
    if os.path.isdir(class_path):
        # Load each image in the class folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                # Resize images to 64x64 pixels
                img = Image.open(img_path).resize((64, 64))
                images.append(np.array(img))  # Convert image to numpy array
                labels.append(class_name)  # Add corresponding label
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

# Check if data is loaded
if len(images) == 0:
    raise ValueError("No data loaded. Check the folder path and data structure.")

# Convert labels to numerical format
label_to_id = {label: idx for idx, label in enumerate(classes)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
labels = [label_to_id[label] for label in labels]

# Convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize pixel values to [0, 1]
images = images / 255.0

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))

# Print dataset statistics
print(f"Loaded {len(images)} images.")
print(f"Classes: {classes}")
print(f"Labels: {np.unique(labels, return_counts=True)}")

# ==================================================
# Data Augmentation
# ==================================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Apply random transformations to training data
datagen = ImageDataGenerator(
    rotation_range=20,  # Random rotation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Random shear
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill missing pixels
)

datagen.fit(X_train)  # Fit the data generator to the training data

# ==================================================
# Model Building
# ==================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # First convolutional layer
    MaxPooling2D((2, 2)),  # First max-pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D((2, 2)),  # Second max-pooling layer
    Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
    MaxPooling2D((2, 2)),  # Third max-pooling layer
    Flatten(),  # Flatten the output
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout for regularization
    Dense(6, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==================================================
# Model Training
# ==================================================

# Train the model using augmented data
batch_size = 32
epochs = 20

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),  # Use data augmentation
    steps_per_epoch=len(X_train) // batch_size,  # Steps per epoch
    epochs=epochs,  # Number of epochs
    validation_data=(X_test, y_test)  # Validation data
)

# ==================================================
# Model Evaluation
# ==================================================

from sklearn.metrics import classification_report, confusion_matrix

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
y_true = np.argmax(y_test, axis=1)  # Convert true labels to class labels

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=id_to_label.values()))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Plot training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ==================================================
# Model Saving and Loading
# ==================================================

# Save the trained model in .keras format
model.save('defect_classification_model.keras')

# Load the saved model
from tensorflow import keras

loaded_model = keras.models.load_model('defect_classification_model.keras')

# Compile the loaded model (if needed)
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the loaded model
loaded_model.evaluate(X_test, y_test)

# ==================================================
# Anomaly Detection
# ==================================================

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Build an autoencoder for anomaly detection
input_img = Input(shape=(64, 64, 3))  # Input layer

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, validation_data=(X_test, X_test))

# Detect anomalies using the autoencoder
reconstructed_imgs = autoencoder.predict(X_test)  # Reconstruct test images
reconstruction_error = np.mean(np.abs(X_test - reconstructed_imgs), axis=(1, 2, 3))  # Calculate reconstruction error

threshold = np.percentile(reconstruction_error, 95)  # Set anomaly threshold (95th percentile)
anomalies = reconstruction_error > threshold  # Identify anomalies
