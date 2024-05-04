import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import shutil

def create_filtered_directory(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Handle potential subdirectories in the training folder
    if 'train' in os.path.basename(source_dir):
        # Handles training directory with subdirectories for genres
        for genre_dir in os.listdir(source_dir):
            genre_path = os.path.join(source_dir, genre_dir)
            if os.path.isdir(genre_path):
                target_genre_dir = os.path.join(target_dir, genre_dir)
                os.makedirs(target_genre_dir, exist_ok=True)  # Ensures target directory exists

                for file in os.listdir(genre_path):
                    if file.endswith('.png'):
                        src_path = os.path.join(genre_path, file)
                        target_path = os.path.join(target_genre_dir, file)
                        shutil.copy(src_path, target_path)  # Copy files instead of creating symbolic links
                        print(f"Copied {src_path} to {target_path}")
    else:
        # Handles test directory which is assumed to be flat
        for file in os.listdir(source_dir):
            if file.endswith('_log.png'):
                src_path = os.path.join(source_dir, file)
                target_path = os.path.join(target_dir, file)
                shutil.copy(src_path, target_path)  # Copy files
                print(f"Copied {src_path} to {target_path}")


filtered_train_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images/train'
filtered_test_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images/test'

# Create an instance of ImageDataGenerator for preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values
    rotation_range=20,  # Random rotation between 0 and 20 degrees
    #HAVE TO TRY IT USING SAME ROTATION FOR BOTH TRAIN AND TEST
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Zoom into images
    horizontal_flip=True,  # Enable random horizontal flips
    fill_mode='nearest',  # Fill in new pixels after a rotation or width/height shift
    validation_split=0.1  # Use 10% of the data for validation
)

test_datagen = ImageDataGenerator(rescale=1. / 255)  # Only rescaling for test data

# Load and iterate training dataset
train_generator = train_datagen.flow_from_directory(
    filtered_train_dir,
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    subset='training',  # Specify this is training data
    color_mode='rgb'  # Images are in RGB
)

validation_generator = train_datagen.flow_from_directory(
    filtered_train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',  # Specify this is validation data
    color_mode='rgb'
)

# Define the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (4, 4), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (4, 4), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (4, 4), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Assuming 10 genres/classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical_crossentropy for multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

# Modify the test generator for prediction
test_generator = test_datagen.flow_from_directory(
    filtered_test_dir,
    target_size=(128, 128),
    batch_size=1,  # Set batch size to 1 to handle files individually
    class_mode=None,  # No labels are available
    shuffle=False,  # Keep data in same order as filenames
    color_mode='rgb'
)

# Predict the classes of the test set
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Map predicted class indices to class names
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())  # Reverse the indices and class names
predicted_labels = [labels[k] for k in predicted_classes]

# Extract filenames and convert to the required format
filenames = test_generator.filenames
ids = [f.split('/')[-1].replace('.png', '.au') for f in filenames]
results = pd.DataFrame({"id": ids, "class": predicted_labels})

# Save the results to a CSV file for submission
results.to_csv('submission.csv', index=False)
