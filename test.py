import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import shutil
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_filtered_directory(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for genre_dir in os.listdir(source_dir):
        genre_path = os.path.join(source_dir, genre_dir)
        if os.path.isdir(genre_path):
            target_genre_dir = os.path.join(target_dir, genre_dir)
            os.makedirs(target_genre_dir, exist_ok=True)
            for file in os.listdir(genre_path):
                if file.endswith('sr22050_nfft512_hop256_nmels64.png'):
                    src_path = os.path.join(genre_path, file)
                    target_path = os.path.join(target_genre_dir, file)
                    shutil.copy(src_path, target_path)
                    print(f"Copied {src_path} to {target_path}")



source_train_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images3/train'
source_test_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images3/test'

# Paths to the filtered directories
filtered_train_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/fil_spectrogram_images/train'
filtered_test_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/fil_spectrogram_images/test'

# Call the function for both train and test directories
create_filtered_directory(source_train_dir, filtered_train_dir)
create_filtered_directory(source_test_dir, filtered_test_dir)


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
#
# # Define the CNN Model
# model = models.Sequential([
#     layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(10, activation='softmax')  # Assuming 10 genres/classes
# ])

# Define the CNN Model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), padding='same', input_shape=(128, 128, 3)),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding='same'),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# model = models.Sequential([
#     layers.Conv2D(128, (3, 3), use_bias=False, input_shape=(128, 128, 3)),  # Removed activation here
#     layers.BatchNormalization(),
#     layers.Activation('relu'),  # Added separate activation layer
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Conv2D(64, (3, 3), use_bias=False),  # Removed activation here
#     layers.BatchNormalization(),
#     layers.Activation('relu'),  # Added separate activation layer
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Conv2D(32, (3, 3), use_bias=False),  # Removed activation here
#     layers.BatchNormalization(),
#     layers.Activation('relu'),  # Added separate activation layer
#     layers.Flatten(),
#
#     layers.Dense(32, activation='relu'),
#     layers.Dense(10, activation='softmax')  # Assuming 10 genres/classes
# ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical_crossentropy for multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator
)

test_generator = test_datagen.flow_from_directory(
    filtered_test_dir,
    target_size=(128, 128),
    batch_size=1,
    class_mode=None,
    shuffle=False,
    color_mode='rgb'
)

# Model prediction
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Class index mapping
class_indices = train_generator.class_indices
labels_dict = dict((v, k) for k, v in class_indices.items())
predicted_labels = [labels_dict[k] for k in predicted_classes]

# Extract filenames and format them correctly for matching and aggregation
filenames = test_generator.filenames
base_ids = [os.path.splitext(os.path.basename(f))[0].split('_')[0] + '.au' for f in filenames]


# Create DataFrame to hold results
results = pd.DataFrame({'id': base_ids, 'class': predicted_labels})
majority_vote = results.groupby('id')['class'].agg(lambda x: x.mode()[0]).reset_index()

# Save the final aggregated predictions to a CSV file for submission
majority_vote.to_csv('submission_sr22050_nfft512_hop256_nmels64.csv', index=False)

# Make predictions on the validation data
validation_generator.reset()  # Resetting generator to ensure the data isn't shuffled, preserving label order
predictions = model.predict(validation_generator, steps=len(validation_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Actual labels (from the generator)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)

# Convert the report to a DataFrame for easy CSV writing
report_df = pd.DataFrame(report).transpose()

# Save the classification report to a CSV file
report_df.to_csv('validation_classification_report_sr22050_nfft512_hop256_nmels64.csv')

# Compute the confusion matrix and save it as well
conf_matrix = confusion_matrix(true_classes, predicted_classes)
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)

# Save the confusion matrix to a CSV file
conf_matrix_df.to_csv('validation_confusion_matrix_sr22050_nfft512_hop256_nmels64.csv')

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusionmatrix_sr22050_nfft512_hop256_nmels64.png')