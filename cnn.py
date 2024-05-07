import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os
import shutil
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filtered_train_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/fil_spectrogram_images/train'
filtered_test_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/fil_spectrogram_images/test'


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,

    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load and iterate training dataset
train_generator = train_datagen.flow_from_directory(
    filtered_train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    #Images are in RGB
    color_mode='rgb'
)

validation_generator = train_datagen.flow_from_directory(
    filtered_train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

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

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1, mode='min', restore_best_weights=True)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical_crossentropy for multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stopping]
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
majority_vote.to_csv('submission.csv', index=False)

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
report_df.to_csv('validation_classification_report.csv')

# Compute the confusion matrix and save it as well
conf_matrix = confusion_matrix(true_classes, predicted_classes)
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)

# Save the confusion matrix to a CSV file
conf_matrix_df.to_csv('validation_confusion_matrix.csv')

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusionmatrix.png')

