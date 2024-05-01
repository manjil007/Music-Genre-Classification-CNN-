from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

filtered_train_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images_filtered/train'
filtered_test_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images_filtered/test'

# Adjust your ImageDataGenerator to use preprocess_input from ResNet50
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Use preprocess_input instead of rescale
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Use preprocess_input for test data

# Adjust target size to match the input size expected by ResNet50
train_generator = train_datagen.flow_from_directory(
    filtered_train_dir,
    target_size=(224, 224),  # Adjusted to 224x224
    batch_size=32,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

validation_generator = train_datagen.flow_from_directory(
    filtered_train_dir,
    target_size=(224, 224),  # Adjusted to 224x224
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

# Load the ResNet50 base model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  # Adjust number of classes

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=30,  # Adjust number of epochs as needed
    validation_data=validation_generator
)

# Use the preprocess_input for the test data, which is specific to ResNet50
test_generator = test_datagen.flow_from_directory(
    filtered_test_dir,
    target_size=(224, 224),  # Adjusted to 224x224, same as training data
    batch_size=1,  # Set batch size to 1 to handle files individually
    class_mode=None,  # No labels are available
    shuffle=False,  # Keep data in same order as filenames
    color_mode='rgb'
)

# Predict the classes of the test set using the transfer learning model
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Map predicted class indices to class names
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())  # Reverse the indices and class names
predicted_labels = [labels[k] for k in predicted_classes]

# Extract filenames and convert them to the required format
filenames = test_generator.filenames
ids = [f.split('/')[-1].replace('_log.png', '.au') for f in filenames]
results = pd.DataFrame({"id": ids, "class": predicted_labels})

# Save the results to a CSV file for submission
results.to_csv('transfer_submission.csv', index=False)
