from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

filtered_train_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/fil_spectrogram_images/train'
filtered_test_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/fil_spectrogram_images/test'

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
    epochs=1,  # Adjust number of epochs as needed
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
ids = [f.split('/')[-1].replace('.png', '.au') for f in filenames]
results = pd.DataFrame({"id": ids, "class": predicted_labels})

# Save the results to a CSV file for submission
results.to_csv('transfer_submission.csv', index=False)

# Reset the validation generator before making predictions
validation_generator.reset()
predictions = model.predict(validation_generator, steps=len(validation_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Extract filenames and base ids for majority voting (if necessary)
filenames = validation_generator.filenames
base_ids = [filename.split('_')[0] for filename in filenames]

# Map predicted classes to their respective files
results = pd.DataFrame({'base_id': base_ids, 'predicted_class': predicted_classes})

# Apply majority voting
majority_vote = results.groupby('base_id')['predicted_class'].agg(lambda x: x.mode()[0]).reset_index()

# Handling true labels
true_labels = [validation_generator.classes[i] for i in range(len(validation_generator.classes))]
labels_df = pd.DataFrame({'base_id': base_ids, 'true_class': true_labels})
majority_true_labels = labels_df.groupby('base_id')['true_class'].agg(lambda x: x.mode()[0]).reset_index()

# Compute the confusion matrix
conf_matrix = confusion_matrix(majority_true_labels['true_class'], majority_vote['predicted_class'])
class_labels = list(validation_generator.class_indices.keys())

# Save the confusion matrix to a CSV file
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
conf_matrix_df.to_csv('transfer_validation_confusion_matrix.csv')

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('transfer_confusion_matrix.png')

# Calculate the classification report
report = classification_report(majority_true_labels['true_class'], majority_vote['predicted_class'], target_names=class_labels, zero_division=0)

print(report)