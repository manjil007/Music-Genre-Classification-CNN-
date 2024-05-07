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

filtered_train_dir = '/Users/dev/NeuralNetworks/neural_network/fil_spectrogram_images/train'
filtered_test_dir = '/Users/dev/NeuralNetworks/neural_network/fil_spectrogram_images/test'

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
    epochs=10,  # Adjust number of epochs as needed
    validation_data=validation_generator
)

# Use the preprocess_input for the test data, which is specific to ResNet50
test_generator = test_datagen.flow_from_directory(
    filtered_test_dir,
    target_size=(224, 224),  # Adjusted to 224x224, same as training data
    batch_size=1,  # Set batch size to 1 to handle files individually
    class_mode=None,  # No labels are available
    shuffle=False,  # Keep data in the same order as filenames
    color_mode='rgb'
)

# Predict the classes of the test set using the transfer learning model
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Condense predictions by taking the most frequent class label every 10 rows
condensed_predictions = [np.bincount(predicted_classes[i:i+10]).argmax() for i in range(0, len(predicted_classes), 10)]

# Map predicted class indices to class names
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())  # Reverse the indices and class names
predicted_labels = [labels[k] for k in condensed_predictions]

# Extract filenames and convert them to the required format
filenames = test_generator.filenames[::10]  # Only taking every 10th filename
ids = [f.split('/')[-1].split('_')[0] + ".au" for f in filenames]  # Corrected filenames
results = pd.DataFrame({"id": ids, "class": predicted_labels})

# Save the condensed results to a CSV file for submission
results.to_csv('/Users/dev/NeuralNetworks/neural_network/Ncondensed_transfer_submission.csv', index=False)

# Reset the validation generator before making predictions
validation_generator.reset()
predictions = model.predict(validation_generator, steps=len(validation_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Condense predictions for validation data
condensed_predictions_val = [np.bincount(predicted_classes[i:i+10]).argmax() for i in range(0, len(predicted_classes), 10)]

# Extract filenames and base ids for majority voting (if necessary)
filenames = validation_generator.filenames[::10]  # Only taking every 10th filename
base_ids = [filename.split('/')[-1].split('_')[0] for filename in filenames]  # Corrected filenames

# Map predicted classes to their respective files
results_val = pd.DataFrame({'base_id': base_ids, 'predicted_class': condensed_predictions_val})


# Apply majority voting
majority_vote = results_val.groupby('base_id')['predicted_class'].agg(lambda x: x.mode()[0]).reset_index()

# Handling true labels
true_labels = [validation_generator.classes[i] for i in range(0, len(validation_generator.classes), 10)]
labels_df = pd.DataFrame({'base_id': base_ids, 'true_class': true_labels})
majority_true_labels = labels_df.groupby('base_id')['true_class'].agg(lambda x: x.mode()[0]).reset_index()

# Compute the confusion matrix
conf_matrix = confusion_matrix(majority_true_labels['true_class'], majority_vote['predicted_class'])
class_labels = list(validation_generator.class_indices.keys())

# Save the confusion matrix to a CSV file
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
conf_matrix_df.to_csv('/Users/dev/NeuralNetworks/neural_network/Ncondensed_transfer_validation_confusion_matrix.csv')

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('Ncondensed_transfer_confusion_matrix.png')

# Calculate the classification report
report = classification_report(majority_true_labels['true_class'], majority_vote['predicted_class'], target_names=class_labels, zero_division=0)

print(report)
