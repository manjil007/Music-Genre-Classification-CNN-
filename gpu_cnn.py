import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import os
import shutil


filtered_train_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images/train'
filtered_test_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images/test'


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def create_filtered_directory(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if 'train' in os.path.basename(source_dir):
        for genre_dir in os.listdir(source_dir):
            genre_path = os.path.join(source_dir, genre_dir)
            if os.path.isdir(genre_path):
                target_genre_dir = os.path.join(target_dir, genre_dir)
                os.makedirs(target_genre_dir, exist_ok=True)

                for file in os.listdir(genre_path):
                    if file.endswith('.png'):
                        src_path = os.path.join(genre_path, file)
                        target_path = os.path.join(target_genre_dir, file)
                        shutil.copy(src_path, target_path)
                        print(f"Copied {src_path} to {target_path}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    filtered_train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
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

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
early_stopping_cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[checkpoint_cb, early_stopping_cb]
)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

test_generator = test_datagen.flow_from_directory(
    filtered_test_dir,
    target_size=(128, 128),
    batch_size=1,
    class_mode=None,
    shuffle=False,
    color_mode='rgb'
)

predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predicted_labels = [labels[k] for k in predicted_classes]

filenames = test_generator.filenames
ids = [f.split('/')[-1].replace('.png', '.au') for f in filenames]
results = pd.DataFrame({"id": ids, "class": predicted_labels})

results.to_csv('submission.csv', index=False)
