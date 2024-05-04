import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import shutil

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

filtered_train_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images/train'
filtered_test_dir = '/nfs/student/m/mpradhan007/PycharmProjects/neural_network/spectrogram_images/test'


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest', validation_split=0.1)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(filtered_train_dir, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training', color_mode='rgb')
validation_generator = train_datagen.flow_from_directory(filtered_train_dir, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation', color_mode='rgb')

activation_functions = ['relu', 'elu', 'leaky_relu']
results = []

for activation in activation_functions:
    for n_layers in [3, 4, 5]:
        print(f"Testing {activation} activation with {n_layers} additional layers.")
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation=activation, padding='same', input_shape=(128, 128, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        for _ in range(n_layers):
            model.add(layers.Conv2D(64, (3, 3), activation=activation, padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation=activation))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_generator, epochs=30, validation_data=validation_generator)
        val_accuracy = history.history['val_accuracy'][-1]  # Get the last validation accuracy
        results.append({'Activation': activation, 'Layers': n_layers + 2, 'Validation Accuracy': val_accuracy})
        print(f"Model with {activation} and {n_layers+2} layers: Validation Accuracy = {val_accuracy}")

# Optionally, save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('model_hyperparameters.csv', index=False)
print(df_results)
