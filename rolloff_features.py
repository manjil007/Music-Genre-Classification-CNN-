import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
import unicodedata

class AudioProcessor:
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, roll_percent=0.85):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.roll_percent = roll_percent

    def plot_spectral_rolloff(self, title, rolloff, output_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(rolloff)
        plt.title(title)
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
            plt.close()

    def extract_and_plot_spectral_rolloff(self, file_path, output_dir):
        # Extract genre from file_path
        genre = file_path.split(os.sep)[-2]

        audio, _ = librosa.load(file_path, sr=self.sample_rate)

        # Compute spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, roll_percent=self.roll_percent)[0]

        # Create subdirectory for the genre if not exists
        genre_output_dir = os.path.join(output_dir, genre)
        os.makedirs(genre_output_dir, exist_ok=True)

        # Generate filename without non-ASCII characters
        filename = os.path.basename(file_path).replace('.au', '_rolloff.png')
        filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('utf-8')

        # Plot spectral rolloff and save image
        rolloff_output_path = os.path.join(genre_output_dir, filename)
        self.plot_spectral_rolloff('Spectral Rolloff', rolloff, output_path=rolloff_output_path)

audio_processor = AudioProcessor()

test_dir = 'sample_data/test'
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith('.au'):
            file_path = os.path.join(root, file)
            audio_processor.extract_and_plot_spectral_rolloff(file_path, 'spectral_rolloff_images/test')

# Process audio files in the train folder
train_dir = 'sample_data/train'
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.endswith('.au'):
            file_path = os.path.join(root, file)
            audio_processor.extract_and_plot_spectral_rolloff(file_path, 'spectral_rolloff_images/train')
