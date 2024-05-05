import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
import unicodedata

class AudioProcessor:
    def __init__(self, sample_rate=22050, frame_length=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

    def plot_zcr(self, title, zcr, output_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(zcr)
        plt.title(title)
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
            plt.close()

    def extract_and_plot_zcr(self, file_path, output_dir):
        # Extract genre from file_path
        genre = file_path.split(os.sep)[-2]

        audio, _ = librosa.load(file_path, sr=self.sample_rate)

        # Compute zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]

        # Create subdirectory for the genre if not exists
        genre_output_dir = os.path.join(output_dir, genre)
        os.makedirs(genre_output_dir, exist_ok=True)

        # Generate filename without non-ASCII characters
        filename = os.path.basename(file_path).replace('.au', '_zcr.png')
        filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('utf-8')

        # Plot zero crossing rate and save image
        zcr_output_path = os.path.join(genre_output_dir, filename)
        self.plot_zcr('Zero Crossing Rate', zcr, output_path=zcr_output_path)

audio_processor = AudioProcessor()

test_dir = 'sample_data/test'
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith('.au'):
            file_path = os.path.join(root, file)
            audio_processor.extract_and_plot_zcr(file_path, 'zcr_images/test')

# Process audio files in the train folder
train_dir = 'sample_data/train'
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.endswith('.au'):
            file_path = os.path.join(root, file)
            audio_processor.extract_and_plot_zcr(file_path, 'zcr_images/train')
