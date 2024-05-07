import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import cv2
import librosa.display
from scipy.signal import wiener
import unicodedata


class AudioProcessor:
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def plot_spectrogram(self, title, y, sr, hop_length, y_axis="linear", output_path=None):
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
        plt.colorbar(format="%+2.f dB")
        plt.title(title)
        if output_path:
            output_path_encoded = output_path.encode('utf-8')  # Encode filename to UTF-8
            with open(output_path_encoded, 'wb') as f:
                plt.savefig(f, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
            plt.close()


    def extract_and_plot(self, file_path, output_dir):
        # Extract genre from file_path
        genre = file_path.split(os.sep)[-2]

        audio, _ = librosa.load(file_path, sr=self.sample_rate)

        # image modifications here ####

        # Apply noise reduction using Wiener filter
        noise_reduced_audio = wiener(audio)

        # Apply contrast enhancement
        contrast_enhanced_audio = exposure.rescale_intensity(noise_reduced_audio, in_range='image', out_range=(0, 1))

        # Apply histogram equalization
        equalized_audio = exposure.equalize_hist(contrast_enhanced_audio)

        ###############################

        stft_audio = librosa.stft(equalized_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        y_audio = np.abs(stft_audio) ** 2
        
        # Create subdirectory for the genre if not exists
        genre_output_dir = os.path.join(output_dir, genre)
        os.makedirs(genre_output_dir, exist_ok=True)

        # Generate filename without non-ASCII characters
        filename = os.path.basename(file_path).replace('.au', '_log.png')
        filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('utf-8')


        # Plot logarithmic spectrogram and save image
        y_log_audio = librosa.power_to_db(y_audio)
        log_output_path = os.path.join(genre_output_dir, os.path.basename(file_path).replace('.au', '_log.png'))
        self.plot_spectrogram('Logarithmic Spectrogram', y_log_audio, self.sample_rate, self.hop_length, output_path=log_output_path)

audio_processor = AudioProcessor()

test_dir = 'sample_data/test'
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith('.au'):
            file_path = os.path.join(root, file)
            audio_processor.extract_and_plot(file_path, 'noise_contrast_hist/test')


# Process audio files in the train folder
train_dir = 'sample_data/train'
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.endswith('.au'):
            file_path = os.path.join(root, file)
            audio_processor.extract_and_plot(file_path, 'noise_contrast_hist/train')
