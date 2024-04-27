import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

class AudioProcessor:
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def plot_spectrogram(self, title, y, sr, hop_length, y_axis="linear", output_path=None):
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
        plt.colorbar(format="%+2.f")
        plt.title(title)
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
        plt.close()

    def extract_and_plot(self, file_path, output_dir):
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        stft_audio = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        y_audio = np.abs(stft_audio) ** 2
        
        # Plot linear spectrogram and save image
        linear_output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.au', '_linear.png'))
        self.plot_spectrogram('Linear Spectrogram', y_audio, self.sample_rate, self.hop_length, output_path=linear_output_path)

        # Plot logarithmic spectrogram and save image
        y_log_audio = librosa.power_to_db(y_audio)
        log_output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.au', '_log.png'))
        self.plot_spectrogram('Logarithmic Spectrogram', y_log_audio, self.sample_rate, self.hop_length, output_path=log_output_path)

        # Plot logarithmic spectrogram with logarithmic y-axis and save image
        log_y_output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.au', '_log_y.png'))
        self.plot_spectrogram('Logarithmic Spectrogram with Logarithmic Y-Axis', y_log_audio, self.sample_rate, self.hop_length, y_axis="log", output_path=log_y_output_path)

# Example usage:
audio_processor = AudioProcessor()
audio_processor.extract_and_plot('sample_data/test/test.00596.au', 'spectrogram_images')
