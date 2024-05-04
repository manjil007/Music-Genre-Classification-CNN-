# import os
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
#
# class AudioProcessor:
#     def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
#         self.sample_rate = sample_rate
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.n_mels = n_mels
#
#     def plot_spectrogram(self, title, y, sr, hop_length, y_axis="linear", output_path=None):
#         plt.figure(figsize=(10, 6))
#         librosa.display.specshow(y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
#         plt.colorbar(format="%+2.f")
#         plt.title(title)
#         if output_path:
#             plt.savefig(output_path)
#         else:
#             plt.show()
#         plt.close()
#
#     def extract_and_plot(self, file_path, output_dir):
#         # Extract genre from file_path
#         genre = file_path.split(os.sep)[-2]
#
#         audio, _ = librosa.load(file_path, sr=self.sample_rate)
#         stft_audio = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
#         y_audio = np.abs(stft_audio) ** 2
#
#         # Create subdirectory for the genre if not exists
#         genre_output_dir = os.path.join(output_dir, genre)
#         os.makedirs(genre_output_dir, exist_ok=True)
#
#         # Plot linear spectrogram and save image
#         linear_output_path = os.path.join(genre_output_dir, os.path.basename(file_path).replace('.au', '_linear.png'))
#         self.plot_spectrogram('Linear Spectrogram', y_audio, self.sample_rate, self.hop_length, output_path=linear_output_path)
#
#         # Plot logarithmic spectrogram and save image
#         y_log_audio = librosa.power_to_db(y_audio)
#         log_output_path = os.path.join(genre_output_dir, os.path.basename(file_path).replace('.au', '_log.png'))
#         self.plot_spectrogram('Logarithmic Spectrogram', y_log_audio, self.sample_rate, self.hop_length, output_path=log_output_path)
#
#         # Plot logarithmic spectrogram with logarithmic y-axis and save image
#         log_y_output_path = os.path.join(genre_output_dir, os.path.basename(file_path).replace('.au', '_log_y.png'))
#         self.plot_spectrogram('Logarithmic Spectrogram with Logarithmic Y-Axis', y_log_audio, self.sample_rate, self.hop_length, y_axis="log", output_path=log_y_output_path)
#
# audio_processor = AudioProcessor()
#
# test_dir = 'data/test'
# for root, dirs, files in os.walk(test_dir):
#     for file in files:
#         if file.endswith('.au'):
#             file_path = os.path.join(root, file)
#             audio_processor.extract_and_plot(file_path, 'spectrogram_images/test')
#
#
# # # Process audio files in the train folder
# train_dir = 'data/train'
# for root, dirs, files in os.walk(train_dir):
#     for file in files:
#         if file.endswith('.au'):
#             file_path = os.path.join(root, file)
#             audio_processor.extract_and_plot(file_path, 'spectrogram_images/train')

# import os
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display

# class AudioProcessor:
#     def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
#         self.sample_rate = sample_rate
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.n_mels = n_mels
#
#     def plot_spectrogram(self, title, y, sr, hop_length, y_axis="linear", output_path=None):
#         plt.figure(figsize=(10, 6))
#         librosa.display.specshow(y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
#         plt.colorbar(format="%+2.f dB")
#         plt.title(title)
#         if output_path:
#             plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#             plt.close()
#         else:
#             plt.show()
#             plt.close()
#
#     def extract_and_plot(self, file_path, output_dir):
#         # Extract genre from file_path
#         if 'train' in file_path:
#             genre = file_path.split(os.sep)[-2]
#             genre_output_dir = os.path.join(output_dir, genre)
#         else:
#             genre_output_dir = os.path.join(output_dir, 'all_images')
#
#         os.makedirs(genre_output_dir, exist_ok=True)
#
#         audio, _ = librosa.load(file_path, sr=self.sample_rate)
#         stft_audio = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
#         y_audio = np.abs(stft_audio) ** 2
#
#         # Logarithmic spectrogram and save image
#         y_log_audio = librosa.power_to_db(y_audio)
#         log_output_path = os.path.join(genre_output_dir, os.path.basename(file_path).replace('.au', '_log.png'))
#         self.plot_spectrogram('Logarithmic Spectrogram', y_log_audio, self.sample_rate, self.hop_length, output_path=log_output_path)
#
# audio_processor = AudioProcessor()
#
# # Process test audio files
# test_dir = 'data/test'
# for root, dirs, files in os.walk(test_dir):
#     for file in files:
#         if file.endswith('.au'):
#             file_path = os.path.join(root, file)
#             audio_processor.extract_and_plot(file_path, 'spectrogram_images_filtered/test')
#
# # Process train audio files
# train_dir = 'data/train'
# for root, dirs, files in os.walk(train_dir):
#     for file in files:
#         if file.endswith('.au'):
#             file_path = os.path.join(root, file)
#             audio_processor.extract_and_plot(file_path, 'spectrogram_images_filtered/train')


# import os
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class AudioProcessor:
#     def __init__(self, sample_rate=22050):
#         self.sample_rate = sample_rate
#
#     def create_spectrogram(self, audio_file, image_file):
#         fig = plt.figure(figsize=(12, 4))
#         ax = fig.add_subplot(1, 1, 1)
#         fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#
#         y, sr = librosa.load(audio_file, sr=self.sample_rate)
#         ms = librosa.feature.melspectrogram(y=y, sr=sr)
#         log_ms = librosa.power_to_db(ms, ref=np.max)
#         librosa.display.specshow(log_ms, sr=sr, ax=ax)
#
#         ax.axis('off')
#         fig.savefig(image_file, bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#     def process_audio_files(self, input_dir, output_dir):
#         # Ensure output directory exists
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#
#         for root, dirs, files in os.walk(input_dir):
#             for file in files:
#                 if file.endswith('.au'):  # Adjust this as needed for '.wav' or other audio formats
#                     audio_file = os.path.join(root, file)
#                     genre = audio_file.split(os.sep)[-2]
#                     genre_output_dir = os.path.join(output_dir, genre)
#                     os.makedirs(genre_output_dir, exist_ok=True)
#
#                     image_file = os.path.join(genre_output_dir, file.replace('.au', '.png'))
#                     self.create_spectrogram(audio_file, image_file)
#
#
# audio_processor = AudioProcessor()
#
# test_dir = 'data/test'
# output_test_dir = 'spectrogram_images/test'
# audio_processor.process_audio_files(test_dir, output_test_dir)
#
# train_dir = 'data/train'
# output_train_dir = 'spectrogram_images/train'
# audio_processor.process_audio_files(train_dir, output_train_dir)


import os
import librosa
import numpy as np
import matplotlib.pyplot as plt


class AudioProcessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def create_spectrogram(self, audio_file, image_file):
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        y, sr = librosa.load(audio_file, sr=self.sample_rate)
        ms = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr, ax=ax)

        ax.axis('off')
        fig.savefig(image_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def process_audio_files(self, input_dir, output_dir):
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.au'):  # Adjust this as needed for '.wav' or other audio formats
                    audio_file = os.path.join(root, file)
                    genre = audio_file.split(os.sep)[-2]
                    genre_output_dir = os.path.join(output_dir, genre)
                    os.makedirs(genre_output_dir, exist_ok=True)

                    image_file = os.path.join(genre_output_dir, file.replace('.au', '.png'))
                    self.create_spectrogram(audio_file, image_file)


audio_processor = AudioProcessor()

test_dir = 'data/test'
output_test_dir = 'spectrogram_images/test'
audio_processor.process_audio_files(test_dir, output_test_dir)

train_dir = 'data/train'
output_train_dir = 'spectrogram_images/train'
audio_processor.process_audio_files(train_dir, output_train_dir)
