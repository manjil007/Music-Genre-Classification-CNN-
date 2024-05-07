import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

class AudioProcessor:
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128, segment_duration=3):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.segment_duration = segment_duration

    def create_spectrogram(self, y, sr, n_fft, hop_length, n_mels, image_file):
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Generate a Mel-spectrogram
        ms = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=sr/2)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel', ax=ax, fmax=sr/2)

        ax.axis('off')
        fig.savefig(image_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def process_audio_files(self, input_dir, output_dir, params):
        # Use parameters passed to the method
        sample_rate = params['sample_rate']
        n_fft = params['n_fft']
        hop_length = params['hop_length']
        n_mels = params['n_mels']

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.au'):
                    audio_file = os.path.join(root, file)
                    genre = os.path.basename(os.path.dirname(audio_file))
                    genre_output_dir = os.path.join(output_dir, genre)
                    os.makedirs(genre_output_dir, exist_ok=True)

                    # Load the entire audio file
                    y, sr = librosa.load(audio_file, sr=sample_rate)
                    duration = librosa.get_duration(y=y, sr=sr)

                    # Process each segment of the audio file
                    for start in np.arange(0, duration, self.segment_duration):
                        end = start + self.segment_duration
                        if end > duration:
                            break
                        y_segment = y[int(start * sr):int(end * sr)]
                        param_string = f"sr{sample_rate}_nfft{n_fft}_hop{hop_length}_nmels{n_mels}"
                        segment_filename = f"{os.path.splitext(file)[0]}_{int(start)}_{int(end)}_{param_string}.png"
                        segment_image_file = os.path.join(genre_output_dir, segment_filename)
                        self.create_spectrogram(y_segment, sr, n_fft, hop_length, n_mels, segment_image_file)

# Example usage with defined parameters
parameters = [
    {'sample_rate': 22050, 'n_fft': 512, 'hop_length': 256, 'n_mels': 64},
    {'sample_rate': 24000, 'n_fft': 1024, 'hop_length': 512, 'n_mels': 128},
    {'sample_rate': 22050, 'n_fft': 4096, 'hop_length': 1024, 'n_mels': 128},
    {'sample_rate': 24000, 'n_fft': 2048, 'hop_length': 256, 'n_mels': 128},
    {'sample_rate': 24000, 'n_fft': 512, 'hop_length': 256, 'n_mels': 128},
    {'sample_rate': 22050, 'n_fft': 1024, 'hop_length': 256, 'n_mels': 64}
]


for params in parameters:
    processor = AudioProcessor(**params)
    processor.process_audio_files('data/test', 'spectrogram_images3/test', params)
    processor.process_audio_files('data/train', 'spectrogram_images3/train', params)



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
#             audio_processor.extract_and_plot(file_path, 'spectrogram_images1/test')
#
#
# # # Process audio files in the train folder
# train_dir = 'data/train'
# for root, dirs, files in os.walk(train_dir):
#     for file in files:
#         if file.endswith('.au'):
#             file_path = os.path.join(root, file)
#             audio_processor.extract_and_plot(file_path, 'spectrogram_images1/train')

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
#
#
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
#         ms = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
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

#
# import os
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display
#
# class AudioProcessor:
#     def __init__(self, sample_rate=16000):
#         self.sample_rate = sample_rate
#
#     def create_spectrogram(self, audio_file, image_file):
#         fig = plt.figure(figsize=(12, 4))
#         ax = fig.add_subplot(1, 1, 1)
#         fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#
#         # Load the audio file with the adjusted sample rate
#         y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=20)  # Limiting to 20-second clips
#         y = librosa.util.normalize(y)  # Normalize amplitude to be between -1 and 1
#
#         # Generate a Mel-spectrogram with the parameters from the paper
#         ms = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=216, fmax=sr/2)
#         log_ms = librosa.power_to_db(ms, ref=np.max)
#         img = librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel', ax=ax, fmax=sr/2)
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
# audio_processor = AudioProcessor()
#
# test_dir = 'data/test'
# output_test_dir = 'spectrogram_images/test'
# audio_processor.process_audio_files(test_dir, output_test_dir)
#
# train_dir = 'data/train'
# output_train_dir = 'spectrogram_images/train'
# audio_processor.process_audio_files(train_dir, output_train_dir)


# import os
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display
#
# class AudioProcessor:
#     def __init__(self, sample_rate=16000):
#         self.sample_rate = sample_rate
#
#     def create_spectrogram(self, audio_file, image_file):
#         fig = plt.figure(figsize=(12, 4))
#         ax = fig.add_subplot(1, 1, 1)
#         fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#
#         # Load and normalize the audio file
#         y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=20)
#         y = librosa.util.normalize(y)
#
#         # Generate a Mel-spectrogram
#         ms = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=216, fmax=sr/2)
#         log_ms = librosa.power_to_db(ms, ref=np.max)
#         librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel', ax=ax, fmax=sr/2)
#
#         ax.axis('off')
#         fig.savefig(image_file, bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#     def process_audio_files(self, input_dir, output_dir):
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#
#         for root, dirs, files in os.walk(input_dir):
#             for file in files:
#                 if file.endswith('.au'):  # Check for the correct audio file extension
#                     audio_file = os.path.join(root, file)
#                     genre = os.path.basename(os.path.dirname(audio_file))
#                     genre_output_dir = os.path.join(output_dir, genre)
#                     os.makedirs(genre_output_dir, exist_ok=True)
#
#                     image_file = os.path.join(genre_output_dir, os.path.splitext(file)[0] + '.png')
#                     self.create_spectrogram(audio_file, image_file)
#
# audio_processor = AudioProcessor()
# audio_processor.process_audio_files('data/test', 'spectrogram_images/test')
# audio_processor.process_audio_files('data/train', 'spectrogram_images/train')


# import os
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display
#
# class AudioProcessor:
#     def __init__(self, sample_rate=16000):
#         self.sample_rate = sample_rate
#
#     def create_spectrogram(self, y, sr, image_file):
#         fig = plt.figure(figsize=(12, 4))
#         ax = fig.add_subplot(1, 1, 1)
#         fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#
#         # Generate a Mel-spectrogram
#         ms = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=216, fmax=sr/2)
#         log_ms = librosa.power_to_db(ms, ref=np.max)
#         librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel', ax=ax, fmax=sr/2)
#
#         ax.axis('off')
#         fig.savefig(image_file, bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#     def augment_and_process(self, y, sr, base_filename, output_dir, augment_type):
#         if augment_type == 'pitch':
#             # Pitch shifting by 2 steps
#             y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
#         elif augment_type == 'stretch':
#             # Time stretching by 10%
#             y = librosa.effects.time_stretch(y, rate=1.1)
#
#         image_file = os.path.join(output_dir, f"{base_filename}_{augment_type}.png")
#         self.create_spectrogram(y, sr, image_file)
#
#     def process_audio_files(self, input_dir, output_dir):
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#
#         for root, dirs, files in os.walk(input_dir):
#             for file in files:
#                 if file.endswith('.au'):
#                     audio_file = os.path.join(root, file)
#                     genre = os.path.basename(os.path.dirname(audio_file))
#                     genre_output_dir = os.path.join(output_dir, genre)
#                     os.makedirs(genre_output_dir, exist_ok=True)
#
#                     y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=20)
#                     y = librosa.util.normalize(y)
#
#                     # Save original spectrogram
#                     base_filename = os.path.splitext(file)[0]
#                     original_image_file = os.path.join(genre_output_dir, f"{base_filename}.png")
#                     self.create_spectrogram(y, sr, original_image_file)
#
#                     # Augment and save augmented spectrograms
#                     self.augment_and_process(y, sr, base_filename, genre_output_dir, 'pitch')
#                     self.augment_and_process(y, sr, base_filename, genre_output_dir, 'stretch')
#
# audio_processor = AudioProcessor()
# audio_processor.process_audio_files('data/test', 'spectrogram_images/test')
# audio_processor.process_audio_files('data/train', 'spectrogram_images/train')


