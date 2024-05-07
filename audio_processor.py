import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


class AudioProcessor:
    def __init__(self, sample_rate=24000, segment_duration=3):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration

    def create_spectrogram(self, y, sr, image_file):
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        ms = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128, fmax=sr/2)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel', ax=ax, fmax=sr/2)

        ax.axis('off')
        fig.savefig(image_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def process_audio_files(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.au'):
                    audio_file = os.path.join(root, file)
                    genre = os.path.basename(os.path.dirname(audio_file))
                    genre_output_dir = os.path.join(output_dir, genre)
                    os.makedirs(genre_output_dir, exist_ok=True)

                    y, sr = librosa.load(audio_file, sr=self.sample_rate)
                    duration = librosa.get_duration(y=y, sr=sr)

                    # Process each segment of the audio file
                    for start in np.arange(0, duration, self.segment_duration):
                        end = start + self.segment_duration
                        if end > duration:
                            break
                        y_segment = y[int(start * sr):int(end * sr)]
                        segment_filename = f"{os.path.splitext(file)[0]}_{int(start)}_{int(end)}.png"
                        segment_image_file = os.path.join(genre_output_dir, segment_filename)
                        self.create_spectrogram(y_segment, sr, segment_image_file)


audio_processor = AudioProcessor()
audio_processor.process_audio_files('data/test', 'fil_spectrogram_images/test')
audio_processor.process_audio_files('data/train', 'fil_spectrogram_images/train')
