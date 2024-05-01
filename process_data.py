import os
import numpy as np
import librosa
import pandas as pd


# def extract_features(file_path, n_mfcc=13, n_fft=1024, hop_length=512, n_mels=40):
#     """
#     Extracts MFCCs and their derivatives from an audio file, according to the parameters suggested in the paper.
#
#     Parameters:
#     - file_path: Path to the audio file.
#     - n_mfcc: Number of MFCCs to extract.
#     - n_fft: FFT window size.
#     - hop_length: Number of samples between successive frames.
#     - n_mels: Number of Mel bands.
#
#     Returns:
#     - features: Concatenated MFCCs, delta MFCCs, and delta-delta MFCCs.
#     """
#     audio, sample_rate = librosa.load(file_path, sr=None, res_type='kaiser_fast')
#     # Compute MFCCs
#     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#     # Compute delta MFCCs
#     delta_mfccs = librosa.feature.delta(mfccs)
#     # Compute delta-delta MFCCs
#     delta2_mfccs = librosa.feature.delta(mfccs, order=2)
#     # Average the MFCCs, delta MFCCs, and delta-delta MFCCs across the time dimension and concatenate
#     avg_mfccs = np.mean(mfccs, axis=1)
#     avg_delta_mfccs = np.mean(delta_mfccs, axis=1)
#     avg_delta2_mfccs = np.mean(delta2_mfccs, axis=1)
#     features = np.hstack((avg_mfccs, avg_delta_mfccs, avg_delta2_mfccs))
#
#     return features

def extract_features(file_path, n_mfcc=20):
    """
    Extracts features from an audio file.

    Parameters:
    - file_path: Path to the audio file.
    - n_mfcc: Number of MFCCs to extract.

    Returns:
    - avg_features: Averaged extracted features across different sampling rates.
    """
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    stft = np.abs(librosa.stft(audio))
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    energy = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    features = np.hstack((mfccs, chroma, spectral_contrast, zero_crossing_rate, energy))

    return features


def process_dataset_for_training(root_dir_train):
    features = []
    labels = []
    j = 0
    genres = os.listdir(root_dir_train)
    for genre in genres:
        genre_path = os.path.join(root_dir_train, genre)
        for filename in os.listdir(genre_path):
            if filename.endswith('.au'):
                file_path = os.path.join(genre_path, filename)
                try:
                    segment_features = (extract_features(file_path))
                    features.append(segment_features)
                    labels.append(genre)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    features = np.array(features)
    labels = np.array(labels)

    labels = labels.reshape(-1, 1)

    combined_data = np.hstack((features, labels))

    feature_columns = ['feature_' + str(i + 1) for i in range(len(features[1]))]
    df = pd.DataFrame(combined_data, columns=feature_columns + ['Genre'])

    name = 'train_features.csv'

    df.to_csv(name, index=False)


def process_dataset_for_testing(root_dir_test):
    """
    Processes the dataset for testing by extracting features from each audio file
    in the root directory.

    Parameters:
    - root_dir_test: Directory where the test audio files are located.
    - n_mfcc: Number of Mel-Frequency Cepstral Coefficients to extract.
    """
    features = []
    filenames = []

    for filename in os.listdir(root_dir_test):
        if filename.endswith('.au'):
            file_path = os.path.join(root_dir_test, filename)
            try:
                # Call the feature extraction function
                audio_features = extract_features(file_path)
                features.append(audio_features)
                filenames.append(filename)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    features = np.array(features)

    feature_columns = ['feature_' + str(i + 1) for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_columns)
    df['id'] = filenames

    name = 'test_features.csv'
    df.to_csv(name, index=False)


process_dataset_for_training('data/train')
process_dataset_for_testing('data/test')
