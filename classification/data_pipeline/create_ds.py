import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
import torch


def preprocess(audio, sr):
    mel_sgram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    dbs = librosa.power_to_db(mel_sgram, ref=np.max)
    norm = (dbs - dbs.min()) / (dbs.max() - dbs.min())
    return norm


def augment(audio):
    # TODO?: pitch shift, time stretch
    return




def encode_labels(label_list):
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(label_list)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    return integer_labels, label_mapping

# NOTE: probably faster to do this vectorized if it's super slow
def slice_audio(audio, n_slices=5):
    # returns a list of "slices" of the input audio
    slice_len = len(audio) // n_slices
    slices = []
    
    for i in range(n_slices):
        s = audio[i * slice_len:(i + 1) * slice_len] # get slice
        if len(s) < slice_len: s = np.pad(s, (0, max(0, slice_len - len(s))))
        slices.append(s)
    
    return slices

def load_data(path, sample_rate, samples_per_track):
    labels = []
    features = []

    for genre in os.listdir(path):
        genre_path = os.path.join(path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(genre_path, file)
                    try:
                        audio, _ = librosa.load(file_path, sr=sample_rate)
                        if len(audio) >= samples_per_track:
                            audio = audio[:samples_per_track]
                        else:
                            audio = np.pad(audio, (0, max(0, samples_per_track - len(audio))), "constant")
                        audio = preprocess(audio, sample_rate)
                        slices = slice_audio(audio)
                        features.extend(slices)
                        labels.extend([genre] * len(slices))
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    return np.array(features), np.array(labels)

def create_dataset(path, test_size=0.2, sample_rate=22050, duration=25):
    X, y = load_data(path, sample_rate, sample_rate * duration)

    # encode labels
    y, label_dict = encode_labels(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val, label_dict

