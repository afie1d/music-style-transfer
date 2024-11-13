import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import kagglehub
import librosa


def preprocess(audio):
    # TODO: extract mel-spectrograms 
    return


def augment(audio):
    # TODO?: pitch shift, time stretch
    return

def slice(audio, duration):
    # TODO: implement
    return

def load_data(path, sample_rate, samples_per_track, slice_duration):
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
                        audio = preprocess(audio)
                        slices = slice(audio, slice_duration)
                        features.extend(slices)
                        labels.extend([genre] * len(slices))
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    return np.array(features), np.array(labels)

def create_dataset(path, test_size=0.2, sample_rate=22050, duration=30):
    X, y = load_data(path, sample_rate, sample_rate * duration)

    # TODO: encode labels
    # TODO: reshape features

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val