import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from tensorflow.keras.models import load_model
import soundfile as sf

# Load the trained genre classification model (ensure this model file exists in your directory)
genre_classifier = load_model("yamnet.h5")

# Load the YAMNet model from TensorFlow Hub
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Function to extract features using YAMNet
def extract_features(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        # Load the audio file
        audio_waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
        audio_waveform = tf.squeeze(audio_waveform, axis=-1)  # Remove the channel dimension

        # Run audio through YAMNet for feature extraction
        scores, embeddings, spectrogram = yamnet_model(audio_waveform)

        # Average the embeddings across all frames to get a fixed-size feature vector
        embedding_avg = np.mean(embeddings.numpy(), axis=0)  # Averaging across time frames

        return embedding_avg
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None  # Or handle accordingly

# Function to predict genre
def predict_genre(features):
    if features is None:
        return "Error: Could not extract features."

    # Reshape the feature vector to match the input shape of the classifier
    features = features.reshape(1, -1)

    # Make prediction
    prediction = genre_classifier.predict(features)
    genre_index = np.argmax(prediction)
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    return genres[genre_index]

# Streamlit UI
st.title("Audio Genre Classification")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join("temp_audio.wav")

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format='audio/wav')

    # Extract features from the uploaded audio
    st.write("Extracting features from the uploaded audio file...")
    features = extract_features(temp_file_path)

    # Display extracted features and prediction
    if features is not None:
        st.write(f"Extracted Features: {features[:10]}...")  # Show first 10 features for example

        # Predict the genre
        genre = predict_genre(features)
        st.write(f"Predicted Genre: {genre}")
    else:
        st.write("Error extracting features.")
