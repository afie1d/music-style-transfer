# Music-Style-Transfer

This repository contains models and scripts for transferring and classifying music styles using various machine learning approaches, including baseline models, CNNs, and transfer learning frameworks like YAMNet and Wav2Vec2. Additionally, it includes a Streamlit-based web application.

## Libraries Used
- `numpy`
- `pandas`
- `os` (part of Python's standard library)
- `tensorflow`
- `scikit-learn`
- `tensorflow-hub`
- `matplotlib`
- `librosa`

## Importing Data
To get started, download the dataset from [this link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).  
Ensure that the path to the data is correctly set for all models below.

## Running Baseline
1. Ensure all required libraries are installed.
2. Run the Jupyter Notebook file located at `./Baseline/random_forest.ipynb`.

## Running CNN
1. Ensure all required libraries are installed.
2. Run the Python file located at `./CNN/classifier_wavenet.py`.

## Running YAMNET (Transfer Learning)
1. Ensure all required libraries are installed.
2. Run the Jupyter Notebook file located at `./YAMNET/yamnet.ipynb`.

## Running Wav2Vec2
1. Ensure all required libraries are installed.
2. Run the Python notebook file located at `./Wav2Vec2/Classifier_wav2vec2.py`.

## Running Website
1. Deploy the app using [Streamlit Cloud](https://streamlit.io/cloud).  
2. Ensure the `requirements.txt` file is correct and the path to `app.py` is provided.'

## Contributers
Jonathan Rogers, Adam Field, Drew Huffman, and Blake Johnson
