import os
import torch
import librosa
from datasets import Dataset
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, Trainer, TrainingArguments
import numpy as np


#model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
model_name = "facebook/wav2vec2-large-xlsr-53"

# Path to the root directory containing class subfolders
root_dir = "./Data/genres_original"

# Initialize the feature extractor from Hugging Face model (this will handle preprocessing)
#extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Define the list to hold audio file paths and their corresponding labels
audio_paths = []
labels = []

# Walk through the directory and load files
class_names = os.listdir(root_dir)  # Class names are subfolder names
class_names = sorted(class_names)  # Ensure classes are sorted (optional but useful)

for label, class_name in enumerate(class_names):  # Assign an integer label for each class
    class_folder = os.path.join(root_dir, class_name)
    if os.path.isdir(class_folder):
        for audio_file in os.listdir(class_folder):
            if audio_file.endswith(".wav"):  # Ensure the file is a .wav file
                audio_path = os.path.join(class_folder, audio_file)
                audio_paths.append(audio_path)
                labels.append(label)  # Assign the corresponding class label

# Create a dataset from the file paths and labels
dataset_dict = {
    "audio_path": audio_paths,
    "label": labels
}

# Convert to a Hugging Face Dataset
dataset = Dataset.from_dict(dataset_dict)

# Check the first few entries in the dataset
#print(dataset[:5])







def preprocess_audio(example):
    # Load audio using librosa (at the correct sampling rate)
    audio_input, sr = librosa.load(example["audio_path"], sr=extractor.sampling_rate)
    
    # Extract features from the audio using the feature extractor
    #features = extractor(audio_input, return_tensors="pt", sampling_rate=extractor.sampling_rate, padding=True, truncation=True)
    #features = extractor(audio_input, return_tensors="pt", sampling_rate=extractor.sampling_rate, padding=True)
    features = extractor(audio_input, return_tensors="pt", sampling_rate=extractor.sampling_rate, padding=True, truncation=True, max_length=100000)
    
    # Ensure that the returned tensor has the correct shape, removing any unnecessary singleton dimensions
    input_values = features['input_values']
    
    # Check the shape before and after squeezing
    # input_values has shape [1, time_steps, frequency_bins], we need to make sure it's in the correct 4D shape for the model
    # Squeeze if necessary to get a shape of [batch_size, channels, height, width]
    
    input_values = input_values.squeeze(0)  # Remove the batch dimension if it's 1
    #input_values = input_values.unsqueeze(0)  # Add batch dimension back to make it [batch_size, channels, height, width]

    return {"input_values": input_values}


# Apply the preprocessing function to the dataset
dataset = dataset.map(preprocess_audio, remove_columns=["audio_path"])

# Check the dataset structure
#print(dataset[0])









# Split dataset into training, validation, and test sets (80/10/10 split)
dataset = dataset.shuffle(seed=42)  # Shuffle the dataset for randomness
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))

train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, len(dataset)))

# Check the splits
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")











# Load the pre-trained model
model = AutoModelForAudioClassification.from_pretrained(model_name)

# The number of classes for your new task (e.g., 10)
num_labels = len(class_names)
model.config.num_labels = num_labels


# Get the size of the hidden layers from the model's configuration (this should be consistent across models)
hidden_size = model.config.hidden_size

# Change the classifier (final linear layer) to match your new number of labels
#model.classifier.dense = torch.nn.Linear(hidden_size, num_labels)

# Replace the classifier (the final layer) to match the number of labels
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(hidden_size // 4, 512),  # You can adjust this size based on your model architecture
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),  # Optional, you can adjust dropout if needed
    torch.nn.Linear(512, num_labels)  # Set the final layer to have 'num_labels' units (e.g., 10 for 10 classes)
)

# =============================================================================
# # Replace the classifier (the final layer) to match the number of labels
# model.classifier = torch.nn.Sequential(
#     torch.nn.Linear(hidden_size // 4, 512),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(0.1),
#     torch.nn.Linear(512, 64),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(0.1),
#     torch.nn.Linear(64, num_labels)  # Set the final layer to have 'num_labels' units (e.g., 10 for 10 classes)
# )
# =============================================================================


# Hyperparameters
num_epochs = 1000
batch_size = 16












# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="no",
    evaluation_strategy="no",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_dir=None,
    logging_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)



# Start fine-tuning
trainer.train()













# Evaluate the model on the test dataset
#results = trainer.evaluate(test_dataset)
pred_results = trainer.predict(test_dataset)

actual = test_dataset["label"]
actual = np.array(actual)
logits = pred_results[0]
predictions = np.argmax(logits, axis=1)



def calculate_accuracy(labels, predictions):
    # Convert labels and predictions to numpy arrays (in case they're lists or other types)
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # Ensure the arrays have the same length
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same length.")
    
    # Calculate the number of correct predictions
    correct_predictions = np.sum(labels == predictions)
    
    # Calculate accuracy
    accuracy = correct_predictions / len(labels)
    return accuracy

accuracy = calculate_accuracy(actual, predictions)






torch.save(model.state_dict(), "transformer_model_1000.pth")

