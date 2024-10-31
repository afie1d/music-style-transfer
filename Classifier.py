import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Set the waveform target size (30 seconds at 22,050 Hz) and set waveform to that size
def resizeWaveform(waveform):
    target_size = 661500  # 22,050Hz * 30seconds
    if waveform.size(1) > target_size:
        trimmed_waveform = waveform[:, :target_size]
    elif waveform.size(1) < target_size:
        padding = target_size - waveform.size(1)
        trimmed_waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        trimmed_waveform = waveform  # Already the correct size
    return trimmed_waveform


# Normalize the waveforms
def normalizeWaveform(waveform):
    waveform = waveform / waveform.abs().max()
    return waveform


class AudioGenreDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = resizeWaveform(waveform)
        spectrum = torch.fft.fft(waveform)
        magnitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        #power_spectrum = magnitude ** 2
        waveform = normalizeWaveform(waveform)
        magnitude = normalizeWaveform(magnitude)
        phase = normalizeWaveform(phase)
        input_tensor = torch.stack((waveform, magnitude, phase)).squeeze(1)
        return input_tensor, label



def prepare_data(folder_path):
    audio_paths = []
    labels = []
    
    # List all subfolders (genres)
    genres = os.listdir(folder_path)
    
    for label, genre in enumerate(genres):
        genre_folder = os.path.join(folder_path, genre)
        if os.path.isdir(genre_folder):
            for file in os.listdir(genre_folder):
                if file.endswith('.wav'):
                    audio_paths.append(os.path.join(genre_folder, file))
                    labels.append(label)
                    
    return audio_paths, labels



# Define the folder containing your genre folders
folder_path = "./Data/genres_original"

# Prepare the dataset
audio_paths, labels = prepare_data(folder_path)

# Split the dataset into training and test sets
train_paths, test_paths, train_labels, test_labels = train_test_split(
    audio_paths, labels, test_size=0.2, random_state=42, stratify=labels)



# Create Dataset instances
train_dataset = AudioGenreDataset(train_paths, train_labels)
test_dataset = AudioGenreDataset(test_paths, test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Calculate post-convolution flattened size
        final_length = input_size
        for _ in range(3):  # for 3 convolution + pooling layers
            final_length = (final_length + 2 - 3) // 1 + 1
            final_length //= 2  # pooling
        
        self.fc1 = nn.Linear(16 * final_length, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, num_classes)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x




# Define parameters
input_size = 22050 * 30
num_classes = 10
learning_rate = 0.01
num_epochs = 100


# Instantiate the model, loss function, and optimizer
model = Classifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Step 1: Training Loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for waveforms, labels in train_loader:
        # Move data to the device
        inputs, labels = waveforms.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track the loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Step 2: Testing Phase
model.eval()  # Set the model to evaluation mode
test_running_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():  # No gradients needed for testing
    for test_waveforms, test_labels in test_loader:
        test_waveforms, test_labels = test_waveforms.to(device), test_labels.to(device)
        test_outputs = model(test_waveforms)
        test_loss = criterion(test_outputs, test_labels)
        test_running_loss += test_loss.item()

        _, test_predicted = torch.max(test_outputs.data, 1)
        test_total += test_labels.size(0)
        test_correct += (test_predicted == test_labels).sum().item()

test_loss = test_running_loss / len(test_loader)
test_accuracy = test_correct / test_total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

