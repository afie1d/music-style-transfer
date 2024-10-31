import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import os
from wavenet import WaveNet

# Custom Dataset class for loading audio files and labels
class AudioGenreDataset(Dataset):
    def __init__(self, audio_files, labels, transform=None):
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        label = self.labels[idx]
        
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:  # Convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if self.transform:
            waveform = self.transform(waveform, sample_rate)
        
        return waveform, label


# Transform function for Mel spectrogram
def mel_spectrogram_transform(waveform, sample_rate):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=128, n_fft=2048, hop_length=512
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    mel_spectrogram = mel_transform(waveform)
    mel_spectrogram = amplitude_to_db(mel_spectrogram)
    return mel_spectrogram


def create_ds(path):
    waves = []
    labels = []

    for f in os.listdir(path):
        f_path = os.path.join(path, f)
        label = f.split(".")[0]
        waves.append(f_path)
        labels.append(label)

    return AudioGenreDataset(waves, labels, transform=mel_spectrogram_transform)


# Training Loop
def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            # Add channel dimension if necessary
            waveforms = waveforms.unsqueeze(1)  # Shape: (batch_size, 1, time)

            # Forward pass
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")



if __name__ == "__main__":
    
    # Parameters
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and data loader
    dataset = create_ds('../../data/GTZAN/genres_original')
    print("Creating dataset ...\n")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = WaveNet(residual_channels=32, skip_channels=64, num_blocks=3, num_layers=10, num_classes=10)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs, device)
