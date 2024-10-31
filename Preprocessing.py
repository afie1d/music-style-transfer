import os
import torch
import torchaudio

base_path = "./Data/genres_original/"
genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]


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
def normalizeTensor(tensor):
    tensor = tensor / tensor.abs().max()
    return tensor


tensors = []
for i in range(len(genres)):
    genre_path = base_path + genres[i]
    for filename in os.listdir(genre_path):
        if filename.endswith('.wav'):

            # Read the wav file and convert to tensor
            file_path = os.path.join(genre_path, filename)
            waveform, sample_rate = torchaudio.load(file_path)

            # Resize the waveform to 30 seconds length (661,500 elements)
            waveform = resizeWaveform(waveform)

            # Fast-Fourier-Transform
            spectrum = torch.fft.fft(waveform)

            # Extract the magnitude and phase
            magnitude = torch.abs(spectrum)
            phase = torch.angle(spectrum)

            # Normalize the tensors
            waveform = normalizeTensor(waveform)
            magnitude = normalizeTensor(magnitude)
            phase = normalizeTensor(phase)

            # Stack the tensors
            data = torch.stack((waveform, magnitude, phase)).squeeze(1)

            # save the waveform to the list
            # tensors.append(data)
            tensors.append(magnitude)


# Save the list of tensors locally
torch.save(tensors, "tensors.pth")
