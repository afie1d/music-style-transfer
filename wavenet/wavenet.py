import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


class ResidualBlock(nn.Module):
    # The connections in the CNN use dilation, which allows temporally distant activations
    # to overlap with future activations/outputs
    def __init__(self, dilation, residual_channels, skip_channels):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(
            in_channels=residual_channels,
            out_channels=2 * residual_channels,
            kernel_size=2,
            dilation=dilation,
            padding=dilation
        )
        self.residual_conv = nn.Conv1d(
            in_channels=residual_channels,
            out_channels=residual_channels,
            kernel_size=1
        )
        self.skip_conv = nn.Conv1d(
            in_channels=residual_channels,
            out_channels=skip_channels,
            kernel_size=1
        )

    def forward(self, x):
        output = self.dilated_conv(x)
        gate, filter = torch.chunk(output, 2, dim=1)

        # WaveNet uses a combination of activation functions
        activated = torch.tanh(filter) * torch.sigmoid(gate)

        skip_out = self.skip_conv(activated)
        residual_out = self.residual_conv(activated) + x

        return residual_out, skip_out


class WaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, num_blocks, num_layers, num_classes):
        super(WaveNet, self).__init__()
        self.input_conv = nn.Conv1d(
            in_channels=1,
            out_channels=residual_channels,
            kernel_size=1
        )

        self.residual_blocks = nn.ModuleList()
        for block in range(num_blocks):
            for layer in range(num_layers):
                dilation = 2 ** layer
                self.residual_blocks.append(
                    ResidualBlock(dilation, residual_channels, skip_channels)
                )

        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=skip_channels, out_channels=skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=skip_channels, out_channels=num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_conv(x)
        skip_connections = []

        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = sum(skip_connections)
        x = self.output_conv(x)
        x = torch.mean(x, dim=2)  # Global average pooling over the time dimension
        return x


def process_audio(file_path):
    # Load audio and convert to mono
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert waveform to Mel spectrogram
    mel_transform = MelSpectrogram(
        sample_rate=sample_rate, n_mels=128, n_fft=2048, hop_length=512
    )
    amplitude_to_db = AmplitudeToDB()

    mel_spectrogram = mel_transform(waveform)
    mel_spectrogram = amplitude_to_db(mel_spectrogram)

    return mel_spectrogram



