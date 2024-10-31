import torch.nn as nn


# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.output_size = output_size
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Calculate post-convolution flattened size
        final_length = input_size
        for _ in range(3):  # for 3 convolution + pooling layers
            final_length = (final_length + 2 - 3) // 1 + 1
            final_length //= 2  # pooling
        
        self.fc1 = nn.Linear(16 * final_length, 256)
        self.fc2 = nn.Linear(256, output_size)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, z):
        z = self.conv1(z)
        z = self.ReLU(z)
        z = self.pool(z)
        
        z = self.conv2(z)
        z = self.ReLU(z)
        z = self.pool(z)
        
        z = self.conv3(z)
        z = self.ReLU(z)
        z = self.pool(z)
        
        z = z.view(z.size(0), -1)
        z = self.fc1(z)
        z = self.ReLU(z)
        
        z = self.fc2(z)
        z = self.Tanh(z)
        
        return z


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
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
        self.fc4 = nn.Linear(16, 1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

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
        
        x = self.fc2(x)
        x = self.ReLU(x)
        
        x = self.fc3(x)
        x = self.ReLU(x)
        
        x = self.fc4(x)
        x = self.Sigmoid(x)
        
        return x