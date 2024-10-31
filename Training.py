import torch
import torch.nn as nn
import torch.optim as optim
from Models import Generator, Discriminator


# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"


# Set size parameters
output_size = 22050 * 30
noise_size = 100


# Set hyperparameters
batch_size = 8
num_epochs = 10
D_learn_rate = 0.0000005
G_learn_rate = 0.000005
kernel_size = 3
stride = 1
padding = kernel_size // 2


# Load in waveform tensors
tensors = torch.load('tensors.pth', weights_only=True)


# Instantiate the models
generator = Generator(noise_size, output_size)
discriminator = Discriminator(output_size)


# Move the models to device
generator = generator.to(device)
discriminator = discriminator.to(device)


# Set the training parameters
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=G_learn_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=D_learn_rate)


# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(tensors), batch_size):
        
        # Get the current batch
        real_waveforms = tensors[i:i + batch_size]
        if len(real_waveforms) < batch_size:
            continue  # Skip the last batch if it's smaller than batch_size
        
        # Stack tensors into a single tensor
        real_waveforms = torch.stack(real_waveforms).to(device)

        # Create labels
        real_labels = torch.ones(real_waveforms.size(0), 1, device=device)
        fake_labels = torch.zeros(real_waveforms.size(0), 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()
        #outputs = discriminator(real_waveforms.view(real_waveforms.size(0), -1).unsqueeze(1))
        outputs = discriminator(real_waveforms)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(real_waveforms.size(0), noise_size, device=device).unsqueeze(1)
        fake_waveforms = generator(z).unsqueeze(1)
        outputs = discriminator(fake_waveforms.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        outputs = discriminator(fake_waveforms)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')



# Save the models locally
torch.save(generator, 'generator.pth')
torch.save(discriminator, 'discriminator.pth')
