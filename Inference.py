import torch
import torchaudio


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Set the parameters
noise_size = 100
sample_rate = 22050
output_size = sample_rate * 30

# Load the model
model = torch.load("generator.pth")

outputs = []
for i in range(100):
    z = torch.randn(1, noise_size, device=device)
    z = torch.randn(1, noise_size, device=device)
    model_output = model(z.unsqueeze(1))
    model_output = model_output.detach().cpu()
    filename = "./TestOutputs/" + "output" + str(i+1) + ".wav"
    torchaudio.save(filename, model_output, sample_rate)
