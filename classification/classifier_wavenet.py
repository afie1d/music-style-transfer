import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_pipeline import create_ds

class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), dilation=(2, 2), padding=(2, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), dilation=(4, 4), padding=(4, 4))
        
        self.residual1 = nn.Conv2d(1, 128, kernel_size=(1, 1))  # To match dimensions
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=(8, 8), padding=(8, 8))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=(4, 4), padding=(4, 4))
        self.conv6 = nn.Conv2d(128, 64, kernel_size=(3, 3), dilation=(2, 2), padding=(2, 2))
        
        self.residual2 = nn.Conv2d(128, 64, kernel_size=(1, 1))  # To match dimensions
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        res1 = self.residual1(x)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = x + res1

        res2 = self.residual2(x)
        
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        
        x = x + res2
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        if epoch % 25 == 0:
            acc = validate(model, val_loader)
            print("VALIDATION ACCURACY", acc, "%")

# Prepare the data for PyTorch
def prepare_dataloader(X_train, X_val, y_train, y_val, batch_size=32):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def validate(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return 100 * correct / total 

# ********************
# Training
# ********************

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on", device)

print("Creating dataset ...")
path = "/work/cssema416/202510/13/data/GTZAN/genres_original"
X_train, X_val, y_train, y_val, label_dict = create_ds.create_dataset(path)

num_classes = len(label_dict)
model = Network(num_classes).to(device)

train_loader, val_loader = prepare_dataloader(X_train, X_val, y_train, y_val)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100)
final_acc = validate(model, val_loader)
print("Final Validation Accuracy", final_acc, "%")

torch.save(model.state_dict(), "./wn_state_dict.pth")

