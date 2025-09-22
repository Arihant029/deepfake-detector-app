import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Transformations
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# Dataset
# ---------------------------
# Make sure you have:
# data/train/real/   -> real faces
# data/train/fake/   -> deepfake faces
train_dataset = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ---------------------------
# Baseline Model
# ---------------------------
# Using pretrained EfficientNet
model = models.efficientnet_b0(pretrained=True)
# Replace classifier for 2 classes: real/fake
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

# ---------------------------
# Loss and Optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# Training Loop
# ---------------------------
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

# ---------------------------
# Save Model
# ---------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/baseline.pth")
print("Training complete, model saved to models/baseline.pth")
