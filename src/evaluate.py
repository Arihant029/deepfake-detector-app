import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = models.efficientnet_b0(weights=None)  # load your trained model if needed
num_classes = 2
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("models/baseline.pth", map_location=device))
model.to(device)
model.eval()

# Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Folders to evaluate
folders = {
    "Real": "data/train/real",
    "Fake": "data/train/fake"
}

# Class labels
labels = ["Real", "Fake"]

for label_name, folder in folders.items():
    if not os.path.exists(folder):
        print(f"⚠ Folder not found: {folder}")
        continue

    images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
    
    if len(images) == 0:
        print(f"⚠ No images found in folder: {folder}")
        continue

    print(f"\nEvaluating folder: {folder}")
    for img_name in images:
        img_path = os.path.join(folder, img_name)
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            predicted_label = labels[pred_idx.item()]
            confidence = conf.item() * 100

        print(f"{img_name} -> {predicted_label} ({confidence:.2f}%)")
