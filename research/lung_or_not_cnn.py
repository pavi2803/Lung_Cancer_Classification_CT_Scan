import torch
from torchvision import models, transforms
from PIL import Image

# Define transform (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Rebuild the ResNet18 model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # For binary classification

# Load the state_dict
model.load_state_dict(torch.load("artifacts/lung_ct_resnet_model1.pth", map_location="cpu"))
model.eval()

# Load test image
# img = Image.open("artifacts/data_ingestion/Chest-CT-Scan-data/normal/6.png").convert("RGB")
img = Image.open("artifacts/cnn_training_data/not_lungs_ct/000000018193.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1).item()

print("Prediction:", pred)
if pred == 0:
    print("✅ This is a lung CT scan.")
else:
    print("🚫 This is NOT a lung CT scan.")
