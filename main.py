from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import io
import tensorflow as tf
from pathlib import Path

app = FastAPI()

# PyTorch Lung Detector class
class LungDetector:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, pil_image):
        img = self.transform(pil_image).unsqueeze(0)  # batch dim
        with torch.no_grad():
            outputs = self.model(img)
            predicted_class = torch.argmax(outputs, dim=1).item()
        return predicted_class  # 1 for lung, 0 for not lung

# Load models once at startup
lung_model_path = Path("artifacts/lung_ct_resnet_model1.pth")
cancer_model_path = Path("artifacts/training/model.h5")

if not lung_model_path.exists() or not cancer_model_path.exists():
    raise RuntimeError("Model files not found at the specified paths.")

lung_detector = LungDetector(lung_model_path)
cancer_model = tf.keras.models.load_model(str(cancer_model_path), compile=False)
cancer_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

def preprocess_image(image_bytes):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    keras_img = pil_img.resize((224, 224))
    keras_img = np.array(keras_img) / 255.0
    keras_img = np.expand_dims(keras_img, axis=0)
    return pil_img, keras_img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        pil_img, keras_input = preprocess_image(image_bytes)

        # Step 1: Lung detection using PyTorch model
        is_lung = lung_detector.predict(pil_img)
        if is_lung != 0:
            return JSONResponse(content={"prediction": "NotLung"})

        # Step 2: Cancer classification using TensorFlow model
        preds = cancer_model.predict(keras_input)
        result = np.argmax(preds, axis=1)[0]
        label = "Normal" if result == 1 else "Adenocarcinoma Cancer"

        return JSONResponse(content={"prediction": label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
