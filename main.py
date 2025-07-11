from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import io
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from pathlib import Path

app = FastAPI()

# Load PyTorch Lung Detector
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
        img = self.transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(img)
            predicted_class = torch.argmax(outputs, dim=1).item()
        return predicted_class  # 1 = Lung, 0 = NotLung

# Load lung detection model
lung_model = LungDetector("artifacts/lung_ct_resnet_model1.pth")

# Load Keras model
with open("artifacts/training/model.json", "r") as json_file:
    model_json = json_file.read()

cancer_model = model_from_json(model_json)
cancer_model.load_weights("artifacts/training/model.h5")
cancer_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                     loss="categorical_crossentropy",
                     metrics=["accuracy"])

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_img, keras_input = preprocess_image(image_bytes)

    try:
        is_lung = lung_model.predict(pil_img)
        if is_lung != 0:
            return JSONResponse(content={"prediction": "NotLung"})

        preds = cancer_model.predict(keras_input)
        result = np.argmax(preds, axis=1)[0]

        label = "Normal" if result == 1 else "Adenocarcinoma Cancer"
        return JSONResponse(content={"prediction": label})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
