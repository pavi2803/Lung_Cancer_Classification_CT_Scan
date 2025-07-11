import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import streamlit as st
from PIL import Image
from io import BytesIO
from pathlib import Path
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model

st.write(f"TensorFlow version: {tf.__version__}")


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
        img = self.transform(pil_image).unsqueeze(0)  # add batch dim
        with torch.no_grad():
            outputs = self.model(img)
            predicted_class = torch.argmax(outputs, dim=1).item()
        return predicted_class  # 1 for lung, 0 for not lung




def cancer_classifier(input_shape=(224, 224, 3), num_classes=2):
    base_model = tf.keras.applications.VGG16(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = tf.keras.layers.Flatten()(base_model.output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    return model


# Main pipeline
class PredictionPipeline:
    def __init__(self, image_bytes):
        self.image_bytes = image_bytes
        self.lung_model_path = Path("artifacts/lung_ct_resnet_model1.pth")
        self.cancer_model_path = Path("artifacts/training/model.h5")

    def img_to_pil(self):
        img = Image.open(BytesIO(self.image_bytes)).convert("RGB")
        return img

    def predict(self):
        if not self.lung_model_path.exists() or not self.cancer_model_path.exists():
            return "Error: Model file(s) not found"

        try:
            pil_img = self.img_to_pil()

            # Step 1: Check if it's a lung CT using PyTorch
            lung_detector = LungDetector(self.lung_model_path)
            is_lung = lung_detector.predict(pil_img)

            if is_lung != 0:
                return "NotLung"

            # Step 2: Predict cancer type using Keras model
            # cancer_model = load_model(self.cancer_model_path)
            keras_img = pil_img.resize((224, 224))
            keras_img = keras_image.img_to_array(keras_img)
            keras_img = np.expand_dims(keras_img, axis=0)
            keras_img = keras_img / 255.0

            #old
            # cancer_model = load_model(self.cancer_model_path)

            # with open("artifacts/training/model.json", "r") as json_file:
            #     model_json = json_file.read()

            # cancer_model = model_from_json(model_json)
            # cancer_model.load_weights("artifacts/training/model.weights.h5")

            # cancer_model = lung_classifier(input_shape=(224,224,3), num_classes=2, freeze_all=True)
            # cancer_model.load_weights("artifacts/training/model.h5")

            # Rebuild and load weights
        
            cancer_model = load_model("artifacts/training/model.h5")

            predictions = cancer_model.predict(keras_img)
            result = np.argmax(predictions, axis=1)

            return "Normal" if result[0] == 1 else "Adenocarcinoma Cancer"

        except Exception as e:
            return f"Error: {str(e)}"


# Streamlit UI
st.title("Chest Cancer Prediction")
st.subheader("Upload a lung CT image (jpg, jpeg or png)")
st.write("This app will first detect if it's a lung CT image, and if yes, classify it.")

uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    pipeline = PredictionPipeline(image_bytes)
    result = pipeline.predict()

    if result == "Normal":
        st.success("✅ It's Normal 😊")
    elif result == "Adenocarcinoma Cancer":
        st.error("🚨 Adenocarcinoma Cancer Detected‼️ Please consult a doctor.")
    elif result == "NotLung":
        st.warning("⚠️ This does not appear to be a lung CT scan.")
    else:
        st.error(f"An error occurred: {result}")
