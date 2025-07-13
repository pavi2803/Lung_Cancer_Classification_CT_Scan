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


            cancer_model = tf.keras.models.load_model(self.cancer_model_path, compile=False)

            # Load weights
            # cancer_model.load_weights("artifacts/training/model.h5")  # match the file name

            # Compile model AFTER loading weights
            cancer_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            predictions = cancer_model.predict(keras_img)
            result = np.argmax(predictions, axis=1)

            return "Normal" if result[0] == 1 else "Adenocarcinoma Cancer"

        except Exception as e:
            return f"Error: {str(e)}"


################################ Header UI ###############################
st.set_page_config(page_title="Chest CT Analysis", page_icon="🫁", layout="centered")

from pathlib import Path

logo_path = "artifacts/3843297.png"

col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo_path, width=60)
with col2:
    st.markdown("<h3 style='text-align: center;'> Chest CT Scan Analysis - Cancer Detection</h3>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .main {
        background-color: #81a6b8;
        padding: 2rem;
    }

    h1, h2, h3, h4, h5, h6,div {
        color: #050505 !important;
    }

    .stApp {
        background-color: #81a6b8;
    }

    .css-18e3th9 {
        padding: 2rem;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    

    .stFileUploader {
        border: 2px dashed #888;
        padding: 10px;
        border-radius: 8px;
        background-color: #fafafa;
    }



    /* Make file uploader box cleaner */
    section[data-testid="stFileUploader"] > div {
        padding: 10px;
        border: 2px dashed #aaa;
        border-radius: 10px;
        background-color: #ffffff;
        color: #333333;
    }

    /* Reduce uploader height and fix font */
    section[data-testid="stFileUploader"] label {
        font-size: 0.9rem;
        color: #222831;
    }

    section[data-testid="stFileUploader"] svg {
        height: 1.2rem;
        width: 1.2rem;
        color: #333;
    }

    </style>
    """,
    unsafe_allow_html=True
)



with st.expander("ℹ️ About this App"):
    st.markdown("""
    This app uses two AI models:

    - A **ResNet18 model (PyTorch)** to check if the uploaded image is a lung CT.
    - A **Keras CNN model (TensorFlow)** to classify the scan as **Normal** or **Adenocarcinoma Cancer**.
    
    All models are trained using **transfer learning**, and experiments are tracked with **MLflow**.
    """)

st.markdown("""<div style='text-align: center;'>
    <p style='font-size: 15px;'>This app will detect if it's a valid lung CT scan and classify it as <strong>Normal</strong> or <strong>Adenocarcinoma Cancer</strong>.</p>
</div>""", unsafe_allow_html=True)


st.markdown("---")

uploaded_file = st.file_uploader("Choose a CT scan image below:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    
    pipeline = PredictionPipeline(image_bytes)
    result = pipeline.predict()
    if result == "Normal":
        st.success("✅ No Signs of Lung Problems Detected, Double Check with a Professional")
    elif result == "Adenocarcinoma Cancer":
        st.error("🚨 Adenocarcinoma Cancer Detected‼️ Please consult a doctor.")
    elif result == "NotLung":
        st.warning("⚠️ This does not appear to be a lung CT scan.")
    else:
        st.error(f"An error occurred: {result}")


########################### Latest ##############################

