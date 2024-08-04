import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
import os
import base64
from PIL import Image
from io import BytesIO

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def img_to_base64(self, img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")


    def base64_to_image(self, base64_str):
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        img = img.convert("RGB")  # Convert image to RGB
        return img

  
    def predict(self):
        model_path = os.path.join("model.h5")
        
        if not os.path.exists(model_path):
            st.error("Model file not found. Please check the path.")
            return "Error: Model file not found"

        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return f"Error: {e}"

        try:
            base64_image = self.img_to_base64(self.filename)
            img = self.base64_to_image(base64_image)
            img = img.resize((224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # Normalize the image

            predictions = model.predict(img)
            result = np.argmax(predictions, axis=1)
            st.write(f"Raw predictions: {predictions}")  # Print raw predictions
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return f"Error: {e}"

        if result[0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'
        return prediction

# Streamlit UI
st.title("Lung Cancer Prediction")
st.subheader("Transfer Learning on VGGNET-16")
st.write("Upload an image to classify it as Normal or Adenocarcinoma Cancer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_dir = Path("temp")
    temp_file_path = temp_dir / uploaded_file.name
    
    # Ensure the temp directory exists
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)
    
    # Save the uploaded file
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    pipeline = PredictionPipeline(temp_file_path)
    prediction = pipeline.predict()
    
    st.write(f"Prediction: {prediction}")
