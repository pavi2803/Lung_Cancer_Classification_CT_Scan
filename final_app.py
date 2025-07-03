import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image
from io import BytesIO
from pathlib import Path

class PredictionPipeline:
    def __init__(self, image_bytes):
        self.image_bytes = image_bytes

    def img_to_image(self):
        img = Image.open(BytesIO(self.image_bytes))
        img = img.convert("RGB")  # Convert image to RGB
        return img

    def predict(self):
        model_path = Path('artifacts/training/model.h5')
        
        # Debugging information

        
        if not model_path.exists():
            st.error("Model file not found. Please check the path.")
            return "Error: Model file not found"

        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return f"Error: {e}"

        try:
            img = self.img_to_image()
            img = img.resize((224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # Normalize the image

            predictions = model.predict(img)
            result = np.argmax(predictions, axis=1)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return f"Error: {e}"

        if result[0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'
        return prediction

# Streamlit UI
st.title("Chest Cancer Prediction")
st.subheader("Upload an image in either jpg, jpeg or png format")
st.write("Upload a CT Scan image to classify it as Normal or presence of Adenocarcinoma Cancer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    pipeline = PredictionPipeline(image_bytes)
    prediction = pipeline.predict()
    
    if(prediction == 'Normal'):
        st.write(f'✅ Its totally Normal 😊')

    else:
        st.write('🚨 Chances of Adenocarcinoma Cancer‼️ - Please Consult a Doctor.')

