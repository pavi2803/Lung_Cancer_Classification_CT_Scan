import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
import os



class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model_path = os.path.join("model_final", "model.h5")
        
        if not os.path.exists(model_path):
            st.error("Model file not found. Please check the path.")
            return "Error: Model file not found"

        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return f"Error: {e}"

        try:
            test_image = image.load_img(self.filename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = np.argmax(model.predict(test_image), axis=1)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return f"Error: {e}"

        print(result)
        if result[0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'
        return prediction

# Streamlit UI
st.title("Image Classification with Keras Model")
st.write("Upload an image to classify it as Normal or Adenocarcinoma Cancer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_file_path = os.path.join("temp", uploaded_file.name)
    
    # Ensure the temp directory exists
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    pipeline = PredictionPipeline(temp_file_path)
    prediction = pipeline.predict()
    
    st.write(f"Prediction: {prediction}")