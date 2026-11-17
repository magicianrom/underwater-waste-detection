import json
import streamlit as st
from PIL import Image
from tensorflow import keras
import numpy as np

st.title("Underwater Waste Classification")

@st.cache_resource
def load_model():
    model = keras.models.load_model("data/mobilenet_phase2.keras")

    with open("data/class_names.json") as f:
        names = json.load(f)

    return model, names

model, class_names = load_model()

file = st.file_uploader(
    "Upload an underwater image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
