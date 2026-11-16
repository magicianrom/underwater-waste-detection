import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

st.title("Underwater Waste Classifier")

@st.cache_resource
def load():

    
    model = keras.models.load_model("mobilenet_phase2.keras")
    with open("class_names.json") as f:
        names = json.load(f)
    return model, names

model, class_names = load()

file = st.file_uploader("Upload a cropped image of one waste item", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Input", width=300)

    # The model already contains Rescaling, so pass raw 0-255 pixels at 224x224
    x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), axis=0)
    probs = model.predict(x, verbose=0)[0]

    best = int(np.argmax(probs))
    st.subheader(f"{class_names[best]} ({probs[best]:.1%})")
    st.bar_chart({n: float(p) for n, p in zip(class_names, probs)})