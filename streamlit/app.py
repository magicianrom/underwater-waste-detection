
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from ultralytics import YOLO

st.title("Underwater Waste Classifier")

@st.cache_resource
def load():
    model = keras.models.load_model("data/mobilenet_phase2.keras")
    with open("data/class_names.json") as f:
        names = json.load(f)
    return model, names

@st.cache_resource
def load_yolo():
    return YOLO("data/best.pt")

model, class_names = load()
yolo_model = load_yolo()

# Classification
file = st.file_uploader("Upload a cropped image of one waste item", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Input", width=300)

    x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), axis=0)
    probs = model.predict(x, verbose=0)[0]

    best = int(np.argmax(probs))
    st.subheader(f"{class_names[best]} ({probs[best]:.1%})")
    st.bar_chart({n: float(p) for n, p in zip(class_names, probs)})

# Detection and segmentation
st.header("Waste Detection and Segmentation")

seg_file = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])

if seg_file:
    img = Image.open(seg_file).convert("RGB")
    results = yolo_model.predict(img, conf=0.25, verbose=False)

    st.image(results[0].plot(), caption="Detection and Segmentation", width=500)
