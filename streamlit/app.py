
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from ultralytics import YOLO

st.title("Underwater Waste Classifier")

# Load classification model
@st.cache_resource
def load():
    model = keras.models.load_model(
        "data/mobilenet_phase2.keras",
        compile=False
    )

    with open("data/class_names.json") as f:
        names = json.load(f)

    return model, names

# Load YOLO model
@st.cache_resource
def load_yolo():
    return YOLO("data/best.pt")


model, class_names = load()
yolo_model = load_yolo()

# Upload image
file = st.file_uploader(
    "Upload an underwater image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:
    img = Image.open(file).convert("RGB")

    st.image(img, caption="Input Image", width=400)

    # Classification
    x = np.array(img.resize((224, 224)), dtype="float32")
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]

    best = int(np.argmax(probs))

    st.subheader("Classification")
    st.write(f"{class_names[best]} ({probs[best]:.1%})")

    st.bar_chart({
        name: float(prob)
        for name, prob in zip(class_names, probs)
    })

    # Detection and Segmentation
    results = yolo_model.predict(
        source=img,
        conf=0.25,
        verbose=False
    )

    st.subheader("Detection and Segmentation")

    detected_image = results[0].plot()[:, :, ::-1]

    st.image(
        detected_image,
        caption="Detected Waste",
        width=500
    )
