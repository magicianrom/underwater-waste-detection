
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from ultralytics import YOLO

st.title("Underwater Waste Detection")

# Load classification model
@st.cache_resource
def load_model():
    model = keras.models.load_model(
        "data/mobilenet_phase2.keras",
        compile=False
    )

    with open("data/class_names.json") as f:
        class_names = json.load(f)

    return model, class_names

# Load YOLO segmentation model
@st.cache_resource
def load_yolo():
    return YOLO("data/best.pt")


model, class_names = load_model()
yolo_model = load_yolo()

# Upload image
file = st.file_uploader(
    "Upload an underwater image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    img = Image.open(file).convert("RGB")

    st.image(img, caption="Uploaded Image", width=400)

    # 1. Classification
    x = np.array(img.resize((224, 224)), dtype="float32")
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x, verbose=0)[0]
    best = int(np.argmax(predictions))

    st.subheader("Classification")
    st.write(f"{class_names[best]}")
    st.write(f"Confidence: {predictions[best]:.2%}")

    st.bar_chart({
        name: float(prob)
        for name, prob in zip(class_names, predictions)
    })

    # 2. Localisation and 3. Segmentation
    st.subheader("Localisation and Segmentation")

    results = yolo_model.predict(
        source=img,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # Display bounding boxes and segmentation masks
    detected_image = result.plot()
    detected_image = detected_image[:, :, ::-1]

    st.image(
        detected_image,
        caption="Detected Objects and Segmentation Masks",
        width=500
    )

    # Display number of detected objects
    if result.boxes is not None:
        st.write(f"Objects detected: {len(result.boxes)}")
    else:
        st.write("No objects detected.")
