
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

st.title("Underwater Waste Classifier")

# Load model
@st.cache_resource
def load_model():
    model = keras.models.load_model(
        "data/mobilenet_phase2.keras",
        compile=False
    )

    with open("data/class_names.json") as f:
        class_names = json.load(f)

    return model, class_names


model, class_names = load_model()

# Upload image
file = st.file_uploader(
    "Upload an underwater image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    img = Image.open(file).convert("RGB")

    st.image(img, caption="Uploaded Image", width=400)

    # Prepare image
    img = img.resize((224, 224))
    x = np.array(img, dtype="float32")
    x = np.expand_dims(x, axis=0)

    # Predict
    predictions = model.predict(x, verbose=0)[0]

    best = np.argmax(predictions)

    # Display result
    st.subheader("Classification Result")
    st.write(class_names[best])

    st.write(f"Confidence: {predictions[best]:.2%}")

    st.bar_chart({
        name: float(prob)
        for name, prob in zip(class_names, predictions)
    })
