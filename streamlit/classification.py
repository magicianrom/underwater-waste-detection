
import json
import streamlit as st
from PIL import Image
from tensorflow import keras
import numpy as np

st.title("🖼️ Underwater Waste Classification")

st.write("""
Upload an underwater image to identify the type of waste
using a trained deep learning model.
""")

st.markdown("### Supported Waste Categories")

categories = [
    "Plastic",
    "Metal",
    "Wood",
    "Fabric",
    "Paper",
    "Fishing Gear",
    "Rubber",
    "Other Waste"
]

st.write(", ".join(categories))

st.divider()

# Load model
@st.cache_resource
def load_model():
    model = keras.models.load_model("data/mobilenet_phase2.keras")

    with open("data/class_names.json") as f:
        names = json.load(f)

    return model, names

model, class_names = load_model()

# Image upload
file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    img = Image.open(file).convert("RGB")

    # Prepare image
    resized_img = img.resize((224, 224))
    img_array = np.array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    # Display results
    st.divider()
    st.subheader("Classification Results")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### Predicted Category")
        st.success(class_names[predicted_class])

        st.markdown("### Confidence")
        st.metric(
            label="Prediction Confidence",
            value=f"{confidence * 100:.2f}%"
        )

        st.progress(float(confidence))

else:
    st.info("👆 Upload an underwater image to get started.")
