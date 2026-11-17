
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("Underwater Waste Segmentation Test")

@st.cache_resource
def load_model():
    return YOLO("data/best.pt")

model = load_model()

file = st.file_uploader(
    "Upload an underwater image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:
    img = Image.open(file).convert("RGB")

    results = model.predict(
        source=img,
        conf=0.25,
        device="cpu",
        verbose=False
    )

    st.image(
        results[0].plot()[:, :, ::-1],
        caption="Detection and Segmentation"
    )
