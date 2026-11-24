
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("🎯 Underwater Waste Localization & Segmentation")

st.write("""
Upload an underwater image to detect waste objects
and highlight their locations using bounding boxes and segmentation masks.
""")

st.divider()

# Load model
@st.cache_resource
def load_model():
    return YOLO("data/best.pt")

model = load_model()

# Image upload
file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    img = Image.open(file).convert("RGB")

    # Make prediction
    results = model.predict(
        source=img,
        conf=0.25,
        device="cpu",
        verbose=False
    )

    # Get results
    result_img = results[0].plot()[:, :, ::-1]
    detected_objects = len(results[0].boxes)

    # Display results
    st.divider()
    st.subheader("Detection Results")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(
            img,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            result_img,
            caption="Detected Waste",
            use_container_width=True
        )

    # Detection summary
    st.divider()
    st.subheader("Detection Summary")

    st.metric(
        label="Detected Objects",
        value=detected_objects
    )

    if detected_objects > 0:
        st.success(f"Successfully detected {detected_objects} object(s).")
    else:
        st.info("No waste objects were detected in this image.")

else:
    st.info("👆 Upload an underwater image to get started.")
