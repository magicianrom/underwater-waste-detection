import streamlit as st

st.set_page_config(
    page_title="Underwater Waste Detection",
    page_icon="🌊"
)

st.title("🌊 Underwater Waste Detection")

st.write("""
This project uses deep learning to identify and locate
underwater waste using computer vision.

It includes two approaches:
""")

st.subheader("1. Image Classification")
st.write("""Classifies an underwater image into one of the trained waste categories.""")

st.subheader("2. Localization & Segmentation")
st.write("""
Detects underwater waste and outlines the objects
within the uploaded image.
""")

st.info("Select a page from the sidebar to try either model.")
