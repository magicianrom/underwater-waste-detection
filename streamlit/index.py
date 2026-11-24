
import streamlit as st

st.set_page_config(
    page_title="Underwater Waste Detection",
    page_icon="🌊",
    layout="wide"
)

# Header
st.title("🌊 Underwater Waste Detection")

st.markdown("""
Using deep learning and computer vision to identify
and locate underwater waste.
""")

st.divider()

# Introduction
st.subheader("Explore the Models")

st.write("""
Choose one of the computer vision models below to explore
how underwater waste can be identified in images.
""")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 🖼️ Image Classification")

    st.write("""
    Identify the type of underwater waste in an uploaded image.

    **Output:** A predicted waste category.
    """)

with col2:
    st.markdown("### 🎯 Detection & Segmentation")

    st.write("""
    Detect waste objects and highlight their locations
    within an uploaded image.

    **Output:** Bounding boxes and segmentation masks.
    """)

st.divider()

# About
st.subheader("About the Project")

st.write("""
This project explores different computer vision approaches
for underwater waste analysis. It demonstrates how deep
learning can help recognise waste categories and locate
individual waste objects in underwater images.
""")

st.info("👈 Select a model from the sidebar to get started.")
