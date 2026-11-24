
import streamlit as st

st.set_page_config(
    page_title="Underwater Waste Detection",
    page_icon="🌊",
    layout="centered"
)

# Main title
st.title("🌊 Underwater Waste Detection")

st.write("""
Underwater waste pollution is a growing environmental problem that
threatens marine ecosystems.

This project uses deep learning and computer vision to identify
underwater waste in images through two different approaches.
""")

st.divider()

# Project approaches
st.header("Explore the Models")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ Image Classification")
    st.write("""
    Classifies underwater images into one of the
    trained waste categories.
    """)

with col2:
    st.subheader("🎯 Detection & Segmentation")
    st.write("""
    Locates underwater waste in images and
    highlights the objects using bounding boxes
    or segmentation masks.
    """)

st.divider()

# Project information
st.header("About the Project")

st.write("""
The goal is to explore how deep learning can help make
underwater waste identification and monitoring easier.

Both approaches provide different ways of analysing
underwater images and recognising waste.
""")

st.info("👈 Select a model from the sidebar to get started.")
