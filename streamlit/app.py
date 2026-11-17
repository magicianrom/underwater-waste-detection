import streamlit as st

st.set_page_config(
    page_title="Underwater Waste Detection",
    page_icon="🌊"
)

# Define the pages
main_page = st.Page(
    "index.py",
    title="Home",
    icon="🏠"
)

page_2 = st.Page(
    "classification.py",
    title="Classification",
    icon="🗂️"
)

page_3 = st.Page(
    "Localization_Segmentation.py",
    title="Localization & Segmentation",
    icon="🌊"
)

# Set up navigation
pg = st.navigation([main_page, page_2, page_3])

# Run selected page
pg.run()
