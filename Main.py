import streamlit as st
from intro import intro
from tabular_classification import tabular_classification
from tabular_regression import tabular_regression
from image_classification import image_classification
from image_segmentation import image_segmentation
from text_classification import text_classification

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select a page",
    ("Introduction", "Tabular Data Classification", "Regression on Tabular Data", 
     "Image Classification", "Image Segmentation", "Text Classification")
)

# Display the selected page
if page == "Introduction":
    intro()
elif page == "Tabular Data Classification":
    tabular_classification()
elif page == "Regression on Tabular Data":
    tabular_regression()
elif page == "Image Classification":
    image_classification()
elif page == "Image Segmentation":
    image_segmentation()
elif page == "Text Classification":
    text_classification()
