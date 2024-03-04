import streamlit as st
from keras.models import load_model
from predict import predict_image_class, load_and_preprocess_image
import os
from PIL import Image

# Title app
st.title("Brain Tumor Classification with Magnetic Resonance Imaging")

# Set header 
st.header('Please upload Brain MRI Slice Image')

# Cache augemented model
@st.cache_resource
def load_keras_model(path):
    """
    path = string
    """
    # Load classifier for mri images
    model = load_model(path)
    return model

# Load classifier
aug_path = "models/op_model1_aug.keras"
aug_model = load_keras_model(aug_path)

# Define Class Names
with open('labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ') for a in f.readlines()]
    f.close()

# Upload File
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
