import streamlit as st
from keras.models import load_model
from predict import classify
from PIL import Image, ImageOps
import numpy as np

# Title app
st.title("Brain Tumor Classification with Magnetic Resonance Imaging")

# Set header for classification
st.header('Please upload Brain MRI Slice Image', divider='blue')

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

# Create statement to predict
if file is not None:
    image = Image.open(file).convert('RGB')
    
    # Make container with three columns 
    with st.container():
        # Divide container in three
        col1, col2, col3 = st.columns(3)
        
        with col2:
            # Display Image in app
            st.image(image, use_column_width=True)
            
    # Classify image
    class_name, prob = classify(image, aug_model, class_names)
    
    probability = round(prob*100, 2)
    
    # Write classification
    st.write(f"### The Brain MRI image is most likely a {class_name[0]} instance")
    st.write(f"### The probability that the image is a {class_name[0]} instance is: {probability}%")

#### Make a section talking about the model 

# Make Section Header
st.header('Model Information', divider='blue')

# Make Secondary Header 
st.write("## Performance of Testing Data")

# Display Confusion matrix
st.image("images/confusion_matrix_augmented.png", use_column_width=True)