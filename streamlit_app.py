import streamlit as st
import pandas as pd
from keras.models import load_model

st.title("Brain Tumor Classification with Magnetic Resonance Imaging")

@st.cache_resource
def load_model():
    # Load classifier for mri images
    model = load_model("models/op_model1_aug.keras")
    
    return model
