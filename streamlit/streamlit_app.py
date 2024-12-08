import pandas as pd
import os
import requests
from PIL import Image
import streamlit as st
import numpy as np
from predict import classify, preprocess_image
from keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
from streamlit_image_select import image_select
import matplotlib.pyplot as plt

# Function to get the root directory
def get_project_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))
root_dir = get_project_root()

# Load model (cached for performance)
@st.cache_resource
def load_keras_model(url, file_path):
    try:
        return load_model(file_path)
    except:
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)
        return load_model(file_path)

@st.cache_resource
def load_lime_explainer(random_state=42):
    return lime_image.LimeImageExplainer(random_state=random_state)

# Model URL and loading
url = "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/op_model1_aug.keras"
file_path = "op_model1_aug.keras"
aug_model = load_keras_model(url, file_path)

if aug_model is None:
    st.error("Failed to load the model.")

class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Display prediction function
def display_prediction(image, model, explainer):
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    class_name, prob = classify(image, model, class_names)
    st.write(f"#### Likely diagnosis: {class_name} ({round(prob * 100, 2)}%) certainty.")

    with st.expander("See Explanation"):
        img = preprocess_image(image)
        explanation = explainer.explain_instance(
            img, model.predict, top_labels=4, num_samples=1500
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True
        )
        
        ind = explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        
        # Plot Lime Mask and Heatmap
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
        axes[0].set_title("Concerning Area", fontsize=20)

        heatmap_plot = axes[1].imshow(heatmap, cmap="RdBu_r", vmin=-heatmap.max(), vmax=heatmap.max())
        axes[1].set_title("Red = More Concerning; Blue = Less Concerning", fontsize=20)
        plt.colorbar(heatmap_plot, ax=axes[1])
        plt.tight_layout()
        st.pyplot(fig)

# Placeholder and example images for the categories
example_images = {
    "glioma": [
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/glioma_ex/gg+(402).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/glioma_ex/gg+(56).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/glioma_ex/gg+(609).jpg",
    ],
    "meningioma": [
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/meningioma_ex/m+(191).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/meningioma_ex/m+(66).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/meningioma_ex/m2+(62).jpg",
    ],
    "pituitary": [
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/pituitary_ex/p+(210).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/pituitary_ex/p+(621).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/pituitary_ex/p+(95).jpg",
    ],
    "no_tumor": [
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/notumor_ex/image(103).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/notumor_ex/image(313).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/notumor_ex/image(82).jpg",
    ],
}

captions = {
    "glioma": ["Example 1", "Example 2", "Example 3"],
    "meningioma": ["Example 1", "Example 2", "Example 3"],
    "pituitary": ["Example 1", "Example 2", "Example 3"],
    "no_tumor": ["Example 1", "Example 2", "Example 3"],
}

# Session state management
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

###########################################################
# App title
st.title("Brain Lesion Classification")

# Add disclaimer
st.write("*Disclaimer: Not all predictions will be correct. The examples provided are the ground truth.")

# Category Selection
category = st.selectbox(
    "Select Category", ["Glioma", "Meningioma", "Pituitary", "No_Tumor"]
)

# Example Selection based on category
if category:
    selected_img = image_select(
        f"Select {category} Example", 
        images=example_images[category.lower()], 
        captions=captions[category.lower()], 
        index=0
    )
    if selected_img:
        st.session_state.selected_category = category
        st.session_state.selected_image = Image.open(requests.get(selected_img, stream=True).raw)

# Trigger Predictions
if st.button("Analyze Image"):
    if st.session_state.selected_image:
        explainer = load_lime_explainer()
        display_prediction(st.session_state.selected_image, aug_model, explainer)
    else:
        st.error(f"Please select an image for {st.session_state.selected_category}.")

#### Model Information Section
# Same as before, or you can add details for the new functionality

#### Make a section talking about the model 
# Model Information Section
def display_model_info():
    st.header("Model Information")
    st.subheader("Model Architecture")
    st.write(
        "The final model architecture consists of 4 convolutional layers, 4 max-pooling layers, 2 dropout layers, "
        "and 4 fully connected layers. The output layer has 4 neurons corresponding to 'no_tumor', 'pituitary', 'meningioma', and 'glioma'."
    )
    st.write(
        "Learn more about the model in this [GitHub repository](https://github.com/cristopher-d-delgado/brain_tumor_classification)."
    )
    image_path_arch = os.path.join(root_dir, "images/model_arch.jpg")
    if os.path.exists(image_path_arch):
        model_arch_image = Image.open(image_path_arch)
        st.image(model_arch_image, use_column_width=True)
    else:
        st.error("Model architecture image not found.")

    st.subheader("Performance on Testing Data")
    st.write(
        "The testing data originates from three Kaggle datasets. The model achieved a 92% accuracy on unseen data. "
        "Below is the data distribution and confusion matrix."
    )
    image_path_dist = os.path.join(root_dir, "images/merged_dist.png")
    if os.path.exists(image_path_dist):
        dist_data = Image.open(image_path_dist)
        st.image(dist_data, caption="Data Distribution", use_column_width=True)
    else:
        st.error("Data distribution image not found.")

    df = pd.DataFrame(
        {
            "Set": ["Training", "Testing", "Validation"],
            "Sensitivity": ["99.94%", "92.37%", "97.92%"],
            "Specificity": ["99.97%", "93.14%", "97.98%"],
            "Accuracy": ["99.97%", "92.37%", "97.92%"],
            "Validation Loss/Generalization": [0.006, 0.584, 0.081],
        }
    )
    st.table(df)

    image_path_cm = os.path.join(root_dir, "images/confusion_matrix_augmented.png")
    if os.path.exists(image_path_cm):
        cm_per = Image.open(image_path_cm)
        st.image(cm_per, caption="Confusion Matrix", use_column_width=True)
    else:
        st.error("Confusion matrix image not found.")

display_model_info()