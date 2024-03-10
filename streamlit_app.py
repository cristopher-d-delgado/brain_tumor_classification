import streamlit as st
from keras.models import load_model
from predict import classify, preprocess_image
from PIL import Image, ImageOps
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import io

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

# Cache Lime explainer
@st.cache_resource
def load_lime_explainer():
    """
    Load LimeImageExplainer
    """
    explainer = lime_image.LimeImageExplainer(random_state=42)
    return explainer

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

    # Lime Explanation
    with st.expander("See Lime Explanation Mask and Importance Heatmap"):
        with st.container():
            # Divide container into 2 columns
            col_1, col_2 = st.columns(2)
            
            # Load Lime explainer
            explainer = load_lime_explainer()
            
            # Define image we want to predict
            image = Image.open(file)
            
            # Preprocess the image for lime explanation and model prediction
            img = preprocess_image(image)
            
            # Develop local model explanation
            explanation = explainer.explain_instance(
                image=img,
                classifier_fn=aug_model.predict,
                top_labels=4,
                num_samples=2000,
                hide_color=0,
                random_seed=42
            )

            # Obtain mask and image from explainer
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],  # Using the top predicted label for visualization
                positive_only=True,
                num_features=5, 
                hide_rest=True, 
                min_weight=0.1
            )

            # Display mask and image in column 2
            with col_1:
                st.image(mark_boundaries(temp / 2 + 0.5, mask), caption="Lime Mask", use_column_width=True)
            
            # Using the same explainer get a heatmap version that explains the areas that contribute most to that decision
            # Select the top label
            with col_2:
                # Select the top label
                ind = explanation.top_labels[0]
        
                # Map each explanation weight to the corressponding superpixel
                dict_heatmap = dict(explanation.local_exp[ind])
                heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

                # Display heatmap 
                # Display normalized heatmap with colorbar
                plt.figure(figsize=(8,6), facecolor='white')
                plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=15)
                plt.title("Blue = More Important; Red = Less Important", fontsize=20)
                plt.axis("off")  # Hide axes
                plt.show()
                
                # Save plot as bytes
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)

                # Display heatmap using Streamlit
                st.image(buf, caption='Heatmap', use_column_width=True)
                
                # Delete buf object to free up memory
                del buf
        # st.image(mark_boundaries(temp / 2 + 0.5, mask), caption="Lime Explanation", use_column_width=True)
        # st.image(heatmap)
        
        

#### Make a section talking about the model 

# Make Section Header
st.header('Model Information', divider='blue')

# Make Secondary Header 
st.write("## Performance of Testing Data")

# Display Confusion matrix
st.image("images/confusion_matrix_augmented.png", use_column_width=True)