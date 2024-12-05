import streamlit as st
from streamlit_image_select import image_select
from keras.models import load_model
from predict import classify, preprocess_image
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os 
import logging
##############################################################################
# # Define root directory for later on in the script 
# def get_project_root():
#     current_dir = os.path.abspath(os.path.dirname(__file__))  # Current script's directory
#     project_root = os.path.abspath(os.path.join(current_dir, ".."))  # Go up one level
#     return project_root

# # Construct the path to the image directory
# root_dir = get_project_root()

# # Title app
# st.title("Brain Lesion Classification with Magnetic Resonance Imaging")

# # Set header for classification
# st.header('Please upload Brain MRI Slice Image')

# ### Cache augemented model
# @st.cache_resource
# def load_keras_model(url, file_path):
#     """Downloads the model from an AWS S3 bucket URL and loads it."""
#     try:
#         loaded_model = load_model(file_path)
#         logging.info("Model loaded successfully from local file.")
#         return loaded_model
#     except Exception as e:
#         logging.error(f"Failed to load model from local file: {e}. Attempting to download.")
#         try:
#             response = requests.get(url)
#             response.raise_for_status()
#             with open(file_path, "wb") as f:
#                 f.write(response.content)
#             logging.info("Model downloaded successfully!")
#             loaded_model = load_model(file_path)
#             logging.info("Model loaded successfully from downloaded file.")
#             return loaded_model
#         except Exception as e:
#             logging.error(f"Error loading Keras model: {e}")
#             return None

# ### Cache Lime explainer
# @st.cache_resource
# def load_lime_explainer(random_state=42):
#     """
#     Load LimeImageExplainer
#     """
#     explainer = lime_image.LimeImageExplainer(random_state=random_state)
#     return explainer

# ### Load classifier from bucket 
# url = "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/op_model1_aug.keras"
# file_path = "op_model1_aug.keras"

# aug_model = load_keras_model(url, file_path)

# # Check if the model was loaded successfully
# if aug_model is None:
#     st.error("Failed to load the model. Please check the file path or URL.")

# # Define Class Names
# class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# # Create Streamlit layout and features
# ##################################################################################################################
# # Function that will pre-process and show LIME explainer 
# def display_prediction(image, model, explainer):
#     """
#     Displays the classification result and LIME explanation for an uploaded image.
#     """
#     # Make container with three columns 
#     with st.container():
#     # Divide container in three
#         col1, col2, col3 = st.columns(3)
    
#     with col2:
#         # Display Image in app
#         st.image(image, caption="Uploaded Image", use_column_width=True)
    
#     # # Display the uploaded image
#     # st.image(image, use_column_width=True, caption="Uploaded Image")

#     # Classify image
#     class_name, prob = classify(image, model, class_names)
#     probability = round(prob * 100, 2)
#     st.write(f"#### The Brain lesion is most likely a {class_name} case")
#     st.write(f"#### The probability associated with {class_name} is: {probability}%")

#     # Lime Explanation
#     with st.expander("See Lime Explanation Mask and Importance Heatmap"):
#         img = preprocess_image(image)
#         explanation = explainer.explain_instance(
#             image=img,
#             classifier_fn=model.predict,
#             top_labels=4,
#             num_samples=1000,
#             hide_color=0,
#             random_seed=42,
#         )
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0],
#             positive_only=True,
#             num_features=5,
#             hide_rest=True,
#             min_weight=0.1,
#         )
#         ind = explanation.top_labels[0]
#         dict_heatmap = dict(explanation.local_exp[ind])
#         heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#         # Plot Lime Mask and Heatmap
#         fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#         axes[0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
#         axes[0].set_title("Concerning Area", fontsize=20)

#         heatmap_plot = axes[1].imshow(heatmap, cmap="RdBu_r", vmin=-heatmap.max(), vmax=heatmap.max())
#         axes[1].set_title("Red = More Concerning; Blue = Less Concerning", fontsize=20)
#         plt.colorbar(heatmap_plot, ax=axes[1])
#         plt.tight_layout()
#         st.pyplot(fig)

# # Add Streamlit session state for managing user actions
# if "prediction_triggered" not in st.session_state:
#     st.session_state["prediction_triggered"] = False

# if "image_source" not in st.session_state:
#     st.session_state["image_source"] = None  # Can be "upload" or "selection"

# if "uploaded_image" not in st.session_state:
#     st.session_state["uploaded_image"] = None

# if "selected_image" not in st.session_state:
#     st.session_state["selected_image"] = None

# # Upload Image File Component
# file = st.file_uploader('Upload a brain MRI image (jpeg, jpg, png)', type=['jpeg', 'jpg', 'png'])

# # Handle uploaded image
# if file:
#     try:
#         # Load the uploaded image
#         uploaded_image = Image.open(file).convert("RGB")
        
#         # Update the session state 
#         st.session_state["image_source"] = "upload"
#         st.session_state["uploaded_image"] = uploaded_image
        
#         # Display uploded image
#         st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
#         # # Add a button to trigger predictions
#         # if st.button("Analyze Uploaded Image"):
#         #     explainer = load_lime_explainer()
#         #     display_prediction(uploaded_image, aug_model, explainer)
#     except Exception as e:
#         st.error(f"Error processing the image: {e}")
#     # st.session_state["image_source"] = "upload"
#     # st.session_state["uploaded_image"] = Image.open(file).convert("RGB")
#     # st.image(st.session_state["uploaded_image"], caption="Uploaded Image", use_column_width=True)
# # if file:
# #     try:
# #         image = Image.open(file).convert("RGB")
# #         explainer = load_lime_explainer()
# #         display_prediction(image, aug_model, explainer)
# #     except Exception as e:
# #         st.error(f"Error processing the image: {e}")
# #######################################################################################################################################

# ### Make another component where if you dont have an image to test you can choose from examples
# # Glioma Example Selection
# st.header("Or Select a Glioma Example Image")

# # Add a placeholder image b/c streamlit component automatically selects an image from selection which is an issue
# placeholder_caption = "Select an image"
# placeholder_image = "https://via.placeholder.com/150?text=Select+Image"

# glioma_img_urls = [
#     placeholder_image, # Placeholder image
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image.jpg",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(89).jpg",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(91).jpg",
# ]

# captions = [placeholder_caption, "Example 1", "Example 2", "Example 3"]

# selected_img = image_select(
#     label="Select Glioma Instance",
#     images=glioma_img_urls,
#     captions=captions,
#     index=0  # Placeholder image selected by default
# )

# # Add a conditional check to see if placeholder is not selected
# # If a valid image is selected, update session state
# if selected_img != placeholder_image:
#     #st.write("Please select a valid image to analyze.")
#     try:
#         # Define the selected image 
#         selected_glioma = Image.open(requests.get(selected_img, stream=True).raw)
        
#         # Updated session state 
#         st.session_state["image_source"] = "selected"
#         st.session_state["selected_image"] = selected_glioma
        
#         # Display the selected image
#         st.image(selected_img, caption="Selected Image", use_column_width=True)

#     # Define error if it occurs 
#     except Exception as e:
#         st.error(f"Errror processing the selected image: {e}")

# # Add a button to trigger predictions based on upload of selection 
# if st.button("Analyze Image"):
#     if st.session_state["image_source"] == "upload" and st.session_state["uploaded_image"]:
#         image = st.session_state["uploaded_image"]
#     elif st.session_state["image_source"] == "selection" and st.session_state["selected_image"]:
#         image = st.session_state["selected_image"]
#     else:
#         st.error("No valid image provided. Please upload or select an image.")
#         st.stop()

#     # Perform prediction
#     explainer = load_lime_explainer()
#     display_prediction(image, aug_model, explainer)
#     st.session_state["prediction_triggered"] = True
    
#         # if st.button("Analyze Selected Image"):
#         #     try:
#         #         selected_image = Image.open(requests.get(selected_img, stream=True).raw)
#         #         explainer = load_lime_explainer()
#         #         display_prediction(selected_image, aug_model, explainer)
#         #     except Exception as e:
#         #         st.error(f"Error processing the selected image: {e}")
#         # # try:
#         #     # Process image and display it 
#         #     image = Image.open(requests.get(selected_img, stream=True).raw)
#         #     explainer = load_lime_explainer()
#         #     display_prediction(image, aug_model, explainer)
#         # except Exception as e:
#         #     st.error(f"Error processing the selected image: {e}")
#     # if selected_img != placeholder_image:
#     #     st.session_state["image_source"] = "selection"
#     #     st.session_state["selected_image"] = Image.open(requests.get(selected_img, stream=True).raw)
#     #     st.image(st.session_state["selected_image"], caption="Selected Image", use_column_width=True)

#     # if selected_img == placeholder_image:
#     #     st.write("Please select and image to analyze")
#     # else:
#     #     try:
#     #         # Process and display image
#     #         image = Image.open(requests.get(selected_img, stream=True).raw)
#     #         explainer = load_lime_explainer()
#     #         display_prediction(image, aug_model, explainer)
#     #     except Exception as e:
#     #         st.error(f"Error processing the selected image: {e}")

# # # Add a button to explicitly trigger the prediction
# # with st.container():
# #     col1, col2, col3 = st.columns(3)
# #     with col2:
# #         if st.button("Analyze Image"):
# #             if st.session_state["image_source"] == "upload" and "uploaded_image" in st.session_state:
# #                 image = st.session_state["uploaded_image"]
# #             elif st.session_state["image_source"] == "selection" and "selected_image" in st.session_state:
# #                 image = st.session_state["selected_image"]
# #             else:
# #                 st.error("No valid image provided. Please upload or select an image.")
# #                 st.stop()

# #     # Trigger prediction and display results
# #     explainer = load_lime_explainer()
# #     display_prediction(image, aug_model, explainer)
# #     st.session_state["prediction_triggered"] = True
# import os
# import requests
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# from tensorflow.keras.models import load_model

# # Define root directory for later on in the script
# def get_project_root():
#     current_dir = os.path.abspath(os.path.dirname(__file__))  # Current script's directory
#     project_root = os.path.abspath(os.path.join(current_dir, ".."))  # Go up one level
#     return project_root

# # Construct the path to the image directory
# root_dir = get_project_root()

# # Title app
# st.title("Brain Lesion Classification with Magnetic Resonance Imaging")

# # Set header for classification
# st.header("Please upload Brain MRI Slice Image")

# # Cache augmented model
# @st.cache_resource
# def load_keras_model(url, file_path):
#     """Downloads the model from an AWS S3 bucket URL and loads it."""
#     try:
#         loaded_model = load_model(file_path)
#         return loaded_model
#     except Exception:
#         response = requests.get(url)
#         response.raise_for_status()
#         with open(file_path, "wb") as f:
#             f.write(response.content)
#         loaded_model = load_model(file_path)
#         return loaded_model

# # Cache Lime explainer
# @st.cache_resource
# def load_lime_explainer(random_state=42):
#     """Load LimeImageExplainer."""
#     return lime_image.LimeImageExplainer(random_state=random_state)

# # Load classifier from bucket
# url = "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/op_model1_aug.keras"
# file_path = "op_model1_aug.keras"
# aug_model = load_keras_model(url, file_path)

# # Define Class Names
# class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# # Function to pre-process and show LIME explainer
# def display_prediction(image, model, explainer):
#     """Displays the classification result and LIME explanation for an uploaded image."""
#     # Display Image in app
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Classify image
#     class_name, prob = classify(image, model, class_names)
#     probability = round(prob * 100, 2)
#     st.write(f"#### The Brain lesion is most likely a {class_name} case")
#     st.write(f"#### The probability associated with {class_name} is: {probability}%")

#     # Lime Explanation
#     with st.expander("See Lime Explanation Mask and Importance Heatmap"):
#         img = preprocess_image(image)
#         explanation = explainer.explain_instance(
#             image=img,
#             classifier_fn=model.predict,
#             top_labels=4,
#             num_samples=1000,
#             hide_color=0,
#             random_seed=42,
#         )
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0],
#             positive_only=True,
#             num_features=5,
#             hide_rest=True,
#             min_weight=0.1,
#         )
#         ind = explanation.top_labels[0]
#         dict_heatmap = dict(explanation.local_exp[ind])
#         heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#         # Plot Lime Mask and Heatmap
#         fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#         axes[0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
#         axes[0].set_title("Concerning Area", fontsize=20)

#         heatmap_plot = axes[1].imshow(heatmap, cmap="RdBu_r", vmin=-heatmap.max(), vmax=heatmap.max())
#         axes[1].set_title("Red = More Concerning; Blue = Less Concerning", fontsize=20)
#         plt.colorbar(heatmap_plot, ax=axes[1])
#         plt.tight_layout()
#         st.pyplot(fig)

# # Initialize session states
# if "image_source" not in st.session_state:
#     st.session_state["image_source"] = None  # Can be "upload" or "selection"

# if "uploaded_image" not in st.session_state:
#     st.session_state["uploaded_image"] = None

# if "selected_image" not in st.session_state:
#     st.session_state["selected_image"] = None

# # Upload Image File Component
# file = st.file_uploader("Upload your MRI image", type=["jpeg", "jpg", "png"])

# # Process uploaded file
# if file:
#     uploaded_image = Image.open(file).convert("RGB")
#     st.session_state["uploaded_image"] = uploaded_image
#     st.session_state["image_source"] = "upload"
#     st.session_state["selected_image"] = None  # Clear selected image
#     st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# # Glioma Example Selection
# st.header("Or Select a Glioma Example Image")

# glioma_img_urls = [
#     "https://via.placeholder.com/150?text=Select+Image",  # Placeholder image
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image.jpg",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(89).jpg",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(91).jpg",
# ]

# captions = ["Select Image", "Example 1", "Example 2", "Example 3"]

# selected_img = st.selectbox("Select Glioma Instance", options=glioma_img_urls, format_func=lambda x: captions[glioma_img_urls.index(x)])

# if selected_img != glioma_img_urls[0]:  # If a valid example is selected
#     selected_image = Image.open(requests.get(selected_img, stream=True).raw)
#     st.session_state["selected_image"] = selected_image
#     st.session_state["image_source"] = "selection"
#     st.session_state["uploaded_image"] = None  # Clear uploaded image
#     st.image(selected_image, caption="Selected Image", use_column_width=True)

# # Trigger prediction
# if st.session_state["image_source"] == "upload" and st.session_state["uploaded_image"]:
#     if st.button("Analyze Uploaded Image"):
#         explainer = load_lime_explainer()
#         display_prediction(st.session_state["uploaded_image"], aug_model, explainer)

# elif st.session_state["image_source"] == "selection" and st.session_state["selected_image"]:
#     if st.button("Analyze Selected Image"):
#         explainer = load_lime_explainer()
#         display_prediction(st.session_state["selected_image"], aug_model, explainer)
# else:
#     st.write("Please upload or select an image to analyze.")

# import os
# import requests
# from PIL import Image
# import streamlit as st
# import numpy as np
# from keras.models import load_model
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import matplotlib.pyplot as plt

# # Function to get the root directory
# def get_project_root():
#     current_dir = os.path.abspath(os.path.dirname(__file__))
#     return os.path.abspath(os.path.join(current_dir, ".."))
# root_dir = get_project_root()

# # Load model (cached for performance)
# @st.cache_resource
# def load_keras_model(url, file_path):
#     try:
#         return load_model(file_path)
#     except:
#         response = requests.get(url)
#         with open(file_path, "wb") as f:
#             f.write(response.content)
#         return load_model(file_path)

# @st.cache_resource
# def load_lime_explainer(random_state=42):
#     return lime_image.LimeImageExplainer(random_state=random_state)

# # Model URL and loading
# url = "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/op_model1_aug.keras"
# file_path = "op_model1_aug.keras"
# aug_model = load_keras_model(url, file_path)

# if aug_model is None:
#     st.error("Failed to load the model.")

# class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# # Display prediction function
# def display_prediction(image, model, explainer):
#     col1, col2, col3 = st.columns(3)
#     with col2:
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#     class_name, prob = classify(image, model, class_names)
#     st.write(f"#### Likely diagnosis: {class_name} ({round(prob * 100, 2)}%)")

#     with st.expander("See Explanation"):
#         img = preprocess_image(image)
#         explanation = explainer.explain_instance(
#             img, model.predict, top_labels=4, num_samples=1000
#         )
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True
#         )
        
#         ind = explanation.top_labels[0]
#         dict_heatmap = dict(explanation.local_exp[ind])
#         heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        
#         # Plot Lime Mask and Heatmap
#         fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#         axes[0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
#         axes[0].set_title("Concerning Area", fontsize=20)

#         heatmap_plot = axes[1].imshow(heatmap, cmap="RdBu_r", vmin=-heatmap.max(), vmax=heatmap.max())
#         axes[1].set_title("Red = More Concerning; Blue = Less Concerning", fontsize=20)
#         plt.colorbar(heatmap_plot, ax=axes[1])
#         plt.tight_layout()
#         st.pyplot(fig)

# # Placeholder and example images
# glioma_img_urls = [
#     "https://via.placeholder.com/150?text=Select+Image",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image.jpg",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(89).jpg",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(91).jpg",
# ]
# captions = ["Select Image", "Example 1", "Example 2", "Example 3"]

# # Session state management
# if "uploaded_image" not in st.session_state:
#     st.session_state.uploaded_image = None
# if "selected_image" not in st.session_state:
#     st.session_state.selected_image = None

# st.title("Brain Lesion Classification")

# # Upload Section
# file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
# if file:
#     # Load the uplaod
#     st.session_state.uploaded_image = Image.open(file).convert("RGB")
    
#     # Reset session state to None whenever a new umage is uploaded
#     st.session_state.selected_image = None  # Clear selected example

# # Example Selection for Glioma
# selected_img = image_select(
#     "Select Glioma Instance", images=glioma_img_urls, captions=captions, index=0
# )
# if selected_img != glioma_img_urls[0]:  # Exclude placeholder
#     st.session_state.selected_image = Image.open(requests.get(selected_img, stream=True).raw)
#     st.session_state.uploaded_image = None  # Clear uploaded image

# # Trigger Predictions
# if st.button("Analyze Image"):
#     image_to_analyze = st.session_state.uploaded_image or st.session_state.selected_image
    
#     # Check if the placeholder image is selected
#     if selected_img == glioma_img_urls[0]:
#         st.error("Please upload an image or select a valid example.")
#     elif image_to_analyze:
#         explainer = load_lime_explainer()
#         display_prediction(image_to_analyze, aug_model, explainer)
#     else:
#         st.error("Please upload or select an image.")

# import os
# import requests
# from PIL import Image
# import streamlit as st
# import numpy as np
# from keras.models import load_model
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import matplotlib.pyplot as plt

# # Function to get the root directory
# def get_project_root():
#     current_dir = os.path.abspath(os.path.dirname(__file__))
#     return os.path.abspath(os.path.join(current_dir, ".."))
# root_dir = get_project_root()

# # Load model (cached for performance)
# @st.cache_resource
# def load_keras_model(url, file_path):
#     try:
#         return load_model(file_path)
#     except:
#         response = requests.get(url)
#         with open(file_path, "wb") as f:
#             f.write(response.content)
#         return load_model(file_path)

# @st.cache_resource
# def load_lime_explainer(random_state=42):
#     return lime_image.LimeImageExplainer(random_state=random_state)

# # Model URL and loading
# url = "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/op_model1_aug.keras"
# file_path = "op_model1_aug.keras"
# aug_model = load_keras_model(url, file_path)

# if aug_model is None:
#     st.error("Failed to load the model.")

# class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# # Display prediction function
# def display_prediction(image, model, explainer):
#     col1, col2, col3 = st.columns(3)
#     with col2:
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#     class_name, prob = classify(image, model, class_names)
#     st.write(f"#### Likely diagnosis: {class_name} ({round(prob * 100, 2)}%)")

#     with st.expander("See Explanation"):
#         img = preprocess_image(image)
#         explanation = explainer.explain_instance(
#             img, model.predict, top_labels=4, num_samples=1000
#         )
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True
#         )
        
#         ind = explanation.top_labels[0]
#         dict_heatmap = dict(explanation.local_exp[ind])
#         heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        
#         # Plot Lime Mask and Heatmap
#         fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#         axes[0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
#         axes[0].set_title("Concerning Area", fontsize=20)

#         heatmap_plot = axes[1].imshow(heatmap, cmap="RdBu_r", vmin=-heatmap.max(), vmax=heatmap.max())
#         axes[1].set_title("Red = More Concerning; Blue = Less Concerning", fontsize=20)
#         plt.colorbar(heatmap_plot, ax=axes[1])
#         plt.tight_layout()
#         st.pyplot(fig)

# # Placeholder and example images
# glioma_img_urls = [
#     "https://via.placeholder.com/150?text=Select+Image",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image.jpg",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(89).jpg",
#     "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(91).jpg",
# ]
# captions = ["Select Image", "Example 1", "Example 2", "Example 3"]

# # Session state management
# if "uploaded_image" not in st.session_state:
#     st.session_state.uploaded_image = None
# if "selected_image" not in st.session_state:
#     st.session_state.selected_image = None

# st.title("Brain Lesion Classification")

# # Upload Section
# file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
# if file:
#     # Upload the new image
#     st.session_state.uploaded_image = Image.open(file).convert("RGB")
    
#     # Reset selected image to None whenever a new image is uploaded
#     st.session_state.selected_image = None  
    
#     # Clear the model and explainer cache
#     st.cache_resource.clear()  

# # Example Selection
# selected_img = image_select(
#     "Select Glioma Instance", images=glioma_img_urls, captions=captions, index=0
# )
# if selected_img != glioma_img_urls[0]:  # Exclude placeholder
#     st.session_state.selected_image = Image.open(requests.get(selected_img, stream=True).raw)
#     st.session_state.uploaded_image = None  # Clear uploaded image if an example is selected

#     # Clear the model and explainer cache
#     st.cache_resource.clear()  

# # Trigger Predictions
# if st.button("Analyze Image"):
#     # Use the uploaded or selected image for prediction
#     image_to_analyze = st.session_state.uploaded_image or st.session_state.selected_image
    
#     # If neither image is available, prompt the user to upload or select
#     if image_to_analyze:
#         explainer = load_lime_explainer()
#         display_prediction(image_to_analyze, aug_model, explainer)
#     else:
#         st.error("Please upload or select an image.")

import os
import requests
from PIL import Image
import streamlit as st
import numpy as np
from keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
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
    st.write(f"#### Likely diagnosis: {class_name} ({round(prob * 100, 2)}%)")

    with st.expander("See Explanation"):
        img = preprocess_image(image)
        explanation = explainer.explain_instance(
            img, model.predict, top_labels=4, num_samples=1000
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
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image.jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(89).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Glioma_ex/image(91).jpg",
    ],
    "meningioma": [
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Meningioma_ex/image(109).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Meningioma_ex/image(16).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Meningioma_ex/Te-me_0057.jpg",
    ],
    "pituitary": [
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Pituitary_ex/image(45).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Pituitary_ex/Te-pi_0036.jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Pituitary_ex/Te-pi_0298.jpg",
    ],
    "no_tumor": [
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Notumor_ex/image(17).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Notumor_ex/image(5).jpg",
        "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/Notumor_ex/image(7).jpg"
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

st.title("Brain Lesion Classification")

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
        st.image(image_path_arch, use_column_width=True)
    else:
        st.error("Model architecture image not found.")

    st.subheader("Performance on Testing Data")
    st.write(
        "The testing data originates from three Kaggle datasets. The model achieved a 92% accuracy on unseen data. "
        "Below is the data distribution and confusion matrix."
    )
    image_path_dist = os.path.join(root_dir, "images/merged_dist.png")
    if os.path.exists(image_path_dist):
        st.image(image_path_dist, caption="Data Distribution", use_column_width=True)
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
        st.image(image_path_cm, caption="Confusion Matrix", use_column_width=True)
    else:
        st.error("Confusion matrix image not found.")

display_model_info()