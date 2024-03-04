from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageOps, Image
import streamlit as st
import numpy as np

# def preprocess_image_gen(img_dims):
#     """Preprocesses a single image based on the same preprocessing steps defined in process_data function.
    
#     Parameters:
#     - img_path (str): Path to the image file.
#     - img_dims (int): Specify the image dimensions in a single number. Ex: 128 will produce (128, 128).
    
#     Returns:
#     - img_array (numpy.ndarray): Preprocessed image as a numpy array.
#     """
    
#     # Creat instance of datagenerator
#     data_generator = ImageDataGenerator(rescale=1./255)
    
#     # Load and preprocess the image using the ImageDataGenerator
#     img = data_generator.flow_from_directory(
#         directory=img_path,
#         target_size=(img_dims, img_dims),
#         batch_size=1,
#         class_mode=None,  # Since we are not using any labels
#         shuffle=False  # No need to shuffle since we're processing a single image
#     )
    
#     image = next(img)
    
#     return image

# def predict_image_class(model, img_path, img_dims):
#     """Predicts the class of an image using a pre-trained model.
    
#     Parameters:
#     - model (tf.keras.Model): Pre-trained Keras model.
#     - img_path (str): Path to the image file.
#     - img_dims (int): Specify the image dimensions in a single number. Ex: 128 will produce (128, 128).
    
#     Returns:
#     - prediction (numpy.ndarray): Predicted class probabilities.
#     """
#     # Preprocess the image
#     img_array = load_and_preprocess_image(img_path, img_dims)
    
#     # Make predictions
#     prediction = model.predict(img_array)
    
#     return prediction

def classify(image, model, class_names):
    """
    Parameters:
    - image (PIL.Image.Image): An image to be classified
    - model (tensorflow.keras.model): Trained Machine Leanring model for image classification
    - class_names (list): A list of class names corressponding to the classes the model can predict

    Returns:
        A tuple of the predicted class name and probability.
    """
    # Convert image to (128, 128)
    img = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
    
    # Convert image to RGB if it's not already in RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert image to numpy array and normalize
    img_array = np.asarray(img)
    norm_img_array = img_array.astype(np.float32)/ 255.0
    
    # Set model input
    data = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)
    data[0] = norm_img_array
    
    # Make prediction
    prediction = model.predict(data)
    
    # Get the predicted class index and probability
    predicted_class_index = np.argmax(prediction)
    predicted_class_prob = prediction[0][predicted_class_index]
    
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name, predicted_class_prob