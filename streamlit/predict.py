from PIL import ImageOps, Image
import numpy as np

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

# Preprocess data into array 
def preprocess_image(image, target_size=(128, 128)):
    # Resize image 
    img = image.resize(target_size)
    # Convert to RGB
    img_rgb = img.convert('RGB')
    # Convert to array and normalize
    img_array = np.array(img_rgb) / 255.0
    return img_array