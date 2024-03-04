from keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(img_path, img_dims):
    """Preprocesses a single image based on the same preprocessing steps defined in process_data function.
    
    Parameters:
    - img_path (str): Path to the image file.
    - img_dims (int): Specify the image dimensions in a single number. Ex: 128 will produce (128, 128).
    
    Returns:
    - img_array (numpy.ndarray): Preprocessed image as a numpy array.
    """
    
    # Creat instance of datagenerator
    data_generator = ImageDataGenerator(rescale=1./255)
    
    # Load and preprocess the image using the ImageDataGenerator
    img = data_generator.flow_from_directory(
        directory=img_path,
        target_size=(img_dims, img_dims),
        batch_size=1,
        class_mode=None,  # Since we are not using any labels
        shuffle=False  # No need to shuffle since we're processing a single image
    )
    
    image = next(img)
    
    return image

def predict_image_class(model, img_path, img_dims):
    """Predicts the class of an image using a pre-trained model.
    
    Parameters:
    - model (tf.keras.Model): Pre-trained Keras model.
    - img_path (str): Path to the image file.
    - img_dims (int): Specify the image dimensions in a single number. Ex: 128 will produce (128, 128).
    
    Returns:
    - prediction (numpy.ndarray): Predicted class probabilities.
    """
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path, img_dims)
    
    # Make predictions
    prediction = model.predict(img_array)
    
    return prediction