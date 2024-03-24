def process_data(img_dims, batch_size, train_data_dir, test_data_dir, val_data_dir):
    """"
    Pre-processes image data and sets up generators for training, testing, and validation. 
    
    Parameters:
    - img_dims (int): Specify the image dimensions in a single number. ex-> 128 will produce (128, 128).
    - batch_size (int): Provide the batch size the image data generators will produce.
    - train_data_dir (str): Provide the train folder directory.
    - test_data_dir (str): Provide the test folder directory.
    - val_data_dir (str): Provide the validation folder directory.
    
    Returns:
    - train_generator (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): image data generator for training images
    - test_generator (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): image data generator for testing images
    - val_generator (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): image data generator for validation images
    
    Example: 
    >>> train_generator, test_generator, validation_generator = process_data(
        img_dims=128, 
        batch_size=32, 
        train_data_dir="data/train_folder", 
        test_data_dir="data/test_folder", 
        val_data_dir="data/validation_folder"
    )
    """
    # import libraries 
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Set up data generators for training, testing, and validation
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True, 
        seed = 42
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',        
        shuffle=False,
        seed = 42
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        seed = 42
    )
    
    return train_generator, test_generator, val_generator


def data_augmentation(img_dims, batch_size, train_data_dir, test_data_dir, val_data_dir):
    """"
    Pre-processes image data and sets up generators for training, testing, and validation.
    
    Parameters:
    - img_dims (int): Specify the image dimensions in a single number. ex-> 128 will produce (128, 128).
    - batch_size (int): Provide the batch size the image data generators will produce.
    - train_data_dir (str): Provide the train folder directory.
    - test_data_dir (str): Provide the test folder directory.
    - val_data_dir (str): Provide the validation folder directory.
    
    Returns:
    - train_generator (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): image data generator for training images
    - test_generator (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): image data generator for testing images
    - val_generator (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): image data generator for validation images
    
    Image Data Generators Configuration:
    - Training and validation data undergo data augmentation, including rotation, width and height shifts,
      vertical and horizontal flips, and nearest filling mode.
    - Testing data is rescaled without augmentation.
    
    Example: 
    >>> train_generator, test_generator, validation_generator = data_augmentation(img_dims=128, 
    ...    batch_size=32, 
    ...    train_data_dir="data/train_folder", 
    ...    test_data_dir="data/test_folder", 
    ...    val_data_dir="data/validation_folder"
    ... )
    """
    # import libraries 
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Set up data generators for training, testing, and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest',
        )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        vertical_flip=True,
        horizontal_flip=True, 
        fill_mode='nearest'        
        )
    
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        seed=42
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        seed=42
    )
    
    return train_generator, test_generator, val_generator 

def get_callbacks():
    """
    Provides training callbacks that will be used for model training. 
    
    Returns:
    - stop: Early stopping callback
    """
    # Import libraries
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Initialize callbacks 
    stop = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=20, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_delta=0.01, patience=5, verbose=1)
    
    # Initialize callbacks 
    return [stop, reduce_lr]



def train_model(model, train_generator, val_gen, total_epochs):
    """
    Trains a Keras model using provided generators for training and validation.
    
    Parameters:
    - model (tf.keras.Model): provide the compiled model.
    - train_generator (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): provide the image train_generator. 
    - val_generator (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): provide the image validation generator.
    - total_epochs (int): provide the total number of epochs desired for training. 
    
    Returns:
    - history (dict): A dictionary containing training and validation metrics over epochs.
    
    >>> Example: history = train_model(
        model=my_model, 
        train_generator=train_data_generator,
        val_generator=val_data_generator, 
        total_epochs=10
    )
    """
    # import required libraries
    import time
    
    # Record the start time for training all epoch range
    start_time = time.time()

    # Train the model for set epochs
    history = model.fit(x=train_generator, validation_data=val_gen, epochs=total_epochs, callbacks=get_callbacks())

    # Record the end time for the current epoch
    end_time = time.time()

    # Print the total training time
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time} seconds")

    # Return the history
    return history.history

def view_history(dictionary, index):
    """
    Visualize training history metrics using matplotlib.

    Parameters:
    - dictionary (list): A list containing dictionaries with training history metrics.
    - index (int): Index specifying which dictionary to visualize.

    Each dictionary in the list should contain the following keys:
    - 'loss': Training loss values.
    - 'val_loss': Validation loss values.
    - 'accuracy': Training accuracy values.
    - 'val_accuracy': Validation accuracy values.
    - 'recall': Training recall values.
    - 'precision': Training precision values.
    - 'val_recall': Validation recall values.
    - 'val_precision': Validation precision values.

    The function generates subplots for the following metrics:
    1. Loss vs Epoch
    2. Accuracy vs Epoch
    3. Precision vs Epoch
    4. Recall vs Epoch
    
    Example:
    >>> view_history(history_list, 0)
    """
    # import required libraries
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define font sizes
    font_label = 15
    font_title = 20 
    font_ticks = 12
    
    # Make Subplots
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.ravel()
    
    ## Plot the Loss vs Epoch graph
    ax[0].plot(np.arange(1, len(dictionary[index]['loss'])+1), dictionary[index]['loss'], label='Training Loss')
    ax[0].plot(np.arange(1, len(dictionary[index]['val_loss'])+1), dictionary[index]['val_loss'], label='Validation Loss')
    ax[0].set_title("Loss vs Epoch", fontsize=font_title)
    ax[0].set_xlabel('Epoch', fontsize=font_label)
    ax[0].set_ylabel('Loss', fontsize=font_label)
    ax[0].legend(fontsize=font_ticks)
    
    ## Plot the Validation Recall/Precision vs Epoch graph
    ax[1].plot(np.arange(1, len(dictionary[index]['val_recall'])+1), dictionary[index]['val_recall'], label='Validation Recall')
    ax[1].plot(np.arange(1, len(dictionary[index]['val_precision'])+1), dictionary[index]['val_precision'], label='Validation Precision')
    ax[1].set_title("Validation Recall & Precision vs Epoch", fontsize=font_title)
    ax[1].set_xlabel('Epoch', fontsize=font_label)
    ax[1].set_ylabel('Performance', fontsize=font_label)
    ax[1].legend(fontsize=font_ticks)
    
    ## Plot the Train Recall/Precision vs Epoch graph
    ax[2].plot(np.arange(1, len(dictionary[index]['recall'])+1), dictionary[index]['recall'], label='Train Recall')
    ax[2].plot(np.arange(1, len(dictionary[index]['precision'])+1), dictionary[index]['precision'], label='Train Precision')
    ax[2].set_title("Train Recall & Precision vs Epoch", fontsize=font_title)
    ax[2].set_xlabel('Epoch', fontsize=font_label)
    ax[2].set_ylabel('Performance', fontsize=font_label)
    ax[2].legend(fontsize=font_ticks)
    
    ## Plot the Accuracies vs Epoch graph
    ax[3].plot(np.arange(1, len(dictionary[index]['accuracy'])+1), dictionary[index]['accuracy'], label='Train Accuracy')
    ax[3].plot(np.arange(1, len(dictionary[index]['val_accuracy'])+1), dictionary[index]['val_accuracy'], label='Validation Accuracy')
    ax[3].set_title("Accuracy vs Epoch", fontsize=font_title)
    ax[3].set_xlabel('Epoch', fontsize=font_label)
    ax[3].set_ylabel('Performance', fontsize=font_label)
    ax[3].legend(fontsize=font_ticks)
    
    plt.tight_layout()
    plt.show()

def model_evaluate(model, train_gen, test_gen, val_gen):
    """
    Evaluate a Keras model on training, testing, and validation sets.

    Parameters:
    - model (tf.keras.Model): The Keras model to be evaluated.
    - train_gen (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): Image data generator for training set.
    - test_gen (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): Image data generator for testing set.
    - val_gen (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): Image data generator for validation set.

    Returns:
    - results (pd.DataFrame): A DataFrame containing evaluation metrics for each dataset.
      Columns: ['Set', 'Loss', 'Precision', 'Recall', 'Accuracy']

    Example:
    >>> model_evaluate(my_model, train_data_generator, test_data_generator, val_data_generator)
    """
    # import libraries 
    import pandas as pd
    
    columns = ['Set', 'Loss', 'Precision', 'Recall', 'Accuracy']
    results = pd.DataFrame(columns=columns)
    
    # Evaluate on the training set
    train_results = model.evaluate(train_gen)
    train_metrics = ['Train'] + train_results[:]
    results = pd.concat([results, pd.DataFrame([dict(zip(columns, train_metrics))])], ignore_index=True)
    
    # Evaluate on the test set
    test_results = model.evaluate(test_gen)
    test_metrics = ['Test'] + test_results[:]
    results = pd.concat([results, pd.DataFrame([dict(zip(columns, test_metrics))])], ignore_index=True)
    
    # Evaluate on the validation set
    validation_results = model.evaluate(val_gen)
    val_metrics = ['Validation'] + validation_results[:]
    results = pd.concat([results, pd.DataFrame([dict(zip(columns, val_metrics))])], ignore_index=True)
    
    # Lets modify the Precision, Recall, Accuracy to percentages
    results['Precision'] = results['Precision']*100
    results['Recall'] = results['Recall']*100
    results['Accuracy'] = results['Accuracy']*100
    
    return results