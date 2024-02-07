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
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        vertical_flip=True,
        horizontal_flip=True, 
        fill_mode='nearest'
        )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
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
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
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


def extract_features_vgg19(model, train_gen, val_gen, test_gen):
    """
    Extracts features from a pre-trained VGG19 model for given data generators.

    Parameters:
    - model (tf.keras.Model): Pre-trained VGG19 model.
    - train_gen (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): Image data generator for training set.
    - val_gen (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): Image data generator for validation set.
    - test_gen (tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory): Image data generator for test set.

    Returns:
    - train_features (numpy.ndarray): Extracted features for the training set.
    - train_labels (numpy.ndarray): Labels for the training set.
    - val_features (numpy.ndarray): Extracted features for the validation set.
    - val_labels (numpy.ndarray): Labels for the validation set.
    - test_features (numpy.ndarray): Extracted features for the test set.
    - test_labels (numpy.ndarray): Labels for the test set.

    The extracted features have shape (number of samples, 4, 4, 512), corresponding to the output shape of the VGG19 model.

    Example:
    >>> train_features, train_labels, val_features, val_labels, test_features, test_labels = extract_features_vgg19(
    ...     model=my_vgg19_model,
    ...     train_gen=train_data_generator,
    ...     val_gen=val_data_generator,
    ...     test_gen=test_data_generator
    ... )
    """
    import numpy as np
    
    # Get the batch size from the data generators
    batch_size = train_gen.batch_size
    
    # Get the number of samples in the training, validation, and test sets
    train_sample_amount = len(train_gen.filenames)
    val_sample_amount = len(val_gen.filenames)
    test_sample_amount = len(test_gen.filenames)

    # Initialize arrays to store features and labels
    train_features = np.zeros(shape=(train_sample_amount, 4, 4, 512)) 
    train_labels = np.zeros(shape=(train_sample_amount))

    val_features = np.zeros(shape=(val_sample_amount, 4, 4, 512)) 
    val_labels = np.zeros(shape=(val_sample_amount))

    test_features = np.zeros(shape=(test_sample_amount, 4, 4, 512)) 
    test_labels = np.zeros(shape=(test_sample_amount))

    # Extract features for the training set
    i = 0
    for inputs_batch, labels_batch in train_gen:
        features_batch = model.predict(inputs_batch)
        train_features[i * batch_size : (i + 1) * batch_size] = features_batch 
        train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= train_sample_amount:
            break

    # Extract features for the validation set
    i = 0
    for inputs_batch, labels_batch in val_gen:
        features_batch = model.predict(inputs_batch)
        val_features[i * batch_size : (i + 1) * batch_size] = features_batch 
        val_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= val_sample_amount:
            break

    # Extract features for the test set
    i = 0
    for inputs_batch, labels_batch in test_gen:
        features_batch = model.predict(inputs_batch)
        test_features[i * batch_size : (i + 1) * batch_size] = features_batch 
        test_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= test_sample_amount:
            break
    
    return train_features, train_labels, val_features, val_labels, test_features, test_labels

def view_history_vgg19(history):
    """
    Visualize performance metrics from a VGG19 model training history.

    Parameters:
    - history (tf.keras.callbacks.History): The training history obtained from model.fit().

    Returns:
    - Plots showing the training and validation performance metrics.

    The function generates subplots for the following metrics:
    1. Loss vs Epoch
    2. Accuracy vs Epoch
    3. Precision vs Epoch
    4. Recall vs Epoch

    Example:
    >>> view_history_vgg19(training_history)
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
    
    # Set background color to white
    fig.patch.set_facecolor('white')
    
    ## Plot the Loss vs Epoch graph
    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title("Loss vs Epoch", fontsize=font_title)
    ax[0].set_xlabel('Epoch', fontsize=font_label)
    ax[0].set_ylabel('Loss', fontsize=font_label)
    ax[0].legend(fontsize=font_ticks)
    
    ## Plot the Accuracy vs Epoch graph
    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title("Accuracy vs Epoch", fontsize=font_title)
    ax[1].set_xlabel('Epoch', fontsize=font_label)
    ax[1].set_ylabel('Accuracy', fontsize=font_label)
    ax[1].legend(fontsize=font_ticks)
    
    ## Plot the Precision vs Epoch graph
    ax[2].plot(history.history['precision'], label='Training Precision')
    ax[2].plot(history.history['val_precision'], label='Validation Precision')
    ax[2].set_title("Precision vs Epoch", fontsize=font_title)
    ax[2].set_xlabel('Epoch', fontsize=font_label)
    ax[2].set_ylabel('Precision', fontsize=font_label)
    ax[2].legend(fontsize=font_ticks)
    
    ## Plot the Recall vs Epoch graph
    ax[3].plot(history.history['recall'], label='Training Recall')
    ax[3].plot(history.history['val_recall'], label='Validation Recall')
    ax[3].set_title("Recall vs Epoch", fontsize=font_title)
    ax[3].set_xlabel('Epoch', fontsize=font_label)
    ax[3].set_ylabel('Recall', fontsize=font_label)
    ax[3].legend(fontsize=font_ticks)
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, train_features, train_labels, val_features, val_labels, test_features, test_labels):
    """
    Evaluate a VGG19 model on training, validation, and test sets using multiple metrics.

    Parameters:
    - model (tf.keras.Model): Pre-trained VGG19 model.
    - train_features (numpy.ndarray): Extracted features for the training set.
    - train_labels (numpy.ndarray): Labels for the training set.
    - val_features (numpy.ndarray): Extracted features for the validation set.
    - val_labels (numpy.ndarray): Labels for the validation set.
    - test_features (numpy.ndarray): Extracted features for the test set.
    - test_labels (numpy.ndarray): Labels for the test set.

    Returns:
    - results (pd.DataFrame): A DataFrame containing evaluation metrics for each dataset.
      Columns: ['Set', 'Loss', 'Accuracy', 'Precision', 'Recall']

    Example:
    >>> evaluation_results = evaluate_model(
    ...     model=my_vgg19_model,
    ...     train_features=train_features,
    ...     train_labels=train_labels,
    ...     val_features=val_features,
    ...     val_labels=val_labels,
    ...     test_features=test_features,
    ...     test_labels=test_labels
    ... )
    """
    
    from sklearn.metrics import precision_score, recall_score
    import pandas as pd
    
    # Initialize an empty DataFrame
    columns = ["Set", "Loss", "Accuracy", "Precision", "Recall"]
    df = pd.DataFrame(columns=columns)

    # Evaluate on the training set
    train_eval = model.evaluate(train_features, train_labels, verbose=0)
    train_predictions = model.predict(train_features)
    train_precision = precision_score(train_labels, (train_predictions > 0.5).astype(int))
    train_recall = recall_score(train_labels, (train_predictions > 0.5).astype(int))

    df = df.append({"Set": "Train", "Loss": train_eval[0], "Accuracy": train_eval[1], "Precision": train_precision, "Recall": train_recall}, ignore_index=True)

    # Evaluate on the validation set
    val_eval = model.evaluate(val_features, val_labels, verbose=0)
    val_predictions = model.predict(val_features)
    val_precision = precision_score(val_labels, (val_predictions > 0.5).astype(int))
    val_recall = recall_score(val_labels, (val_predictions > 0.5).astype(int))

    df = df.append({"Set": "Validation", "Loss": val_eval[0], "Accuracy": val_eval[1], "Precision": val_precision, "Recall": val_recall}, ignore_index=True)

    # Evaluate on the test set
    test_eval = model.evaluate(test_features, test_labels, verbose=0)
    test_predictions = model.predict(test_features)
    test_precision = precision_score(test_labels, (test_predictions > 0.5).astype(int))
    test_recall = recall_score(test_labels, (test_predictions > 0.5).astype(int))

    df = df.append({"Set": "Test", "Loss": test_eval[0], "Accuracy": test_eval[1], "Precision": test_precision, "Recall": test_recall}, ignore_index=True)

    return df