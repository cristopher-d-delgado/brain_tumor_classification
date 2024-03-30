# import dash components
from dash import Input, Output, State, html, dcc, Dash
# Import warnings to ignore warnings
import warnings
warnings.filterwarnings('ignore')
from io import BytesIO
import numpy as np
from PIL import Image
import base64
# import load_model from keras
import tensorflow as tf
# Import visualization
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
# Import Explainer 
from lime import lime_image
# import requests
import requests

# Function to retrieve image information using requests
def get_image_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print("Failed to retrieve content:", response.status_code)
        return None

# Function to retrieve model from AWS bucket
def load_keras_model(url, file_path):
    """
    url = url to request
    filer_path = file to write to. In other words save the file to.
    """
    s3_url = url
    try:
        # Download the model from AWS bucket
        response = requests.get(s3_url)
        
        # Save the model to the file path
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        # Load the saved model
        loaded_model = tf.keras.models.load_model(file_path)
        return loaded_model
    
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None
# Lets retrieve model from AWS container
url_model = "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/op_model1_aug.keras"
file_path = "op_model1_aug.keras"
model = load_keras_model(url_model, file_path)

# Define style sheet 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create the dash app
app = Dash(__name__, external_stylesheets=external_stylesheets)

# General Model information
markdown_1 = dcc.Markdown("""
    # Brain Tumor Lesion Assessment
""", style={'fontSize': '35px', 'margin': '20px'}
)

markdown_2 = dcc.Markdown("""
    The model integrated into this dashboard has the capability to predict four different brain lesions.
""", style={'fontSize': '20px', 'margin': '20px'}
)

markdown_3 = dcc.Markdown("""
    The four supported brain lesion classifications are Meningioma, Pituitary, Glioma tumors. 
    The fourth possible prediction would be a No Tumor classification.
""", style={'fontSize': '20px', 'margin': '20px'}
)
markdown_4 = dcc.Markdown("""
    ## Lime Image Diagnostic
""", style={'fontSize': '35px', 'margin': '20px'}
)

# Model Information Section
markdown_5 = dcc.Markdown("""
    ## Model Information
""", style={'fontSize': '35px', 'margin': '20px'}
)
markdown_6 = dcc.Markdown("""
    The final model architecture is found in the 'Model Architecture' section below. 
    This model architecture is also the same model that is being used for the model classification that is utilized in this app for image predictions.
    The model uses 4 Convolutional layers, 4 Maxpooling layers, 2 Dropout layers, and 4 Fully Connected layers. The output layer is a 4 neuron output. In order to classify 'no_tumor', 'pituitary', 'meningioma', and 'glioma'.
""", style={'fontSize': '20px', 'margin': '20px'}
)
markdown_7 = dcc.Markdown("""
    The detailed dive into the model training and development can be found in the following 
    [repository]('https://github.com/cristopher-d-delgado/brain_tumor_classification').
""", style={'fontSize': '20px', 'margin': '20px'}
)

# Header for model architecture section
markdown_7 = dcc.Markdown("""
    ### Model Architecture
""", style={'fontSize': '35px', 'margin': '20px'}
)

# Retrieve model image from github repo
github_model_arch = "https://github.com/cristopher-d-delgado/brain_tumor_classification/blob/0e93e6c4dfdf0a01890822db3447f3d7d6cf3873/images/model_arch.jpg?raw=true"
# Retrieve image data
model_arch = get_image_from_github(github_model_arch)
# Create an Img component to display the image
if model_arch:
    img_model_arch = html.Img(src='data:image/jpeg;base64,{}'.format(base64.b64encode(model_arch).decode()),
                            style={'width': '40%', 'height': '40%'})
else:
    img_model_arch = html.Div("Failed to load image from GitHub", style={'color': 'red'})

# Header for the testing confusion Matrix 
markdown_8 = dcc.Markdown(
    """
    ### Performance on testing Data
    """,
style={'fontSize': '24px', 'margin': '20px'}
)
github_confusion_matrix = "https://github.com/cristopher-d-delgado/brain_tumor_classification/blob/0e93e6c4dfdf0a01890822db3447f3d7d6cf3873/images/confusion_matrix_augmented.png?raw=true"
# Retrieve image data
model_confusion_matrix = get_image_from_github(github_confusion_matrix)
# Create an Img component to display the image
if model_confusion_matrix:
    img_confusion = html.Img(src='data:image/jpeg;base64,{}'.format(base64.b64encode(model_confusion_matrix).decode()),
                            style={'width': '40%', 'height': '40%'})
else:
    img_confusion = html.Div("Failed to load image from GitHub", style={'color': 'red'})


# Upload image
upload_img = dcc.Upload(
    id='upload-image',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select File')
    ]),
        style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
    },
    multiple=True, 
)

# Set the layout
app.layout = html.Div(children=[
    html.Div(markdown_1),
    html.Div(markdown_2),
    html.Div(markdown_3),
    html.Div(style={'textAlign': 'center'}, children=[
        html.Div([upload_img, html.Div(id='output-image-upload')]), # Upload image and update image
    ]),
    html.Hr(),
    html.Div(markdown_4),
    html.Div(id='prediction-output', style={'textAlign': 'center', 'margin': '20px'}),
    html.Div(style={'textAlign': 'center'}, children=[
        html.Div(id='lime-container'), # Display Lime Mask and Display Importance Heatmap
    ]),
    html.Hr(), 
    html.Div(markdown_5), # Model Information Section
    html.Div(markdown_6),
    html.Hr(),
    html.Div(markdown_7), # Model Architecture Section
    html.Div(img_model_arch, style={'textAlign': 'center'}), 
    html.Hr(),
    html.Div(markdown_8), # Model Confusion Matrix
    html.Div(img_confusion, style={'textAlign': 'center'}),
])


############################################################################################################################
# function to parse file path 
def parse_contents(contents, filename):
    """
    Parse image object to display it at a consistent size 
    """
    # Decode the content string
    content_type, content_string = contents.split(',')
    # Decode the base64 encoded image
    decoded_image = base64.b64decode(content_string)
    # Open the image using PIL
    img = Image.open(BytesIO(decoded_image))
    # Resize the image to 128x128 pixels
    img_resized = img.resize((200, 200))
    # Convert image to bytes
    img_byte_array = BytesIO()
    img_resized.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()
    # Convert bytes to base64 string
    img_base64 = base64.b64encode(img_byte_array).decode('utf-8')

    return html.Div([
        html.H5(filename),
        # Display the resized image
        html.Img(
            src='data:image/png;base64,' + img_base64, 
            style={'width': '20%', 'height': '20%'},
            ),
        html.Hr()
    ])
    
    # return html.Div([
    #     html.H5(filename),

    #     # HTML images accept base64 encoded strings in the same format
    #     # that is supplied by the upload
    #     html.Img(src=contents),
    #     html.Hr()
    # ])
# Define callback to change image upload
@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
)
# Define an update function for the uploaded image
def update_output(list_of_contents, list_of_names):
    print("Update Output Callback Triggered")
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        print("Children:", children)
        return children 
    else:
        print("No contents uploaded.")
        return None

# Define the callback to preprocess the image and make predictions
@app.callback(
    [Output('prediction-output', 'children'), 
    Output('lime-container', 'children')],
    [Input('upload-image', 'contents')]
)
# Define function to update the prediction
def update_prediction_output(contents):
    print("Update Prediction Output Callback Triggered")
    if contents is not None:
        max_prediction_label = None
        max_prediction_value = 0
        
        for content in contents:
            content_type, content_string = content.split(',')
            
            # Decode the uploaded image
            decoded_image = base64.b64decode(content_string)
            
            # Preprocess the image
            img = Image.open(BytesIO(decoded_image))
            img = img.convert('RGB') # Convert image to RGB
            img = img.resize((128, 128)) # Resize the image to expected model image dimensions  
            img = np.array(img) / 255.0  # Normalize the image
            
            # Make prediction
            prediction = model.predict(np.expand_dims(img, axis=0))
            
            # Get the index of the class with the highest probability
            max_index = np.argmax(prediction)
            
            # Map the index to the corresponding class label
            if max_index == 0:
                max_prediction_label = 'glioma'
            elif max_index == 1:
                max_prediction_label = 'meningioma'
            elif max_index == 2:
                max_prediction_label = 'no_tumor'
            elif max_index == 3:
                max_prediction_label = 'pituitary'
            
            # Get the probability of the predicted class
            max_prediction_prob = prediction[0, max_index] * 100
            
            # If the max probability for this image is higher, update the label
            if max_prediction_prob > max_prediction_value:
                max_prediction_value = max_prediction_prob
            
            # Generate Lime explanation
            # Load the Lime explainer
            explainer = lime_image.LimeImageExplainer(random_state=42)
            
            # Develop local model explanation
            explanation = explainer.explain_instance(
                image=img,
                classifier_fn=model.predict,
                top_labels=4,
                num_samples=2000,
                hide_color=0,
                random_seed=42
            )
            
            # Obtain mask and image from the explainer
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],  # Using the top predicted label for visualization
                positive_only=True,
                num_features=5,
                hide_rest=True,
                min_weight=0.1
            )
            
            # Obtaining components to Diplay Heatmap on second subplot
            ind = explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
            
            # Create the Lime Mask Figure with the Heatmap in a single Figure
            # Lime Mask
            fig, axes = plt.subplots(1, 2, figsize=(14,6), facecolor='white')
            axes[0].imshow(mark_boundaries(temp / 2 + 0.5, mask)) # Plots image
            axes[0].set_title("Concerning Area", fontsize=20)
            
            # Display heatmap on second subplot
            heatmap_plot = axes[1].imshow(heatmap, cmap='RdBu_r', vmin=-heatmap.max(), vmax=heatmap.max())
            axes[1].set_title("Red = More Concernig; Blue = Less Concerning", fontsize=20)
            axes[1].set_xlim(0, img.shape[1]) # Set x-axis to equal the image width
            axes[1].set_ylim(img.shape[0], 0) # Set y-axis to equal the image height
            colorbar = plt.colorbar(heatmap_plot, ax=axes[1]) # Add colorbar
            
            # Create tight layout for figure
            plt.tight_layout()
            
            # Save the figure as html
            diagnostic_fig = 'diagnostic.png'
            fig.savefig(diagnostic_fig)    
            
            # Save the figure as bytes in memory
            buf = BytesIO()
            fig.savefig(buf, format='jpeg')
            buf.seek(0)
            
            # Encode the bytes as base64
            fig_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Make html Image 
            lime_fig = html.Img(
                src=f'data:image/png;base64, {fig_base64}', 
                style={'width': '40%', 'height': '40%'}
            )
            
        # Return label with highest probablity 
        if max_prediction_label:
            statement = dcc.Markdown(f'''
                ### The predicted lesion of the following image is most likely: {max_prediction_label.capitalize()} 
                ### The associated probability of the predicted lesion is {max_prediction_prob:.2f}%
                ''')
            print("Prediction statement:", statement)
            return statement, lime_fig
        else:
            statement = dcc.Markdown("### Please upload an image above.")
            print("Prediction statement:", statement)
            return statement, []
    else:
        print("No contents uploaded.")
        statement = dcc.Markdown("### Please upload an image above.")
        print("Prediction statement:", statement)
        return statement, []

##############################################################################################################################
if __name__ == '__main__':
    app.run_server(mode='external', host='localhost', port=5000)
