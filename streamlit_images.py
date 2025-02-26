import pandas as pd
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import zipfile
import io
from io import BytesIO
import gdown
import os
from tensorflow.keras.preprocessing import image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Bird Image Classifier", layout="wide")
st.title("🎶 Bird Species Images Classifier from Audio 🎶")
st.markdown("Record bird calls directly in your browser or upload a .wav file!")
model_filename = 'bird_resnet_model.h5'

# Directory to store the model file
model_directory = './models'
model_path = os.path.join(model_directory, model_filename)

# Download and cache the model file
@st.cache_resource
def download_and_load_model():
    # Check if model directory exists
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # If the model file is already present, skip the download
    if not os.path.exists(model_path):
        st.write("Downloading model...")
        file_id = '15IRvAg31XQAhn45bUsDXCQRDttIqjSt9'
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, model_path, quiet=False)
        st.write("Model downloaded successfully!")

    # Load the model from the local path
    model = load_model(model_path)
    return model

# Load the model (this will be cached)
model = download_and_load_model()

# List of class names (this can be defined based on your specific model)
class_names = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 
               'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 
               'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 
               'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 
               'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 
               'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 
               'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 
               'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 
               'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 
               'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 
               'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 
               'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 
               'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 
               'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 
               'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 
               'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 
               'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 
               'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 
               'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 
               'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 
               'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 
               'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 
               'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 
               'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 
               'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 
               'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 
               'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 
               'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 
               'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 
               'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 
               'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 
               'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 
               'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 
               'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 
               'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 
               'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 
               'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 
               'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 
               'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 
               'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 
               'Winter_Wren', 'Common_Yellowthroat']

# Preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = image.load_img(uploaded_image, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  
    return img, img_array

# Make prediction on the image
def make_prediction(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)
    return predicted_class[0]

st.title("Image Classifier")
st.write("Upload an image to classify.")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    img, img_array = preprocess_image(uploaded_image)
    predicted_class = make_prediction(img_array)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {class_names[predicted_class]}')
