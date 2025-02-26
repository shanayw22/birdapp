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
import gdown
import os
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from st_audiorec import st_audiorec

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set page configuration
st.set_page_config(page_title="Bird Species Classifier", layout="wide")

# Model file paths
model_filename_image = 'bird_resnet_model.h5'
model_filename_audio = 'best_resnet50.h5'
model_dir = './models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Define your model path
model_path = os.path.join(model_dir, 'image_model.h5')

# Download the model if it doesn't already exist
if not os.path.exists(model_path):
    st.write("Downloading image model...")
    file_id = '15IRvAg31XQAhn45bUsDXCQRDttIqjSt9'
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, model_path, quiet=False)
    st.write("Image model downloaded successfully!")

# Now load the model
model = load_model(model_path)
# Tab options
tab = st.radio("Select the functionality", ("Classify Bird Species from Images", "Classify Bird Species from Audio"))

# Function for loading the image model
@st.cache_resource
def load_image_model():
    model_path = os.path.join(model_directory, model_filename_image)
    if not os.path.exists(model_path):
        st.write("Downloading image model...")
        file_id = '15IRvAg31XQAhn45bUsDXCQRDttIqjSt9'
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, model_path, quiet=False)
        st.write("Image model downloaded successfully!")

    model = load_model(model_path)
    return model

# Function for loading the audio model
@st.cache_resource
def load_audio_model():
    zip_filename = 'model.zip'
    if not os.path.exists(zip_filename):
        st.write("Downloading audio model...")
        file_id = '1uQ66PLifiZUboGR9KMjAlf7TAOJOKSdy'
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, zip_filename, quiet=False)

    if not os.path.exists(model_filename_audio):
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            zipf.extract(model_filename_audio, './')

    model = load_model(model_filename_audio)
    return model

# Image Classification Tab
if tab == "Classify Bird Species from Images":
    st.title("🎶 Bird Species Image Classifier")
    st.markdown("Upload a bird image and get it classified!")

    model_image = load_image_model()

    # List of class names (for image classification)
    class_names_image = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani',
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
               'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird',
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


    def preprocess_image(uploaded_image):
        img = load_img(uploaded_image, target_size=(224, 224)) 
        img_array = img_to_array(img)
        img_array = img_array / 255.0 
        img_array = np.expand_dims(img_array, axis=0)  
        return img, img_array

    def make_prediction(img_array):
        predictions = model_image.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)
        return predicted_class[0]

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        img, img_array = preprocess_image(uploaded_image)
        predicted_class = make_prediction(img_array)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write(f'Prediction: {class_names_image[predicted_class]}')

# Audio Classification Tab
elif tab == "Classify Bird Species from Audio":
    st.title("🎶 Bird Species Classifier from Audio 🎶")
    st.markdown("Record or upload a bird call and get it classified!")

    model_audio = load_audio_model()

    class_names_audio = ["American Robin", "Bewick's Wren", "Northern Cardinal", "Northern Mockingbird", "Song Sparrow"]

    def audio_to_spectrogram(audio_data, sr=22050):
        y = np.array(audio_data)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  

        fig, ax = plt.subplots(figsize=(10, 6))
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)
        ax.set_title('Spectrogram')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

    def process_and_predict(audio_data):
        spectrogram_buf = audio_to_spectrogram(audio_data)

        spectrogram_path = "./temp_spectrogram.png"
        with open(spectrogram_path, "wb") as f:
            f.write(spectrogram_buf.getvalue())

        image = load_img(spectrogram_path, color_mode="grayscale", target_size=(224, 224))
        image = img_to_array(image) / 255.0  
        image = np.repeat(image, 3, axis=-1)  
        image = np.expand_dims(image, axis=0)

        prediction = model_audio.predict(image)

        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names_audio[predicted_class_index]
        confidence = np.max(prediction) * 100  

        return predicted_class_name, confidence

    # Option to upload an audio file
    audio_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])

    def process_and_predict_from_recording(wav_audio_data):
        audio_data = np.frombuffer(wav_audio_data, dtype=np.int16)  
        audio_data = audio_data / np.max(np.abs(audio_data))  
        return process_and_predict(audio_data)

    if audio_file:
        audio_data = audio_file.read()
        predicted_class_name, confidence = process_and_predict_from_recording(audio_data)
        st.audio(audio_file, format="audio/wav")
        st.write(f"Prediction: {predicted_class_name}")
        st.write(f"Confidence: {confidence:.2f}%")

    # Option to record audio directly
    st.markdown("Or, record your audio:")
    audio_data = st_audiorec()

    if audio_data:
        predicted_class_name, confidence = process_and_predict_from_recording(audio_data)
        st.audio(audio_data, format="audio/wav")
        st.write(f"Prediction: {predicted_class_name}")
        st.write(f"Confidence: {confidence:.2f}%")
