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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

st.set_page_config(page_title="Bird Species Classifier", layout="wide")

# Download the model.zip file from Google Drive
def download_model_from_drive():
    # Google Drive file ID (extract this from the shared URL)
    file_id = '1C_1VBe1KC_oQqbfgYog1-elQNd6zIrSv'
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # Download the zip file
    output = 'model.zip'
    gdown.download(download_url, output, quiet=False)

# Extract the model from the zip file
def extract_model(zip_filename, model_filename):
    if not os.path.exists(model_filename):  # Extract only if model doesn't exist
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            zipf.extract(model_filename, './')  # Extract model file to current directory

# Load the model from the extracted file
def load_model_from_file(model_filename):
    # Load the model using Keras
    model = load_model(model_filename)
    return model

# Streamlit cache for model loading
@st.cache_resource
def load_model_from_zip(zip_filename, model_filename):
    # Check if model already exists
    if not os.path.exists(model_filename):
        # Download model from Google Drive if not present
        download_model_from_drive()
        
        # Extract the model from the zip file
        extract_model(zip_filename, model_filename)
    
    # Load the model from the extracted file
    model = load_model_from_file(model_filename)
    
    return model

# Define the class names
class_names = ["American Robin", "Bewick's Wren", "Northern Cardinal", "Northern Mockingbird", "Song Sparrow"]

# Load the model
zip_filename = 'model.zip'
model_filename = 'best_resnet50.h5'

model = load_model_from_zip(zip_filename, model_filename)

# Function to convert audio to spectrogram
def audio_to_spectrogram(audio_data, sr=22050):
    y = np.array(audio_data)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  # Convert to dB
    
    # Plot spectrogram directly into memory
    fig, ax = plt.subplots(figsize=(10, 6))
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Save the plot to a BytesIO object instead of saving to a file
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# Function to process and predict using the model
def process_and_predict(audio_data):
    # Generate spectrogram
    spectrogram_buf = audio_to_spectrogram(audio_data)
    
    spectrogram_path = "./temp_spectrogram.png"
    with open(spectrogram_path, "wb") as f:
        f.write(spectrogram_buf.getvalue())

    # Load the spectrogram image as grayscale and resize it to match the model input size
    image = load_img(spectrogram_path, color_mode="grayscale", target_size=(224, 224))  # Resize to match model input size
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    
    # Convert grayscale to RGB by repeating the grayscale channels
    image = np.repeat(image, 3, axis=-1)  # Convert to RGB
    
    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image)
    
    # Decode the prediction to class label and confidence percentage
    predicted_class_index = np.argmax(prediction)  # Get the index of the highest predicted probability
    predicted_class_name = class_names[predicted_class_index]  # Get the corresponding class name
    confidence = np.max(prediction) * 100  # Convert to percentage
    
    return predicted_class_name, confidence


st.title("🎶 Bird Species Classifier from Audio 🎶")
st.markdown("Classify bird species from your audio recordings or uploaded files.")

# Option to upload audio
audio_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])

if audio_file is not None:
    # Save the uploaded audio file temporarily
    temp_audio_path = "./temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Read the audio file for prediction
    audio_data, sr = librosa.load(temp_audio_path, sr=None)
    predicted_class_name, confidence = process_and_predict(audio_data)
    
    # Show the prediction result with confidence
    st.markdown(f"### Prediction Result")
    st.markdown(f"#### Bird Species: **{predicted_class_name}**")
    st.markdown(f"#### Confidence: **{confidence:.2f}%**")

# Allow the user to record audio directly
st.subheader("Record a Bird Call")

# Streamlit WebRTC setup for real-time audio recording
webrtc_streamer(
    key="bird-recording",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessorBase,
    media_stream_constraints={"audio": True},
    async_processing=True
)
