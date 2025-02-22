import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import zipfile
import os
import base64
import tempfile
import gdown
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Bird Species Classifier", layout="wide")

# Download the model.zip file from Google Drive
def download_model_from_drive():
    file_id = '1C_1VBe1KC_oQqbfgYog1-elQNd6zIrSv'  # Google Drive file ID
    download_url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.zip'
    gdown.download(download_url, output, quiet=False)

# Extract the model from the zip file
def extract_model(zip_filename, model_filename):
    if not os.path.exists(model_filename):  
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            zipf.extract(model_filename, './')

# Load the model from the extracted file
def load_model_from_file(model_filename):
    model = load_model(model_filename)
    return model

# Streamlit cache for model loading
@st.cache_resource
def load_model_from_zip(zip_filename, model_filename):
    if not os.path.exists(model_filename):
        download_model_from_drive()
        extract_model(zip_filename, model_filename)
    
    model = load_model_from_file(model_filename)
    return model

class_names = ["American Robin", "Bewick's Wren", "Northern Cardinal", "Northern Mockingbird", "Song Sparrow"]
zip_filename = 'model.zip'
model_filename = 'best_resnet50.h5'
model = load_model_from_zip(zip_filename, model_filename)

# Convert audio to spectrogram
def audio_to_spectrogram(audio_data):
    try:
        # Load the audio using librosa
        y, sr = librosa.load(audio_data, sr=None)

        # Ensure that the audio data is not empty
        if y.size == 0:
            raise ValueError("Audio data is empty")

        # Compute the Short-Time Fourier Transform (STFT)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        # Create a plot for the spectrogram
        fig, ax = plt.subplots(figsize=(10, 6))
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)
        ax.set_title('Spectrogram')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        
        # Save the figure to a buffer
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)  # Go to the start of the buffer
        
        return buf
    except Exception as e:
        print(f"Error in audio_to_spectrogram: {e}")
        return None

# Function to process and predict using the model
def process_and_predict(audio_data):
    spectrogram_buf = audio_to_spectrogram(audio_data)
    spectrogram_path = "./temp_spectrogram.png"
    with open(spectrogram_path, "wb") as f:
        f.write(spectrogram_buf.getvalue())

    image = load_img(spectrogram_path, color_mode="grayscale", target_size=(224, 224))
    image = img_to_array(image) / 255.0  
    image = np.repeat(image, 3, axis=-1)  # Convert to RGB
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(prediction) * 100  
    
    return predicted_class_name, confidence

# Audio recording and playback
st.title("🎶 Bird Species Classifier from Audio 🎶")
st.markdown("Record bird calls directly in your browser or upload a .wav file!")

# Option to upload .wav file
uploaded_audio = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_audio is not None:
    # Save the uploaded file temporarily to avoid errors
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_audio.getvalue())
        temp_filename = temp_file.name

    try:
        # Load the uploaded .wav file and process it
        st.audio(temp_filename, format='audio/wav')
        
        audio_data, sr = librosa.load(temp_filename, sr=None)
        predicted_class_name, confidence = process_and_predict(audio_data)
        
        # Display the results
        st.markdown(f"### Prediction Result")
        st.markdown(f"#### Bird Species: **{predicted_class_name}**")
        st.markdown(f"#### Confidence: **{confidence:.2f}%**")

        # Provide download link for the audio file
        def get_audio_download_link(audio_data, filename="recording.wav"):
            b64 = base64.b64encode(audio_data).decode()
            href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download the recording</a>'
            return href

        st.markdown(get_audio_download_link(uploaded_audio.getvalue()))
    except Exception as e:
        st.error(f"Error processing the audio file: {e}")
else:
    # Allow the user to record audio
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        # Show the audio player for playback
        st.audio(wav_audio_data, format='audio/wav')

        # Process and classify the audio
        audio_data = np.frombuffer(wav_audio_data, dtype=np.float32)
        predicted_class_name, confidence = process_and_predict(audio_data)

        # Display the results
        st.markdown(f"### Prediction Result")
        st.markdown(f"#### Bird Species: **{predicted_class_name}**")
        st.markdown(f"#### Confidence: **{confidence:.2f}%**")

        # Provide download link for the audio file
        def get_audio_download_link(audio_data, filename="recording.wav"):
            b64 = base64.b64encode(audio_data).decode()
            href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download the recording</a>'
            return href

        st.markdown(get_audio_download_link(wav_audio_data))
