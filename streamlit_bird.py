from st_audiorec import st_audiorec
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

st.set_page_config(page_title="Bird Species Classifier", layout="wide")

# Download the model.zip file from Google Drive
def download_model_from_drive():
    file_id = '1C_1VBe1KC_oQqbfgYog1-elQNd6zIrSv'
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

    prediction = model.predict(image)

    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(prediction) * 100  

    return predicted_class_name, confidence

st.title("🎶 Bird Species Classifier from Audio 🎶")
st.markdown("Record bird calls directly in your browser or upload a .wav file!")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

    audio_data = np.frombuffer(wav_audio_data, dtype=np.float32)
    predicted_class_name, confidence = process_and_predict(audio_data)

    st.markdown(f"### Prediction Result")
    st.markdown(f"#### Bird Species: **{predicted_class_name}**")
    st.markdown(f"#### Confidence: **{confidence:.2f}%**")
