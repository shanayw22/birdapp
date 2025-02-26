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

# Download the model.zip file from Google Drive
def download_model_from_drive():
    file_id = '15IRvAg31XQAhn45bUsDXCQRDttIqjSt9'
    download_url = f'https://drive.google.com/uc?id={file_id}'
    output = model_filename
    gdown.download(download_url, output, quiet=False)


# Load the model from the extracted file
def load_model_from_file(model_filename):
    model = load_model(model_filename)
    return model



download_model_from_drive()
class_names = ["American Robin", "Bewick's Wren", "Northern Cardinal", "Northern Mockingbird", "Song Sparrow"]

model = load_model_from_file(model_filename)

def preprocess_image(uploaded_image):
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def make_prediction(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)
    return predicted_class[0]

st.title("Image Classifier")
st.write("Upload an image to classify it using the trained model.")

Image_file = st.file_uploader("Upload an image file (PNG)", type=["png"])

if Image_file is not None:
    st.image(Image_file, caption="Uploaded Image", use_column_width=True)

    img, img_array = preprocess_image(Image_file)

    predicted_class = make_prediction(img_array)

    st.write(f"Predicted class: {class_names[predicted_class]}")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Prediction: {class_names[predicted_class]}")
    st.pyplot(fig)

