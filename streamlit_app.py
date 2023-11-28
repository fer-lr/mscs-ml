from io import BytesIO, StringIO
import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from streamlit_image_select import image_select
import urllib.request
import random
import tensorflow as tf
from keras.models import load_model
import pickle
import glob
from PIL import Image
import requests

def get_image_path(img):
    # Create a directory and save the uploaded image.
    file_path = f"data/uploadedImages/{img.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as img_file:
        img_file.write(img.getbuffer())
    return file_path

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

@st.cache_resource()
def load_model():
     return load_model("model/model_1500_400.keras")

st.title('Autoencoder image denoiser')

stats_for_nerds = st.toggle('Stats for nerds')

if stats_for_nerds:
    st.write(f"{tf.__version__=}")
    st.write(f"model: model_1500_400.keras")

# # Using "with" notation
# with st.sidebar:
#     st.markdown("Hello world")

md_intro = """*_Autoencoding_* is a data compression algorithm where the compression and decompression functions are:

1) data-specific
	- Which means that they will only be able to compress data similar to what they have been trained on.
2) lossy
	- Which means that the decompressed outputs will be degraded compared to the original inputs
3) learned automatically from examples rather than engineered by a human.
	- Which is a useful property: it means that it is easy to train specialized instances of the algorithm that will perform well on a specific type of input. It doesn't require any new engineering, just appropriate training data.

> Today two interesting practical applications of autoencoders are **data denoising** and **dimensionality reduction for data visualization**.
"""
st.image("https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/autoencoder_schema.jpg")

model = st.session_state["model"]

#pickled_model = pickle.load(open('model/model_1500_400.pkl', 'rb'))

selected_image = image_select("256x256 celebrity faces sample", ["https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15240.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15241.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15242.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15243.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15244.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15245.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15246.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15247.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15248.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15249.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/15250.jpg",
                                        "https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/picker/mcqeen.jpeg"
                                        ])

noise_density = st.slider('Noise density', 0.0, 1.0, 0.1)

# req = urllib.request.urlopen(selected_image)
# arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
# selected_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
# selected_image = cv2.cvtColor(selected_image , cv2.COLOR_BGR2RGB)
# noisy_image = sp_noise(selected_image, noise_density)

# noisy_image = noisy_image[None,...]

# st.write(selected_image.shape, noisy_image.shape)


response = requests.get(selected_image)
imageeee = np.asarray(Image.open(BytesIO(response.content)))
#imageeee = np.asarray(Image.open('images/picker/15240.jpg'))
noisy_imagee = sp_noise(imageeee,noise_density)
noisy_imagee = noisy_imagee[None, ...]
noisy_imagee = noisy_imagee / 255.0

#images = np.array(images)

col1, col2 = st.columns(2)
with col1: 
    st.subheader("Noisy Image")
    st.image(noisy_imagee)
    if stats_for_nerds:
        st.write(st.write(noisy_imagee.shape))
        st.write(f"Noisy vs. Original MSE: {mse(imageeee, noisy_imagee)/100}")

with col2:
    st.subheader("Cleaned Image")
    prediction = model.predict(noisy_imagee)
    st.image(prediction)
    if stats_for_nerds:
        st.write(st.write(prediction.shape))
        st.write(f"Cleaned vs. Original MSE: {mse(imageeee, noisy_imagee)/100}")

st.divider()

uploaded_file = st.file_uploader("Choose an image",type=['jpg'])
if uploaded_file is not None:
    bytes_data = get_image_path(uploaded_file)
    file_bytes = np.asarray(bytearray(uploaded_file.read()))
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    opencv_image = cv2.cvtColor(opencv_image , cv2.COLOR_BGR2RGB)
    # ReSize
    resized = cv2.resize(opencv_image,dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    # ReScale Values
    resized = resized / 255
    st.image(resized)

st.divider()

st.subheader("References")