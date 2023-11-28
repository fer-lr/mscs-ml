from io import StringIO
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

# @st.cache_resource()
# def load_model(embed_size, loss_type):
#     return loadPretrainedModel(embed_size, loss_type)

st.title('Autoencode image denoiser')

st.write(tf.__version__)

# # Using "with" notation
# with st.sidebar:
#     st.markdown("Hello world")

st.write("BRIEF DESCRIPTION")
st.image("https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/autoencoder_schema.jpg")

model = load_model("model/model200-2-15.keras")

pickled_model = pickle.load(open('model/Model15.pkl', 'rb'))

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

req = urllib.request.urlopen(selected_image)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
selected_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
selected_image = cv2.cvtColor(selected_image , cv2.COLOR_BGR2RGB)
noisy_image = sp_noise(selected_image, noise_density)

noisy_image = noisy_image[None,...]

st.write(selected_image.shape, noisy_image.shape)


imageeee = np.asarray(Image.open('images/picker/15240.jpg'))
noisy_imagee = sp_noise(imageeee,noise_density)
noisy_images = np.asarray([noisy_imagee])
noisy_images = noisy_images / 255.0

#images = np.array(images)

col1, col2 = st.columns(2)
with col1: 
    st.image(imageeee)
    st.image(noisy_images[0])

with col2:
    prediction = pickled_model.predict(noisy_image)
    st.write(prediction)
    st.write(prediction.shape)
    st.image(model.predict(noisy_imagee))



st.write(mse(selected_image, noisy_image)/100)



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