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
    if prob == 0:
        return image
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


st.title('Autoencoder image denoiser')

stats_for_nerds = st.toggle('Stats for nerds')

if stats_for_nerds:
    st.write(f"{tf.__version__=}")
    st.write(f"model: model_1500_400.keras")


md_intro = """*_Autoencoding_* is a data compression algorithm where the compression and decompression functions are:

**Data-specific**

Which means that they will only be able to compress data similar to what they have been trained on.

**Lossy**

Which means that the decompressed outputs will be degraded compared to the original inputs

**Learned automatically (unsupervised)**

Which is a useful property: it means that it is easy to train specialized instances of the algorithm that will perform well on a specific type of input. It doesn't require any new engineering, just appropriate training data.

**Today two interesting practical applications of autoencoders are data denoising and dimensionality reduction for data visualization.** [1]
"""
st.markdown(md_intro)
st.image("https://raw.githubusercontent.com/fer-lr/mscs-ml/main/images/autoencoder_schema.jpg")

model = load_model("model/model_1500_400.keras")

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
imageeee = imageeee[None,...]
imageeee = imageeee / 255.0

#images = np.array(images)

col1, col2 = st.columns(2)
with col1: 
    st.subheader("Noisy Image")
    st.image(noisy_imagee)
    mse_noisy_original = mse(imageeee, noisy_imagee)/100
    if stats_for_nerds:
        st.write(noisy_imagee.shape)
        st.write("Noisy vs. Original MSE:", mse_noisy_original)
    st.write("Likeness:", '{:.4%}'.format(1 - mse_noisy_original))

with col2:
    st.subheader("Cleaned Image")
    prediction = model.predict(noisy_imagee)
    st.image(prediction)
    mse_processed_original = mse(imageeee, prediction)/100
    if stats_for_nerds:
        st.write(prediction.shape)
        st.write("Cleaned vs. Original MSE:", mse_processed_original)
    st.write("Likeness:", '{:.4%}'.format(1 - mse_processed_original))

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

st.subheader("The process")
st.markdown("#### Image selection")

col3, col4 = st.columns([6,2],gap="medium")

with col3:
    st.write("A sample of **3,004 256x256** images of celebrity faces from a bank of 30,000 was selected to train and validate the model.")

with col4:
    st.caption("Collection dimensions: ") 
    st.write((3004, 256, 256, 3))
st.image(image="images/content/initial_sample.png", caption="Image Bank Sample")

st.markdown("#### Noise Introduction")

col5, col6 = st.columns([6,2], gap = "medium")
with col5:
    st.write("Salt & pepper noise following a Gaussian distribution was introduced into copies of the training split.")
    st.code('''import random
import cv2

def sp_noise(image,prob):
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
    return output''',language="python")
    st.write("[3]")

with col6:
    st.caption("Split dimensions")
    st.write((2403, 256, 256, 3))
    st.write((601, 256, 256, 3))
    st.caption("Training Noise level")
    st.write(0.1)

st.image(image="images/content/clean_sample.png", caption="Clean Train Sample")
st.image(image="images/content/noisy_sample.png", caption="Noisy Train Sample")

st.markdown("#### The Model")

with st.expander("Model as Table"):
    st.text("""Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 256, 256, 3)]     0         
                                                                 
 conv2d (Conv2D)             (None, 256, 256, 64)      1792      
                                                                 
 max_pooling2d (MaxPooling2  (None, 128, 128, 64)      0         
 D)                                                              
                                                                 
 batch_normalization (Batch  (None, 128, 128, 64)      256       
 Normalization)                                                  
                                                                 
 conv2d_1 (Conv2D)           (None, 128, 128, 32)      18464     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 64, 64, 32)        0         
 g2D)                                                            
                                                                 
 batch_normalization_1 (Bat  (None, 64, 64, 32)        128       
 chNormalization)                                                
                                                                 
 conv2d_2 (Conv2D)           (None, 64, 64, 16)        4624      
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 32, 32, 16)        0         
 g2D)                                                            
                                                                 
 conv2d_3 (Conv2D)           (None, 32, 32, 64)        9280      
                                                                 
 up_sampling2d (UpSampling2  (None, 64, 64, 64)        0         
 D)                                                              
                                                                 
 conv2d_4 (Conv2D)           (None, 64, 64, 32)        18464     
                                                                 
 up_sampling2d_1 (UpSamplin  (None, 128, 128, 32)      0         
 g2D)                                                            
                                                                 
 conv2d_5 (Conv2D)           (None, 128, 128, 16)      4624      
                                                                 
 up_sampling2d_2 (UpSamplin  (None, 256, 256, 16)      0         
 g2D)                                                            
                                                                 
 conv2d_6 (Conv2D)           (None, 256, 256, 3)       435       
                                                                 
=================================================================
Total params: 58067 (226.82 KB)
Trainable params: 57875 (226.07 KB)
Non-trainable params: 192 (768.00 Byte)
_________________________________________________________________""")

st.caption("Model plot:")
col7, col8, col9 = st.beta_columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.image(image="images/content/model.png",caption="")

with col3:
    st.write("")

st.divider()

st.subheader("References")