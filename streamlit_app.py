from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image


def get_image_path(img):
    # Create a directory and save the uploaded image.
    file_path = f"data/uploadedImages/{img.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as img_file:
        img_file.write(img.getbuffer())
    return file_path


st.title('Autoencode image denoiser')


# Using "with" notation
with st.sidebar:
    st.markdown("Hello world")

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.subheader('Raw data')
st.write(data)

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]


uploaded_file = st.file_uploader("Choose an image",type=['jpg'])
if uploaded_file is not None:
    bytes_data = get_image_path(uploaded_file)
    st.image(bytes_data)
    file_bytes = np.asarray(bytearray(uploaded_file.read()))
    st.image(bytes_data)
    opencv_image = cv2.imdecode(file_bytes, cv2.COLOR_BGR2RGB)
    # ReSize
    resized = cv2.resize(opencv_image,dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    # ReScale Values
    resized = resized / 255
    st.image(resized)
    im = Image.fromarray(file_bytes)
    st.image(im)

st.bar_chart(hist_values)

st.subheader('Map of all pickups')

st.map(data)
