import streamlit as st
# from transformers import pipeline
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from preprocess import create_ela_image

IMG_SIZE = (224, 224)
MODEL_KERAS  = "forgery_detector.keras"


st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🕵️",
    layout="centered"
)


# Title & description
st.title("Deepfake Image Detector")
st.write(
    "Upload an image and the model will predict whether it is **REAL** or **FAKE**."
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_KERAS)

model = load_model()


# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    with open("temp_input.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    ela_img = create_ela_image("temp_input.jpg")
    st.image(uploaded_file, caption="Image", clamp=True)

    ela_img = np.expand_dims(ela_img, axis=0)
    pred = model.predict(ela_img)[0][0]

    if pred > 0.5:
        st.success(f"✅ REAL ({pred:.2%})")
    else:
        st.error(f"🚨 FAKE ({1 - pred:.2%})")

