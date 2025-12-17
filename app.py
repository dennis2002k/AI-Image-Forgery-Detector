import streamlit as st
# from transformers import pipeline
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

IMG_SIZE = (224, 224)

def create_ela_image(image_path, quality=90, scale=10):
    original = Image.open(image_path).convert("RGB")
    buffer = "temp.jpg"
    original.save(buffer, 'JPEG', quality=quality)
    compressed = Image.open(buffer)
    ela_image = ImageChops.difference(original, compressed)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale_factor = scale * 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor)
    
    ela_image = ela_image.resize(IMG_SIZE)
    return np.array(ela_image).astype("float32") / 255.0

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
    return tf.keras.models.load_model("forgery_detector2_old.keras")

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

# # Run prediction
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")

#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     with st.spinner("Analyzing image..."):
#         result =  1

#     st.subheader("Prediction")

#     for pred in result:
#         label = pred["label"]
#         score = pred["score"]

#         st.write(f"**{label}**: {score:.2%}")
